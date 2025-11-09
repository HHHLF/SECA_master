

import copy
import pdb
from omegaconf import DictConfig
import os
import json

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .utils import get_class_ids_per_task, get_class_names
from .continual_adapter import Adapter, ContinualAdapter
from .continual_prompt import VisualPrompt, ContinualPrompt
from clip.model import VisionTransformer, Transformer, ResidualAttentionBlock, LayerNorm
from .template import template_dict

from einops import rearrange, reduce, repeat
import time

def pairwise_distance(x, y):
   
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_mat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(x, y.t(), beta=1, alpha=-2)
        
    return dist_mat


class Mlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

def shrink_cov(cov):
    diag_mean = torch.mean(torch.diagonal(cov))
    off_diag = cov.clone()
    off_diag.fill_diagonal_(0.0)
    mask = off_diag != 0.0
    off_diag_mean = (off_diag*mask).sum() / mask.sum()
    iden = torch.eye(cov.shape[0], device=cov.device)
    alpha1 = 1
    alpha2  = 1
    cov_ = cov + (alpha1*diag_mean*iden) + (alpha2*off_diag_mean*(1-iden))
    return cov_
def sample(mean, cov, size, shrink=False):
    vec = torch.randn(size, mean.shape[-1], device=mean.device)
    if shrink:
        cov = shrink_cov(cov)
    sqrt_cov = torch.linalg.cholesky(cov)
    vec = vec @ sqrt_cov.t()
    vec = vec + mean
    return vec



class VisualPrototypeRefinement(nn.Module):

    def __init__(self, embed_dim: int = 512, device = None, clip_dtype = None):
        super().__init__()

        self.clip_dtype = clip_dtype
        self.embed_dim = embed_dim
        self.ln_text = LayerNorm(self.embed_dim)
        self.ln_visual = LayerNorm(self.embed_dim)

        self.linear_projector = nn.Linear(self.embed_dim, self.embed_dim, bias=False ,device=device)
        self.ln_post = LayerNorm(self.embed_dim)

        self.gamma = math.sqrt(self.embed_dim) * 2

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor):

        text_features = self.ln_text(text_features) ## K,d
        image_features = self.ln_visual(image_features)

        text_features_proj = self.linear_projector(text_features.to(torch.float)).to(self.clip_dtype)
        M = pairwise_distance(text_features_proj, text_features_proj)

        affinity_matrix = torch.softmax(-M / self.gamma, dim=1)
        image_features = affinity_matrix @ image_features
        out: torch.Tensor = self.ln_post(image_features)
        
        return out.to(self.clip_dtype)
    


class KLDivergenceLoss(nn.Module):

    def __init__(self, kd_temperature: float = 1.0, num_previous_classes: int = 0):
        super().__init__()
        self.eps = 1e-7
        self.kd_temperature = kd_temperature
        self.num_previous_classes = num_previous_classes
        self.label_smoothing = False
        self.smoothing_factor = 0.1

    def forward(self, image_features: torch.Tensor, 
                      image_features_aux: torch.Tensor, 
                      text_features: torch.Tensor, 
                      curr_targets: torch.Tensor,
                      per_sample: bool = False):

        q_pred = image_features @ text_features.detach().t() * self.kd_temperature
        p_pred = image_features_aux.detach() @ text_features.detach().t() * self.kd_temperature

        q_pred = torch.softmax(q_pred[:, self.num_previous_classes:], dim=1)
        p_pred = torch.softmax(p_pred[:, self.num_previous_classes:], dim=1).detach()

        if self.label_smoothing:
            p_pred_oh = F.one_hot(curr_targets, text_features.shape[0] - self.num_previous_classes)
            p_pred = self.smoothing_factor * p_pred_oh + (1.0 - self.smoothing_factor) * p_pred

        loss_kl = (p_pred * (p_pred.log() - q_pred.log())).sum(dim=1) ## b

        if per_sample:
            return loss_kl

        return loss_kl.mean()



class IntermAggregation(nn.Module):

    def __init__(self, visual_dim: int = 512, text_dim: int = 512, device = None, clip_dtype = None):
        super().__init__()

        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.device = device
        self.clip_dtype = clip_dtype
        
        self.q_projector = nn.Linear(self.text_dim, self.visual_dim, bias=False, device=self.device)
        self.k_projector = nn.Linear(self.visual_dim, self.visual_dim, bias=False, device=self.device)

        self.ln_q = LayerNorm(self.text_dim)
        self.ln_k = LayerNorm(self.visual_dim)

        self.ln_post = LayerNorm(self.visual_dim)
        self.lambda_ = 1.0 / math.sqrt(self.visual_dim)

        self.accumulated_scores = None


    def forward(self, text_prototypes: torch.Tensor, visual_features: torch.Tensor):
        
        b, T, d = text_prototypes.shape

        ## to torch.float
        text_prototypes = text_prototypes.to(torch.float)
        visual_features = visual_features.to(torch.float)

        text_prototypes = self.ln_q(text_prototypes) ## b, t, 512
        visual_features = self.ln_k(visual_features) ## b, t - 1, 768

        q_features: torch.Tensor = self.q_projector(text_prototypes) ## b, t, 512 --> b, t, 768
        k_features: torch.Tensor = self.k_projector(visual_features) ## b, t - 1, 768
        v_features: torch.Tensor = visual_features

        relevance_score = torch.softmax(torch.bmm(q_features, k_features.transpose(-1, -2)) * self.lambda_, dim=-1) ## b, t, t - 1
        o_features: torch.Tensor = torch.bmm(relevance_score, v_features).mean(dim=1) ## b, t - 1, 768

        o_features = self.ln_post(o_features)
        
        return o_features.to(self.clip_dtype), relevance_score.mean(dim=1).to(self.clip_dtype) ## b, t - 1


class ClassIncrementalCLIP(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.cfg = cfg

        self.prompt_template = template_dict[str(self.cfg.dataset) + "_templates"] if self.cfg.many_templates else self.cfg.prompt_template

        self.device = device
        self.classes_names = None
        model, self.transforms = clip.load(cfg.model_name, device=device, jit=jit)
        self.visual = model.visual
        self.transformer = model.transformer
        self.positional_embedding = model.positional_embedding
        self.token_embedding = model.token_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.logit_scale = model.logit_scale
        # pdb.set_trace()
        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.text_tokens = None
        self.dtype = torch.float16 if cfg.fp16 else torch.float32
        self.clip_dtype = model.dtype

        # old adapter
        self.old_adapter = None
        self.old_edge_samples = []
        self.old_edge_samples_labels = []
        self.old_edge_samples_nearest_labels = []

        # class stat
        self.class_mean_list = []
        self.class_cov_list = []

        self.class_diff = None
        self.nearest_class = None
        self.class_edge_distance = []
        self.mix_b = cfg.mix_bias

        self.adapter_width = 16
        self.pool_size = self.cfg.pool_size
        self.continual_adapters = nn.ModuleList([])
        for (i,blk) in enumerate(self.visual.transformer.resblocks):
            adapter = ContinualAdapter(embed_dim=self.visual.transformer.width, 
                                    middle_dim=self.adapter_width, dropout=0.1, 
                                    pool_size=self.pool_size,
                                    clip_dtype=self.clip_dtype,
                                    ensemble_type=self.cfg.ensemble_type)
            self.continual_adapters.append(adapter)

        self.prompt_len = 4
        self.continual_prompts = nn.ModuleList([])
        for (i,blk) in enumerate(self.visual.transformer.resblocks):
            vpt = ContinualPrompt(prompt_len=self.prompt_len, 
                                embed_dim=self.visual.transformer.width, 
                                clip_dtype=self.clip_dtype)
            self.continual_prompts.append(vpt)

        self.learnable_text_prompts = nn.ParameterList([])
        self.learnable_text_adapters = nn.ModuleList([])
        self.text_prompt_len = 8
        self.learnable_text_prompts_template = 'X ' * self.text_prompt_len + '{}'
        self.text_embed_dim = 512

        ## cross-attention layers
        self.cls_gamma = 0.5
        self.agg_net = IntermAggregation(visual_dim=768, text_dim=512, device=self.device, clip_dtype=self.clip_dtype)
        self.vpr = VisualPrototypeRefinement(embed_dim=512, device=self.device, clip_dtype=self.clip_dtype)

        self.text_interm_features = None
        self.vp_type = 'VPR'

    def encode_text(self, text, prompt=False):
        x: torch.Tensor
        x = self.token_embedding(text).type(self.clip_dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.clip_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


    def encode_leanable_text(self, text: torch.Tensor, task_id: int=0, get_interm_features: bool = False):

        interm_features = []
        x: torch.Tensor = self.token_embedding(text).type(self.clip_dtype)  # [batch_size, n_ctx, d_model]
        
        if self.cfg.text_peft_module == 'prompt':
            prompt = self.learnable_text_prompts[task_id].expand(text.shape[0], -1, -1).to(self.device).type(self.clip_dtype)
            x = torch.cat([x[:, :1, :], prompt, x[:, 1+self.text_prompt_len:, :]], dim=1)

        x = x + self.positional_embedding.type(self.clip_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.transformer(x)

        for (layer_idx,blk) in enumerate(self.transformer.resblocks):
            x = x + blk.attention(blk.ln_1(x))
            if self.cfg.text_peft_module == 'adapter' and layer_idx >= self.cfg.text_start_block and layer_idx <= self.cfg.text_end_block:
                x = x + blk.mlp(blk.ln_2(x)) + self.learnable_text_adapters[task_id][layer_idx](x)
            else:
                x = x + blk.mlp(blk.ln_2(x))

        interm_features.append(x.permute(1, 0, 2).detach()) ## b, 77, 512

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        if get_interm_features:
            return x, interm_features

        return x
    
    def encode_image(self, image, mode:str='visual', 
                    get_interm_features: bool = False,
                    get_ori_features: bool = False):
        
        interm_features = []
        image = image.to(self.clip_dtype)

        ## pre-transformer
        x: torch.Tensor
        x = self.visual.conv1(image)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.visual.class_embedding.to(x.dtype) +\
                    torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2) 

        ## transformer-blocks
        for (layer_idx,blk) in enumerate(self.visual.transformer.resblocks):
            blk: ResidualAttentionBlock
            if self.cfg.peft_module == "vpt" and\
                layer_idx >= self.cfg.start_block and layer_idx <= self.cfg.end_block and mode is not None:

                sub_module: ContinualPrompt = self.continual_prompts[layer_idx]
                x = x + blk.attention(blk.ln_1(sub_module(x, mode)))[:sub_module.train_vpt.pre_tokens]
            else:
                x = x + blk.attention(blk.ln_1(x))

            if self.cfg.peft_module == 'adapter' and\
                layer_idx >= self.cfg.start_block and layer_idx <= self.cfg.end_block and mode is not None:

                ori_x = x + blk.mlp(blk.ln_2(x))
                adapt_x: torch.Tensor = self.continual_adapters[layer_idx](x, mode)                    
                x = ori_x + adapt_x ## 197, b, 768

                interm_features.append(x)
            else:
                x = x + blk.mlp(blk.ln_2(x))

        ## post-transformer
        x = x.permute(1, 0, 2)  # LND -> NLD
        if mode is not None and 'task_specific' in mode:
            return x[:, 0, :].detach(), x[:, 1:, :].detach() ## [b, 768], [b, N-1, 768]

        patch_features = x[:, 1:, :]
        x = self.visual.ln_post(x[:, 0, :])

        if get_ori_features:
            return x @ self.visual.proj
        
        if self.visual.proj is not None:
            x = x @ self.visual.proj

        if get_interm_features:
            return x, patch_features
        
        return x

    
    @torch.no_grad()
    def get_class_name_features(self):
        class_name_features = self.encode_text(self.text_tokens)
        return class_name_features.type(torch.float32)

    def forward(self, image, mode: str=None, target: torch.Tensor = None):

        image = image.type(torch.float16)
        ori_image_features = None
        patch_features = None

        if self.cfg.zero_shot == True:
            text_features: torch.Tensor = self.frozen_text_features.mean(dim=1, keepdim=False) if self.cfg.many_templates\
                                                                                            else self.frozen_text_features
            text_features = text_features
            ori_image_features = self.encode_image(image, mode=None)
            self.cfg.eval_prompt_tool = 'mean'
            self.cfg.post_attention = False
            # text_features: torch.Tensor = self.description_features
        else:
            learnable_text_tokens = self.learnable_text_tokens
            if mode == 'text':
                text_tokens = learnable_text_tokens if self.cfg.text_peft_module == 'prompt' else self.text_tokens
                text_features: torch.Tensor = self.encode_leanable_text(learnable_text_tokens.to(self.device), self.task_id)
                ori_image_features = self.encode_image(image, mode=None)
            elif mode == 'visual':
                text_features: torch.Tensor = self.frozen_text_features
                ori_image_features = self.encode_image(image, mode) 
            elif 'visual_text' in mode:
                text_tokens = learnable_text_tokens if self.cfg.text_peft_module == 'prompt' else self.text_tokens
                text_features, text_interm_features =\
                self.encode_leanable_text(learnable_text_tokens.to(self.device), 
                                          self.task_id,
                                          get_interm_features=True,)
                self.text_interm_features = text_interm_features                
                
                ori_image_features, patch_features = self.encode_image(image, mode, get_interm_features=True)

            elif mode == 'cross_attention':
                ori_image_features, patch_features = self.encode_image(image, mode, 
                                                   get_interm_features=True,
                                                   target = target,
                                                   text_interm_features=self.text_interm_features)
                return ori_image_features, patch_features

            elif mode in ['inference']:
                if self.cfg.train_text == True:
                    text_feature_list: list = []
                    for t in range(self.task_id+1):
                        text_tokens = self.learnable_text_tokens if self.cfg.text_peft_module == 'prompt' else self.text_tokens
                        text_feature_list.append(self.encode_leanable_text(text_tokens.to(self.device), t))
                    text_features: torch.Tensor = torch.nn.functional.normalize(torch.stack(text_feature_list), dim=-1)
                    ori_image_features = self.encode_image(image, mode)    
                else:
                    text_features: torch.Tensor = self.frozen_text_features
                    ori_image_features = self.encode_image(image, mode)   
                    self.cfg.eval_prompt_tool = 'mean' 

        image_features = ori_image_features / ori_image_features.norm(dim=1, keepdim=True)
        temperature = self.cfg.eval_temperature
        
        if mode in ['inference']:
            if self.cfg.eval_prompt_tool == 'exp_mean':

                assert text_features.dim() == 3 and text_features.shape[0] == self.task_id+1 ## T, C, d
                text_features = rearrange(text_features, 't c d -> (t c) d', t=self.task_id+1, c=len(self.current_class_names))
                logits_per_image = temperature * image_features @ text_features.t().type(image_features.dtype)
                logits_per_image = torch.exp(logits_per_image - logits_per_image.max())
                logits_per_image = rearrange(logits_per_image, 'b (t c) -> b t c', 
                                            b=image_features.shape[0], t=self.task_id+1, c=len(self.current_class_names))
                logits_per_image = logits_per_image.mean(dim=1)

            else:
                if self.cfg.eval_prompt_tool == 'mean':
                    text_features = text_features.mean(dim=0, keepdim=False) if text_features.dim() == 3 else text_features
                else: ## 'last'
                    text_features = text_features[-1]
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                logits_per_image = self.logit_scale.exp() * image_features @ text_features.t().type(image_features.dtype)
                
        else:
            text_features = text_features.mean(dim=0, keepdim=False) if text_features.dim() == 3 else text_features
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t().type(image_features.dtype)

        if mode in ['inference'] and self.cfg.post_attention:

            logit_scale = self.logit_scale.exp()
            # logits_per_image_ = temperature * image_features @ self.visual_prototoypes.t().type(image_features.dtype)

            if self.vp_type == 'clip':
                m = self.clip_mean.to(self.device)
            elif self.vp_type == 'adapt':
                m = self.feature_mean / self.feature_mean.norm(dim=1, keepdim=True)
                m = m.to(self.device)
            else:
                m = self.visual_prototoypes
            logits_per_image_ = temperature * image_features @ m.t().type(image_features.dtype)
            logits_per_image_ = torch.exp(logits_per_image_ - logits_per_image_.max())

            probs = self.cls_gamma * logits_per_image +\
                    (1.0 - self.cls_gamma) * logits_per_image_

            return probs, image_features, text_features, ori_image_features

        probs = logits_per_image

        return probs, image_features, text_features, ori_image_features

    def adaptation(self, task_id, threshold=0):

        self.task_id = task_id
        if task_id == 0:
            self.classes_names = [name.replace('_', ' ') for name in self.classes_names]
        # print(self.classes_names)
        
        self.num_previous_classses = len(self.current_class_names)
        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.num_known_classes = len(self.current_class_names)
        self.num_increment_classes = self.num_known_classes - self.num_previous_classses

        ## learnable textual prompts
        prompt_vector = nn.Parameter(torch.zeros(1, self.text_prompt_len, self.text_embed_dim))
        nn.init.normal_(prompt_vector, std=0.02)
        self.learnable_text_prompts.append(prompt_vector)
        self.learnable_text_tokens = clip.tokenize(
            [self.learnable_text_prompts_template.format(c) for c in self.current_class_names]
        ).to(self.device)
        self.learnable_text_end = self.learnable_text_tokens.max(dim=-1)[1]

        ## learnable textual adapters
        text_adapters = nn.ModuleList([])
        for i in range(len(self.transformer.resblocks)):
            adapter = Adapter(embed_dim=512, dropout=0.1, middle_dim=16, clip_dtype=self.clip_dtype)
            text_adapters.append(adapter)
        self.learnable_text_adapters.append(text_adapters)

        if self.vp_type == 'linear':
            self.classifier =\
                nn.Linear(in_features=512, out_features=self.cfg.initial_increment, bias=False) if task_id==0 else \
                nn.Linear(in_features=512, out_features=self.cfg.increment, bias=False)

        # frozen prompts
        if self.cfg.many_templates:
            with torch.no_grad():
                text_tokens = []
                frozen_text_features = []
                for _c in self.current_class_names:
                    cls_text_tokens = clip.tokenize([_t.format(_c)  
                                                    for _t in self.prompt_template]).to(self.device)
                    text_tokens.append(cls_text_tokens)
                    cls_text_features: torch.Tensor = self.encode_text(cls_text_tokens)
                    cls_text_features = cls_text_features / cls_text_features.norm(dim=1, keepdim=True)
                    frozen_text_features.append(cls_text_features.detach())

                self.text_tokens = torch.cat(text_tokens, dim=0)
                self.frozen_text_features: torch.Tensor = torch.cat(frozen_text_features, dim=0)
                # self.frozen_text_features = self.frozen_text_features / self.frozen_text_features.norm(dim=1, keepdim=True)
                self.frozen_text_features = rearrange(self.frozen_text_features, '(c p) d -> c p d', 
                                        p=len(self.prompt_template), c=len(self.current_class_names)).to(self.device)
        else:
            self.text_tokens = clip.tokenize(
                [self.prompt_template.format(_c) for _c in self.current_class_names]
            ).to(self.device)
            frozen_text_features: torch.Tensor = self.encode_text(self.text_tokens)
            self.frozen_text_features = frozen_text_features.detach().to(self.device)

        self.text_end = self.text_tokens.max(dim=-1)[1]
        self.queue_empty = True
        self.hard_pairs = None

    def get_text_features(self, max_task_id):
        
        text_feature_list: list = []
        for t in range(max_task_id):
            text_tokens = self.learnable_text_tokens if self.cfg.text_peft_module == 'prompt' else self.text_tokens
            text_feature_list.append(self.encode_leanable_text(text_tokens.to(self.device), t))
        text_features: torch.Tensor = torch.nn.functional.normalize(torch.stack(text_feature_list), dim=-1)
        ## t, k, 512

        return text_features
    
    def get_visual_features(self, features: torch.Tensor, labels):
        
        mean, cov = [], []
        label = torch.sort(torch.unique(labels))[0]
        for l in label:
            index = torch.nonzero(labels == l).squeeze()
            class_data: torch.Tensor = features[index].cpu()
            ## compute mean
            mean.append(class_data.mean(dim=0))
            ## compute cov
            _cov = torch.cov(class_data.clone().detach().to(torch.float64).T) + torch.eye(class_data.shape[-1]) * 1e-4
            cov.append(_cov)
        
        mean = torch.stack(mean)
        cov = torch.stack(cov)        
        if not hasattr(self, 'clip_mean'):
            self.clip_mean, self.clip_cov = mean, cov
        else:
            self.clip_mean = torch.cat([self.clip_mean, mean], dim=0)
            self.clip_cov = torch.cat([self.clip_cov, cov], dim=0)

        return (self.clip_mean / self.clip_mean.norm(dim=1, keepdim=True)).cuda()
    

    def get_statistics(self, features: torch.Tensor, labels):
        
        mean, cov = [], []
        label = torch.sort(torch.unique(labels))[0]
        for l in label:
            index = torch.nonzero(labels == l).squeeze()
            class_data: torch.Tensor = features[index].cpu()
            ## compute mean
            mean.append(class_data.mean(dim=0))
            ## compute cov
            _cov = torch.cov(class_data.clone().detach().to(torch.float64).T) + torch.eye(class_data.shape[-1]) * 1e-4
            cov.append(_cov)
        
        mean = torch.stack(mean)
        cov = torch.stack(cov)
        if not hasattr(self, 'feature_mean'):
            self.feature_mean, self.feature_cov = mean, cov
        else:
            self.feature_mean = torch.cat([self.feature_mean, mean], dim=0)
            self.feature_cov = torch.cat([self.feature_cov, cov], dim=0)

        return None

    
    def get_old_edge_samples(self, batch_size):
        random_select = torch.randperm(self.old_edge_samples.shape[0])[:batch_size]
        return self.old_edge_samples[random_select], self.old_edge_samples_labels[random_select], self.old_edge_samples_nearest_labels[random_select]

    def build_visual_prototypes(self):
        
        if self.vp_type == 'VPR':
            with torch.no_grad():
                # text_features: torch.Tensor = self.get_text_features(self.task_id)
                text_features: torch.Tensor = self.encode_leanable_text(self.learnable_text_tokens.to(self.device), self.task_id)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                clip_mean = (self.clip_mean / self.clip_mean.norm(dim=1, keepdim=True)).cuda()
                visual_prototoypes: torch.Tensor = self.vpr(clip_mean, 
                                                            text_features)
                visual_prototoypes = visual_prototoypes / visual_prototoypes.norm(dim=1, keepdim=True)

            if not hasattr(self, 'visual_prototoypes'):
                self.visual_prototoypes = visual_prototoypes
            else:
                if not self.cfg.gaussian_sampling:
                    self.visual_prototoypes = torch.cat([self.visual_prototoypes, 
                                                        visual_prototoypes[self.num_previous_classses:]], dim=0)
                else:
                    self.visual_prototoypes = visual_prototoypes

        elif self.vp_type == 'linear':
            if not hasattr(self, 'visual_prototoypes'):
                self.visual_prototoypes = F.normalize(self.classifier.weight.data.to(self.clip_dtype), dim=1)
            else:
                self.visual_prototoypes = torch.cat([self.visual_prototoypes, 
                                                     F.normalize(self.classifier.weight.data.to(self.clip_dtype), dim=1)], dim=0)

        return None


class DomainIncrementalCLIP(nn.Module):
    def __init__(self, cfg, device, jit=False) -> None:
        super().__init__()
        self.model, self.transforms = clip.load(cfg.model_name, device=device, jit=jit)
        self.text_tokens = None
        self.prompt_template = cfg.prompt_template
        self.device = device

    def forward(self, image):
        with torch.no_grad():
            logits_per_image, _ = self.model(image, self.text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return probs

    def tokenize(self, class_names):
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in class_names]
        ).to(self.device)



class TaskAgnosticCLIP(nn.Module):
    pass



def load_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    r"""Load a CLIP model in different continual scenarios.
    
    Arguments:
        cfg (DictConfig): Experiment configurations.
        device (torch.device): Device to train (or) evaluate the model on.
        
    Returns:
        nn.Module: Return scenario specific CLIP model.
    """
    if cfg.scenario == "class":
        return ClassIncrementalCLIP(cfg, device)
    elif cfg.scenario == "domain":
        return DomainIncrementalCLIP(cfg, device)
    elif cfg.scenario == "task-aganostic":
        return TaskAgnosticCLIP(cfg, device)
    else:
        raise ValueError(f"""
            `{cfg.scenarios}` is not a valid scenario, 
            Please choose from ['class', "domain', 'task-agnostic']
        """)
    
