
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import pdb
import random
import hydra
import logging
from omegaconf import DictConfig

import torch
import time
import statistics
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from continuum.metrics import Logger
from torch.distributions.multivariate_normal import MultivariateNormal

from tqdm import tqdm
from continual_clip import utils
from continual_clip.models import load_model, sample, ClassIncrementalCLIP, KLDivergenceLoss, shrink_cov
from continual_clip.continual_adapter import ContinualAdapter
from continual_clip.continual_prompt import ContinualPrompt
from continual_clip.classes_names import classes_name_dict

from continual_clip.datasets import build_cl_scenarios
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

import numpy as np
import math
from einops import rearrange, reduce, repeat


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_model_mode(model: ClassIncrementalCLIP, mode: str = 'eval'):

    if mode == 'eval':
        for n, p in model.named_parameters():
            p.requires_grad == False

    return model



def train_text_encoder(cfg, model: ClassIncrementalCLIP, train_dataset, train_loader, task_id: int, device):

    num_known_classes = train_dataset.initial_increment + train_dataset.increment * task_id
    num_previous_classes = train_dataset.initial_increment + train_dataset.increment * (task_id-1) if task_id >= 1 else 0
    num_current_classes = train_dataset.increment if task_id >= 1 else train_dataset.initial_increment

    cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    print('***** Training text encoder *****')
    training_string = ['learnable_text_prompts.' + str(task_id)] if cfg.text_peft_module == 'prompt'\
                                                                else ['learnable_text_adapters.' + str(task_id)]
    
    for n, p in model.named_parameters():
        p.requires_grad = True if any(_s in n for _s in training_string) else False
    
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)

    current_lr = cfg.prompt_lr
    prompt_epochs = cfg.prompt_epochs
    prompt_weight_decay = cfg.prompt_weight_decay

    if cfg.prompt_optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, weight_decay=prompt_weight_decay)
        warmup_epochs = 1
        def warmup_lr_scheduler(optimizer, warmup_epochs, base_lr, target_lr):
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return base_lr + (target_lr - base_lr) * (epoch / warmup_epochs)
                else:
                    return 1.0
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_epochs, base_lr=1e-2, target_lr=1.0)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=prompt_epochs - warmup_epochs, eta_min=0.0)
    
    elif cfg.prompt_optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, weight_decay=prompt_weight_decay)
        prompt_milestones = cfg.prompt_milestones
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, prompt_milestones, gamma=0.1, last_epoch=-1)

    model.cuda()
    model.train()
    loss_ce = torch.tensor(0.).to(device)
    loss_kd = torch.tensor(0.).to(device)

    for i_epoch in range(prompt_epochs):
        loss = torch.tensor(0.0).to(device)
        loss_ce = torch.tensor(0.0).to(device)
        tqdm_loader = tqdm(train_loader)
        batch_id = -1
        current_lr = optimizer.param_groups[0]['lr']
        for inputs, targets, task_ids in tqdm_loader:
            batch_id += 1
            inputs, targets = inputs.to(device), targets.to(device)

            outputs, image_features, text_features, _ = model(inputs, mode='text')
            curr_outputs = outputs[:, num_previous_classes:]
            curr_targets = targets - num_previous_classes
            loss_ce = F.cross_entropy(curr_outputs, curr_targets)

            loss = loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tqdm_loader.set_description(f"Epoch {i_epoch + 1}/{cfg.prompt_epochs} | Loss: {loss.item():.4f} | Loss_ce: {loss_ce.item():.4f} | Loss_kd: {loss_kd.item():.4f} | lr: {current_lr:.5f}")
        
        if cfg.prompt_optimizer == 'sgd':
            if i_epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
        elif cfg.prompt_optimizer == 'adam':
            scheduler.step()

    return model




def train_visual_encoder(cfg, model: ClassIncrementalCLIP, train_dataset, train_loader, task_id: int, device):

    num_known_classes = train_dataset.initial_increment + train_dataset.increment * task_id
    num_previous_classes = train_dataset.initial_increment + train_dataset.increment * (task_id-1) if task_id >= 1 else 0
    num_current_classes = train_dataset.increment if task_id >= 1 else train_dataset.initial_increment

    print('***** Training visual encoder *****')
    if cfg.peft_module == "vpt":
        training_string = ['train_vpt'] 
    elif cfg.peft_module == "adapter":
        training_string = ['s_adapter'] 

    for n, p in model.named_parameters():
        p.requires_grad = True if any(_s in n for _s in training_string) else False
    
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)

    model.cuda()
    model.train()
    
    current_lr = cfg.lr if task_id == 0 else cfg.lr * cfg.lr_scale
    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, weight_decay=5e-4)

    milestones = cfg.milestones
    epochs = cfg.epochs

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
    for i_epoch in range(epochs):
        loss = torch.tensor(0.0).to(device)
        loss_ce = torch.tensor(0.0).to(device)
        loss_kd = torch.tensor(0.0).to(device)
        tqdm_loader = tqdm(train_loader)

        if task_id >0:
            random_class_order_list = list(range(cfg.initial_increment+(task_id-1)*cfg.increment))
            random.shuffle(random_class_order_list)
        batch_id = -1
        for inputs, targets, task_ids in tqdm_loader:
            batch_id += 1
            inputs, targets = inputs.to(device), targets.to(device)

            outputs, image_features, text_features, _ = model(inputs, mode='visual')
            loss_ce = torch.nn.functional.cross_entropy(outputs[:, num_previous_classes:], targets - num_previous_classes)
            
            loss = loss_ce + loss_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tqdm_loader.set_description(f"Epoch {i_epoch + 1}/{cfg.epochs} | Loss: {loss.item():.4f} | Loss_ce: {loss_ce.item():.4f} | Loss_kd: {loss_kd.item():.4f} | lr: {scheduler.get_last_lr()[0]:.4f}")
        
        scheduler.step()

    return model



def extract_visual_features(model:ClassIncrementalCLIP, train_loader, device, mode: str=None):

    print('Extract features')
    sample_data = []
    sample_target = []
    for input, target, task_ids in tqdm(train_loader):
        input, target = input.to(device), target.to(device)
        with torch.no_grad():
            # _ = model.encode_image(input, mode='forward_interm')
            ori_ima_feat = model.encode_image(input, mode=mode)
        sample_data.append(ori_ima_feat)
        sample_target.append(target)
    sample_target = torch.cat(sample_target, dim=0)
    sample_data = torch.cat(sample_data, dim=0)

    return sample_data, sample_target


def train_visual_text_encoder(cfg, model: ClassIncrementalCLIP, train_dataset, train_loader, task_id: int, device):

    num_known_classes = train_dataset.initial_increment + train_dataset.increment * task_id
    num_previous_classes = train_dataset.initial_increment + train_dataset.increment * (task_id-1) if task_id >= 1 else 0
    num_current_classes = train_dataset.increment if task_id >= 1 else train_dataset.initial_increment
    
    cosine_similarity = torch.nn.CosineSimilarity()
    cosine_similarity_3D = torch.nn.CosineSimilarity(dim=-1)

    with torch.no_grad():
        image_features, targets = extract_visual_features(model, train_loader, device, mode=None)
        clip_means = model.get_visual_features(image_features, targets)
        prev_text_features = model.get_text_features(task_id) if task_id >= 1 else None
        if prev_text_features is not None:
            print(prev_text_features.shape)
    torch.cuda.empty_cache()

    print('***** Training visual encoder *****')
    if cfg.peft_module == "vpt":
        visual_training_string = ['train_vpt'] 
    elif cfg.peft_module == "adapter":
        visual_training_string = ['s_adapter'] 

    if cfg.text_peft_module == 'prompt':
        text_training_string = ['learnable_text_prompts.' + str(task_id)] 
    elif cfg.text_peft_module == 'adapter':
        text_training_string = ['learnable_text_adapters.' + str(task_id)]

    lr_scale_patterns = [] if task_id == 0 else ['s_adapter']
    interm_training_string = ['agg_net']
    post_training_string = ['vpr', 'classifier']

    for n, p in model.named_parameters():
        p.requires_grad = True if any(_s in n for _s in 
                                    visual_training_string+\
                                    text_training_string+\
                                    interm_training_string+\
                                    post_training_string)\
                                else False

    param_lr_groups = [{'params': [], 'lr': cfg.lr},        ## visual encoder
                       {'params': [], 'lr': cfg.prompt_lr}, ## text prompt
                       {'params': [], 'lr': cfg.interm_lr}, ## interm attention block
                       {'params': [], 'lr': cfg.post_lr},   ## post attention block
                       {'params': [], 'lr': cfg.lr_scale * cfg.lr}]
    lr_param_dict = {_p['lr']: [] for _p in param_lr_groups}
    for n, p in model.named_parameters():
        if p.requires_grad:
            if any(_s in n for _s in lr_scale_patterns):
                _group_idx = 4
            else:
                if any(_s in n for _s in visual_training_string):
                    _group_idx = 0
                elif any(_s in n for _s in text_training_string):
                    _group_idx = 1
                elif any(_s in n for _s in interm_training_string):
                    _group_idx = 2
                elif any(_s in n for _s in post_training_string):
                    _group_idx = 3
            param_lr_groups[_group_idx]['params'].append(p)
            lr_param_dict[param_lr_groups[_group_idx]['lr']].append(n)
    optimizer = torch.optim.Adam(param_lr_groups, weight_decay=5e-4)

    model.cuda()
    model.train()

    milestones = cfg.milestones
    epochs = cfg.epochs

    torch.autograd.set_detect_anomaly(True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

    criterion_kd = KLDivergenceLoss(kd_temperature=cfg.train_temperature, 
                                    num_previous_classes=num_previous_classes)
    model.agg_net.accumulated_scores = torch.ones(model.pool_size) / model.pool_size

    sum = 0
    for n,p in model.named_parameters():
        if p.requires_grad and 'agg_net' not in n:
            print(n)
            sum += p.numel()
    print(sum / 1e6)
    
    for i_epoch in range(epochs):

        loss = torch.tensor(0.0).to(device)
        loss_stab = torch.tensor(0.0).to(device)
        loss_aux = torch.tensor(0.0).to(device)
        loss_kd = torch.tensor(0.0).to(device)
        loss_gs = torch.tensor(0.0).to(device)

        relevance_scores = None
        tqdm_loader = tqdm(train_loader)

        if task_id >0:
            random_class_order_list = list(range(cfg.initial_increment+(task_id-1)*cfg.increment))
            random.shuffle(random_class_order_list)
        batch_id = -1
        for inputs, targets, task_ids in tqdm_loader:
            batch_id += 1
            inputs, targets = inputs.to(device), targets.to(device)
            
            image_features: torch.Tensor
            text_features: torch.Tensor

            ## stability loss
            curr_targets = targets - num_previous_classes
            outputs, image_features, text_features, ori_image_features = model(inputs, mode='visual_text')  

            curr_outputs = outputs[:, num_previous_classes:]

            loss_tp_stab = F.cross_entropy(curr_outputs, curr_targets)

            loss_vp_stab = torch.tensor(0.).to(device)
            if cfg.visual_proto:
                if model.vp_type == 'VPR':
                    attn_means: torch.Tensor = model.vpr(clip_means, text_features)
                    attn_means = attn_means / attn_means.norm(dim=1, keepdim=True)
                    logits = model.logit_scale.exp() * image_features @ attn_means.t()
                    loss_vp_stab = F.cross_entropy(logits[:, num_previous_classes:], curr_targets)

                    if task_id >= 1:
                        L_kd: torch.Tensor = cosine_similarity(model.visual_prototoypes.detach(), 
                                                            attn_means[:num_previous_classes])
                        loss_vp_stab += torch.mean(1.0 - L_kd)

                elif model.vp_type == 'linear':
                    attn_means = F.normalize(model.classifier.weight, dim=1).to(model.clip_dtype)
                    attn_means = attn_means / attn_means.norm(dim=1, keepdim=True)
                    logits = model.logit_scale.exp() * image_features @ attn_means.t()
                    loss_vp_stab = F.cross_entropy(logits, curr_targets)

            loss_stab = loss_tp_stab + loss_vp_stab

            ## knowledge distillation loss
            if task_id >= 1 and cfg.loss_kd:

                kd_type = 'ta-akt'

                if kd_type == 'clip':
                    image_features_aux = model.encode_image(inputs, mode=None)
                    image_features_aux = image_features_aux / image_features_aux.norm(dim=1, keepdim=True)
                else:
                    ## extract previous features
                    prev_list = []
                    for t in range(min(model.pool_size, task_id)):
                        prev_features, _ = model.encode_image(inputs, 
                                                            mode='task_specific.' + str(t), 
                                                            get_interm_features=True)
                        prev_features = prev_features / prev_features.norm(dim=1, keepdim=True)
                        prev_list.append(prev_features)
                    
                    image_feature_list = torch.stack(prev_list).permute(1, 0, 2) ## b, t, 768
                    
                    if kd_type == 'avg-kd':
                        image_features_aux: torch.Tensor = image_feature_list.mean(dim=1)
                    elif kd_type == 'vanilla-kd':
                        image_features_aux: torch.Tensor = image_feature_list[:, -1, :] ## (t-1)-th model
                    elif kd_type == 'ta-akt':
                        text_feature_list: torch.Tensor = torch.cat([
                            prev_text_features.permute(1, 0, 2), ## k, t-1, 512 
                            text_features.unsqueeze(1),          ## k, 1, 512
                            ], dim=1)[targets]                   ## k, t, 512
                        
                        relevance_scores: torch.Tensor
                        image_features_aux, relevance_scores = model.agg_net(text_feature_list, image_feature_list) ## [b, 768], [b, t - 1]
                    else:
                        raise KeyError('**** Invalid distillation type *****')
                    
                    image_features_aux = model.visual.ln_post(image_features_aux) @ model.visual.proj
                    image_features_aux = image_features_aux / image_features_aux.norm(dim=1, keepdim=True)

                ## auxiliary loss function
                logits_tp_aux = model.logit_scale.exp() * image_features_aux @ text_features.detach().t()
                loss_aux = F.cross_entropy(logits_tp_aux[:, num_previous_classes:], curr_targets)

                # knowledge distillation loss function                
                loss_kd = criterion_kd(image_features, image_features_aux, text_features, curr_targets)

            if cfg.gaussian_sampling and task_id >= 1:
                
                ## gaussian sampling
                sample_size = 10
                smp_inp_list = []
                smp_tgt_list = []

                ## pseudo features
                for class_idx, (class_mean, class_cov) in enumerate(zip(model.feature_mean, model.feature_cov)):
                    m = MultivariateNormal(class_mean.float(), class_cov.float())
                    _smp = m.sample(sample_shape=(sample_size,))
                    smp_inp_list.append(_smp)
                    smp_tgt_list.append(torch.as_tensor([class_idx,] * sample_size, dtype=torch.long))

                smp_inp: torch.Tensor = torch.cat(smp_inp_list).to(torch.half).to(device)
                smp_tgt = torch.cat(smp_tgt_list).to(device)
                smp_inp = torch.cat([image_features.detach(), smp_inp], dim=0)
                smp_tgt = torch.cat([targets, smp_tgt], dim=0)

                smp_inp = smp_inp / smp_inp.norm(dim=1, keepdim=True)

                ## cross entropy
                logits_tp: torch.Tensor = model.logit_scale.exp() * smp_inp @ text_features.t()
                logits_vp: torch.Tensor = model.logit_scale.exp() * smp_inp @ attn_means.t()
                loss_gs = F.cross_entropy(logits_tp, smp_tgt) + F.cross_entropy(logits_vp, smp_tgt)
                
            loss = loss_stab + loss_aux + loss_kd * task_id + loss_gs

            momentum_factor = 0.50
            if relevance_scores is not None and relevance_scores.shape[1] == model.pool_size:
                model.agg_net.accumulated_scores =\
                model.agg_net.accumulated_scores * momentum_factor +\
                relevance_scores.cpu().mean(dim=0) * (1.0 - momentum_factor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tqdm_loader.set_description(f"Epoch {i_epoch + 1}/{cfg.epochs} | Loss: {loss.item():.4f} | Loss_stab: {loss_stab.item():.4f} | Loss_aux: {loss_aux.item():.4f} | Loss_kd: {loss_kd.item():.4f} | Loss_GS: {loss_gs.item():.4f} | lr: {scheduler.get_last_lr()[0]:.4f}")
        
        scheduler.step()

    return model


def save_statistics(cfg, train_dataset, model: ClassIncrementalCLIP, task_id, device):

    sample_loader = DataLoader(train_dataset[task_id], batch_size=128, shuffle=False, num_workers=cfg.num_workers)
    sample_data, sample_target = [], []
    for input, target, _ in tqdm(sample_loader):
        input, target = input.to(device), target.to(device)
        with torch.no_grad():
            features = model.encode_image(input, mode='inference', get_ori_features=True)

        sample_data.append(features)
        sample_target.append(target)

    sample_data = torch.cat(sample_data, dim=0)
    sample_target = torch.cat(sample_target, dim=0)
    model.get_statistics(sample_data, sample_target)

    return model


def run_class_incremental(cfg, device, trial):

    cfg.class_order = utils.get_class_order(os.path.join('SECA_master', 
                                                        'class_orders', 
                                                        str(cfg.dataset)+'_order' + '.yaml'), 
                                            str(cfg.class_order_idx[trial]))
    
    model: ClassIncrementalCLIP = load_model(cfg, device)
    eval_dataset, classes_names = build_cl_scenarios(
        cfg, is_train=False, transforms=model.transforms
    )

    train_dataset, _ = build_cl_scenarios(
        cfg, is_train=True, transforms=model.transforms
    )
    model.classes_names = classes_names
    acc_list = []
    metric_logger = Logger(list_subsets=["test"])
    for task_id, _ in enumerate(eval_dataset):

        logging.info(f"Train for task {task_id} has started.")
        model.adaptation(task_id, threshold=cfg.threshold)

        if cfg.zero_shot == True:
            cfg.peft_module = None
        else:
            train_loader = DataLoader(train_dataset[task_id], 
                                    batch_size=cfg.train_batch_size,
                                    shuffle=True, 
                                    num_workers=cfg.num_workers)

            for sub_module in model.continual_adapters:
                sub_module: ContinualAdapter
                sub_module.training_adaptation(task_id)

            torch.cuda.empty_cache()
            if cfg.co_optimizer:
                cfg.train_visual, cfg.train_text = True, True
                model = train_visual_text_encoder(cfg, model, train_dataset, train_loader, task_id, device)
            else:
                if cfg.train_visual:
                    if task_id == 0 or not cfg.visual_first_session:
                        model = train_visual_encoder(cfg, model, train_dataset, train_loader, task_id, device)
                if cfg.train_text:
                    model = train_text_encoder(cfg, model, train_dataset, train_loader, task_id, device)

            torch.cuda.empty_cache()
            model = set_model_mode(model)

            if cfg.loss_kd:
                update_idx = model.agg_net.accumulated_scores.argmin()
                print(model.agg_net.accumulated_scores)
                for sub_module in model.continual_adapters:
                    sub_module: ContinualAdapter
                    sub_module.inference_adaptation(task_id, update_idx = update_idx)

            if cfg.visual_proto:
                model.build_visual_prototypes()

            torch.cuda.empty_cache()

        if cfg.gaussian_sampling or model.vp_type == 'adapt':
            model = save_statistics(cfg, train_dataset, model, task_id, device)

        model.eval()
        eval_loader = DataLoader(eval_dataset[:task_id + 1], batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        # eval_loader = DataLoader(eval_dataset[:], batch_size=cfg.batch_size, num_workers=cfg.num_workers)

        num, num_correct = 0, 0
        save_img, save_tgt, save_txt = [], [], []
        for inputs, targets, task_ids in eval_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs, image_features, text_features, _ = model(inputs, mode='inference')
                save_img.append(image_features.cpu())
                save_tgt.append(targets.cpu())
                if len(save_txt) == 0:
                    save_txt.append(text_features.cpu())
                torch.nn.functional.softmax(outputs, dim=-1)

            metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")
        torch.cuda.empty_cache()
        
        from pathlib import Path
        save_dir = Path("SERA/save_tensors").resolve()
        print(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        np.save('SERA/save_tensors/img.npy', torch.cat(save_img, dim=0).numpy())
        np.save('SERA/save_tensors/txt.npy', save_txt[0].numpy())
        np.save('SERA/save_tensors/proto.npy', model.visual_prototoypes.cpu().numpy())
        np.save('SERA/save_tensors/tgt.npy', torch.cat(save_tgt, dim=0).cpu().numpy())
        print('Finish saving tensors for visualization')

        acc_list.append(100 * metric_logger.accuracy)
        with open(cfg.log_path, 'a+') as f:
            f.write(json.dumps({
                'task': task_id,
                'acc': round(100 * metric_logger.accuracy, 2),
                'avg_acc': round(100 * metric_logger.average_incremental_accuracy, 2),
                'forgetting': round(100 * metric_logger.forgetting, 6),
                'acc_per_task': [round(100 * acc_t, 2) for acc_t in metric_logger.accuracy_per_task],
                'bwt': round(100 * metric_logger.backward_transfer, 2),
                'fwt': round(100 * metric_logger.forward_transfer, 2),
            }) + '\n')
            metric_logger.end_task()


    with open(cfg.log_path, 'a+') as f:
        f.write(json.dumps({
            'last': round(acc_list[-1], 2), 
            'avg': round(statistics.mean(acc_list), 2)
        }) + '\n')


@hydra.main(config_path='configs/class', config_name='imagenet_a.yaml', version_base="1.1") 
def continual_clip(cfg: DictConfig) -> None:

    seed_everything(cfg.seed)

    cfg.workdir = 'SECA_master'
    logs_dir = os.path.join(cfg.workdir, 'logs', cfg.dataset, cfg.log_suffix)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)

    trials = [0, 1, 2]
    for trial in trials:
        
        cfg.log_path = os.path.join(logs_dir, 'trial='+str(trial)+'.txt')

        utils.save_config(cfg)
        with open(cfg.log_path, 'w+') as f: 
            pass
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if cfg.scenario == "class":
            run_class_incremental(cfg, device, trial)


if __name__ == "__main__":
    continual_clip()