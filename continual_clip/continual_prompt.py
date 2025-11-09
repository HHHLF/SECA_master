import torch
import torch.nn as nn
import copy
import math


class VisualPrompt(nn.Module):
    def __init__(self, prompt_len: int, 
                embed_dim: int = 768, 
                prompt_init: str = 'uniform', 
                replace_token: bool = False,
                clip_dtype = None):
        
        super().__init__()
        assert prompt_len >= 0
        self.prompt_len = prompt_len
        self.embed_dim = embed_dim
        self.prompt_init = prompt_init
        self.replace_token = replace_token
        self.clip_dtype = clip_dtype

        if prompt_init == 'zero':
            self.prompt = nn.Parameter(torch.zeros(1, prompt_len, embed_dim))
        elif prompt_init in ('uniform', ''):
            self.prompt = nn.Parameter(torch.randn(1, prompt_len, embed_dim))
            nn.init.uniform_(self.prompt, -1, 1)
        else:
            raise NameError(f"prompt_init: {prompt_init}")
        
        self.pre_tokens = 197
        

    def forward(self, x: torch.Tensor, stop_grad: bool=False):

        x = x.type(self.prompt.dtype)
        batch_prompt = self.prompt.expand(x.shape[1], -1, -1) ## 1,4,768 --> B,4,768
        if stop_grad:
            batch_prompt = batch_prompt.detach()

        if self.replace_token:
            out = torch.concat([x[:, :self.pre_tokens], batch_prompt.permute(1,0,2)], dim=0)
            assert out.shape[1] == x.shape[1]
        else:
            out = torch.concat([x, batch_prompt.permute(1,0,2)], dim=0) ## cat [x:B, 197, 768, prompts:B, 4, 768]
            assert out.shape[0] == x.shape[0] + self.prompt_len

        return out.type(self.clip_dtype)
    


class ContinualPrompt(nn.Module):

    def __init__(self, prompt_len: int, 
                embed_dim: int = 768, 
                prompt_init: str = 'uniform',
                clip_dtype = None):
        
        super().__init__()
        assert prompt_len >= 0
        self.prompt_len = prompt_len
        self.embed_dim = embed_dim
        self.prompt_init = prompt_init
        self.clip_dtype = clip_dtype
        self.ensemble_type = 'ema'
        self.eta = None
    
    def init_modules_for_new_task(self, task_id: int = 0):

        self.taskid: int = task_id

        if task_id == 0:
            self.train_vpt: VisualPrompt = VisualPrompt(prompt_len = self.prompt_len, embed_dim = self.embed_dim,
                                            prompt_init = self.prompt_init, clip_dtype=self.clip_dtype)
            self.ema_vpt: VisualPrompt = VisualPrompt(prompt_len = self.prompt_len, embed_dim = self.embed_dim,
                                            prompt_init = self.prompt_init, clip_dtype=self.clip_dtype)
            

    def emsemble_modules(self, taskid: int=0, ensemble_type: str=None):

        if ensemble_type is None:
            ensemble_type = self.ensemble_type

        current_device = self.train_vpt.prompt.data.device
        with torch.no_grad():

            if taskid == 0:
                self.train_vpt.prompt.data = self.train_vpt.prompt.data.clone()
                
            if taskid >= 1 and ensemble_type != 'None':
                ## weighted ensemble
                if ensemble_type == 'sequential':

                    self.train_vpt.prompt.data = self.train_vpt.prompt.data.clone()

                elif ensemble_type == 'ema':
                    eta = self.eta if self.eta is not None else torch.tensor(1.0 / (taskid + 1.0)).to(torch.float32)

                    self.train_vpt.prompt.data =\
                    (1-eta) * self.train_vpt.prompt.data.clone() +\
                    eta * self.ema_vpt.prompt.data.clone()

            self.ema_vpt.prompt.data = self.train_vpt.prompt.data.clone()
        
        return None
    


    def forward(self, x: torch.Tensor, mode: str = 'visual'):

        current_device = self.train_vpt.prompt.device
        
        if mode == 'inference':
            output = self.ema_vpt(x)
        else:
            output = self.train_vpt(x)

        return output