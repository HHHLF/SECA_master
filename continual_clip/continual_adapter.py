import torch
import torch.nn as nn
import copy
import math

class Adapter(nn.Module):

    def __init__(self, embed_dim: int = 768, 
                dropout: float = 0.0, 
                middle_dim: int = 64, 
                init_option:str = "lora", 
                adapter_layernorm_option: str = 'none', 
                adapter_scalar: float=0.1, 
                relu: bool = True,
                clip_dtype = None):
        
        super().__init__()

        self.clip_dtype = clip_dtype
        self.embed_dim = embed_dim
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None

        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.embed_dim)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.relu = relu
        is_bias = False
        self.middle_dim = middle_dim
        self.down_proj = nn.Linear(self.embed_dim, self.middle_dim, bias=is_bias)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.middle_dim, self.embed_dim, bias=is_bias)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                if self.down_proj.bias is not None and self.up_proj.bias is not None:
                    nn.init.zeros_(self.down_proj.bias)
                    nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor, add_residual=False, residual=None):

        x = x.to(self.down_proj.weight.dtype)

        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in': #  none
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x) ## 768 --> 64
        if self.relu:
            down = self.non_linear_func(down) ## relu
        down = nn.functional.dropout(down, p=self.dropout, training=self.training) ## dropout
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out': #  none
            up = self.adapter_layer_norm_before(up)

        output: torch.Tensor = up

        return output.to(self.clip_dtype)



class ContinualAdapter(nn.Module):

    def __init__(self, embed_dim: int = 768, middle_dim: int = 64, dropout: float = 0.0, pool_size: int = 100,
                clip_dtype = None, ensemble_type = None):
        super().__init__()

        self.taskid = 0
        self.clip_dtype = clip_dtype

        self.embed_dim = embed_dim
        self.middle_dim = middle_dim

        self.dropout = dropout
        self.relu = True
        self.ensemble_type = ensemble_type
        self.eta = 0.5

        self.s_adapter = None
        self.p_adapter = None
        self.prev_adapters = nn.ModuleList([])

        self.pool_size = pool_size

    def training_adaptation(self, task_id: int):

        self.taskid: int = task_id

        if task_id == 0:
            self.s_adapter: Adapter = Adapter(embed_dim = self.embed_dim, dropout = self.dropout,
                                            middle_dim = self.middle_dim, clip_dtype=self.clip_dtype)
            self.s_: Adapter = Adapter(embed_dim = self.embed_dim, dropout = self.dropout,
                                            middle_dim = self.middle_dim, clip_dtype=self.clip_dtype)
            
        else:            
            with torch.no_grad(): 
                self.s_.down_proj.weight.data = self.s_adapter.down_proj.weight.data.clone()
                self.s_.up_proj.weight.data = self.s_adapter.up_proj.weight.data.clone()

        return None


    def inference_adaptation(self, taskid: int, update_idx: torch.Tensor, ensemble_type: str=None):

        if ensemble_type is None:
            ensemble_type = self.ensemble_type

        with torch.no_grad():
            
            prev_adapter: Adapter = Adapter(embed_dim = self.embed_dim, dropout = self.dropout,
                                            middle_dim = self.middle_dim, clip_dtype=self.clip_dtype)
            prev_adapter.down_proj.weight.data = self.s_adapter.down_proj.weight.data.clone()
            prev_adapter.up_proj.weight.data = self.s_adapter.up_proj.weight.data.clone()

            self.prev_adapters: list[Adapter]
            if len(self.prev_adapters) + 1 > self.pool_size:
                self.prev_adapters.pop(update_idx)
                self.prev_adapters.append(prev_adapter)
            else:
                self.prev_adapters.append(prev_adapter)
            print(f'Adapter Pool Size: {len(self.prev_adapters)}')

            if taskid == 0:
                self.s_adapter.down_proj.weight.data = self.s_adapter.down_proj.weight.data.clone()
                self.s_adapter.up_proj.weight.data = self.s_adapter.up_proj.weight.data.clone()
                
            if taskid >= 1 and ensemble_type != 'None':
                ## weighted ensemble
                if ensemble_type == 'seq':

                    self.s_adapter.down_proj.weight.data = self.s_adapter.down_proj.weight.data.clone()
                    self.s_adapter.up_proj.weight.data = self.s_adapter.up_proj.weight.data.clone()

                elif ensemble_type == 'ema':
                    eta = self.eta if self.eta is not None else torch.tensor(1.0 / (taskid + 1.0)).to(torch.float32)

                    self.s_adapter.down_proj.weight.data =\
                    (1-eta) * self.s_.down_proj.weight.data.clone() +\
                    eta * self.s_adapter.down_proj.weight.data.clone()

                    self.s_adapter.up_proj.weight.data =\
                    (1-eta) * self.s_.up_proj.weight.data.clone() +\
                    eta * self.s_adapter.up_proj.weight.data.clone()

        return None

    
    def forward(self, x, mode: str = 'visual', add_residual: bool = False):
        
        if mode == 'inference' or 'visual' in mode:
            output = self.s_adapter(x, add_residual=add_residual)

        elif mode == 'cross_attention':
            output = self.s_(x, add_residual=add_residual)

        elif 'task_specific' in mode:
            output = self.prev_adapters[int(mode[-1])](x, add_residual=add_residual)
        
        else:
            raise KeyError('***** Invalid Mode *****')
        
        return output