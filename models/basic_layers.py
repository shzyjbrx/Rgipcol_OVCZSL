import torch 
import torch.nn as nn
import math

class MLP(nn.Module):
    def __init__(
            self,
            inp_dim,            # Input dimension.
            latent_dim,         # Hidden layer dimension.
            out_dim,            # Output dimension.
            num_layers=2,       # Number of layers (incl. input & output).
            bias=True,          # Bias term in Linear layers.
            batchnorm=True,     # Use BatchNorm.
            layernorm=False,    # Use LayerNorm.
            dropout=0,          
            end_relu=False,     # Use ReLU at the end.
            drop_input=0,       # Dropout at input.
            drop_output=0,       # Dropout at output.
            final_linear_bias=True
        ):
        super(MLP, self).__init__()
        mod = []

        if drop_input > 0:
            mod.append(nn.Dropout(drop_input))

        mod.append(nn.Linear(inp_dim, latent_dim, bias=bias))
        if batchnorm:
            mod.append(nn.BatchNorm1d(latent_dim))
        if layernorm:
            mod.append(nn.LayerNorm(latent_dim))
        mod.append(nn.ReLU(True))

        for L in range(num_layers-2):
            mod.append(nn.Linear(latent_dim, latent_dim, bias=bias))
            if batchnorm:
                mod.append(nn.BatchNorm1d(latent_dim))
            if layernorm:
                mod.append(nn.LayerNorm(latent_dim))
            mod.append(nn.ReLU(True))
        
        if dropout > 0:
            mod.append(nn.Dropout(dropout))

        mod.append(nn.Linear(latent_dim, out_dim, bias=final_linear_bias))

        if end_relu:
            mod.append(nn.ReLU(True))

        if drop_output > 0:
            mod.append(nn.Dropout(drop_output))

        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        output = self.mod(x)
        return output
    
class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=8, lora_alpha=16):
        super(LoRALinear, self).__init__()
        self.original_layer = original_layer # 冻结的原始 CLIP 层
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # LoRA 矩阵
        self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))
        self.scaling = lora_alpha / rank
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    @property
    def weight(self):
        return self.original_layer.weight

    @property
    def bias(self):
        return self.original_layer.bias

    def forward(self, x):
        # 结果 = 原始分支 + (x * A^T * B^T) * scaling
        lora_output = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return self.original_layer(x) + lora_output