import torch 
import torch.nn as nn
import math
import torch.nn.functional as F

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
    
class AdapterLinear(nn.Module):
    """
    并联瓶颈适配器 (Parallel Bottleneck Adapter)
    用于替代 LoRALinear。包含非线性激活函数，能更好地拟合复杂的属性-物体组合偏移。
    """
    def __init__(self, linear_layer: nn.Linear, reduction_factor: int = 4, scale: float = 1.0):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        
        # 继承并冻结原有的 Linear 权重
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # 计算瓶颈维度 (Bottleneck dimension)
        self.bottleneck_dim = self.in_features // reduction_factor
        
        # 构建 Adapter 旁路: 降维 -> 非线性激活 -> 升维
        self.down_proj = nn.Linear(self.in_features, self.bottleneck_dim, bias=False)
        self.act = nn.GELU() # 引入非线性，这是区别于 LoRA 的关键
        self.up_proj = nn.Linear(self.bottleneck_dim, self.out_features, bias=False)
        
        self.scale = scale
        
        # 初始化权重
        self.reset_parameters()
        
        # 确保 Adapter 参数的数据类型与原线性层一致（例如 float16 或 float32）
        self.to(self.weight.dtype)

    def reset_parameters(self):
        # 降维层使用正态分布初始化
        nn.init.normal_(self.down_proj.weight, std=0.02)
        # 升维层初始化为 0，确保训练初期 Adapter 的输出为 0，等价于原始 CLIP 模型
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x):
        # 1. 原始的冻结线性层前向传播
        base_out = F.linear(x, self.weight, self.bias)
        
        # 2. Adapter 旁路前向传播
        adapter_out = self.up_proj(self.act(self.down_proj(x)))
        
        # 3. 特征融合
        return base_out + self.scale * adapter_out