import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import clip
import os

# --- 新增 Adapter 模块 ---
class AdapterLinear(nn.Module):
    """
    并联瓶颈适配器 (Parallel Bottleneck Adapter)
    包含降维 -> GELU激活 -> 升维 的非线性过程
    """
    def __init__(self, original_layer, reduction_factor=4, scale=1.0):
        super(AdapterLinear, self).__init__()
        self.original_layer = original_layer
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # 冻结原始线性层的权重
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        # 计算瓶颈维度
        self.bottleneck_dim = in_features // reduction_factor
        
        # 构建 Adapter 旁路
        self.down_proj = nn.Linear(in_features, self.bottleneck_dim, bias=False)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(self.bottleneck_dim, out_features, bias=False)
        self.scale = scale
        
        # 初始化权重
        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.zeros_(self.up_proj.weight) # 初始化为0，确保初期行为等价于原模型
        
        # 确保数据类型与原模型一致 (如 float16)
        self.to(self.original_layer.weight.dtype)

    @property
    def weight(self):
        return self.original_layer.weight

    @property
    def bias(self):
        return self.original_layer.bias

    def forward(self, x):
        # 原路输出 + Adapter输出
        adapter_output = self.up_proj(self.act(self.down_proj(x)))
        return self.original_layer(x) + self.scale * adapter_output


# --- 替换注入函数 ---
def inject_adapter(model, reduction_factor=4, scale=1.0):
    """
    将 Adapter 注入到 CLIP 的 MLP 层中
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 过滤策略：只对视觉编码器的 MLP 层注入，避开注意力矩阵的投影
            if "visual" in name and "mlp" in name and "out_proj" not in name:
                attrs = name.split('.')
                submodule = model
                for attr in attrs[:-1]:
                    submodule = getattr(submodule, attr)
                
                original_layer = getattr(submodule, attrs[-1])
                # 替换为 AdapterLinear
                setattr(submodule, attrs[-1], AdapterLinear(original_layer, reduction_factor, scale))
                print(f"[Adapter] 成功注入: {name}")

    # 启用 Adapter 梯度，冻结其他参数
    for name, param in model.named_parameters():
        if "down_proj" in name or "up_proj" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    return model
    
class Backbone(nn.Module):
    def __init__(self, backbone='resnet50'):
        super(Backbone, self).__init__()
        self.backbone_name = backbone

        if 'ViT' in backbone:
            if 'ViT-L/14' in backbone:
                local_weights_path = "/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/checkpoints/ViT-L-14.pt"
            else:
                local_weights_path = "/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/checkpoints/ViT-B-32.pt"

            print(f"=> Loading CLIP weights ({backbone}) from {local_weights_path}")
            
            try:
                state_dict = torch.load(local_weights_path, map_location='cpu')
                # 这里的参数必须用兼容 clip 库的格式，如 "ViT-L/14"
                clip_model, _ = clip.load(backbone, device='cpu') 
                clip_model.load_state_dict(state_dict)
                self.visual = clip_model.visual.float()
            except Exception as e:
                print(f"=> Local load failed: {e}. Trying online load...")
                clip_model, _ = clip.load(backbone, device='cpu')
                self.visual = clip_model.visual.float()

            self.visual = self.visual.to("cuda")
            return 

        # ResNet 逻辑保持不变
        if backbone == 'resnet18':
            resnet = torchvision.models.resnet.resnet18(pretrained=True)
        elif backbone == 'resnet50':
            resnet = torchvision.models.resnet.resnet50(pretrained=True)
        elif backbone == 'resnet101':
            resnet = torchvision.models.resnet.resnet101(pretrained=True)
        else:
            raise ValueError(f"Backbone {backbone} is not supported.")

        self.block0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.block1, self.block2, self.block3, self.block4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

    def forward(self, x, returned=[4]):
        if hasattr(self, 'visual'):
            return [self.encode_image(x)]
        blocks = [self.block0(x)]
        for i in range(1, 5):
            blocks.append(getattr(self, f'block{i}')(blocks[-1]))
        return [blocks[i] for i in returned]

    def encode_image(self, x):
        if hasattr(self, 'visual'):
            # 1. 卷积层投影 [B, C, H, W] -> [B, Width, Grid, Grid]
            x = self.visual.conv1(x)  
            # 2. 展平并转置 [B, Width, Grid*Grid] -> [B, Patches, Width]
            x = x.reshape(x.shape[0], x.shape[1], -1)  
            x = x.permute(0, 2, 1)  
            
            # 💡 核心修复：使用 x.shape[2] (Width) 确保维度匹配 (768)
            # 添加 Class Embedding
            cls_token = self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[2], dtype=x.dtype, device=x.device)
            x = torch.cat([cls_token, x], dim=1) 
            
            # 3. 加上位置编码 (CLIP 内部会自动适配 50 或 257 个 token)
            x = x + self.visual.positional_embedding.to(x.dtype)
            x = self.visual.ln_pre(x)

            # 4. Transformer 运算
            x = x.permute(1, 0, 2)  # LND
            x = self.visual.transformer(x)
            x = x.permute(1, 0, 2)  # NLD
            
            # 5. 返回空间 Patch 特征 (去掉 CLS token)
            # L/14 结果维度: [B, 256, 768]
            return x[:, 1:, :].float()
        
        return self.forward(x)[-1]