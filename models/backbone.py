import torch
import torch.nn as nn
import torchvision
import clip
import os

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