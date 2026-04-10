import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import os
import torch.utils.checkpoint as cp 
from .backbone import inject_adapter

class CLIPSoftPrompt(nn.Module):
    """
    CLIPSoftPrompt: 残差特征融合 + 三分支软提示模型
    核心逻辑: Final_Feature = Normalize(Base_CLIP_Feature + alpha * Learned_SoftPrompt_Feature)
    1. Base_CLIP_Feature: 原始手工提示 "a photo of [class]" 的固定特征。
    2. Learned_SoftPrompt_Feature: 可学习提示词产生的修正特征。
    3. alpha: 可学习的残差缩放因子 (初始化为 0.01)，确保起步稳健。
    """
    def __init__(self, dset, cfg):
        super().__init__()
        self.cfg = cfg
        self.dset = dset
        self.n_ctx = getattr(cfg.MODEL, 'n_ctx', 16)
        clip_type = cfg.TRAIN.clip_type 

        # 1. 加载 CLIP 骨干网络
        # 注意：这里使用类内部定义的 _load_clip 方法，它会处理本地路径
        clip_model = self._load_clip(clip_type)

        # --- 💡 注入 Adapter ---
        # 必须在拆解模型组件之前进行注入，以确保 visual 包含 Adapter 层
        from .backbone import inject_adapter
        
        # 获取压缩因子，如果没有配置则默认设为 4 (等效于把 768 维降到 192 维)
        reduction_factor = getattr(cfg.MODEL, 'adapter_reduction', 4)
        
        # 注入 Adapter 层并返回修改后的模型
        clip_model = inject_adapter(clip_model, reduction_factor=reduction_factor, scale=1.0)
        # ----------------------------------------

        # 2. 拆解并保存模型组件
        self.visual = clip_model.visual
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

        # 3. 彻底冻结 CLIP 原始参数 (Adapter 参数的 requires_grad 已在 inject_adapter 中处理)
        # 我们只在这里二次确认非 Adapter 参数被冻结
        for name, p in self.named_parameters():
            if "down_proj" not in name and "up_proj" not in name:
                p.requires_grad = False

        self.emb_dim = self.token_embedding.embedding_dim 

        # 4. 💡 预计算原始 CLIP 的“基准特征” (Base Anchor Features)
        # 这些特征作为“锚点”，在训练中保持不变
        print(f"[CLIPSoftPrompt] 正在预计算原始 CLIP 基准特征作为锚点...")
        self._precompute_base_features(dset, clip_model)

        # 5. 初始化可学习参数
        # (1) 软提示上下文 ctx (用 "a photo of a" 初始化)
        ctx_init = "a photo of a"
        tokens = clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = self.token_embedding(tokens).float()
        init_vec = embedding[0, 1: 1 + self.n_ctx, :]
        
        self.ctx_attr = nn.Parameter(init_vec.clone())
        self.ctx_obj  = nn.Parameter(init_vec.clone())
        self.ctx_comp = nn.Parameter(init_vec.clone())

        # (2) 类别嵌入参数 (使用 CLIP 均值初始化，参考 Troika)
        self.attr_embeds = nn.Parameter(self._init_class_embeds(dset.all_attrs))
        self.obj_embeds  = nn.Parameter(self._init_class_embeds(dset.all_objs))
        self.attr_embeds.requires_grad = False
        self.obj_embeds.requires_grad = False

        # (3) 💡 残差缩放因子 alpha (初始化为很小的值)
        self.alpha = nn.Parameter(torch.tensor([0.01])) 

        # (4) 可学习温度
        self.logit_scale = nn.Parameter(clip_model.logit_scale.data.clone())

        # 6. 辅助 Buffer 与映射
        self._build_index_maps(dset)
        self.register_buffer('sos_token', torch.tensor([49406]))
        self.register_buffer('eos_token', torch.tensor([49407]))
        
        self.all_pairs1 = dset.pairs
        self.test_text_features = None

    # ────────────────────────────────────────────────────────────
    # 核心功能模块
    # ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _precompute_base_features(self, dset, clip_model):
        """
        增加缓存机制：自动保存和加载预计算特征
        """
        # 1. 定义缓存路径（建议根据数据集命名）
        cache_path = f"cache_{self.cfg.DATASET.name}_base_features.pt"
        rank = int(os.environ.get('RANK', 0))

        # 2. 检查缓存是否存在
        if os.path.exists(cache_path):
            if rank == 0:
                print(f"[CLIPSoftPrompt] 发现缓存文件 {cache_path}，正在直接加载...")
            
            # 加载缓存数据
            cache_data = torch.load(cache_path, map_location='cpu')
            
            # 注册到 Buffer
            self.register_buffer('base_attr_all', cache_data['attr'])
            self.register_buffer('base_obj_all', cache_data['obj'])
            self.register_buffer('base_pair_tr', cache_data['pair_tr'])
            
            if rank == 0:
                print("[CLIPSoftPrompt] 缓存加载完成，秒级启动。")
            return

        # 3. 如果没有缓存，则执行计算（保持之前的分批逻辑）
        if rank == 0:
            print(f"[CLIPSoftPrompt] 未发现缓存，开始预计算 (Attrs/Objs/Pairs)...")

        def get_anchor_feats(names, desc):
            all_feats = []
            batch_size = 128
            for i in range(0, len(names), batch_size):
                if rank == 0:
                    print(f"  -> 正在编码 {desc}: {i}/{len(names)}...", flush=True)
                batch_names = names[i : i + batch_size]
                texts = [f"a photo of {n.replace('_', ' ')}" for n in batch_names]
                tokens = clip.tokenize(texts).to(next(clip_model.parameters()).device)
                
                x = clip_model.token_embedding(tokens).float()
                x = x + clip_model.positional_embedding.float()
                x = x.permute(1, 0, 2)
                x = clip_model.transformer(x)
                x = x.permute(1, 0, 2)
                x = clip_model.ln_final(x)
                x = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)] @ clip_model.text_projection.float()
                all_feats.append(F.normalize(x, dim=-1).cpu())
            return torch.cat(all_feats, dim=0)

        # 执行计算
        attr_feats = get_anchor_feats(dset.all_attrs, "属性")
        obj_feats = get_anchor_feats(dset.all_objs, "对象")
        tr_pair_names = [f"{p[0]} {p[1]}" for p in dset.train_pairs]
        pair_tr_feats = get_anchor_feats(tr_pair_names, "训练组合")

        # 4. 💡 核心：仅由主进程保存到硬盘
        if rank == 0:
            print(f"[CLIPSoftPrompt] 正在将特征保存至 {cache_path}...")
            save_dict = {
                'attr': attr_feats,
                'obj': obj_feats,
                'pair_tr': pair_tr_feats
            }
            torch.save(save_dict, cache_path)
            print("[CLIPSoftPrompt] 预计算并保存完成。")

        # 5. 所有进程同步注册 Buffer
        self.register_buffer('base_attr_all', attr_feats)
        self.register_buffer('base_obj_all', obj_feats)
        self.register_buffer('base_pair_tr', pair_tr_feats)

    @torch.no_grad()
    def _init_class_embeds(self, names):
        """参考 Troika：提取类别名称的语义均值作为初始化"""
        embeds = []
        for name in names:
            tokens = clip.tokenize(name.replace('_', ' '))
            eos_idx = tokens[0].argmax()
            tok_emb = self.token_embedding(tokens).float()
            # 取 SOS 和 EOS 之间的单词向量均值
            mean_emb = tok_emb[0, 1:eos_idx, :].mean(dim=0)
            embeds.append(mean_emb)
        return torch.stack(embeds)

    def _encode_text(self, ctx, class_embeds, base_anchor):
        """
        残差编码逻辑：Normalize(Base + alpha * Learned)
        """
        N = class_embeds.shape[0]
        D = self.emb_dim
        
        # 1. 构造软提示序列特征 [SOS][ctx][Label][EOS]
        with torch.no_grad():
            sos_emb = self.token_embedding(self.sos_token).float()
            eos_emb = self.token_embedding(self.eos_token).float()
        
        sos_exp = sos_emb.expand(N, -1, -1)
        ctx_exp = ctx.unsqueeze(0).expand(N, -1, -1)
        cls_exp = class_embeds.unsqueeze(1)
        eos_exp = eos_emb.expand(N, -1, -1)
        
        # 拼接长度为 77 的序列
        prefix = torch.cat([sos_exp, ctx_exp, cls_exp, eos_exp], dim=1) # (N, n_ctx+3, D)
        pad_len = 77 - prefix.shape[1]
        pad_emb = torch.zeros(N, pad_len, D, device=prefix.device, dtype=prefix.dtype)
        x = torch.cat([prefix, pad_emb], dim=1)

        x = x + self.positional_embedding.float()
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2).float()
        x = self.ln_final(x)

        # 提取 EOS 位置对应的投影特征
        eos_pos = 1 + self.n_ctx + 1 
        learned_feat = x[:, eos_pos] @ self.text_projection.float()
        learned_feat = F.normalize(learned_feat, dim=-1)

        # 2. 💡 残差连接核心 (类似于 Troika)
        # 将原始特征与学习到的特征线性相加
        combined = base_anchor + self.alpha * learned_feat
        
        return F.normalize(combined, dim=-1)

    # ────────────────────────────────────────────────────────────
    # 运行逻辑
    # ────────────────────────────────────────────────────────────

    def forward(self, batch):
        if self.training:
            self.test_text_features = None # 训练时清空缓存
            return self._train_forward(batch)
        
        # 测试阶段缓存优化：大幅提升评估速度
        if self.test_text_features is None:
            with torch.no_grad():
                # 预计算全量测试组合的锚点 (由于上面的修改，这里已经安全了)
                test_pair_names = [f"{p[0]} {p[1]}" for p in self.all_pairs1]
                base_c_test = self._compute_manual_feats_live(test_pair_names).to(self.alpha.device)
                
                # 构造学习部分的类别嵌入
                test_attr_idx = [self.dset.attr2idx[p[0]] for p in self.all_pairs1]
                test_obj_idx  = [self.dset.obj2idx[p[1]] for p in self.all_pairs1]
                t_comp_embeds = (self.attr_embeds[test_attr_idx] + self.obj_embeds[test_obj_idx]) / 2.0
                
                # 💡 新增：融合生成最终测试特征 (增加分块处理防止 OOM)
                chunk_size = 256
                test_feats_list = []
                for i in range(0, t_comp_embeds.shape[0], chunk_size):
                    chunk_embeds = t_comp_embeds[i : i+chunk_size]
                    chunk_base = base_c_test[i : i+chunk_size]
                    chunk_out = self._encode_text(self.ctx_comp, chunk_embeds, chunk_base)
                    test_feats_list.append(chunk_out)
                    
                self.test_text_features = torch.cat(test_feats_list, dim=0)
        
        return self._val_forward(batch)

    def _construct_prompts(self, ctx, class_embeds):
        """辅助函数：构造进入 Transformer 前的 Embedding 序列"""
        N = class_embeds.shape[0]
        with torch.no_grad():
            sos_emb = self.token_embedding(self.sos_token).float()
            eos_emb = self.token_embedding(self.eos_token).float()
        
        sos_exp = sos_emb.expand(N, -1, -1)
        ctx_exp = ctx.unsqueeze(0).expand(N, -1, -1)
        cls_exp = class_embeds.unsqueeze(1)
        eos_exp = eos_emb.expand(N, -1, -1)
        
        prefix = torch.cat([sos_exp, ctx_exp, cls_exp, eos_exp], dim=1)
        pad_len = 77 - prefix.shape[1]
        pad_emb = torch.zeros(N, pad_len, self.emb_dim, device=prefix.device, dtype=prefix.dtype)
        return torch.cat([prefix, pad_emb], dim=1)

    def _train_forward(self, batch):
        imgs = batch['img']
        p_idx_tr = batch['pair']
        
        # 视觉编码 (正常执行)
        v = self._encode_visual(imgs)

        # 1. 准备所有文本输入
        attr_emb = self.attr_embeds[self.tr_attr_idx]
        x_a = self._construct_prompts(self.ctx_attr, attr_emb)
        obj_emb = self.obj_embeds[self.tr_obj_idx]
        x_o = self._construct_prompts(self.ctx_obj, obj_emb)
        pair_a_emb = self.attr_embeds[self.train_pair_attr_indices]
        pair_o_emb = self.obj_embeds[self.train_pair_obj_indices]
        comp_emb = (pair_a_emb + pair_o_emb) / 2.0
        x_c = self._construct_prompts(self.ctx_comp, comp_emb)

        n_a, n_o, n_c = x_a.shape[0], x_o.shape[0], x_c.shape[0]
        x_all = torch.cat([x_a, x_o, x_c], dim=0)

        # --------------------------------------------------------------------
        # 💡 核心修复：定义 Transformer 的前向逻辑 (为了配合 checkpoint)
        # --------------------------------------------------------------------
        def transformer_forward(x_in):
            # 将原来 transformer 及其前后相关的逻辑写在一起
            x_in = x_in + self.positional_embedding.float()
            x_in = x_in.permute(1, 0, 2)
            # 这里的 self.transformer 包含了你注入的 LoRA 层
            x_in = self.transformer(x_in) 
            x_in = x_in.permute(1, 0, 2).float()
            x_in = self.ln_final(x_in)
            return x_in

        # 2. 分段进入 Transformer 并开启 Checkpoint
        from torch.utils.checkpoint import checkpoint # 确保导入
        
        sub_batch_size = 64  # 根据 40G A100 的压力，64-128 之间比较合适
        all_learned_feats = []
        
        for i in range(0, x_all.shape[0], sub_batch_size):
            x_chunk = x_all[i : i + sub_batch_size]
            
            # 💡 关键：使用 checkpoint 运行分块，而不是直接运行
            # use_reentrant=False 是新版 PyTorch 推荐的、与 DDP 兼容性最好的写法
            x_chunk_out = checkpoint(transformer_forward, x_chunk, use_reentrant=False)
            
            # 提取 EOS 位置特征并投影
            eos_pos = 1 + self.n_ctx + 1 
            learned_chunk = x_chunk_out[:, eos_pos] @ self.text_projection.float()
            all_learned_feats.append(learned_chunk)
        
        # --------------------------------------------------------------------

        # 拼接分段结果
        learned_feats_all = torch.cat(all_learned_feats, dim=0)
        learned_feats_all = F.normalize(learned_feats_all, dim=-1)

        # 3. 拆分回分支并计算残差 (保持原逻辑)
        t_a_learned, t_o_learned, t_c_learned = torch.split(learned_feats_all, [n_a, n_o, n_c], dim=0)
        
        t_a = F.normalize(self.base_attr_all[self.tr_attr_idx] + self.alpha * t_a_learned, dim=-1)
        t_o = F.normalize(self.base_obj_all[self.tr_obj_idx] + self.alpha * t_o_learned, dim=-1)
        t_c = F.normalize(self.base_pair_tr + self.alpha * t_c_learned, dim=-1)

        # 4. 计算损失
        scale = self.logit_scale.exp()
        logits_c = scale * (v @ t_c.T)
        logits_a = scale * (v @ t_a.T)
        logits_o = scale * (v @ t_o.T)

        loss_c = F.cross_entropy(logits_c, p_idx_tr)
        loss_a = F.cross_entropy(logits_a, batch['attr'])
        loss_o = F.cross_entropy(logits_o, batch['obj'])

        loss = loss_c + self.cfg.MODEL.w_loss_attr * loss_a + self.cfg.MODEL.w_loss_obj * loss_o

        return {
            'loss_total': loss,
            'acc_pair': (logits_c.argmax(1) == p_idx_tr).float().mean(),
        }

    def _val_forward(self, batch):
        v = self._encode_visual(batch['img'])
        logits = self.logit_scale.exp() * (v @ self.test_text_features.T)
        scores = {pair: logits[:, i] for i, pair in enumerate(self.all_pairs1)}
        return {'scores': scores}

    # ────────────────────────────────────────────────────────────
    # 工具函数
    # ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_visual(self, images):
        feats = self.visual(images.float())
        return F.normalize(feats.float(), dim=-1)

    def _compute_manual_feats_live(self, texts_raw):
        """测试时实时计算手工提示词锚点 (增加分块防止 OOM)"""
        texts = [f"a photo of {t.replace('_', ' ')}" for t in texts_raw]
        tokens = clip.tokenize(texts).to(self.alpha.device)
        
        chunk_size = 256 # 分块大小，256对于A100非常安全
        all_feats = []
        with torch.no_grad():
            for i in range(0, len(tokens), chunk_size):
                batch_tokens = tokens[i : i+chunk_size]
                x = self.token_embedding(batch_tokens).float()
                x = x + self.positional_embedding.float()
                x = x.permute(1, 0, 2)
                x = self.transformer(x)
                x = x.permute(1, 0, 2)
                x = self.ln_final(x)
                x = x[torch.arange(x.shape[0]), batch_tokens.argmax(dim=-1)] @ self.text_projection.float()
                all_feats.append(F.normalize(x, dim=-1))
                
        return torch.cat(all_feats, dim=0)

    def _load_clip(self, clip_type):
        local_path = '/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/checkpoints/ViT-L-14.pt'
        if os.path.exists(local_path):
            print(f'[CLIPSoftPrompt] 加载本地模型: {local_path}')
            model, _ = clip.load(local_path, device='cpu')
            return model.float()
        model, _ = clip.load(clip_type, device='cpu')
        return model.float()

    def _build_index_maps(self, dset):
        self.register_buffer('tr_attr_idx', torch.tensor([dset.attr2idx[a] for a in dset.train_attrs]))
        self.register_buffer('tr_obj_idx', torch.tensor([dset.obj2idx[o] for o in dset.train_objs]))
        self.register_buffer('train_pair_attr_indices', torch.tensor([dset.attr2idx[p[0]] for p in dset.train_pairs]))
        self.register_buffer('train_pair_obj_indices', torch.tensor([dset.obj2idx[p[1]] for p in dset.train_pairs]))