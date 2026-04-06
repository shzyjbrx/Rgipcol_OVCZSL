import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import os

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
        clip_model = self._load_clip(clip_type)
        self.visual = clip_model.visual
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

        # 彻底冻结 CLIP 原始参数
        for p in self.parameters():
            p.requires_grad = False

        self.emb_dim = self.token_embedding.embedding_dim 

        # 2. 💡 预计算原始 CLIP 的“基准特征” (Base Anchor Features)
        # 这些特征作为“锚点”，在训练中保持不变
        print(f"[CLIPSoftPrompt] 正在预计算原始 CLIP 基准特征作为锚点...")
        self._precompute_base_features(dset, clip_model)

        # 3. 初始化可学习参数
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
        # 建议将其设为不可训练以保持语义纯净，防止 ua_uo_acc 崩溃
        self.attr_embeds = nn.Parameter(self._init_class_embeds(dset.all_attrs))
        self.obj_embeds  = nn.Parameter(self._init_class_embeds(dset.all_objs))
        self.attr_embeds.requires_grad = False
        self.obj_embeds.requires_grad = False

        # (3) 💡 残差缩放因子 alpha (参考 Troika 中的 lamda)
        # 初始化为很小的值，确保 Epoch 1 等同于原始 CLIP
        self.alpha = nn.Parameter(torch.tensor([0.01])) 

        # (4) 可学习温度
        self.logit_scale = nn.Parameter(clip_model.logit_scale.data.clone())

        # 4. 辅助 Buffer 与映射
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
                # 预计算全量测试组合的锚点 (实时计算或缓存)
                test_pair_names = [f"{p[0]} {p[1]}" for p in self.all_pairs1]
                base_c_test = self._compute_manual_feats_live(test_pair_names).to(self.alpha.device)
                
                # 构造学习部分的类别嵌入
                test_attr_idx = [self.dset.attr2idx[p[0]] for p in self.all_pairs1]
                test_obj_idx  = [self.dset.obj2idx[p[1]] for p in self.all_pairs1]
                t_comp_embeds = (self.attr_embeds[test_attr_idx] + self.obj_embeds[test_obj_idx]) / 2.0
                
                # 融合生成最终测试特征
                self.test_text_features = self._encode_text(self.ctx_comp, t_comp_embeds, base_c_test)
        
        return self._val_forward(batch)

    def _train_forward(self, batch):
        imgs = batch['img']
        p_idx_tr = batch['pair']

        v = self._encode_visual(imgs)

        # 分支 1: 属性
        t_a = self._encode_text(self.ctx_attr, self.attr_embeds[self.tr_attr_idx], self.base_attr_all[self.tr_attr_idx])
        # 分支 2: 对象
        t_o = self._encode_text(self.ctx_obj,  self.obj_embeds[self.tr_obj_idx], self.base_obj_all[self.tr_obj_idx])
        # 分支 3: 组合
        pair_a_emb = self.attr_embeds[self.train_pair_attr_indices]
        pair_o_emb = self.obj_embeds[self.train_pair_obj_indices]
        t_c = self._encode_text(self.ctx_comp, (pair_a_emb + pair_o_emb) / 2.0, self.base_pair_tr)

        scale = self.logit_scale.exp()
        logits_c = scale * (v @ t_c.T)
        logits_a = scale * (v @ t_a.T)
        logits_o = scale * (v @ t_o.T)

        # 损失计算 (假设 train.py 已将全局索引转换为训练本地索引)
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
        """测试时实时计算手工提示词锚点"""
        texts = [f"a photo of {t.replace('_', ' ')}" for t in texts_raw]
        tokens = clip.tokenize(texts).to(self.alpha.device)
        with torch.no_grad():
            x = self.token_embedding(tokens).float()
            x = x + self.positional_embedding.float()
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_final(x)
            x = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)] @ self.text_projection.float()
            return F.normalize(x, dim=-1)

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