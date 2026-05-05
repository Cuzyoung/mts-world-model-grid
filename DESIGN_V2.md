# PatchTST-Dreamer v2: 架构改进设计文档

**Created**: 2026-04-12
**Status**: 设计阶段，等待 v1 实验结果后实施

---

## 1. v1 架构回顾与问题诊断

### 1.1 当前信息流

```
Encoder (B, nvars, d_model=128, patch_num=12)   [seq_len=96, patch_len=16, stride=8]
  → Flatten ALL patches: (B*nvars, 128×12=1536)
  → MLP: 1536 → 512 → 256 → z_0
  → GRU Rollout K步 (K = min(pred_len/16, 8)):
      For t = 0..K-1:
        input = step_embed[t]     ← 可学习参数，跟输入数据无关
        z_fast = FastGRU(input, z_fast)
        z_slow = SlowGRU(input, z_slow)   [每 slow_interval 步更新]
        z_t = Gate(z_fast, z_slow) + residual + LayerNorm
  → Cross-Attention(z_seq, encoder_patches)   ← 一次性，rollout结束后
  → Deep Decoder → prediction
```

### 1.2 三个根本缺陷

#### 缺陷 A: 信息瓶颈 — Flatten 压缩

`d_model × patch_num → d_latent` 是暴力降维：
- seq_len=96:  `128 × 12 = 1536 → 256`，6 倍压缩
- seq_len=336: `128 × 42 = 5376 → 256`，21 倍压缩 → **直接崩溃**（实验证实，Setting C MSE +50~66%）

Encoder 学了 12 个 patch 的分布式表示，每个 patch 在 128 维空间有自己的语义。Flatten 把它们拼成一个长向量再压缩，**破坏了 patch 之间的结构关系**。

类比：好比把一本书的每一页拍照，然后把所有照片像素拼成一行，再用 PCA 压到 1/6。你丢的不只是细节，更是"页与页之间的顺序和关系"。

#### 缺陷 B: GRU 输入无意义 — "盲飞"

```python
step_emb = self.step_embeds[t]  # shape: (d_latent,)，是 nn.Parameter
z_fast_new = self.fast_gru(step_emb, z_fast)
```

GRU 的 input 是一个跟输入数据**完全无关**的可学习向量。唯一的数据信息在 hidden state (z_fast) 里，而且随着 rollout 步数增加不断衰减。

对比 Dreamer 原版（RL）：
- GRU input = **action embedding + observation embedding**
- 每步都有外部信号注入，GRU 不是在"自说自话"

对比 Transformer decoder：
- decoder 在每步通过 cross-attention 从 encoder 获取信息
- 不存在"断开数据连接"的问题

我们的 GRU 就像一个**没有眼睛的人在走路**：初始位置知道（z_0 来自 encoder），之后完全靠记忆前进，每一步都会偏离更多。

#### 缺陷 C: Cross-Attention 位置错误 — 事后补救

当前 cross-attention 在 GRU rollout **全部完成后**才执行：

```python
z_seq = self.dynamics(z_init, self.num_pred_steps)  # 先全部rollout
z_seq = self.cross_attn(z_seq, enc_features)         # 再一次性attend
```

问题：
- GRU rollout 过程中没有任何外部校准，误差持续累积
- 等 rollout 完再做 attention，z_seq 已经偏了，attention 只是在"修补坏的 query"
- 这也解释了为什么 pred_len=720（rollout 8 步）时 MSE 暴涨 8.5%

正确的做法：**每步 rollout 都与 encoder 交互**，像 Dreamer 里的"每步观测修正后验"。

### 1.3 为什么之前还能赢部分实验？

Setting A 中 Dreamer 5W/6L，说明在某些情况下还是有优势：
- **中等 horizon (192, 336)**: rollout 步数不多（2-4步），误差累积不严重，multi-scale GRU 的时序建模能力还能发挥
- **ETTh1, ETTm1**: 这两个数据集的 OT（油温）变量有较强的时序动态，GRU 能捕获
- **消融证实 multi-scale 关键**: single-scale 比 multi-scale 差 20-30%，说明 fast+slow 的设计本身是有效的

所以：**multi-scale GRU 的核心设计是好的，问题出在信息流的组织方式上。**

---

## 2. v2 改进方案

### 2.1 核心思路：每步 rollout 都与 encoder 交互

这是对 Dreamer 世界模型更忠实的实现：

- Dreamer 原版: `posterior = GRU(action + observation, prior)`，每步都有观测校准
- 我们的 v1: `z_{t+1} = GRU(step_embed, z_t)`，没有观测
- **v2**: `z_{t+1} = GRU(cross_attn(z_t, encoder), z_t)`，每步从 encoder 提取信息

### 2.2 改进点

#### 改进 1: Attention Pooling 替代 Flatten（修复缺陷 A）

```
v1: Flatten(d_model × patch_num) → MLP → z_0
v2: LearnableQuery attend to encoder patches → z_0
```

具体实现：
```python
# 一个可学习的 query token
self.latent_query = nn.Parameter(torch.randn(1, 1, d_latent) * 0.02)

# Attention pooling
Q = self.latent_query.expand(B, -1, -1)  # (B, 1, d_latent)
K = self.proj_k(enc_patches)              # (B, patch_num, d_latent)
V = self.proj_v(enc_patches)              # (B, patch_num, d_latent)
z_0 = MultiHeadAttention(Q, K, V)         # (B, 1, d_latent) → squeeze → (B, d_latent)
```

好处：
- 没有 patch_num 和 d_latent 之间的硬性限制
- 适应任意 seq_len（12 patches 或 42 patches 都可以）
- z_0 是 encoder 信息的**语义加权聚合**，而非暴力展平
- 这也为未来扩展 seq_len=336 打下基础

#### 改进 2: Cross-Attention 移入 GRU 循环（修复缺陷 B + C）

```
v1:
  for t in range(K):
    z[t] = GRU(step_embed[t], z[t-1])   # 盲飞
  z = CrossAttn(z, encoder)               # 事后补

v2:
  for t in range(K):
    c[t] = CrossAttn(z[t-1], encoder)     # 每步从encoder取信息
    z[t] = GRU(c[t], z[t-1])              # GRU input 有意义了
```

这样 GRU 的 input 不再是无意义的 step_embed，而是**通过 attention 从 encoder 提取出的、与当前预测步相关的信息**。

关于 step_embed：可以**保留但改为加法**，在 cross-attention output 上加 step embedding，提供位置信号。也可以改为不用 step embed，直接靠 cross-attention 的 query 差异来区分步骤。需要消融实验决定。

#### 改进 3: Multi-Scale GRU 保持不变

fast+slow GRU 的设计已被消融实验证明有效（single-scale 差 20-30%），不需要改。只是信息输入方式变了：

```
v1: FastGRU(step_embed, z_fast), SlowGRU(step_embed, z_slow)
v2: FastGRU(cross_attn_out, z_fast), SlowGRU(cross_attn_out, z_slow)
```

### 2.3 v2 完整架构

```
Encoder Output (B, nvars, patch_num, d_model)

Phase 1 — Latent Initialization:
  z_0 = AttentionPooling(learnable_query, encoder_patches)  # (B*nvars, d_latent)

Phase 2 — Dynamics Rollout (K steps):
  For t = 0, 1, ..., K-1:
    # Step 1: 从 encoder 提取当前步相关信息
    c_t = CrossAttention(query=z_t, key/value=encoder_patches)  # 每步attend
    c_t = c_t + step_embed[t]  (可选: 提供位置信号)
    
    # Step 2: Multi-Scale GRU 更新
    z_fast = FastGRU(input=c_t, hidden=z_fast)
    if t % slow_interval == 0:
      z_slow = SlowGRU(input=c_t, hidden=z_slow)
    z_{t+1} = Gate(z_fast, z_slow) + residual + LayerNorm
    
    # Step 3: 解码当前步
    pred_t = Decoder(z_{t+1})

Phase 3 — 拼接输出:
  pred = Concat(pred_0, ..., pred_{K-1})[:pred_len]
```

### 2.4 对比 v1 vs v2

| 方面 | v1 | v2 |
|------|-----|-----|
| Latent 初始化 | Flatten + MLP (暴力压缩) | Attention Pooling (软选择) |
| GRU input | step_embed (跟数据无关) | cross-attention output (来自encoder) |
| Cross-Attention | rollout 后一次性 | 每步 rollout 内 |
| 长 seq_len 支持 | ✗ (seq=336 崩溃) | ✓ (attention 不受 patch_num 限制) |
| 误差累积 | 严重 (pred=720 +8.5%) | 应该缓解 (每步校准) |
| 类比 Dreamer | 不完整 (缺少观测) | 忠实 (每步有观测校准) |
| 参数量变化 | — | 略增 (attention pooling 层) |

### 2.5 预期效果

1. **短 horizon (96)**: 可能持平或小幅提升。v1 在 96 上其实已经不错（只输 1-3%），改进空间不大。
2. **中等 horizon (192, 336)**: v1 本来就赢，v2 应该赢更多，因为 GRU 每步都有校准。
3. **长 horizon (720)**: 最大的期望改进点。v1 输 8.5%，v2 的每步 cross-attention 应该显著缓解误差累积。
4. **seq_len=336**: 有望从"完全崩溃"变为"可用"，但需要实验验证。
5. **ETTh2**: v1 全败。需要分析 ETTh2 的数据特性，看 v2 能否改善。

### 2.6 风险和注意事项

1. **计算开销增加**: 每步 rollout 内做 cross-attention，训练时间约增加 30-50%。但 ETT 数据集小，应该可接受。
2. **过拟合风险**: 增加了参数（attention pooling），ETT 数据集小。需要适当 dropout。
3. **step_embed 是否保留**: 需要消融。可能 cross-attention 的 query 差异已经足够区分步骤。
4. **slow_interval 的最优值**: v2 中 slow GRU 的 input 也变了，最优 interval 可能不再是 2。

---

## 3. 消融实验设计

在 ETTh1 上做，4 个 horizon：

| 实验 | 说明 |
|------|------|
| v2-full | 完整 v2 (attention pooling + 循环内 cross-attn + step_embed) |
| v2-no-step-embed | 去掉 step embedding，纯靠 cross-attention |
| v2-flatten-init | 保留 Flatten 初始化，但 cross-attn 在循环内 |
| v2-post-attn (=v1) | 保留 attention pooling 初始化，但 cross-attn 在循环外 |
| v1 原版 | Flatten + 循环外 cross-attn（当前版本） |

这个消融可以独立验证每个改进点的贡献。

---

## 4. 实施优先级

### Phase 1: 等 v1 Final 结果（当前）

- 4 GPU 正在跑 v1 的最终实验（正确官方超参数 + 训练策略改进）
- 预计还需 2-3 小时
- 结果出来后，精确定位 v1 在哪些点上输、输多少

### Phase 2: 实现 v2 核心改进

1. 新建 `PatchTST_Dreamer_v2.py`，不修改 v1 代码
2. 实现 AttentionPooling、循环内 CrossAttention
3. 在 ETTh1 上快速验证（4 个 horizon）
4. 与 v1 对比

### Phase 3: 全面实验

- 如果 v2 在 ETTh1 上明显优于 v1，扩展到 ETTh2/ETTm1/ETTm2
- 消融实验
- 整理最终论文表格

### Phase 4: 根据结果微调

- 分析哪些数据集/horizon 还不够好
- 针对性调参或进一步改进

---

## 5. 备选改进（如果 v2 仍不够）

以下方向可在 v2 基础上叠加：

### 5.1 Decoder 端改进
- 当前 decoder 是简单的 MLP，每步独立解码
- 可以改为 **autoregressive decoder**：上一步的解码结果作为下一步 decoder 的输入
- 或者用 1D Conv 在 step 维度做平滑

### 5.2 KL 正则化
- 加 Dreamer 风格的 KL 散度 loss，鼓励 latent 平滑
- `loss = MSE + β * KL(z_posterior || z_prior)`
- 可能对长 horizon 有帮助

### 5.3 多尺度 slow interval
- 不只是 fast+slow，加一个 "ultra-slow" GRU (interval=4 或 8)
- 更好地捕获长期趋势

### 5.4 频域辅助
- 在 latent space 加 FFT 分解
- 让 fast GRU 建模高频，slow GRU 建模低频
- 更显式的多尺度分工

---

## 6. 思考日志（持续更新）

### 2026-04-12

**初始分析**：仔细阅读了 PatchTST_Dreamer.py 全部代码和 EXPERIMENTS.md。

核心发现：v1 的三个缺陷不是独立的，它们相互加剧：
- Flatten 把 patch 结构信息丢了 → z_0 质量差
- GRU 没有外部输入 → 只能靠 z_0 自回归，z_0 越差 rollout 越快偏
- Cross-attention 在最后才做 → 没法在 rollout 过程中纠偏

这三个问题形成了一个**恶性循环**。v2 的核心是打破这个循环：用 attention pooling 提升 z_0 质量，用循环内 cross-attention 在每步纠偏，让 GRU 有真正有意义的输入。

**待观察**：v1 Final 的结果（正确超参 + grad clip + type3 LR + patience 20）能改善多少。如果训练策略优化就能让 v1 赢大多数，那 v2 的紧迫性降低；如果 v1 还是输，则 v2 必须上。

**ETTh2 的疑问**：v1 在 ETTh2 上全败（4 个 horizon 全输）。ETTh2 跟 ETTh1 的区别是什么？需要分析数据特性。可能的原因：
- ETTh2 的动态比较"平"（低信噪比），GRU 建模反而引入噪声
- ETTh2 在 TSLib 官方脚本中用 e_layers=3（而 ETTh1 用 e_layers=1），说明 ETTh2 需要更深的 encoder
- v1 的 Final 实验已经改成 e_layers=3 for ETTh2，看看结果是否改善
