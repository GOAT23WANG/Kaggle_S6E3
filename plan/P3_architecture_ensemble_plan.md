# P3: 架构与集成创新实验计划

> **状态**: 待审核 (DESIGN ONLY — 用户审核后再执行)  
> **日期**: 2026-03-22  
> **基线**: EXP-P0-003 = 0.919170 OOF ROC-AUC (10-fold XGBoost V16-style)  
> **来源**: blamerx GitHub Memory 文件 (V1-V44 完整开发历史分析)

---

## 一、分析摘要：blamerx 做了什么，我们该学什么

### 1.1 blamerx 的成绩排行

| 版本 | 方法 | OOF AUC | LB Score | 备注 |
|------|------|---------|----------|------|
| V16b | XGB 单模 20-fold | 0.91925 | 0.91680 | 最强 XGB 单模 |
| V21 | TabM (k=32) + V16特征 | 0.91898 | 0.91682 | 最强 NN |
| V27 | Two-Stage Ridge→XGB | 0.91920 | 0.91683 | Ridge 预测作为特征 |
| V37 | Two-Stage Ridge→XGB (V36特征) | 0.91921 | 0.91684 | 含隐藏特征 |
| **V39** | **Two-Stage Ridge→XGB (10 seeds)** | **0.91934** | **0.91687** | **最佳 LB (单架构)** |
| V42 | NODE Diverse Meta-Model (6模型) | 0.91922 | **0.91700** | **最佳 LB (集成)** |
| V44 | RealMLP Optimized | 0.91913 | 0.91660 | 最新 NN |

### 1.2 我们 vs blamerx 对比

| 维度 | 我们 (当前) | blamerx (最佳) | 差距 |
|------|------------|---------------|------|
| XGB 单模 OOF | 0.91917 (10-fold) | 0.91925 (20-fold) | -0.00008 |
| 最佳单架构 OOF | 0.91917 | 0.91934 (V39 multi-seed) | -0.00017 |
| 是否有 Two-Stage | ❌ | ✅ Ridge→XGB | 缺失 |
| 是否有 Multi-Seed | ❌ | ✅ 10-seed 平均 | 缺失 |
| 是否有 NN 模型 | ❌ | ✅ TabM / RealMLP | 缺失 |
| 是否有 集成 | ❌ | ✅ 简单平均 / Meta-Model | 缺失 |

### 1.3 核心发现

**可行方向（blamerx 已验证有效）：**
1. **Two-Stage Ridge→XGB** — Ridge 线性预测作为 XGB 特征，ridge_pred 排名第 3-7 位重要性，提供了 XGB 无法自行捕获的正交线性信号
2. **Multi-Seed 平均** — 10 seeds 平均在 XGB 上有效 (+0.00013 OOF)，但在 LightGBM/CatBoost 上无效
3. **20-Fold CV** — 从 10→20 折，每折训练数据从 90%→95%，稳定获得 +0.00008 OOF
4. **TabM 神经网络** — 不同归纳偏差，OOF 0.91898，提供集成多样性
5. **简单平均集成** — XGB + TabM + Two-Stage 等多模型简单平均，OOF ~0.91933

**已确认的死胡同（不要再试）：**
- 特征扩展/剪枝（V16 特征集已饱和）
- CatBoost / LightGBM 替代 XGB（在重 FE 数据集上均弱于 XGB）
- 参数微调（我们的 P2 和 blamerx 的 Optuna 均确认参数已近最优）
- 伪标签（blamerx 0/18+ 次尝试全部失败）
- 复杂 Meta-Model（NODE / CCP-Net 勉强追平简单平均）
- DART / Focal Loss / Label Smoothing / 噪声剪枝

---

## 二、P3 实验队列设计

### 停止规则（沿用 P1/P2）

| 规则 | 阈值 |
|------|------|
| 最小有效增益 | ≥ 0.0002 ROC-AUC |
| 严重下降 | ≤ -0.0010 ROC-AUC |
| 连续无提升暂停 | 3 次连续 |

### 实验总览

| 实验 ID | 方向 | 预期增益 | 复杂度 | 优先级 |
|---------|------|----------|--------|--------|
| P3-001 | Two-Stage Ridge→XGB | +0.0001~0.0003 | 中 | 🔴 高 |
| P3-002 | 20-Fold CV | +0.0001~0.0001 | 低 | 🟡 中 |
| P3-003 | Multi-Seed XGB (5 seeds) | +0.0001~0.0002 | 低 | 🟡 中 |
| P3-004 | P3-001 + P3-003 组合 | +0.0002~0.0004 | 中 | 🟡 中 (依赖 P3-001) |
| P3-005 | TabM 神经网络 | 不直接提升 XGB | 高 | 🔵 低 (集成储备) |

> **注意**：P3 实验与 P1/P2 不同，不再是 "微调同一架构" 而是 "引入新的架构维度"。
> 预期增益可能较小但来源于正交方向，因此累加性更强。

---

### P3-001: Two-Stage Ridge→XGB（最高优先级）

**灵感来源**: blamerx V27 (OOF 0.91920, LB 0.91683)

**核心思路**: 
训练一个 Ridge 线性回归模型，用其 OOF 预测值 `ridge_pred` 作为额外特征加入 XGB。Ridge 捕获全局线性关系（如 tenure 与 churn 的线性相关），XGB 单靠树分裂很难高效重建这些模式。

**实现方案**:
1. 在每个外层 CV fold 内部，用内层 5-fold CV 生成 Ridge 的 OOF 预测（防止泄漏）
2. Ridge 输入特征：所有数值型特征 + ORIG_proba 映射（StandardScaler 预处理）
3. 将 `ridge_pred` 作为一列新特征加入现有 V16 特征集
4. XGB 训练参数不变

**预期**:
- blamerx V27 显示 ridge_pred 重要性排名第 3（0.043）
- OOF 增益预期 +0.0001~0.0003
- 训练时间增加 ~30%（Ridge 训练极快，主要开销在嵌套 CV 结构）

**风险**:
- 我们的 OOF 0.91917 vs blamerx V27 OOF 0.91920，差值很小
- blamerx 用 nested CV 修复后（V28c→LightGBM 无增益），但 XGB 确实有增益

**代码改动**:
- `src/features.py`: 新增 `build_ridge_predictions()` 函数
- `src/train.py`: 在 `run_unified_v16()` 中添加 Ridge 特征生成步骤

**环境变量控制**:
```bash
EXPERIMENT_ID=EXP-P3-001
PROFILE=unified_v16
ENABLE_RIDGE_FEATURE=1
RIDGE_INNER_FOLDS=5
```

---

### P3-002: 20-Fold CV

**灵感来源**: blamerx V16b (OOF 0.91925, LB 0.91680)

**核心思路**:
从 10-fold 切换到 20-fold，每个 fold 获得 95% 训练数据（vs 90%），且最终预测取 20 个模型的平均值（降低方差）。

**实现方案**:
1. 将 `N_FOLDS=20` 作为环境变量传入
2. 其余完全不变

**预期**:
- blamerx 获得 +0.00008 OOF, +0.00001 LB
- 训练时间约 2x（20 folds vs 10 folds）
- 收益虽小但近乎 "免费"，无任何负面风险

**环境变量控制**:
```bash
EXPERIMENT_ID=EXP-P3-002
PROFILE=unified_v16
N_FOLDS=20
```

---

### P3-003: Multi-Seed XGBoost 平均（5 seeds）

**灵感来源**: blamerx V39 (在 Two-Stage 基础上 multi-seed, +0.00013 OOF)

**核心思路**:
使用 5 个不同的随机种子训练相同的 XGB 管道，最终预测取平均。降低随机性带来的方差。

**实现方案**:
1. 新增 `MULTI_SEED=1` 环境变量开关
2. 种子列表: `[42, 123, 456, 789, 2026]`
3. 对每个种子跑完整 10-fold CV，得到 5 组 OOF
4. 最终 OOF = 5 组的算术平均

**预期**:
- blamerx V39 显示 XGB multi-seed 10 seeds → +0.00013 OOF
- 5 seeds 预期 +0.00005~0.00010 OOF
- 训练时间 5x

**注意**:
- blamerx 在 LightGBM 和 CatBoost 上尝试 multi-seed 均无增益
- **只对 XGB 有效**

**环境变量控制**:
```bash
EXPERIMENT_ID=EXP-P3-003
PROFILE=unified_v16
MULTI_SEED=1
N_SEEDS=5
```

---

### P3-004: Two-Stage Ridge→XGB + Multi-Seed（组合实验）

**灵感来源**: blamerx V39 = 完整组合 (OOF 0.91934, LB 0.91687)

**前提**: P3-001 和 P3-003 至少有一个达到最小有效增益

**核心思路**:
组合 Two-Stage Ridge 特征 + Multi-Seed 平均，直接复刻 blamerx 的最佳单架构方案。

**实现方案**:
- P3-001 + P3-003 的代码合并

**预期**:
- OOF 目标: 0.9193+ (接近 blamerx V39 的 0.91934)
- 训练时间约 5x (multi-seed) + 30% (ridge) ≈ 6.5x 基线时间

**环境变量控制**:
```bash
EXPERIMENT_ID=EXP-P3-004
PROFILE=unified_v16
ENABLE_RIDGE_FEATURE=1
MULTI_SEED=1
N_SEEDS=5
```

---

### P3-005: TabM 神经网络（集成储备）

**灵感来源**: blamerx V21 (OOF 0.91898, LB 0.91682)

**核心思路**:
TabM (ICLR 2025 BatchEnsemble MLP) 使用与 XGB 完全不同的归纳偏差（平滑决策边界 vs 阶梯形分裂）。不直接提升 XGB 分数，但为未来集成提供多样性锚点。

**实现方案**:
1. 安装 `pytabkit` 包
2. 使用 `tabm-mini-normal` 配置
3. k=32 BatchEnsemble heads（blamerx 确认 k=64 更差）
4. PiecewiseLinear 数值嵌入
5. 16 个原始分类特征用字符串类型保留（cat_col_names），其余数值特征用 float32
6. 10-fold CV，每 fold 训练 ~40 min

**预期**:
- OOF ~0.918-0.919（接近但不超过 XGB）
- 与 XGB 的相关性较低（blamerx 显示 TabM 在 fold 3/7 上优于 XGB）
- 简单平均 XGB + TabM 可能达到 0.919+ OOF

**注意**:
- 这是一个 "储备" 实验，直到需要集成时才产生价值
- 建议在 P3-001/002/003 完成后再考虑是否执行

**环境变量控制**:
```bash
EXPERIMENT_ID=EXP-P3-005
PROFILE=tabm_v16
```

---

## 三、执行顺序建议

```
P3-001 (Two-Stage Ridge→XGB)     ← 最关键，新增正交线性信号
    ↓
P3-002 (20-Fold CV)              ← 最简单，近乎零风险
    ↓
P3-003 (Multi-Seed 5 seeds)     ← 降低方差，需更多计算时间
    ↓
[评估点: 若 P3-001 有增益]
    ↓
P3-004 (组合: Ridge + Multi-Seed) ← 直接复刻 blamerx V39
    ↓
[评估点: 是否需要集成]
    ↓
P3-005 (TabM 神经网络)           ← 集成储备，仅在需要时执行
```

---

## 四、与 P1/P2 的差异

| 维度 | P1/P2 | P3 |
|------|-------|-----|
| 方向 | 在同一架构内微调（开关特征/调参数） | 引入新架构维度（两阶段/多种子/NN） |
| 预期增益来源 | 减少冗余/调整偏差 | 增加正交信号/降低方差 |
| 累加性 | 互相抵消（同一模型的不同视角） | 可能叠加（不同来源的增益） |
| 风险 | 低（已知不太可能有大提升） | 中（新代码复杂度更高） |

---

## 五、停止条件与退出策略

1. 如果 P3-001 + P3-002 + P3-003 连续 3 个均无效增益 → 暂停 P3 队列
2. 如果 P3-004 组合后 OOF > 0.9193 → 成功达到 blamerx 水平，考虑是否进入集成阶段
3. 如果所有尝试后 OOF 仍停留在 0.9191-0.9192 → 说明我们的特征工程与 blamerx V16b 等价，差距仅来自 20-fold / multi-seed 的微优化

---

## 六、代码改动清单（预估）

| 文件 | 改动 | 涉及实验 |
|------|------|---------|
| `src/features.py` | 新增 `build_ridge_predictions()` | P3-001, P3-004 |
| `src/train.py` | 支持 `ENABLE_RIDGE_FEATURE`、`N_FOLDS` 环境变量、`MULTI_SEED` 逻辑 | P3-001~004 |
| `src/train.py` | 新增 TabM profile | P3-005 |
| `requirements.txt` | 添加 `pytabkit` (仅 P3-005) | P3-005 |

---

*等待用户审核后再开始执行。*
