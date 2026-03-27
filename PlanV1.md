# PLAN: Kaggle S6E3 Customer Churn — 全自动化实验迭代

## 项目背景

你正在帮我参加 Kaggle Playground Series S6E3 比赛（Predict Customer Churn）。
- 比赛页面：https://www.kaggle.com/competitions/playground-series-s6e3
- 任务类型：二分类（预测客户是否流失）
- 评估指标：F1-score
- 原始数据来源：IBM Telco Customer Churn 数据集（7043行 × 21列），竞赛数据是基于此生成的合成数据
- 当前代码仓库：基于 https://github.com/BlamerX/Kaggle-Playground-Predection-Competition/tree/main/S6E3 整理
- 我的队友仓库：https://github.com/GOAT23WANG/Kaggle_S6E3

## 你的角色

你是一个高级数据科学家 + ML 工程师。你需要：
1. 完全自主地编写、运行、调试代码
2. 系统化地进行实验迭代，每轮实验只改变一个变量
3. 详细记录每次实验的结果
4. 所有改动都通过 Git 分支管理，方便我审核

## 核心规则（必须严格遵守）

1. **每次实验只改一个变量**：要么改特征，要么改模型，要么改参数。不要同时改多个东西，否则无法判断是哪个改动带来了提升。
2. **所有实验必须记录**：在 `experiments/log.csv` 中记录每次实验的编号、日期、改动内容、CV F1 分数、备注。
3. **Git 分支管理**：每个重要实验创建独立分支 `exp/xxx`，我审核通过后才合并到 main。
4. **CV 优先于 LB**：以 5 折 StratifiedKFold 的 CV F1 分数为主要判断依据，不要追 LB 分数。
5. **统一随机种子**：所有实验使用 `SEED = 42`，CV 折划分使用 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`。
6. **不要删除已有代码**：在已有代码基础上扩展，不要重写整个项目。

---

## 阶段一：跑通 Baseline（最高优先级）

### 目标
让现有代码能成功运行并生成 submission.csv，记录 baseline CV 分数。

### 步骤

1. **探索项目结构**
   - 列出仓库所有文件和目录结构
   - 找到主入口文件（.ipynb 或 .py）
   - 找到数据读取路径，确认 train.csv / test.csv / sample_submission.csv 是否存在

2. **修复数据路径**
   - 如果代码中使用 Kaggle 路径（如 `/kaggle/input/...`），改为本地相对路径
   - 如果数据文件不存在，提示我手动下载并告诉我放到哪个目录

3. **修复依赖问题**
   - 检查所有 import，确认依赖包已安装
   - 如果缺少包，列出需要安装的命令

4. **运行 baseline**
   - 运行主代码文件
   - 如果报错，逐个修复
   - 成功运行后，记录：
     - baseline CV F1 分数
     - 用了什么模型和参数
     - 用了哪些特征
     - 生成的 submission.csv 行数是否正确

5. **建立实验框架**
   - 创建 `experiments/log.csv`，格式如下：
     ```
     exp_id,date,description,features_version,model,cv_f1,lb_f1,notes
     001,2024-xx-xx,baseline,v0_raw,LGB_default,0.xxxx,,initial baseline
     ```
   - 创建 `experiments/` 目录存放每次实验的配置和结果

6. **Git 提交**
   ```
   git checkout -b baseline
   git add .
   git commit -m "baseline: 跑通初始代码, CV F1 = 0.xxxx"
   ```

### 完成标准
- [ ] submission.csv 已生成且格式正确
- [ ] CV F1 分数已记录
- [ ] experiments/log.csv 已创建
- [ ] baseline 分支已提交

---

## 阶段二：特征工程迭代

### 目标
在 baseline 基础上，通过特征工程提升 CV F1 分数。

### 实验队列（按优先级排列，逐一执行）

#### EXP-002：合并原始 IBM Telco 数据集
- 下载 IBM Telco Customer Churn 原始数据（如果尚未包含在项目中）
- 对比原始数据和竞赛数据的列名差异，做对齐
- 将原始数据合并到训练集中
- 训练模型，记录 CV F1
- 分支：`exp/002-merge-original-data`

#### EXP-003：交互特征
- 对数值特征两两组合，生成乘法、除法、差值特征
- 例如：MonthlyCharges × tenure, TotalCharges / tenure 等
- 训练模型，记录 CV F1
- 分支：`exp/003-interaction-features`

#### EXP-004：Target Encoding
- 对所有分类特征做 Target Encoding（注意必须在 CV 折内做，防止泄露）
- 替代原有的 Label Encoding 或 One-Hot Encoding
- 训练模型，记录 CV F1
- 分支：`exp/004-target-encoding`

#### EXP-005：分组聚合统计特征
- 按分类特征分组，对数值特征计算 mean / std / median / count
- 例如：按 Contract 类型分组的平均 MonthlyCharges
- 训练模型，记录 CV F1
- 分支：`exp/005-groupby-stats`

#### EXP-006：频率编码
- 对分类特征做频率编码（该类别出现次数 / 总行数）
- 训练模型，记录 CV F1
- 分支：`exp/006-frequency-encoding`

#### EXP-007：特征筛选
- 汇总 EXP-002 到 006 中所有有效的特征
- 用 feature importance 排名 + 逐一删除实验确定最优特征子集
- 将最终特征集封装为 `features.py` 模块
- 分支：`exp/007-feature-selection`

### 每个实验的标准流程

```
1. git checkout main
2. git checkout -b exp/xxx-描述
3. 修改代码（只改一个变量）
4. 运行实验，获取 CV F1
5. 将结果追加到 experiments/log.csv
6. git add . && git commit -m "exp/xxx: 描述, CV F1 = 0.xxxx"
7. 输出实验总结：改了什么、分数变化、是否值得保留
8. 等待我审核后再合并到 main
```

---

## 阶段三：模型调优

### 前提
阶段二完成后，最优特征集已确定。

#### EXP-008：多模型对比
- 使用最优特征集，分别训练：
  - LightGBM
  - XGBoost
  - CatBoost
- 每个模型用默认参数，记录各自的 CV F1
- 分支：`exp/008-multi-model-compare`

#### EXP-009：Optuna 超参搜索 - LightGBM
- 对 LightGBM 做 200 trials 的 Optuna 搜索
- 搜索空间包括：learning_rate, n_estimators, max_depth, num_leaves, subsample, colsample_bytree, reg_alpha, reg_lambda
- 记录最优参数和 CV F1
- 分支：`exp/009-lgb-optuna`

#### EXP-010：Optuna 超参搜索 - XGBoost
- 同上，对 XGBoost 做 200 trials
- 分支：`exp/010-xgb-optuna`

#### EXP-011：Optuna 超参搜索 - CatBoost
- 同上，对 CatBoost 做 200 trials
- 分支：`exp/011-cb-optuna`

#### EXP-012：多 Seed 平均
- 对每个调好参的模型，用 5 个不同 seed (42, 123, 456, 789, 2024) 训练
- 对预测概率取平均，减少方差
- 记录 CV F1
- 分支：`exp/012-multi-seed`

---

## 阶段四：集成与提交优化

#### EXP-013：简单加权平均
- 对 LGB / XGB / CatBoost 的预测概率做加权平均
- 尝试多组权重，找到 CV F1 最高的组合
- 分支：`exp/013-weighted-blend`

#### EXP-014：Stacking
- Layer 1：LGB + XGB + CatBoost 的 OOF 预测
- Layer 2：Ridge 或 LogisticRegression 作为元学习器
- 对比 Stacking vs Blending 的 CV F1
- 分支：`exp/014-stacking`

#### EXP-015：F1 阈值精细搜索
- 在最终 Ensemble 的 OOF 预测上，以 0.001 粒度搜索最优分类阈值
- 不要默认用 0.5，搜索范围 0.30 ~ 0.70
- 记录最优阈值和对应的 CV F1
- 分支：`exp/015-threshold-tuning`

#### EXP-016：最终提交
- 用最终方案（最优特征 + 最优模型 + 最优集成 + 最优阈值）生成 submission.csv
- 检查 submission.csv 格式：无 NaN，行数与 test.csv 匹配，列名正确
- 分支：`exp/016-final-submission`

---

## 项目目录结构（期望）

```
Kaggle_S6E3/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── telco_original.csv          # IBM 原始数据（如果合并使用）
├── models/
│   ├── (队友原有的代码文件)
├── src/                             # 新增模块化代码
│   ├── features.py                  # 特征工程模块
│   ├── train.py                     # 训练框架
│   ├── ensemble.py                  # 集成模块
│   └── utils.py                     # 工具函数（CV、阈值搜索等）
├── experiments/
│   ├── log.csv                      # 实验记录表
│   └── configs/                     # 每次实验的参数配置
├── submissions/
│   └── submission_expXXX.csv        # 各版本提交文件
├── notebooks/                       # 探索性分析
└── README.md
```

---

## 输出规范

每次实验完成后，输出以下格式的总结：

```
========================================
实验 EXP-XXX: [描述]
========================================
改动内容: [具体改了什么]
Baseline CV F1: [之前的分数]
当前 CV F1:     [这次的分数]
分数变化:       [+0.xxxx 或 -0.xxxx]
结论:           [保留 / 丢弃 / 待定]
分支:           exp/xxx-描述
下一步建议:     [接下来应该尝试什么]
========================================
```

---

## 重要提醒

- 数据路径：代码中不要使用绝对路径，全部使用相对路径
- 如果遇到缺少数据文件的情况，不要自己生成假数据，而是暂停并提示我下载
- 如果某个实验导致 CV F1 下降超过 0.005，立即停止该方向并回退
- 每完成一个阶段（不是每个实验），暂停并等待我确认再进入下一阶段
- 不要一次性把所有实验都跑完，按顺序逐个执行，每个实验完成后告诉我结果