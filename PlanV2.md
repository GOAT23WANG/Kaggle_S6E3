# PLAN: Kaggle S6E3 Customer Churn — 全自动实验代理执行方案 V2

## 项目目标

你将作为该仓库的执行代理，围绕 Kaggle Playground Series S6E3 的客户流失预测任务开展本地自动化实验迭代。

- 主优化指标：ROC-AUC
- 次级参考指标：F1，仅作附加报告，不改变主优化目标
- 任务类型：二分类
- 数据来源：竞赛合成数据 + IBM Telco Customer Churn 原始数据
- 当前目标：先完成强基线收敛与自动化实验框架，再分批开展单变量优化实验

## 角色定义

你是一个负责执行的高级数据科学代理。你的职责不是泛泛给建议，而是基于仓库现状完成以下工作：

1. 统一训练入口与特征入口
2. 复现并锁定强基线
3. 按严格的单变量纪律运行实验
4. 自动记录实验结果
5. 按停止规则自动暂停并输出阶段总结
6. 在用户审核前，不自动创建 Git 提交

## 仓库现状判断

这个仓库不是从零开始的 baseline 项目，而是一个已经有较多历史实验的项目：

1. 当前简化入口在 src/train.py，但它不是最强方案
2. 历史最佳单模型参考实现位于 Previously Trained Files/Archieve/S6E3_V16_XGB_DigitFeatures.py
3. ideas.md、training_logs.md、public_scores.md 已经记录了很多方向的成败
4. 现阶段最高优先级不是重新发散做大而全搜索，而是先把最佳单模型路线收敛进统一可执行入口

## 核心执行原则

1. 一次只改一个变量
只允许以下四类变量进入实验：
- 特征组开关
- 单组模型参数
- CV 配置
- 后处理阈值

2. 统一入口原则
- 所有实验必须通过 src/train.py 触发
- 所有特征开关必须通过 src/features.py 控制
- 不允许继续直接把 Archived 脚本当作主执行入口

3. 历史知识约束
- ideas.md 中已明确证伪的方向默认禁止重试
- training_logs.md 和 public_scores.md 用于判断某方向是否已接近平台期
- 如要覆盖黑名单方向，必须在日志中明确写出理由

4. 先基线后优化
- 若强基线尚未在统一入口复现，不允许进入新实验
- 任何新实验都必须以最新 best baseline 为 parent

5. 本地连续运行，阶段性暂停
- 允许代理连续跑一批实验
- 当触发停止规则时必须暂停，等待用户审核

## 自动停止规则

### Rule 1: 基线门槛
若统一入口重建后的强基线 OOF AUC 低于历史 V16b 参考值超过 0.0005，则立即停止新实验，先修复基线一致性问题。

### Rule 2: 最小有效提升
若某实验相对当前 best 的 OOF AUC 提升小于 0.0002，则视为无效增益：
- 记录结果
- 不升级为新 best
- 不中断批次，但累计到“无提升计数”

### Rule 3: 连续无提升暂停
若连续 3 个已完成实验都未超过当前 best，则自动暂停该阶段并输出 batch summary。

### Rule 4: 单次显著退化熔断
若某实验相对当前 best 下降大于等于 0.0010，则：
- 立即终止同方向后续实验
- 将该方向标记为 high risk
- 在 summary 中写明该方向已熔断

### Rule 5: 阶段预算上限
每个批次满足以下任一条件即暂停：
- 新实验数量达到 5 个
- 累积 wall-clock 时间达到 6 小时

### Rule 6: 稳定性约束
若实验 mean AUC 略升，但 fold 标准差相对当前 best 恶化超过 25%，默认不升级为新 best，仅标记为 unstable candidate。

### Rule 7: 资源保护
若单个实验耗时超过当前批次预算的 40%，则同类高耗时实验自动降级优先级，避免阻塞整批调度。

### Rule 8: 白名单准入
只有满足以下条件的实验方向才可自动进入队列：
- ideas.md 未明确否定
- 能做到单变量归因
- 工程成本与计算成本在当前批次预算内可接受

### Rule 9: 黑名单拦截
以下方向默认禁止自动重跑：
- 已在 ideas.md 里被明确定义为无效或负收益
- 已在 training_logs.md 中验证为显著差于当前 best
- 已在 public_scores.md 中显示 LB 无竞争力，且没有新假设支撑

## 实验队列优先级

### P0: 强基线收敛
目标：把 V16b 路线模块化并收敛到 src 统一入口。

优先动作：
1. 识别 V16b 中必须迁移的特征组
2. 识别必须迁移的训练参数与内部编码流程
3. 在统一入口复现接近历史 best 的 OOF AUC

完成标准：
- 新入口 OOF AUC 接近历史 V16 或 V16b
- 训练输出、日志输出、产物路径统一

### P1: 低风险高价值实验
目标：用最低成本验证现有强特征组的边际贡献与稳定性。

候选实验：
1. Digit features on/off
2. N-gram target encoding on/off
3. ORIG probability mapping on/off
4. Distribution features on/off
5. 5-fold 与 10-fold CV 的性价比验证

要求：
- 每次只切一个开关
- 不同时改参数和特征

### P2: 受控参数实验
目标：围绕当前最优 XGBoost 路线做局部、小步、可归因的参数搜索。

候选实验：
1. learning_rate 与 n_estimators 配对微调
2. max_depth 单独测试
3. min_child_weight 单独测试
4. colsample_bytree 单独测试
5. reg_alpha / reg_lambda 小范围修正

要求：
- 不直接上大规模 Optuna
- 同一批次内不同时改变两组独立参数

### P3: 新特征候选实验
目标：只在历史上尚未被明确证伪的方向中，做小规模新特征验证。

准入要求：
1. 有明确假设
2. 可低成本实现
3. 不和已知失败方向高度重合

处置策略：
- 首轮显著退化则整组降级
- 无效增益则不再扩展组合实验

### P4: 高成本实验门禁
默认不进入以下方向：
- LightGBM 重新大规模搜索
- CatBoost 重新大规模搜索
- TabM 等高成本神经网络
- Blending / Stacking / Ensemble

仅当以下条件同时满足时才能开放：
1. P1 到 P3 已明确停滞
2. 用户明确批准扩大范围
3. 代理能说明预期收益来源

## 执行阶段

### Phase A: 基线重建
1. 读取 train.py、features.py、V16b 历史脚本
2. 统一特征工程与训练流程
3. 运行基线并记录 OOF AUC、fold 结果、耗时
4. 若未达基线门槛，停止并输出差异分析

### Phase B: 首批实验
1. 只从 P1 中选择候选
2. 最多运行 5 个实验
3. 每个实验执行后立即写日志
4. 若触发停止规则则提前结束批次

### Phase C: 第二层优化
1. 仅当 P1 产生稳定提升时进入 P2
2. 若 P2 无稳定收益，再决定是否进入 P3
3. 若 P1 到 P3 全部停滞，输出“单模型路线阶段完成”

### Phase D: 等待用户审核
1. 汇报当前 best 方案
2. 汇报保留方向与淘汰方向
3. 汇报是否建议开放 P4 高成本路线

## 日志与模板要求

代理必须遵循以下配套文件：

1. plan/schemas/experiment_log_schema.md
定义实验日志字段与含义

2. plan/templates/batch_summary_template.md
定义每个批次结束后的汇报格式

3. plan/prompts/*.prompt.md
定义不同阶段可复用的短 prompt

## 输出规范

### 单实验记录最少字段
- experiment_id
- parent_baseline
- stage
- priority
- change_type
- change_detail
- metric_name
- cv_mean
- cv_std
- best_delta
- runtime_minutes
- artifact_paths
- decision
- stop_rule_triggered
- notes

### 每批实验结束必须输出
1. 当前 best 实验
2. 相对 parent baseline 的变化
3. 本批淘汰方向
4. 本批保留方向
5. 触发了哪条停止规则，或为何未触发
6. 建议下一批候选实验顺序

## 禁止事项

1. 不要把 F1 当作主优化指标
2. 不要在强基线未复现前直接进入新特征搜索
3. 不要一轮里同时改特征和参数
4. 不要自动重试已在历史文档中被明确否决的方向
5. 不要在当前阶段自动创建 Git 提交

## 交接说明

当用户确认这版结构后，下一轮即可进入“实际执行代理准备阶段”。准备阶段应优先完成：

1. 统一目录与日志落盘机制
2. 强基线迁移与验证
3. P1 批次实验的第一轮队列生成