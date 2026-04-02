# TianShouDSSAT 可执行改造路线图

## 1. 目标

本路线图的目标不是直接“换算法”或“堆更多功能”，而是先把项目重构成一个更适合农业强化学习研究的基础平台，使以下几类内容彼此解耦：

- 基础能力：DSSAT 环境接入、作物配置、原始状态获取、动作下发、episode 汇总
- 任务定义：reward、observation、action 设计
- 研究方案：算法、训练器、采样方式、超参数、baseline
- 实验管理：seed、日志、评估、结果导出、train/test 划分

如果这四层没有解耦，项目后续会很快进入“每做一个实验都要改环境底层代码”的状态，既不利于研究，也不利于维护。

## 2. 我对“解耦”的理解

### 2.1 什么叫解耦

在本项目里，解耦不是简单地“拆文件”，而是要做到：

- 换一个奖励函数，不需要改 DSSAT 接口层
- 换一个观测组合，不需要改训练入口主逻辑
- 换一个动作定义，不需要改底层 `DssatPdi`
- 换一个算法，不需要改作物配置或 reward 代码
- 做 baseline、rule-based、离线评估时，复用同一套环境与指标导出

### 2.2 什么不该耦合

当前最不应该耦合在一起的内容有：

- 原始环境适配 与 研究任务定义
- 作物资源装配 与 动作桶离散化
- episode 运行 与 reward 聚合逻辑
- 训练器选择 与 观测工程
- 评估指标导出 与 TensorBoard 日志

### 2.3 什么应该稳定

项目里最应该长期稳定的是“基础能力层”，即：

- 能正确启动 DSSAT
- 能得到一致的原始状态
- 能把动作可靠地下发给 DSSAT
- 能在回合结束时导出统一 KPI

而 reward、observation、action、algorithm 都应该是可插拔的研究层对象，而不是写死在包装器或训练入口里。

## 3. 当前耦合点定位

基于当前仓库实现，主要耦合点如下：

### 3.1 `dssat_generic_wrapper.py` 同时承担了过多职责

当前文件同时负责：

- 作物资源路径拼装
- Wheat 兼容补丁
- 观测筛选
- 历史拼接
- 动作离散化
- reward 合并

这意味着任何研究问题一变化，都容易回到这个核心文件改逻辑，后续会越来越难维护。

### 3.2 `train_dssat.py` 把“实验协议”和“算法实例化”硬编码在一起

当前训练入口直接固定：

- 历史长度
- 环境工厂
- seed 用法
- train/test env 数量
- Rainbow/C51 结构
- 训练轮数与采样参数

这使得项目当前更像一个单实验脚本，而不是研究平台入口。

### 3.3 reward 仍留在底层 vendored 环境配置中

`lib/.../configs/rewards.py` 目前属于底层环境的一部分，但 reward 本质上是研究任务定义，不应该埋在 DSSAT 接口层内部。否则：

- 换一个 reward 要动底层环境
- baseline 与 RL 用不同 reward 时不方便复用
- 多目标评估与训练目标难以分离

### 3.4 observation/action 设计没有形成显式接口

当前观测和动作主要隐含在：

- `CROP_SPECIFIC_CONFIGS`
- `_extract_features`
- `_map_action`

这会导致“当前任务定义”和“作物基础配置”混在一起，不利于后续快速试验多个 observation / action 方案。

## 4. 第一阶段最值得改的代码点

第一阶段不建议大改算法，而是优先重构“基础能力层”和“任务定义层”的边界。最值得先改的代码点，按优先级如下。

### 4.1 第一优先：把 `dssat_generic_wrapper.py` 从“大一统包装器”收缩为“基础环境适配器”

目标：

- 只负责 DSSAT 启动、作物资源选择、原始状态交互、episode 生命周期管理
- 不再在这里写死 reward 设计、动作桶、研究型观测裁剪

建议保留在该文件中的职责：

- crop resource resolution
- `DssatPdi` 初始化
- reset / step / close 基础流程
- 原始状态与 context 的安全返回
- episode summary 收集

建议移出的职责：

- `_map_action`
- `_extract_features`
- `_update_history`
- reward list 合并

为什么这一点最重要：

- 它能直接决定后续所有 reward/observation/action 试验是否能低成本进行
- 它是当前整个项目最大的研究耦合点

### 4.2 第二优先：把 reward 从底层环境配置中上移为项目级可插拔模块

目标：

- 让 reward 成为训练任务的一个配置项
- 允许“训练 reward”和“评估 KPI”分离

建议原则：

- DSSAT 环境只输出原始状态与过程量
- reward 由项目层 `RewardFn` 计算
- 评估时无论训练用什么 reward，都统一输出 agronomic KPI

### 4.3 第三优先：把 observation/action 定义改成“任务对象”

目标：

- 同一个 crop，可以挂不同 observation schema
- 同一个 crop，可以挂不同 action schema
- 算法入口只依赖 schema 输出的 `observation_space` 和 `action_space`

这样就能自然支持：

- 低维专家特征版
- 含天气预报版
- 周尺度动作版
- 阶段门控版

### 4.4 第四优先：把训练入口改成“实验装配器”

目标：

- `train_dssat.py` 不再直接承载所有研究假设
- 它只负责按配置装配 env、task、policy、collector、logger、evaluator

这样以后换算法时，不需要回头改环境逻辑。

### 4.5 第五优先：建立统一评估与 episode KPI 导出层

目标：

- 训练日志之外，再有一套稳定的农业指标导出
- 用统一结构支持 RL、baseline、rule-based、ablation 的对比

没有这个层，后续研究会出现“训练能跑，但结果很难横向比较”的问题。

## 5. 每一项改动对应哪些文件

下面分“当前应修改文件”和“建议新增文件”两部分说明。这里的“新增”是为了形成稳定边界，不是为了过度设计。

### 5.1 改动一：收缩基础环境适配器

**当前应修改文件**

- `dssat_generic_wrapper.py`
- `fix_dssat_pdi.py`

**建议新增文件**

- `envs/dssat_base_env.py`
- `envs/crop_registry.py`
- `envs/resource_resolver.py`

**改造意图**

- `dssat_generic_wrapper.py` 当前逻辑拆分后，`dssat_base_env.py` 只提供原始 dict state + info + episode summary
- `crop_registry.py` 只管理作物元数据、模板、辅助文件、默认实验编号
- `resource_resolver.py` 只处理路径、模板 fallback、环境定位

### 5.2 改动二：把 observation 设计独立出来

**当前应修改文件**

- `dssat_generic_wrapper.py`

**建议新增文件**

- `tasks/schemas.py`
- `tasks/forecast.py`

**改造意图**

- observation schema 决定“选哪些变量”
- forecast provider 决定“未来天气从哪里来、如何做摘要”
- normalizer 决定“怎么缩放”

这样未来改成 forecast-aware observation 或 stage-aware observation，不需要碰 DSSAT 基础层。

### 5.3 改动三：把 action 定义独立出来

**当前应修改文件**

- `dssat_generic_wrapper.py`

**建议新增文件**

- `tasks/action_schemas.py`
- `tasks/action_constraints.py`

**改造意图**

- `action_schemas.py` 定义离散桶、连续动作、参数化动作
- `action_constraints.py` 负责阶段约束、预算约束、action masking

这样未来从“逐日固定桶”改成“周尺度 + 预算约束”时，不需要重新设计整个 env。

### 5.4 改动四：把 reward 与 KPI 计算独立出来

**当前应修改文件**

- `lib/gym_dssat_pdi_official/gym-dssat-pdi/gym_dssat_pdi/envs/configs/rewards.py`
- `dssat_generic_wrapper.py`

**建议新增文件**

- `tasks/reward_schemas.py`
- `tasks/kpi_extractors.py`
- `evaluation/episode_report.py`

**改造意图**

- reward schema 只负责训练用 reward
- kpi extractor 只负责汇总 yield、water、nitrogen、leaching、profit 等指标
- `episode_report.py` 统一生成对比友好的结果记录

理想状态下，底层 vendored `rewards.py` 应逐步退出主训练链路，只保留兼容用途。

### 5.5 改动五：把训练入口改成实验装配器

**当前应修改文件**

- `train_dssat.py`
- `train_rainbow.py`

**当前主链路文件**

- `experiments/build_env.py`
- `experiments/build_policy.py`

**改造意图**

- `build_env.py` 负责把 base env + task schema 装配成训练环境，并接受轻量对象、字典或直接参数覆盖
- `build_policy.py` 单独处理 Tianshou/Rainbow 等算法
- `train_dssat.py` 只负责训练协议、collector、logger 与保存逻辑

这样项目以后支持 PPO、SAC、rule-based、MPC baseline 时，不会污染环境层代码。

### 5.6 改动六：重构测试，使其覆盖农业行为而非只覆盖 API

**当前应修改文件**

- `tests/test_suite_v2.py`

**建议新增文件**

- `tests/test_action_schema.py`
- `tests/test_observation_schema.py`
- `tests/test_reward_schema.py`
- `tests/test_episode_kpi.py`
- `tests/test_seed_reproducibility.py`

**改造意图**

- 把“接口是否可跑”与“农业任务定义是否正确”分开测试
- 先保护 schema 层，再保护 env 层，再保护训练装配层

## 6. 更适合农业 RL 的 reward / observation / action 重构方案

这里给出一个“第一阶段可落地”的方案，重点是合理、稳健、可比较，而不是一开始就追求最复杂。

### 6.1 Observation 重构方案

建议把 observation 拆成四组特征。

#### A. Crop state

- `istage` / `vstage`
- `xlai` 或 `lai`
- `topwt`
- `grnwt`
- `rtdep`
- `swfac`
- `nstres`

#### B. Soil-water / nitrogen state

- `sw` 的聚合统计，例如根层平均含水量、剖面最小/最大含水量
- `wtdep`
- `cumsumfert`
- `totir`
- `trnu`
- `tleachd`
- `tnoxd`

#### C. Weather state

- 当日 `rain`, `tmin`, `tmax`, `srad`
- 未来 3～7 天天气预报摘要
- 可选：累计降雨或阶段累计热量

#### Weather forecast 接入原则

- 原始 forecast 序列保留在 `info` / metadata 中
- 当前 MLP 训练默认只接收 forecast 摘要进入 observation
- 第一阶段推荐窗口就是未来 3 天和未来 7 天
- 每个窗口建议先压缩为：
  - 累积降雨
  - 平均气温
  - 平均辐射
- 如果没有在线天气 API，允许先从 `.WTH` 文件按当前 `yrdoy` 截取未来窗口，作为离线 forecast provider

这样做的核心原因是：forecast 属于研究层输入，而不是 DSSAT 基础层状态；因此它应该由 task 层 provider 注入，而不是写死在底层 env 中。

#### D. Management memory

- 距离上次施肥天数
- 距离上次灌溉天数
- 当前累计施氮
- 当前累计灌溉
- 剩余预算

**第一阶段建议做法**

- 先不要直接把全部土壤剖面原样塞给网络
- 先做“专家摘要特征版”
- 所有特征统一做 normalization

### 6.2 Action 重构方案

第一阶段建议从“逐日 36 个组合动作”改为“周尺度决策 + 条件约束”。

#### 方案 A：最稳妥的第一阶段方案

- 每 7 天决策一次
- 动作为二元组合：
  - nitrogen: `[0, 30, 60, 90]`
  - irrigation: `[0, 10, 20, 30]`
- 阶段约束：
  - 生育早期允许小剂量施氮
  - 特定后期阶段禁施或限施
- 预算约束：
  - season nitrogen budget
  - season irrigation budget

优点：

- 实现难度低
- 训练难度显著下降
- 更接近农业管理节律

#### 方案 B：面向后续扩展的方案

- 动作拆为：
  - 是否行动
  - 行动类型
  - 行动剂量

这更接近参数化动作空间，适合后续转 PPO/SAC 或自定义 actor-critic。

**第一阶段不建议**

- 一上来做完全连续动作
- 一上来做复杂多头动作网络

因为当前项目最先需要的是“任务定义稳定”，不是“动作表达最强”。

### 6.3 Reward 重构方案

第一阶段建议采用“终局目标 + 轻量过程惩罚”的结构。

#### 推荐训练 reward

可以用如下形式：

`R_t = 阶段奖励 + 终局奖励`

其中：

- 阶段奖励：小权重，用于抑制明显不合理管理
  - 过量施氮惩罚
  - 过量灌溉惩罚
  - 氮淋失/反硝化惩罚
- 终局奖励：大权重，基于 episode 结束时计算
  - yield 或 profit

更具体一点：

`EpisodeReward = YieldValue - FertilizerCost - IrrigationCost - LeachingPenalty - EmissionPenalty`

如果缺少可靠经济参数，第一阶段也可以先用：

`EpisodeReward = StandardizedYield - λ1 * TotalN - λ2 * TotalIrrigation - λ3 * Leaching`

#### 训练 reward 与评估 KPI 分离

无论训练 reward 选哪一种，评估必须固定导出：

- final yield
- total nitrogen
- total irrigation
- nitrogen use efficiency
- water use efficiency
- leaching / denitrification
- optional profit

这样才能避免“reward 看起来更高，但农业表现未必更好”的问题。

## 7. 第一阶段建议的落地顺序

### Step 1：先拆边界，不先换算法

先完成：

- base env 层
- task schema 层
- episode KPI 层

不要先急着把 Rainbow 改成别的算法。

### Step 2：先做一个稳定的 baseline task

建议先固定一个任务：

- 作物先选 wheat 或 maize 之一
- 周尺度动作
- 专家摘要 observation
- yield-cost-leaching reward

先把这一套打通，再扩展到 tomato 或多任务。

### Step 3：评估层先于大规模训练

在开始大量训练前，先保证：

- 随机策略可跑
- 零投入策略可跑
- 固定日历策略可跑
- episode KPI 导出稳定

这样后面训练出来的结果才有参照物。

### Step 4：最后再比较不同算法

当 task 定义稳定后，再去比较：

- Rainbow
- PPO
- SAC
- rule-based

否则当前比较出来的结果，往往是在比“哪种算法更适应当前不合理任务定义”，而不是真正比农业决策能力。

## 8. 我建议的第一阶段里程碑

### M1：基础层解耦完成

完成标志：

- DSSAT base env 独立可运行
- crop config 独立管理
- episode summary 可导出

### M2：任务层解耦完成

完成标志：

- observation schema 可切换
- action schema 可切换
- reward schema 可切换

### M3：评估层稳定

完成标志：

- baseline 与 RL 共享同一 KPI 导出
- train/test split 清晰
- 多 seed 结果可比较

当前推进：

- `EpisodeEvaluator` 已支持 callable actor、`compute_action` actor，以及按名称组织的 actor suite 评估
- `TaskEnv` 已向评估链路暴露 `action_schema` 与 observation feature 元数据，rule-based baseline 可以直接复用任务装配结果
- `episode_report.py` 已稳定导出 nitrogen use efficiency 与 water use efficiency，便于 baseline / RL 横向比较

### M4：研究层开始扩展

完成标志：

- 在不改 env 基础层的前提下切换算法或任务定义
- 能较低成本开展 ablation study

## 9. 本轮已推进内容

本轮已经按路线图继续推进了训练装配链路：

- 新增 `envs/dssat_base_env.py`、`envs/crop_registry.py`、`envs/resource_resolver.py`
- 新增 `tasks/schemas.py`，把 observation / action / reward 变成显式 schema
- 新增 `tasks/forecast.py`，把天气预报接入做成 provider，而不是塞进基础 env
- 新增 `evaluation/episode_report.py`，统一导出 episode 级指标
- 新增 `experiments/build_env.py` 与 `experiments/build_policy.py`
- 将 `train_dssat.py` 改成通过 `build_phase1_task_env` 与 `build_rainbow_components` 构建训练链路
- 将 `build_env.py` 扩展为可接受嵌套字典、轻量对象与参数覆盖，减少训练入口对单一配置类的依赖
- 为 phase1 配置解析补充回归测试
- 在 `experiments/build_evaluator.py` 中补充 `zero` / `reactive` baseline actor 构造与 actor suite 评估能力
- 在 `TaskEnv` 信息流中补充 action schema 元数据，降低 baseline 与 task 层之间的耦合
- 在 `episode_report.py` 中补充 NUE / WUE 导出，并在 phase1 测试中加入 baseline 评估覆盖

当前最重要的收益是：

- forecast 数据现在可以作为可选任务输入接入
- 未来 3 天和 7 天 forecast 已有稳定摘要接口
- 训练入口已不再直接在本地重复定义 env 组装与 Rainbow 实例化
- `TaskEnv` 已成为训练主链路的一部分，后续接 baseline 的成本明显下降
- baseline 与 RL 已可以共用一套 episode KPI 导出与 evaluator 协议
- 下一步进入多 seed 对比时，不必再为 rule-based 单独维护一套评估脚本

## 10. 一句话建议

下一步最应该做的，是把训练入口的评估阶段正式切到 baseline + RL 共用的 actor suite 协议，而不是回到旧 wrapper 上叠更多 crop-specific 逻辑。只有让训练、baseline、评估共用同一套任务装配接口，后续农业 RL 对比实验才会真正可重复、可解释。
