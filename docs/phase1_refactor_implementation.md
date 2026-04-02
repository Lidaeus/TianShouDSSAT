# TianShouDSSAT 第一阶段代码改造清单

## 1. 第一阶段目标

第一阶段不是追求“最强算法”，而是把项目先改造成一个可稳定承载农业 RL 研究的基础平台。具体目标如下：

- 把 DSSAT 接入层和研究任务层拆开
- 让 observation、action、reward 成为可插拔对象
- 让 episode 级农业 KPI 独立于训练 reward
- 为后续更换算法、增加 baseline、扩展作物保留稳定接口

当前已在仓库中落下第一版目录骨架与接口定义，用于承接这一阶段重构。

## 2. 第一阶段代码改造清单

### 2.1 基础环境层

- 新增 `envs/crop_registry.py`
- 新增 `envs/resource_resolver.py`
- 新增 `envs/dssat_base_env.py`

本层职责：

- 维护作物注册表
- 解析模板、辅助文件与运行路径
- 启动 `DssatPdi`
- 输出原始状态、原生 reward、基础 episode summary

本层不再承载：

- 研究用 observation 裁剪
- 研究用动作桶设计
- 研究用 reward 逻辑

### 2.2 任务定义层

- 新增 `tasks/schemas.py`
- 新增 `tasks/forecast.py`

本层职责：

- 定义 observation schema
- 定义 action schema
- 定义 reward schema
- 定义组合后的 `TaskEnv`

本层是农业 RL 研究的主要试验面，未来大部分任务设计变化都应优先在这一层完成，而不是回到底层 DSSAT 适配器。

### 2.3 评估层

- 新增 `evaluation/episode_report.py`

本层职责：

- 汇总 episode 级农业指标
- 统一输出 RL 与 baseline 都可复用的 KPI 结构

### 2.4 实验装配层

- 新增 `experiments/build_env.py`
- 新增 `experiments/build_policy.py`
- 已接入 `train_dssat.py`

本层职责：

- 用轻量配置对象或字典装配 base env、task schema、reporter
- 用独立工厂装配 policy / algorithm
- 让训练入口复用同一套 phase1 任务装配逻辑

### 2.5 测试层

- 新增 `tests/test_phase1_scaffold.py`

本层当前覆盖：

- observation schema 输出形状
- weather forecast 摘要计算
- action schema 的时间间隔与预算裁剪
- reward schema 与 episode reporter 的基础行为
- phase1 任务配置解析

## 3. 逐文件接口设计

### 3.1 `envs/crop_registry.py`

目标接口：

- `CropConfig`
- `get_crop_config(crop_name: str) -> CropConfig`
- `list_supported_crops() -> list[str]`

职责边界：

- 只放作物元数据
- 不放训练超参数
- 不放 reward 逻辑
- 不放 observation 逻辑

### 3.2 `envs/resource_resolver.py`

目标接口：

- `ResolvedCropResources`
- `resolve_crop_resources(crop_name: str, run_dssat_location: str, extra_auxiliary_files: list[str] | None) -> ResolvedCropResources`

职责边界：

- 负责模板路径、辅助文件、Wheat fallback、运行资源装配
- 不负责动作、观测、奖励定义

### 3.3 `envs/dssat_base_env.py`

目标接口：

- `class DssatBaseEnv(gym.Env)`
- `reset() -> tuple[dict, dict]`
- `step(action_dict: dict[str, float]) -> tuple[dict, float, bool, bool, dict]`

返回约定：

- observation 保持原始 dict 结构
- reward 返回 native reward 的标量化版本
- info 中附带：
  - `crop_name`
  - `native_reward`
  - `applied_action`
  - `episode_summary`

后续应继续演进为：

- 原始 state 输出更稳定
- context 与 summary 字段定义固定
- 不再让上层依赖 vendored reward

### 3.4 `tasks/schemas.py`

当前包含四类接口：

- `BaseObservationSchema`
- `BaseActionSchema`
- `BaseRewardSchema`
- `TaskEnv`

以及三个第一阶段实现：

- `AgronomicObservationSchema`
- `WeeklyDiscreteActionSchema`
- `YieldCostRewardSchema`

本轮新增的天气接口：

- `BaseForecastProvider`
- `NullForecastProvider`
- `WeatherWindowForecastProvider`

### 3.5 `evaluation/episode_report.py`

目标接口：

- `EpisodeReport`
- `EpisodeReporter.reset(crop_name)`
- `EpisodeReporter.record_transition(...)`
- `EpisodeReporter.finalize()`

输出约定：

- `crop_name`
- `episode_steps`
- `task_return`
- `native_return`
- `total_fertilizer`
- `total_irrigation`
- `final_grain_weight`
- `final_biomass`
- `final_leaching`
- `final_denitrification`

### 3.6 phase1 任务配置约定

目标接口：

- `crop_name`
- `mode`
- `seed`
- `observation.include_forecast`
- `observation.forecast_horizons`
- `action.*`
- `reward.*`

职责边界：

- 只描述任务配置
- 不直接创建环境
- 不直接实例化算法

### 3.7 `experiments/build_env.py`

目标接口：

- `resolve_phase1_task_config(config=None, **overrides) -> dict`
- `build_phase1_task_env(config=None, **overrides) -> TaskEnv`

职责边界：

- 只负责拼装
- 接受字典、轻量对象或扁平参数覆盖
- 不承担训练循环
- 不承担算法实例化

### 3.8 `experiments/build_policy.py`

目标接口：

- `build_rainbow_components(state_shape, action_shape, hidden_size, device)`

职责边界：

- 只负责算法与 policy 实例化
- 不承担 env 装配
- 不承担训练循环

## 4. observation 在本项目中的定义

这是第一阶段最需要仔细定义的部分。对农业 RL 来说，observation 不是“从状态里随便挑几个变量”，而是“给智能体什么信息，才能让它做出有农艺意义的管理决策”。

我建议在本项目中，把 observation 明确定义为：

> 在每一个管理决策时刻，智能体可见的、足以支持水肥管理决策的低维农艺状态摘要。

这个定义有四个关键点：

- 是为管理决策服务，而不是为复现 DSSAT 全状态服务
- 是低维摘要，不是把所有变量直接塞给网络
- 必须覆盖作物、土壤、天气、管理记忆四类信息
- 必须允许未来扩展天气预报，但不强绑在当前基础环境层

## 5. 第一阶段 observation 的具体内容

当前落地的 `AgronomicObservationSchema` 采用了 24 维主特征；当启用 forecast-aware 模式时，再增加未来 3 天和未来 7 天各 3 维摘要，共 6 维天气预报特征。

### 5.1 作物状态

- `istage_norm`
- `vstage_norm`
- `canopy_norm`
- `grnwt_norm`
- `topwt_norm`
- `rtdep_norm`
- `swfac_norm`
- `nstres_norm`

这组特征回答的是：

- 作物现在处于什么发育阶段
- 冠层和生物量积累到了什么水平
- 根系是否足够深
- 当前是否处于明显的水分或氮素胁迫

### 5.2 土壤与水氮状态

- `root_zone_sw_ratio`
- `profile_sw_ratio`
- `wtdep_norm`
- `trnu_norm`
- `tleachd_norm`
- `tnoxd_norm`

这组特征回答的是：

- 根层是否还有可利用水
- 整个剖面是否偏干或偏湿
- 当前是否存在较高环境损失风险
- 当前氮吸收是否有效

其中 `root_zone_sw_ratio` 不是直接把 `sw` 原样输入，而是结合 `ll`、`dul`、`dlayr` 和 `rtdep` 计算出的根层相对可利用水比例。这比直接输入土壤剖面数组更适合第一阶段的稳定训练。

### 5.3 管理状态

- `cumulative_fertilizer_norm`
- `cumulative_irrigation_norm`
- `days_since_fertilizer_norm`
- `days_since_irrigation_norm`
- `remaining_fertilizer_ratio`
- `remaining_irrigation_ratio`

这组特征非常关键，因为农业决策不是单步最优，而是季节约束下的序贯最优。没有这组变量，智能体很难学到“资源预算管理”。

### 5.4 当日天气状态

- `rain_norm`
- `tmin_norm`
- `tmax_norm`
- `srad_norm`

这组特征回答的是：

- 当前日是否自然供水
- 当前热量条件如何
- 光照水平是否支持生长响应

### 5.5 天气预报摘要应该如何进入系统

未来 3 天、未来 7 天的天气数据，建议采用“两层表示”：

- 原始逐日 forecast 不直接塞进基础环境层
- 面向当前 MLP 策略的 forecast 摘要进入 observation
- 原始逐日 forecast 保留在 `info` / metadata 中，供日志、评估和未来序列模型使用

当前已经落地的 forecast-aware 设计如下：

- provider 层：`tasks/forecast.py`
- summary 层：`weather_forecast`
- observation 层：`AgronomicObservationSchema`
- 数据来源优先级：外部 forecast daily → `info` 中显式 forecast → `.WTH` 天气文件窗口摘要

当前第一阶段建议放入 observation 的，不是完整逐日序列，而是两个窗口摘要：

- 未来 3 天：
  - `forecast_rain_3d_norm`
  - `forecast_tmean_3d_norm`
  - `forecast_srad_3d_norm`
- 未来 7 天：
  - `forecast_rain_7d_norm`
  - `forecast_tmean_7d_norm`
  - `forecast_srad_7d_norm`

这么做的理由是：

- 对当前全连接策略最稳定
- 不会让 observation 维度因为 forecast 长度快速膨胀
- 保留“短期天气冲击”和“中短期趋势”两种时间尺度
- 不把天气预报数据源硬耦合到 `DssatBaseEnv`

如果未来切换到 Transformer、RNN 或者显式时序 encoder，再把逐日 forecast 作为单独输入通道，而不是继续堆在当前低维 observation 中。

## 6. 为什么第一阶段不直接使用完整土壤剖面

当前项目底层已经能提供 `sw`、`ll`、`dul`、`dlayr` 等数组信息，但第一阶段不建议把完整剖面直接拼进 observation，原因有三点：

- 维度快速增加，样本效率下降
- 不同作物和土层配置下维度稳定性较差
- 当前项目的训练与评估协议还未稳定，先用专家摘要更容易做出可解释基线

因此第一阶段的策略是：

- 保留剖面原始信息在 raw state 层
- 在 observation 层先压缩成根层和全剖面摘要
- 等 task 定义稳定后，再实验更细粒度土壤输入

## 7. 第一阶段 action 接口定义

当前实现的 `WeeklyDiscreteActionSchema` 定义为：

- 决策频率：每 7 天一次
- 氮肥桶：`[0, 30, 60, 90]`
- 灌溉桶：`[0, 10, 20, 30]`
- 预算约束：season-level fertilizer / irrigation budget
- 生育阶段约束：超过阶段阈值后禁施氮

这一定义的意义在于：

- 比逐日动作更接近农业管理节律
- 允许预算受限策略学习
- 给后续 action masking 和参数化动作留出边界

## 8. 第一阶段 reward 接口定义

当前实现的 `YieldCostRewardSchema` 采用：

- 小权重的过程奖励：谷重增量
- 显式投入成本：施氮与灌溉成本
- 环境惩罚：淋失与反硝化
- 终局奖励：回合结束时的 grain weight 奖励

这仍然只是第一阶段版本，但优于直接把底层 native reward 当成研究目标，因为它已经体现了：

- 生产目标
- 投入成本
- 环境代价

## 9. 下一步逐文件改造建议

### P0

- 将 `train_dssat.py` 重构为通过 `experiments/build_env.py` 装配环境
- 增加 `experiments/build_policy.py`，把 Rainbow 实例化从训练入口中拆出
- 在训练和评估阶段统一使用 `episode_report`

当前状态：

- `train_dssat.py` 已改为通过 `build_phase1_task_env` 创建训练与测试环境
- `train_dssat.py` 已直接复用 `build_rainbow_components`
- `build_env.py` 已不再强依赖单一配置类，可接受嵌套字典与参数覆盖
- forecast provider 已接入 task 层

### P1

- 让 `DssatGenericWrapper` 逐步退场，保留兼容用途
- 用 `TaskEnv(DssatBaseEnv + schema)` 替代当前“大一统 wrapper”
- 把 baseline 策略接入同一套 `TaskEnv`

当前状态：

- `TaskEnv` 的 reset / step 信息中已暴露 `action_schema` 元数据，baseline actor 不需要再反向猜测动作桶定义
- `experiments/build_evaluator.py` 已新增 `build_baseline_actor`，可直接构造 `zero` 与 `reactive` 两类基线
- `EpisodeEvaluator` 已支持 `evaluate_actor_suite`，便于在同一评估协议下批量对比 baseline 与 RL actor
- `episode_report.py` 已补充 nitrogen / water use efficiency，评估导出不再只停留在 yield 与投入总量

下一轮建议：

- 先用 `zero`、`reactive` 两个 baseline 固化单作物评估协议
- 再把训练脚本的评估阶段接到 `evaluate_actor_suite`，形成 RL vs baseline 的统一报表
- 最后再考虑把固定日历策略或 crop-specific 规则迁移到同一 actor 接口

### P2

- 引入天气预报提供器接口
- 扩展 observation schema 到 forecast-aware 版本
- 扩展 action schema 到参数化动作或连续动作版本

## 10. 当前已落地且仍在主链路使用的骨架文件

- [crop_registry.py](file:///home/lidaeus/TianShouDSSAT/envs/crop_registry.py)
- [resource_resolver.py](file:///home/lidaeus/TianShouDSSAT/envs/resource_resolver.py)
- [dssat_base_env.py](file:///home/lidaeus/TianShouDSSAT/envs/dssat_base_env.py)
- [schemas.py](file:///home/lidaeus/TianShouDSSAT/tasks/schemas.py)
- [forecast.py](file:///home/lidaeus/TianShouDSSAT/tasks/forecast.py)
- [episode_report.py](file:///home/lidaeus/TianShouDSSAT/evaluation/episode_report.py)
- [build_env.py](file:///home/lidaeus/TianShouDSSAT/experiments/build_env.py)
- [build_policy.py](file:///home/lidaeus/TianShouDSSAT/experiments/build_policy.py)
- [train_dssat.py](file:///home/lidaeus/TianShouDSSAT/train_dssat.py)
- [test_phase1_scaffold.py](file:///home/lidaeus/TianShouDSSAT/tests/test_phase1_scaffold.py)

## 11. 一句话结论

第一阶段最重要的不是继续在旧 wrapper 上叠逻辑，而是先让 observation、action、reward、baseline evaluation 都成为明确的研究接口对象，并让 DSSAT 基础环境层保持稳定、薄而清晰。
