# 示例任务集

本文档配合 `examples/phase1_example_tasks.py` 使用，目标是让示例任务既能展示项目能力，也能作为一套高覆盖率的冒烟测试。

## 任务一：环境装配与 Forecast 观测

命令：

```bash
./venv/bin/python examples/phase1_example_tasks.py --crop wheat --tasks env
```

覆盖能力：

- `build_phase1_task_env` 装配底层 DSSAT 环境
- `AgronomicObservationSchema` 编码 agronomic observation
- `WeatherWindowForecastProvider` 生成未来天气摘要
- `WeeklyDiscreteActionSchema` 对动作空间做离散化映射

输出文件：

- `logs/example_tasks/env_showcase.json`

重点关注字段：

- `observation_dim`
- `forecast_keys`
- `action_schema`
- `transitions`
- `episode_report`

## 任务二：统一评估器与多 Actor 对比

命令：

```bash
./venv/bin/python examples/phase1_example_tasks.py --crop wheat --tasks evaluate --episodes 1
```

覆盖能力：

- `EpisodeEvaluator` 统一执行 episode
- baseline actor `zero` 与 `reactive`
- `None` 代表的 random actor
- `build_rainbow_components` 生成的 untrained Rainbow policy

输出文件：

- `logs/example_tasks/evaluator_showcase.json`

重点关注字段：

- 每个 actor 的 `aggregate`
- 每个 actor 的 `reports`
- `mean_task_return`
- `mean_final_grain_weight`
- `mean_nitrogen_use_efficiency`

## 任务三：微型训练入口

命令：

```bash
./venv/bin/python examples/phase1_example_tasks.py --crop wheat --tasks train
```

覆盖能力：

- `train_dssat.py` 的训练主入口
- Tianshou Rainbow 组件初始化
- collector、replay buffer 与 trainer 参数联动
- 训练结束后的统一评估与结果落盘

输出文件：

- `logs/example_tasks/training_showcase.json`
- `logs/example_tasks/training_logs/<timestamp>/policy_best.pth`
- `logs/example_tasks/training_logs/<timestamp>/policy_final.pth`
- `logs/example_tasks/training_logs/<timestamp>/evaluation_summary.json`

## 一次性运行全部示例

命令：

```bash
./venv/bin/python examples/phase1_example_tasks.py --crop wheat
```

该命令会顺序执行 env、evaluate、train 三个子任务，并在 `logs/example_tasks/summary.json` 汇总结果。

## 推荐作为回归检查的命令组合

```bash
./venv/bin/python -m pytest tests/test_phase1_scaffold.py
./venv/bin/python examples/phase1_example_tasks.py --crop wheat
./venv/bin/python -m compileall train_dssat.py experiments evaluation tasks tests examples
```

这组命令分别覆盖：

- 单元与集成层 scaffold 验证
- 真实示例任务链路验证
- 主要 Python 模块语法与导入校验
