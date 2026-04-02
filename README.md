# TianShouDSSAT

一个围绕 DSSAT 作物模拟器与 Tianshou 强化学习框架构建的农业决策实验项目。当前仓库已经完成 Phase1 方向的分层重构：环境装配、观测编码、动作离散化、奖励设计、评估报告与 Rainbow 策略构建彼此解耦，可以独立组合与验证。

## 项目能力

- 使用 `DssatBaseEnv` 封装底层 DSSAT PDI 环境
- 使用 `TaskEnv` 组合观测、动作、奖励与 episode report
- 支持 forecast-aware observation，可从天气文件推导未来 3/7 天摘要
- 提供 weekly discrete 动作空间，统一管理施肥与灌溉预算
- 提供 baseline actor、Rainbow policy 与统一 evaluator
- 提供训练入口与可运行的示例任务

## 目录结构

- `envs/`：底层 DSSAT 环境封装与 crop 资源解析
- `tasks/`：观测 schema、动作 schema、奖励 schema、forecast provider
- `evaluation/`：episode report 与聚合指标
- `experiments/`：环境装配、policy 构建、评估器构建、实验配置
- `train_dssat.py`：Rainbow 训练入口
- `examples/phase1_example_tasks.py`：覆盖主要能力的示例任务集合
- `tests/`：现有 smoke test、wrapper test 与 phase1 scaffold test
- `docs/`：重构说明、路线图与补充文档

## 环境准备

推荐直接使用项目内虚拟环境：

```bash
/home/lidaeus/TianShouDSSAT/venv/bin/python -V
```

常用命令：

```bash
./venv/bin/python -m pytest tests/test_phase1_scaffold.py
./venv/bin/python -m compileall train_dssat.py experiments evaluation tasks tests examples
./venv/bin/python train_dssat.py --crop wheat
```

如果运行时遇到 matplotlib 缓存目录权限问题，可以临时指定：

```bash
MPLCONFIGDIR=/tmp/mpl ./venv/bin/python examples/phase1_example_tasks.py
```

## 快速开始

### 1. 构造 phase1 环境

```bash
./venv/bin/python - <<'PY'
from experiments.build_env import build_phase1_task_env

env = build_phase1_task_env(crop_name="wheat", include_forecast=True, forecast_horizons=(3, 7))
obs, info = env.reset(seed=42)
print(obs.shape)
print(info["observation_features"])
print(info["weather_forecast"])
env.close()
PY
```

### 2. 运行示例任务集合

```bash
./venv/bin/python examples/phase1_example_tasks.py --crop wheat
```

该脚本会依次执行：

- 环境装配与 forecast observation 展示
- baseline actor、random actor、untrained Rainbow 的统一评估
- 微型训练任务，并输出训练日志与评估汇总

所有结果会写入 `logs/example_tasks/`。

### 3. 运行训练入口

标准训练：

```bash
./venv/bin/python train_dssat.py --crop wheat --include-forecast --forecast-horizons 3 7
```

更适合冒烟测试的小规模训练：

```bash
./venv/bin/python train_dssat.py \
  --crop wheat \
  --epochs 1 \
  --epoch-num-steps 10 \
  --collection-step-num-env-steps 5 \
  --test-step-num-episodes 1 \
  --eval-episodes 1 \
  --replay-buffer-size 128
```

训练完成后会在日志目录输出：

- `policy_best.pth`
- `policy_final.pth`
- `evaluation_summary.json`

## 示例任务文档

更完整的示例任务说明见 [docs/example_tasks.md](file:///home/lidaeus/TianShouDSSAT/docs/example_tasks.md)。

## 核心代码入口

- [build_env.py](file:///home/lidaeus/TianShouDSSAT/experiments/build_env.py)
- [build_policy.py](file:///home/lidaeus/TianShouDSSAT/experiments/build_policy.py)
- [build_evaluator.py](file:///home/lidaeus/TianShouDSSAT/experiments/build_evaluator.py)
- [train_dssat.py](file:///home/lidaeus/TianShouDSSAT/train_dssat.py)
- [phase1_example_tasks.py](file:///home/lidaeus/TianShouDSSAT/examples/phase1_example_tasks.py)
