# AGENTS

## 常用路径

- 项目根目录：`/home/lidaeus/TianShouDSSAT`
- 虚拟环境：`/home/lidaeus/TianShouDSSAT/venv`
- 虚拟环境 Python：`/home/lidaeus/TianShouDSSAT/venv/bin/python`
- 训练入口：`/home/lidaeus/TianShouDSSAT/train_dssat.py`
- 实验装配：`/home/lidaeus/TianShouDSSAT/experiments`
- 环境层：`/home/lidaeus/TianShouDSSAT/envs`
- 任务层：`/home/lidaeus/TianShouDSSAT/tasks`
- 评估层：`/home/lidaeus/TianShouDSSAT/evaluation`
- 测试目录：`/home/lidaeus/TianShouDSSAT/tests`
- 文档目录：`/home/lidaeus/TianShouDSSAT/docs`
- 训练日志：`/home/lidaeus/TianShouDSSAT/logs`
- DSSAT vendored 目录：`/home/lidaeus/TianShouDSSAT/lib/gym_dssat_pdi_official`

## 常用命令

- 运行 phase1 测试：`./venv/bin/python -m pytest tests/test_phase1_scaffold.py`
- 运行训练入口：`./venv/bin/python train_dssat.py --crop wheat`
- 编译校验主要模块：`./venv/bin/python -m compileall train_dssat.py experiments evaluation tasks tests`
