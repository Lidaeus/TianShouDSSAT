# 农业强化学习研究与复现路线图 (2026-02-12)

> **目标**：通过深度研读与复现 10 篇核心文献，攻克“奖励稀疏”与“气候波动风险”两大难题，为 v3.0 版本提供理论支撑。

## 1. 核心研究课题 (Research Topics)

### Topic A: 潜在奖励塑形 (Potential-based Reward Shaping)
- **核心思想**：利用作物生长指标（LAI, Biomass）的日增量作为势能函数 $F(s, s')$。
- **复现目标**：在 `DssatGenericWrapper` 中实现一个动态奖励计算器，支持从生理特征自动合成中间奖励。

### Topic B: 风险敏感的分布强化学习 (Risk-sensitive Distributional RL)
- **核心思想**：不仅关注期望产量的最大化，更关注产量分布的“左尾”（即最差情况下的表现）。
- **复现目标**：将当前的 Rainbow 升级为支持 CVaR 优化的 QR-DQN 架构，并测试其在极端干旱年份的生存率。

## 2. 复现里程碑 (Milestones)

### 第一阶段：基准对齐 (Week 1)
- **任务**：完全复现 *Gym-DSSAT (2022)* 中的标准奖励函数，建立 baseline 性能指标。
- **产出**：`tests/repro_baseline.py`。

### 第二阶段：奖励塑形实验 (Week 2)
- **任务**：基于 *Policy invariance (1999)* 理论，对比“纯收成奖励”与“LAI 引导奖励”的收敛速度。
- **产出**：奖励函数对比报告。

### 第三阶段：稳健性压力测试 (Week 3)
- **任务**：复现 *Climate uncertainty (2023)* 中的环境随机化方法，使用不同气候年份数据进行交叉验证。
- **产出**：模型在异常气候下的鲁棒性评价模型。

## 3. 文献与代码关联矩阵

| 文献编号 | 提取的想法 | 对应的代码实现位置 |
| :--- | :--- | :--- |
| #3, #10 | 知识引导的奖励塑形 | `DssatGenericWrapper.step()` |
| #1, #2 | 分布式风险度量 (CVaR) | `train_dssat.py` -> Policy Config |
| #5, #8 | 气候领域随机化 | `train_dssat.py` -> Env Factory |
| #9 | 多目标权重动态调节 | `config/reward_config.yml` |

---
**提示**：准备好开始第一篇文献的深度解析了吗？我们将从 **#3 Reward Shaping 理论** 开始，因为它直接决定了我们的环境包装器该如何重构。
