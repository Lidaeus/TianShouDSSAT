# 文献研读 #4：知识引导的农业管理强化学习 (2024)

## 1. 文献基本信息
- **标题**: Knowledge-guided reinforcement learning for agricultural management
- **发表日期**: 2024
- **核心理念**: KGRL (Knowledge-guided Reinforcement Learning)

## 2. 核心价值解析
- **解决“不合逻辑”决策**: 通过动作掩码 (Action Masking) 解决强化学习在生理非活跃期盲目施肥的问题。
- **提升样本效率**: 利用专家经验进行热启动 (Warm-start)，避免了 RL 在海量动作空间中的随机无效搜索。

## 3. 对本项目的启示
- **显式约束**: 我们可以根据 DSSAT 的 `istage` 变量，在 `DssatGenericWrapper` 中硬编码一套“生物安全边界”。例如：在作物成熟后 (istage > 5) 禁止所有水肥操作。
- **启发式探索**: 我们可以编写一个简单的 Python 函数作为“影子专家”，当环境状态触发特定阈值（如极度缺氮）时，强制引导 Agent 尝试施肥动作。

## 4. 实验方案：DSSAT-KG-Action-Filtering
- **核心逻辑**: 在 `step()` 方法中加入逻辑判断，根据生理阶段过滤无效动作，并注入启发式采样引导。
- **目标**: 消除“违反常识”的错误策略，将训练收敛步数压缩至原来的 1/2。
- **评分**: 9.5 (本阶段最高分，具有极强的落地指导意义)
