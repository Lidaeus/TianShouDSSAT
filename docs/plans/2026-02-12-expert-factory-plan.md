# 专家工厂 (Expert Factory) 子项目开发方案

> **状态**：方案已评审 / 开发启动
> **版本**：v0.2.0 (策略改进版)
> **核心目标**：利用遗传算法 (GA) 在华北平原历史气象背景下，通过 DSSAT 仿真生产帕累托最优轨迹数据集，建立强化学习的“合成专家经验”。

---

## 1. 核心改进策略 (v0.2.0)

### 1.1 分阶段触发式编码 (Stage-based Encoding)
- **文献依据**：参考 *MDPI Water (2022)* 华北番茄优化研究。
- **设计**：染色体不再是全季统一参数，而是将生长季划分为：
    - **苗期 (Seedling)**: 侧重根系下扎，设定较高的干旱忍受度。
    - **开花坐果期 (Flowering)**: 水分敏感期，设定极低的灌溉触发阈值。
    - **果实膨大/熟前期 (Fruiting)**: 侧重产量积累，平衡水肥投入。
- **参数量**：每个阶段 3 个参数 (Irrigation Threshold, Amount, Nitrogen Threshold)，总共 12-15 维。

### 1.2 风险感知适应度函数 (Risk-aware Fitness)
- **目标**：不仅追求高产，更要确保“不绝收”。
- **惩罚项**：引入死亡风险惩罚 $P_{death}$。若模拟过程中 `LAI < 0.1` 持续超过 5 天，则该方案得分极低。
- **目标向量**：$f = [\max(Yield), \min(Water), \min(N\_Leaching)]$。

### 1.3 气象数据策略：华北平原 (North China Plain)
- **数据源**：NASA POWER API。
- **样点选择**：选取华北平原代表性坐标 ($35.0^{\circ}N, 115.0^{\circ}E$)。
- **聚类分析**：对过去 30 年降水进行聚类，从“丰、平、枯”三类年份中各选 5 年进行重点优化，构建覆盖全气候谱的专家集。

---

## 2. 任务执行序列

### Task 0: 华北平原气象数据自动化抓取 (NASA POWER API)
- **目标**：生成 1991-2021 年华北平原连续 30 年的 DSSAT \`.WTH\` 文件库。
- **产出**：\`data/weather/NCP_1991_2021.WTH\`。

### Task 1: 基础设施搭建与版本化 (Ongoing)
- **目标**：确立独立版本生命周期。
- **文件**：\`VERSION\`, \`CHANGELOG.md\`, \`requirements.txt\`。

### Task 2: 分阶段触发编码器实现
- **目标**：编写 \`StageBasedPolicy\` 类，将 GA 染色体转化为 DSSAT 每日动作。

### Task 3: GA 优化引擎实现 (基于 Pymoo)
- **目标**：实现多目标 NSGA-II 算法与 DSSAT 包装器的桥接。

### Task 4: 专家轨迹序列化
- **目标**：将帕累托最优方案导出为 \`tianshou.data.Batch\` 格式。

---

## 3. 评审要点 (v0.2.0)
1. **API 配额**：NASA POWER API 有日请求限制，需实现分批抓取。
2. **格式对齐**：NASA 原始数据（SRAD, TMAX, TMIN, RAIN）需严格转换为 DSSAT 的 ICASA 格式。
