# 小麦强化学习优化系统开发文档 (Design Document)

## 1. 系统架构

系统采用三层架构并深度集成防御与引导机制：
1.  **引擎层 (Engine Layer)**: `gym-dssat-pdi` 集成的 DSSAT 核心引擎（基于 v4.7.5 修改版），负责小麦生长物理过程模拟。
2.  **接口层 (Interface Layer)**: 基于 `gym-dssat-pdi` 的增强型 `WheatEnv`。
    *   **Wrapper 逻辑**: 负责动作裁剪、异常捕获、时序特征拼接和 Reward 计算。
3.  **算法层 (Algorithm Layer)**: 基于天授 (Tianshou) 的 RL 框架。
    *   **主算法**: Rainbow DQN (应对离散空间)。
    *   **预训练**: 行为克隆 (Behavioral Cloning) 模块。

## 2. 核心模块设计

### 2.1 增强型环境封装 (`WheatEnv`)
*   **动作处理**: 
    *   输入 Agent 的索引值，映射到预设的 [0, 10, ... 200] 物理值。
    *   **动作裁剪**: 强制限制单日总施用量，防止 DSSAT 崩溃。
*   **状态增强引擎**:
    *   **时序 Buffer**: 内部维护一个队列，存储过去 14 天的历史状态。
    *   **预报接口**: 提前读取 `.WTH` 文件，注入未来 7 天的气象向量。
*   **错误处理 (Safety Mechanisms)**:
    *   `try-except` 捕获引擎底层错误，返回 `done=True` 和 `Penalty`。
    *   监测 `LAI <= 0` 或 `istage == 99`（作物死亡），触发即时惩罚。

### 2.2 复合奖励函数计算器
$$R_{total} = R_{yield} + R_{dense} - C_{resources} - P_{death}$$
*   $R_{dense}$: 包含 LAI 增量奖励和基于先验知识的物候期区间奖励。
*   $P_{death}$: 针对系统崩溃或作物死亡的重大负反馈。

### 2.3 先验知识集成模块
*   **Expert Buffer**: 预先运行标准专家策略（Heuristic rules），生成格式化的 `Batch` 数据。
*   **Action Masker**: 根据 `istage` 和 `weather` 生成布尔掩码，传递给策略网络。

## 3. 关键算法流程

1.  **初始化**: 环境初始化，加载滑动窗口配置。
2.  **专家模仿阶段**: 
    *   加载 `Expert Buffer`。
    *   进行行为克隆预训练，使策略网络输出接近专家决策。
3.  **强化学习阶段**:
    *   **采样**: 带有 Action Mask 的并行环境探索。
    *   **学习**: Rainbow DQN 从 Buffer 中采样，同时学习 Q 值和分布式收益。
    *   **评估**: 在独立的测试年份上测试泛化性。

## 4. 接口规范 (Gymnasium)

```python
# 增强型状态空间
observation_space = Dict({
    "current_state": Box(...),  # 当前土壤与作物指标
    "history_14d": Box(...),    # 过去14天时序特征
    "forecast_7d": Box(...),   # 未来7天气象预报
    "action_mask": Box(binary)  # 可用动作掩码
})

# 离散动作空间
action_space = Discrete(56) # 灌溉(7级) x 施肥(8级) 的组合索引
```

## 5. 监控与可视化
*   **TensorBoard**: 监控奖励曲线、LAI 轨迹、水分/肥料累积消耗。
*   **对比实验**: 自动生成与“专家基准方案”的对比图表（产量 vs 水肥效率）。
