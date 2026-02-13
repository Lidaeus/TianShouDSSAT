# DssatGenericWrapper 开发计划书

## 1. 核心目标
设计并实现一个通用的 `DssatGenericWrapper`，支持 Maize (玉米) 和 Tomato (番茄) 的强化学习训练，解决官方库中硬编码品种列表导致的扩展性问题。

## 2. 关键技术方案

### 2.1 品种列表注入 (Cultivar Injection)
官方 `DssatPdi` 在 `__init__` 中硬编码了 `self.cultivars_fileX`。
*   **策略**: 在包装器构造时，动态扩展支持列表，例如 `{"tomato": "UFGA0602"}`。

### 2.2 真实模板解耦 (Real Template Decoupling)
为了保证农学准确性，模板必须从该作物的真实 FileX 派生。
*   **策略**: 
    1. Maize 模板源自 `UFGA8201.MZX`。
    2. Tomato 模板源自 `UFGA0602.TMX`（已处理种植密度与移栽逻辑）。
    3. 实例化时通过路径注入模板。

### 2.3 状态空间归一化 (Observation Normalization)
不同作物的观测值含义和量级不同（例如番茄果实重 vs 玉米谷物重）。
*   **策略**: 
    1. 引入 `obs_config` 映射表，为每种作物定义特征索引（如 `yield_key`: "SDWT" 或 "GRNWT"）。
    2. 统一输出固定长度的向量（14天历史 + 7天预报 + 5个核心指标）。

### 2.4 并行训练准备
为后续 `SubprocVectorEnv` 做准备。
*   **策略**: 为每个环境实例生成唯一的临时工作目录名，规避 `/opt/dssat_env` 下的文件写入冲突。

## 3. 实施步骤

### 第一步：环境模板准备
1. 复制官方玉米模板到项目 `templates/maize/`。
2. 制作番茄（TM）和草莓（SB）的模板，重点修改 FileX 中的作物代码和基因型文件名。

### 第二步：编写 `DssatGenericWrapper`
1. 集成 `maize_wrapper.py` 中的所有补丁（路径补丁、ZMQ补丁）。
2. 实现 `_get_obs_config(cultivar)` 辅助函数。
3. 实现更具健壮性的 `step` 方法（包含 `info` 字典保底）。

### 第三步：验证
1. 运行 `test_generic_wrapper.py --cultivar maize`。
2. 运行 `test_generic_wrapper.py --cultivar tomato`（这是第一个真考点）。

## 4. 风险预估
*   **PDI 映射不匹配**: 如果番茄的 PDI 变量名与玉米不同，会导致 `get_state` 崩溃。
*   **基因型文件缺失**: 虽然 `TM*` 文件存在，但如果缺少 `.SPE` 里的特定定义，物理引擎会报错。
