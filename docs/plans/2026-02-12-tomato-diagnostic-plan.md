# Tomato 环境失效诊断计划 (2026-02-12)

## 1. 核心假设 (Hypotheses)
1.  **H1: 变量名失配**：`CROPGRO` (番茄) 在 Fortran 层通过 PDI 导出的变量标识符与 `CERES-Maize` 不同，导致 PDI 插件无法识别 YAML 中的变量，从而阻塞通讯。
2.  **H2: 临时目录缺陷**：`DssatPdi` 的 `_copy_auxiliary_files` 逻辑可能遗漏了番茄运行所需的特定支持文件（如 `.SPE`, `.CUL`, `.ECO`），导致物理引擎弹出交互式报错。
3.  **H3: PDI YAML 模板死锁**：官方 `dssat_pdi.jinja2` 模板中可能硬编码了针对玉米的逻辑。

## 2. 诊断实验设计

### 实验 A：物理引擎“脱水”测试 (Standalone Run)
- **目的**：排除 Python 干扰，直接观察番茄在 PDI 模式下的物理引擎表现。
- **步骤**：
    1. 使用 `DssatGenericWrapper` 生成番茄的临时目录。
    2. **修改源码**，阻止 `DssatPdi` 在出错或退出时删除 `/tmp/tmp*` 目录。
    3. 手动进入该目录，运行 `/opt/dssat_env/inst/run_dssat B fileX.MZX`。
    4. 观察是否有控制台输出或 `ERROR.OUT` 生成。

### 实验 B：PDI 通讯“侵入式”日志 (Intercepting ZMQ)
- **目的**：确认 ZMQ 握手是否发生，以及卡在哪个消息。
- **步骤**：
    1. 修改 `lib/.../dssat_pdi.py`，在 `self._server.recv()` 前后增加时间戳日志。
    2. 修改子进程启动逻辑，将物理引擎的 `stdout` 和 `stderr` 重定向到本地文件而非 `/dev/null`。

### 实验 C：Fortran 源码验证 (Source Audit)
- **目的**：核实 `CROPGRO.for` 中真正的 PDI 定义。
- **步骤**：
    1. 使用 `grep` 在 `lib/gym_dssat_pdi_official/dssat-csm-os/Plant/CROPGRO/` 下寻找 `PDI_` 开头的宏或函数调用。
    2. 对比番茄和玉米的标识符。

## 3. 补丁计划
若确认是变量名问题，我们将：
1. 在 `templates/` 下创建 `dssat_pdi_tomato.jinja2`，专门映射番茄标识符。
2. 修改 `DssatPdi` 以支持从外部注入该 YAML 模板。
