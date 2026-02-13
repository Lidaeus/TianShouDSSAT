# Changelog

## [v2.1.0] - 2026-02-12
### Added
- 正式支持 **Tomato (番茄)** 环境适配（基于 CROPGRO 模型）。
- 实现通用包装器 `DssatGenericWrapper` 的动作离散化映射逻辑，支持 Rainbow DQN 训练。
- 增加 `train_dssat.py` 统一训练脚本，支持参数化切换作物和配置。
- 增加多作物自动运行脚本 `run_experiments.sh`。

### Fixed
- 修复了 Maize (玉米) 环境因 PDI 模板逻辑错误导致的“单步即终止”关键 Bug。
- 解决了 PDI 握手过程中的 YRDOY 变量同步失效问题。
- 汉化了所有核心文档和代码注释，对齐项目长期记忆规范。

### Changed
- 重构了 `dssat_pdi.jinja2` 模板，采用更鲁棒的 `flag_undone` 和 `interact` 逻辑。
- 升级了 `DssatGenericWrapper`，支持全路径资源自动补全和历史状态拼接。

## [2.0.0-rc2] - 2026-02-12

### Added
- **通用包装器**: 实现了 `DssatGenericWrapper`，支持通过配置矩阵驱动不同作物的观测空间和物理逻辑。
- **配置矩阵**: 预置了玉米和番茄的农学参数（种植密度、移栽逻辑）。
- **冒烟测试**: 建立了番茄和玉米通用的环境验证脚本。

### Fixed
- 优化了 Wrapper 初始化顺序，通过在 `super().__init__` 前预置属性彻底解决了 `reset()` 导致的属性缺失报错。
- 增强了 Tianshou 2.0 兼容性补丁（NoneType info 检查）。

### Known Issues
- 番茄 (Tomato) 环境在特定物理引擎配置下存在初始化阻塞问题，待后续通过土壤/气象数据对齐修复。

### Added
- **天授 2.0 适配**: 完成了从 Tianshou 1.x 到 2.0 (Algorithm/Policy/Params 架构) 的全面迁移。
- **多作物支持架构**: 设计并初步实现了 `DssatGenericWrapper` 架构，支持玉米 (Maize) 和番茄 (Tomato)。
- **真实物理模板**: 基于 `UFGA0602.TMX` 建立了符合农学逻辑的番茄 Jinja2 模板。
- **TensorBoard 集成**: 实现了训练指标的实时监控。
- **排坑文档**: 建立了详尽的 `docs/troubleshooting_log.md` 记录。

### Changed
- **Wrapper 重构**: 修复了 `DssatPdi` 初始化死锁及 `NoneType` 字典赋值崩溃问题。
- **路径策略**: 采用本地短路径基准 (`/opt/dssat_env`) 彻底规避了 Fortran 字符串长度限制导致的段错误。

### Fixed
- 修复了 Python 3.12 下 `pkgutil` 触发的 Gym/Gymnasium 兼容性冲突。
- 解决了 ZMQ 端口残留导致的孤儿进程占用问题。

### Removed
- 移除了未经过物理验证的草莓 (Strawberry) 支持。

## [2.0.0-rc3] - 2026-02-12
### Added
- 为 DssatPdi 类增加 pdi_template_path 参数，支持外部动态注入 PDI YAML 模板。
- 为 Tomato (番茄) 建立独立的 PDI YAML 和资源注入逻辑。
- 在 DssatGenericWrapper 中实现辅助文件 (.WTH, .CLI, .SOL) 的全路径自动补全。

### Fixed
- 彻底解决了 PDI 握手死锁问题（通过 crop-specific YAML）。
- 解决了 PDI 内部 Python 环境找不到 Numpy/PDI 绑定的问题。
- 修复了 Numpy 2.0 导致的 AttributeError: itemset 兼容性问题。
- 解决了库源码中的循环引用问题（通过重构导入机制）。
- 确保了 env.reset() 和 env.step() 返回值符合 Gymnasium 1.x 标准。
