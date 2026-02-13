# Changelog

## [2.0.0-rc1] - 2026-02-12

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
