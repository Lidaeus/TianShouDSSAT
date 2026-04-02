# Wheat 支持开发计划 (2026-04-01)

## 1. 目标

为 TianShouDSSAT 增加 `Wheat` 作物支持，使项目能够像当前的 `maize`、`tomato` 一样：

- 通过 `DssatGenericWrapper` 直接实例化小麦环境
- 正确加载 Wheat 对应的 FileX 模板、PDI 模板与状态配置
- 支持训练入口中的作物切换
- 具备最小可回归的测试与文档闭环

## 2. 当前审计结论

### 2.1 已确认的现状

- 项目封装层 [dssat_generic_wrapper.py](file:///home/lidaeus/TianShouDSSAT/dssat_generic_wrapper.py) 已显式支持 `wheat`，并为 Wheat 增加资源定位与运行时兼容处理。
- 训练入口 [train_dssat.py](file:///home/lidaeus/TianShouDSSAT/train_dssat.py) 的 `--crop` 选项已包含 `wheat`。
- vendored [dssat_pdi.py](file:///home/lidaeus/TianShouDSSAT/lib/gym_dssat_pdi_official/gym-dssat-pdi/gym_dssat_pdi/envs/dssat_pdi.py) 已登记 `wheat -> KSAS8101`，并允许从外部传入 `env_config_path`。
- 支持矩阵 [crop_support_matrix.md](file:///home/lidaeus/TianShouDSSAT/docs/crop_support_matrix.md) 已更新为项目层已接入。

### 2.2 代码与运行时线索

- vendored DSSAT Fortran 源码中的 NWHEAT 目录未检索到新的 `pdi_expose` 调用，说明仓库内源码层面没有完成一套与 `maize/rice/cotton/tomato` 同等级的专用小麦插桩。
- 但当前环境中已经存在 Wheat 相关实验数据与天气文件：
  - [KSAS8101.WHX](file:///home/lidaeus/TianShouDSSAT/lib/gym_dssat_pdi_official/dssat-csm-data/Wheat/KSAS8101.WHX)
  - `/opt/dssat_env/data/KSAS8101.WTH`
  - `/opt/dssat_env/inst/Genotype/WHAPS048.{CUL,ECO,SPE}`
- 项目虚拟环境目录中已经出现 `gym_dssat_pdi/envs/configs/wheat/`，说明现有运行环境曾经具备一版小麦配置，可作为参考来源，但不能直接替代仓库内缺失的实现。

### 2.3 当前开发判断

- 本轮优先目标不是改造 DSSAT Fortran 物理内核，而是补齐项目层支持链路：
  - wrapper 配置
  - Wheat 模板与 PDI 配置
  - 训练入口
  - 回归测试
  - 开发文档
- 若验证表明现有 `run_dssat` 二进制无法支撑 Wheat 通讯，再单独规划 Fortran/PDI 插桩路线；本轮先以“项目集成支持”作为交付边界。

## 3. 设计方案

### 3.1 支持边界

本轮以 [KSAS8101.WHX](file:///home/lidaeus/TianShouDSSAT/lib/gym_dssat_pdi_official/dssat-csm-data/Wheat/KSAS8101.WHX) 作为 Wheat 的默认实验模板来源，并优先复用环境中已存在的 Wheat 配置资源：

- `lib/gym_dssat_pdi_official/dssat-csm-data/Wheat/KSAS8101.WHX`
- `venv/lib/python*/site-packages/gym_dssat_pdi/envs/configs/wheat/env_config.yml`
- `venv/lib/python*/site-packages/gym_dssat_pdi/envs/configs/wheat/dssat_pdi.jinja2`

### 3.2 观测与动作策略

- 动作仍沿用项目统一接口：
  - `anfer`
  - `amir`
- 观测选择遵循当前 wrapper 的固定长度特征抽取方式，优先复用已有跨作物通用变量：
  - `swfac`
  - `istage`
  - `vstage`
  - `grnwt`
  - `topwt`
  - `xlai`
  - `nstres`
  - `rtdep`

### 3.3 资源装配策略

- Wheat 默认附带：
  - `/opt/dssat_env/data/SOIL.SOL`
  - `/opt/dssat_env/data/KSAS8101.WTH`
- Genotype 文件仍由 `/opt/dssat_env/inst/Genotype/WHAPS048.*` 提供，不复制进临时目录。

## 4. 实施任务

1. 补齐 Wheat 模板、PDI 配置与环境配置
2. 扩展 wrapper、训练入口与修补脚本
3. 更新作物支持矩阵与系统设计文档
4. 执行 Wheat 冒烟验证与回归测试

## 5. 验证方案

- Wheat 环境初始化
- Wheat 环境单步交互
- 现有 `maize` / `tomato` 回归不退化
- Python 语法级检查
- 若仓库内可用 `pytest`，执行定向测试

## 6. 进度记录

### Round 1

- 已完成文档立项与代码审计
- 已确认 Wheat 当前缺口主要位于项目集成层，而不是需求定义层
- 下一步进入代码实现

### Round 2

- 已完成 `wheat` 在 wrapper、训练入口与底层 cultivar 映射中的接入
- 已确认 Wheat 运行依赖 `KSAS8101.WHX`、`KSAS8101.WTH`、`SOIL.SOL` 与 `WHAPS048`
- 已完成 Wheat 初始化与单步交互的可运行验证

### Round 3

- 已定位 Wheat 运行时两个关键兼容性问题：
  - `random_weather=True` 时会误走 `.CLI` 路径，导致查找 `KSAS.CLI`
  - 现有 Wheat PDI 模板包含 `ndarray.itemset` 与未初始化变量，和当前 NumPy/PyCall 组合不兼容
- 已在 wrapper 中改为固定天气模式，并在运行前对 Wheat PDI 模板做兼容化处理
- 已验证 `maize`、`tomato`、`wheat` 均可连续 reset/step，且 Wheat 不再首步即终止
