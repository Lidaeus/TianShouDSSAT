# 作物 PDI 插桩支持矩阵 (Crop Support Matrix)

本文件详细记录了 DSSAT Fortran 源码中针对不同作物模型的 PDI 插桩（Data Exchange Points）支持情况。

## 1. 通用模块 (Global Support)
以下变量在 DSSAT 的管理或环境模块中定义，对**所有作物**有效。

| 变量名 | 描述 | 类型 | 源码位置 |
| :--- | :--- | :--- | :--- |
| **AMIR** | 灌溉量 (Irrigation) | Input | `Management/IRRIG.for` |
| **ANFER** | 施肥量 (Fertilization) | Input | `Management/Fert_Place.for` |
| **DAP** | 播后天数 (Days After Planting) | Output | `CSM_Main/CSM.for` |
| **YRDOY** | 年份与天数 (Year + DOY) | Output | `CSM_Main/CSM.for` |
| **RAIN/TMIN/TMAX** | 气象数据 | Output | `Weather/weathr.for` |
| **SW** | 土壤含水量 | Output | `Soil/SoilWater/WATBAL.for` |

## 2. 专用作物模型 (Specific Crop Models)

### 2.1 CERES-Maize (玉米)
*   **状态**: 深度支持。
*   **关键变量**: `NSTRES` (氮胁迫), `SWFAC` (水分胁迫), `ISTAGE` (生长阶段), `VSTAGE` (营养阶段), `LAI` (叶面积指数), `GRNWT` (谷物重量)。

### 2.2 CROPGRO 模板类 (番茄、大豆等)
*   **状态**: **通过专用实验模板深度支持**。
*   **适用作物**: 番茄 (Tomato, TM), 大豆 (Soybean, SB) 等。
*   **关键变量**:
    *   `XLAI`: 叶面积指数
    *   `SDWT`: 果实重量 (在番茄中即为产量)
    *   `RTDEP`: 根系深度
*   **模板来源**: 番茄模版已基于 `UFGA0602.TMX` 真实实验数据派生，确保了种植密度（1.2株/m²）和移栽逻辑（Transplanted）的准确性。
*   **源码验证**: 插桩点位于 `Plant/CROPGRO/CROPGRO.for` 和 `GROW.for`。

### 2.3 CERES-Rice (水稻)
*   **状态**: 支持。
*   **关键变量**: `TILNO` (分蘖数), `DYIELD` (日产量), `GPP` (每平方米粒数)。

### 2.4 NWheat (小麦)
*   **状态**: **暂不支持 (PDI缺失)**。
*   **现状**: DSSAT 拥有 NWheat 物理逻辑，但 `gym-dssat-pdi` 的开发者未在 `Plant/NTEF` 目录下插入 PDI 调用代码。
*   **对策**: 
    1.  手动在 `TF_PHENOL.for` 等文件中插入 `pdi_expose`。
    2.  或仅依赖通用模块（Global Support）中的土壤与水资源数据进行简单训练。

## 3. 开发优先级
1.  **Maize (玉米)**: 已调通，作为 Baseline。
2.  **Tomato (番茄)**: 利用 CROPGRO 通用性，优先迁移。
3.  **Strawberry (草莓)**: 利用 CROPGRO 通用性，次优先迁移。
4.  **Wheat (小麦)**: 待评估是否需要进行 Fortran 层的手动插桩。
