#!/bin/bash

# 小麦强化学习项目环境配置脚本 (gym-dssat-pdi 内置内核版)

echo "Starting environment setup..."

# 1. 检查基础编译器
for cmd in gfortran cmake; do
    if ! command -v $cmd >/dev/null 2>&1; then
        echo "Error: $cmd is not installed. Please install it first."
        exit 1
    fi
done

# 2. 创建 Conda 环境
if command -v conda >/dev/null 2>&1; then
    echo "Creating conda environment 'wheat-rl'..."
    conda env create -f environment.yml
else
    echo "Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

echo "----------------------------------------------------"
echo "NOTE: gym-dssat-pdi will attempt to build its internal DSSAT kernel."
echo "If compilation fails, ensure you have libpdi-dev installed or follow:"
echo "https://github.com/gym-dssat/gym-dssat-pdi"
echo "----------------------------------------------------"

echo "Setup script finished. Run 'conda activate wheat-rl' to start."
