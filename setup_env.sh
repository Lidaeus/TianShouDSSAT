#!/bin/bash
# 小麦强化学习项目环境配置脚本 (venv 版)
echo "Starting environment setup..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Setup finished. Run 'source venv/bin/activate' to start."
