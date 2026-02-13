#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/lib/gym_dssat_pdi_official/gym-dssat-pdi
VENV_PYTHON="./venv/bin/python3"

echo "Starting Maize training..."
$VENV_PYTHON train_dssat.py --crop maize --epochs 100 --seed 1 --log-dir logs > logs/maize_train.out 2>&1 &
MAIZE_PID=$!

echo "Starting Tomato training..."
$VENV_PYTHON train_dssat.py --crop tomato --epochs 100 --seed 1 --log-dir logs > logs/tomato_train.out 2>&1 &
TOMATO_PID=$!

echo "Training launched. PIDs: Maize=$MAIZE_PID, Tomato=$TOMATO_PID"
echo "Check logs/maize_train.out and logs/tomato_train.out for progress."
