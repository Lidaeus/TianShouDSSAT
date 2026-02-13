# Rainbow Training & Infrastructure Upgrade Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable multi-crop (Maize & Tomato) Rainbow DQN training by upgrading the generic wrapper with action discretization and parameterizing the training script.

**Architecture:** Tianshou 2.0 (Rainbow) -> DssatGenericWrapper (Discrete Action Mapping) -> DssatPdi (Box Action Dict) -> DSSAT Binary.

**Tech Stack:** Python 3.12, Tianshou, Gymnasium, PyTorch.

---

### Task 1: Upgrade DssatGenericWrapper with Action Discretization

**Files:**
- Modify: `dssat_generic_wrapper.py`
- Test: `tests/test_action_discretization.py` (New)

**Step 1: Write the failing test**
Create `tests/test_action_discretization.py` that initializes `DssatGenericWrapper` and checks:
1. `env.action_space` is `Discrete`.
2. `env.step(0)` accepts an integer and doesn't crash (mocking the underlying env or running real one).

**Step 2: Run test to verify it fails**
Run pytest. Expect failure because current wrapper has Box/Dict action space from underlying env.

**Step 3: Implement Discretization Logic**
Modify `dssat_generic_wrapper.py`:
- Define `ANFER_BUCKETS` (e.g., `[0, 50, 100, 150, 200]`) and `AMIR_BUCKETS` (e.g., `[0, 10, 20, 30, 40, 50]`).
- Set `self.action_space = spaces.Discrete(len(ANFER) * len(AMIR))`.
- Implement `map_action(action_idx) -> dict`.
- Update `step()` to convert index to dict.

**Step 4: Run test to verify it passes**
Run pytest.

**Step 5: Commit**
Commit changes.

### Task 2: Parameterize Training Script

**Files:**
- Create: `train_dssat.py` (Refactored from `train_rainbow.py`)
- Remove: `train_rainbow.py` (After verification)

**Step 1: Create `train_dssat.py`**
- Copy logic from `train_rainbow.py`.
- Add `argparse` for:
    - `--crop`: 'maize' or 'tomato'
    - `--seed`: int
    - `--epochs`: int
    - `--logdir`: str
- Replace `make_maize_env` with `lambda: DssatGenericWrapper(args.crop, ...)` (handled correctly for pickling).
- Ensure log paths differentiate by crop.

**Step 2: Dry Run Test**
Run `python train_dssat.py --crop maize --epochs 1` to ensure it runs without crashing.

**Step 3: Commit**
Commit the new script.

### Task 3: Launch Training

**Files:**
- Create: `run_experiments.sh`

**Step 1: Create script**
Script to run maize and tomato training in background/parallel or sequence.

**Step 2: Execute**
Start the actual training.
