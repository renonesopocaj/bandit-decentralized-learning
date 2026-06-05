# BanditDL Refactoring Plan

This document outlines the strategy for improving the maintainability, readability, and consistency of the BanditDL codebase.

## 1. Clean Up & Standardization

### 1.1. Dead Code Removal
- **Aggregators:** Remove `geometric_median_old`, `krum_old`, `nnm_old`, and `nearest_neighbor_mixing_old` from `banditdl/core/robustness/aggregators.py`.
- **Imports:** Run `ruff check --fix` across the repository to remove unused imports and fix minor linting issues (e.g., multiple imports on one line).
- **Result Files:** Remove redundant saving of `accuracies.npy` in `engine.py` (prefer `validation_accuracies.npy`).

### 1.2. Naming & Key Consistency
- Standardize on `_` (underscores) for all internal dictionary keys and configuration parameters.
- Audit `config_adapter.py` to ensure it only performs necessary translations, moving toward a single point of truth for parameter naming.

## 2. Structural Refactoring

### 2.1. Simplified Worker Initialization
- **Action:** Create `banditdl/core/worker/config.py` containing a `WorkerConfig` dataclass.
- **Impact:** Collapse the 20+ arguments in `HonestWorker`, `DynamicWorker`, and `FixedGraphWorker` constructors into a single `config: WorkerConfig` object.
- **Benefit:** Improves readability and makes adding new hyperparameters significantly easier.

### 2.2. Unified Experiment Engine
- **Action:** Refactor `banditdl/experiments/engine.py`.
- **Strategy:**
    - Extract common logic (setup, evaluation, result saving, logging) into a shared `BaseEngine` class or helper functions.
    - Implement a unified training loop that handles both fixed and dynamic topologies via a "RoundStrategy".
- **Benefit:** Eliminates ~70% code duplication between `run_fixed` and `run_dynamic`.

### 2.3. Configuration Management Improvements
- **Action:** Refactor `config_adapter.py` to use `OmegaConf` more idiomatically.
- **Goal:** Minimize manual dictionary building. Use Hydra's `_target_` where possible for aggregators and samplers to reduce the "if-else" factory patterns.

## 3. Bug Fixes & Robustness

### 3.1. Topology Weights
- **Action:** Add validation in `CommunicationNetwork.weights` to ensure normalized weights don't become negative or diverge if `sum(res) > 1`.

### 3.2. Error Handling in Parsers
- **Action:** Improve `_read_metric_file_max` in `sweep.py` to log specific warnings when corruption is detected instead of silent failure, helping diagnose collided Hydra runs.

## 4. Documentation & Maintenance
- Update `AGENTS.md` and `README.md` if any major CLI or config entry points change.
- Ensure all new modules have appropriate unit tests in `tests/`.

## 5. Implementation Phases
1. **Phase 1 (Cleanup):** [COMPLETED] Dead code removal and linting.
2. **Phase 2 (Configuration):** [COMPLETED] Introduce `WorkerConfig` and update worker constructors.
3. **Phase 3 (Engine):** [COMPLETED] Unify `run_fixed` and `run_dynamic`.
4. **Phase 4 (Validation):** [COMPLETED] Final audit of weights and error handling.
