"""
Central Configuration & Reproducibility Settings
Hybrid Quantum-Classical Term Deposit Campaign Pipeline (IBM Qiskit)
"""

# Reproducibility Seeds (Section 8.2)
SEEDS = {
    "data_split": 42,
    "model": 42,
    "optimizer": 42,
    "simulator": 42,
    "transpiler": 42,
    "monte_carlo": 42,
}

# Stage 2 Constraint Parameters (bank_campaign_participant_release.pdf)
STAGE2_CONSTRAINTS = {
    "candidate_count_N": 10,
    "exact_selection_K": 6,
    "budget_limit_C": 25.0,
    "capacity_limit_H": 22.0,
    "qubo_penalty_A": 500.0,
}

# Stage 3 Quantum Simulation Settings
STAGE3_CONFIG = {
    "backend_name": "IBM Qiskit StatevectorSampler / AerSimulator",
    "shots": 10000,
    "mc_trials": 100000,
    "target_min_subscriptions": 3,
    "depolarizing_error_rate": 0.005,
    "readout_error_p0given1": 0.015,
    "readout_error_p1given0": 0.010,
}

# File Paths
FILES = {
    "train_csv": "Stage1_Train.csv",
    "test_csv": "Stage1_Test.csv",
    "stage2_data_csv": "Stage2_Data.csv",
    "stage2_constraints_csv": "Stage2_Constraints.csv",
    "stage1_out": "stage1_results.csv",
    "stage2_out": "stage2_results.csv",
    "stage2_top10_out": "stage2_top10_candidates.csv",
    "stage3_out": "stage3_results.csv",
}
