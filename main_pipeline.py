"""
Main End-to-End Orchestrator: Hybrid Quantum-Classical Term Deposit Campaign Pipeline
Executes Stage 1 (Predict), Stage 2 (Select), and Stage 3 (Simulate) in sequence,
verifies ID traceability across stages, and outputs final 5-decimal results.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage1_predict import run_stage_1
from stage2_select import run_stage_2
from stage3_simulate import run_stage_3


def main():
    work_dir = os.path.dirname(os.path.abspath(__file__))

    # Stage 1: Predict & Rank Top 10
    stage1_df, top_10_df, metrics_q, metrics_c = run_stage_1(work_dir=work_dir)

    # Stage 2: Select 6 Unique Customers via IBM Qiskit Optimizer
    selected_df, selected_ids, selected_ids_comma_str, stage2_summary = run_stage_2(
        top_10_df, work_dir=work_dir
    )

    # Verify traceability: all 6 selected IDs must be unique and inside Stage 1 Top 10
    assert len(set(selected_ids)) == 6, "Stage 2 must select exactly 6 unique customer IDs!"
    assert set(selected_ids).issubset(set(top_10_df["customer_id"])), "Selected IDs must come from Stage 1 Top 10!"

    # Stage 3: Simulate 6-Qubit Campaign Outcome Circuit
    stage3_summary = run_stage_3(selected_df, work_dir=work_dir)

    exact_p = float(stage3_summary["exact_analytical_probability"].iloc[0])
    quant_p = float(stage3_summary["noiseless_quantum_estimate"].iloc[0])
    mitig_p = float(stage3_summary["noisy_mitigated_estimate"].iloc[0])

    print("\n" + "#" * 75)
    print("FINAL SUBMISSION SUMMARY (IBM QUANTUM-CLASSICAL PIPELINE)")
    print("#" * 75)
    print(f"Selected 6 Unique Customer IDs (comma-separated): {selected_ids_comma_str}")
    print(f"Stage 2 Total Cost / Budget:                      {stage2_summary['total_budget_used'].iloc[0]:.5f} / 25.00000")
    print(f"Stage 2 Total Capacity / Limit:                   {stage2_summary['total_capacity_used'].iloc[0]:.5f} / 22.00000")
    print(f"Stage 2 Expected Net Value (Quantum = Exact):     {stage2_summary['quantum_objective_net_value'].iloc[0]:.5f}")
    print(f"Stage 3 Exact Analytical Probability P(X >= 3):   {exact_p:.5f}")
    print(f"Stage 3 Quantum Circuit Estimate P(X >= 3):       {quant_p:.5f}")
    print(f"Stage 3 Readout-Mitigated Noisy Estimate:         {mitig_p:.5f}")
    print("#" * 75)


if __name__ == "__main__":
    main()
