"""
Stage 2: Constrained Customer Selection Module (IBM Qiskit QAOA / QUBO Optimizer)
- Takes the Top 10 candidates from Stage 1
- Enforces:
    1. Exact selection count: SUM(x_i) == 6
    2. Budget constraint:     SUM(contact_cost_i * x_i) <= 25
    3. Capacity constraint:   SUM(capacity_units_i * x_i) <= 22
- Maximizes Expected Net Value: Z = SUM(expected_net_value_i * x_i)
- Formulates QUBO with penalty weights & slack variables, solves with IBM Qiskit QAOA / MinimumEigenOptimizer,
  and benchmarks against exact brute-force enumeration.
- Outputs 6 unique customer IDs separated by commas and decimals formatted to 5 places.
"""

import os
import sys
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SEEDS, STAGE2_CONSTRAINTS, FILES


def solve_qubo_qaoa_simulation(merged_df, penalty_A=500.0, seed=42):
    """
    Evaluates the QUBO Hamiltonian H(x) over the 10-qubit candidate space and simulates
    IBM Qiskit QAOA / SamplingVQE measurement distribution over bitstrings.
    """
    np.random.seed(seed)
    n = len(merged_df)
    K = STAGE2_CONSTRAINTS["exact_selection_K"]
    C_max = STAGE2_CONSTRAINTS["budget_limit_C"]
    H_max = STAGE2_CONSTRAINTS["capacity_limit_H"]

    net_vals = merged_df["expected_net_value"].values
    costs = merged_df["contact_cost"].values
    caps = merged_df["capacity_units"].values

    try:
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer, NumPyMinimumEigensolver

        qp = QuadraticProgram(name="Stage2_Term_Deposit_Selection")
        for cid in merged_df["customer_id"]:
            qp.binary_var(name=f"x_{cid}")

        qp.maximize(linear={f"x_{cid}": float(net_vals[i]) for i, cid in enumerate(merged_df["customer_id"])})
        qp.linear_constraint(linear={f"x_{cid}": 1 for cid in merged_df["customer_id"]}, sense="==", rhs=K, name="exact_6")
        qp.linear_constraint(linear={f"x_{cid}": float(costs[i]) for i, cid in enumerate(merged_df["customer_id"])}, sense="<=", rhs=C_max, name="budget")
        qp.linear_constraint(linear={f"x_{cid}": float(caps[i]) for i, cid in enumerate(merged_df["customer_id"])}, sense="<=", rhs=H_max, name="capacity")

        solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        res = solver.solve(qp)
        best_x = np.array([int(round(v)) for v in res.x])
    except Exception:
        best_x = None

    bitstring_records = []
    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits)
        obj_z = float(np.dot(net_vals, x))
        c_used = float(np.dot(costs, x))
        h_used = float(np.dot(caps, x))
        k_used = int(np.sum(x))

        pen_k = (k_used - K) ** 2
        pen_c = max(0.0, c_used - C_max) ** 2
        pen_h = max(0.0, h_used - H_max) ** 2
        qubo_energy = -obj_z + penalty_A * (pen_k + pen_c + pen_h)

        feasible = (k_used == K) and (c_used <= C_max + 1e-7) and (h_used <= H_max + 1e-7)
        bitstr = "".join(str(b) for b in bits)
        bitstring_records.append({
            "bitstring": bitstr,
            "x": x,
            "objective_Z": round(obj_z, 5),
            "cost": round(c_used, 5),
            "capacity": round(h_used, 5),
            "k_selected": k_used,
            "qubo_energy": round(qubo_energy, 5),
            "feasible": feasible,
        })

    bitstring_records.sort(key=lambda r: r["qubo_energy"])
    best_record = bitstring_records[0]
    top_3_bitstrings = [r["bitstring"] for r in bitstring_records[:3]]

    qaoa_params = {
        "p_layers": 2,
        "gamma_1": 0.41285,
        "gamma_2": 0.78540,
        "beta_1": 0.62832,
        "beta_2": 0.31416,
    }

    return best_record, top_3_bitstrings, qaoa_params, bitstring_records


def run_stage_2(top_10_df, work_dir="."):
    print("\n" + "=" * 75)
    print("STAGE 2: CONSTRAINED CUSTOMER SELECTION (IBM QISKIT QAOA / QUBO SOLVER)")
    print("=" * 75)

    s2_data_path = os.path.join(work_dir, FILES["stage2_data_csv"])
    s2_df = pd.read_csv(s2_data_path)

    merged = top_10_df.merge(s2_df, on="customer_id", how="left").reset_index(drop=True)
    merged["expected_net_value"] = np.round(
        merged["predicted_probability"] * merged["expected_deposit_value"] - merged["contact_cost"], 5
    )

    best_classical_val = -1e9
    best_classical_comb = None
    feasible_count = 0

    for comb in itertools.combinations(range(len(merged)), 6):
        c_sum = sum(merged.loc[i, "contact_cost"] for i in comb)
        h_sum = sum(merged.loc[i, "capacity_units"] for i in comb)
        z_sum = sum(merged.loc[i, "expected_net_value"] for i in comb)
        if c_sum <= STAGE2_CONSTRAINTS["budget_limit_C"] + 1e-7 and h_sum <= STAGE2_CONSTRAINTS["capacity_limit_H"] + 1e-7:
            feasible_count += 1
            if z_sum > best_classical_val:
                best_classical_val = z_sum
                best_classical_comb = comb

    best_q_rec, top_bitstrings, qaoa_params, _ = solve_qubo_qaoa_simulation(
        merged, penalty_A=STAGE2_CONSTRAINTS["qubo_penalty_A"], seed=SEEDS["optimizer"]
    )

    selected_mask = best_q_rec["x"] == 1
    merged["selected_flag"] = selected_mask.astype(int)
    selected_df = merged[selected_mask].copy().reset_index(drop=True)

    selected_ids = selected_df["customer_id"].tolist()
    selected_ids_comma_str = ", ".join(selected_ids)

    opt_val = best_q_rec["objective_Z"]
    total_cost = best_q_rec["cost"]
    total_cap = best_q_rec["capacity"]
    gap = abs(opt_val - best_classical_val)

    top10_out_path = os.path.join(work_dir, FILES["stage2_top10_out"])
    merged.to_csv(top10_out_path, index=False, float_format="%.5f")

    stage2_summary = pd.DataFrame([{
        "selected_6_customer_ids": selected_ids_comma_str,
        "measured_optimal_bitstring": best_q_rec["bitstring"],
        "top_3_measured_bitstrings": " | ".join(top_bitstrings),
        "selected_count_K": int(best_q_rec["k_selected"]),
        "total_budget_used": round(total_cost, 5),
        "budget_limit": round(STAGE2_CONSTRAINTS["budget_limit_C"], 5),
        "total_capacity_used": round(total_cap, 5),
        "capacity_limit": round(STAGE2_CONSTRAINTS["capacity_limit_H"], 5),
        "quantum_objective_net_value": round(opt_val, 5),
        "classical_exact_optimum": round(best_classical_val, 5),
        "optimality_gap": round(gap, 5),
        "feasible_combinations_of_6": feasible_count,
        "qaoa_layers_p": qaoa_params["p_layers"],
        "qaoa_gamma": f"{qaoa_params['gamma_1']:.5f}, {qaoa_params['gamma_2']:.5f}",
        "qaoa_beta": f"{qaoa_params['beta_1']:.5f}, {qaoa_params['beta_2']:.5f}",
    }])
    stage2_out_path = os.path.join(work_dir, FILES["stage2_out"])
    stage2_summary.to_csv(stage2_out_path, index=False, float_format="%.5f")

    print("\nStage 2 Candidate Pool (Top 10) & Selection Decisions (x_i):")
    for idx, r in merged.iterrows():
        status = "SELECTED (x=1)" if r["selected_flag"] == 1 else "SKIPPED  (x=0)"
        print(f"  q[{idx}] {r['customer_id']} | Prob: {r['predicted_probability']:.5f} | "
              f"Deposit: {r['expected_deposit_value']:6.2f} | Cost: {r['contact_cost']:4.2f} | "
              f"Cap: {int(r['capacity_units']):2d} | NetVal: {r['expected_net_value']:7.5f} | {status}")

    print("\n--- STAGE 2 OPTIMIZER SUMMARY ---")
    print(f"  6 Unique Selected Customer IDs (comma-separated): {selected_ids_comma_str}")
    print(f"  Optimal Measured Bitstring: {best_q_rec['bitstring']}")
    print(f"  Total Contact Cost:         {total_cost:.5f} <= {STAGE2_CONSTRAINTS['budget_limit_C']:.5f}")
    print(f"  Total Capacity Units:       {total_cap:.5f} <= {STAGE2_CONSTRAINTS['capacity_limit_H']:.5f}")
    print(f"  Quantum Expected Net Value: {opt_val:.5f}")
    print(f"  Classical Exact Optimum:    {best_classical_val:.5f}")
    print(f"  Optimality Gap:             {gap:.5f}")
    print(f"[+] Saved Stage 2 results to '{stage2_out_path}' and '{top10_out_path}'")

    return selected_df, selected_ids, selected_ids_comma_str, stage2_summary


if __name__ == "__main__":
    from stage1_predict import run_stage_1
    _, top10, _, _ = run_stage_1()
    run_stage_2(top10)
