# Hybrid Quantum-Classical Term Deposit Campaign Pipeline

## 1. Executive Summary & Final Answers
- **Selected 6 Unique Customer IDs (comma-separated)**: `C096, C014, C007, C078, C020, C062`
- **Stage 2 Constraint Verification**:
  - Selection Count: `6` (`SUM(x_i) = 6`)
  - Total Contact Cost: `24.80000` $\le 25.00000$ (`PASS`)
  - Total Capacity Units: `20.00000` $\le 22.00000$ (`PASS`)
  - Quantum Expected Net Value ($Z$): `688.93291` (Classical Exact Optimum: `688.93291`, Optimality Gap: `0.00000`)
- **Stage 3 Conversion Target Probability ($P(X \ge 3)$, 5 Decimals)**:
  - Exact Analytical (Poisson Binomial): `0.99975`
  - Ideal Statevector Probability: `0.99975`
  - IBM Quantum Circuit Estimate (`10,000` shots, noiseless): `0.99990` (Abs Error: `0.00015`)
  - Noisy Quantum Circuit Unmitigated (`10,000` shots): `0.99880` (Abs Error: `0.00095`)
  - Noisy Quantum Circuit Mitigated (Readout Inversion): `0.99973` (Abs Error: `0.00002`)
  - Classical Monte Carlo (`100,000` trials): `0.99973` (Abs Error: `0.00002`)

## 2. Modular File Architecture
| File | Role in Pipeline |
| :--- | :--- |
| `config.py` | Central configuration, reproducibility seeds (`seed=42`), Stage 2 constraints ($N=10, K=6, C=25, H=22$), and IBM Qiskit backend settings |
| `stage1_predict.py` | Training-only preprocessing (`StandardScaler`), Stratified 5-Fold Cross-Validation, Hybrid Quantum Variational Classifier (`ZZFeatureMap` Pauli expectations) + Classical L2 Logistic Baseline, and 5-decimal ranking of 100 test customers |
| `stage2_select.py` | Constrained binary optimization (`QuadraticProgram` / QUBO penalty Hamiltonian / QAOA $p=2$) selecting 6 unique customer IDs separated by commas, validated against exact classical brute force over $\binom{10}{6}=210$ combinations |
| `stage3_simulate.py` | 6-qubit IBM Qiskit circuit preparing $R_y(\theta_i)$ rotations ($\theta_i = 2\arcsin\sqrt{p_i}$), evaluating ideal statevector, noiseless shots, noisy depolarizing/readout shots, readout mitigation, exact Poisson Binomial, and Monte Carlo |
| `main_pipeline.py` | End-to-end orchestrator running Stages 1, 2, and 3 sequentially with automated customer ID traceability assertions |
| `requirements.txt` | Pinned package versions for local or cloud execution |
| `stage1_results.csv` | All 100 test customers with `customer_id`, `predicted_probability` (5 decimals), `rank`, and `top_10_flag` |
| `stage2_results.csv` | Top-level Stage 2 optimization summary including comma-separated 6 customer IDs, optimal bitstring (`1111001100`), budget/capacity usage, objective value, and `0.00000` optimality gap |
| `stage2_top10_candidates.csv` | Full breakdown of the 10 Stage 2 candidates with Stage 1 probabilities, deposit values, contact costs, capacity units, expected net values, and binary selection decisions $x_i \in \{0,1\}$ |
| `stage3_results.csv` | Full Stage 3 circuit specification, qubit-to-customer mapping, bit-ordering convention, gate/depth counts, and 5-decimal probabilities across all simulation regimes |

## 3. Quantum vs. Classical Split & Technical Approach
1. **Stage 1 (Predict)**:
   - **Classical**: Feature standardization (`StandardScaler`) fit strictly on training folds (`N=300`), L2-regularized logistic regression baseline (`ROC-AUC: 0.75881`, `PR-AUC: 0.73234`, `Precision@10: 1.00000`, `Brier: 0.19773`).
   - **Quantum**: 5-qubit `ZZFeatureMap` encoding single-qubit Bloch rotations ($\cos(2\arctan x_i), \sin(2\arctan x_i)$) and two-qubit $ZZ$ entanglement correlations ($\cos(2\arctan x_i - 2\arctan x_j)$) into a calibrated hybrid readout (`ROC-AUC: 0.71551`, `Precision@10: 1.00000`).
2. **Stage 2 (Select)**:
   - **Formulation**: Binary decision variables $x_i \in \{0,1\}$ for $i \in \{0,\dots,9\}$. Objective maximizes $Z = \sum_{i=1}^{10} \text{expected\_net\_value}_i \cdot x_i$, transformed to a QUBO minimization Hamiltonian:
     $$H(x) = -\sum_{i=1}^{10} v_i x_i + A\left(\sum_{i=1}^{10} x_i - 6\right)^2 + A\,\max\left(0, \sum_{i=1}^{10} c_i x_i - 25\right)^2 + A\,\max\left(0, \sum_{i=1}^{10} h_i x_i - 22\right)^2$$
     with penalty weight $A = 500.0$.
   - **Why Rank 5 (`C009`) and Rank 6 (`C065`) Are Skipped**: Naively picking the top 6 customers by probability (`C096, C014, C007, C078, C009, C065`) costs $25.50 > 25.00$ (violating the budget limit!). The quantum QUBO optimizer skips `C009` (high capacity = 5, cost = 4.20) and `C065` (low deposit value = 89.61) in favor of `C020` (cost = 2.50, cap = 3) and `C062` (deposit value = 142.31), achieving the exact global optimum $Z = 688.93291$ within budget (`24.80000`) and capacity (`20.00000`).
3. **Stage 3 (Simulate)**:
   - **Circuit Design**: 6 logical/physical qubits (`q[0]` to `q[5]`), 1 single-qubit gate layer of $R_y(\theta_i)$ rotations ($\theta_i = 2\arcsin\sqrt{p_i}$) + 1 measurement layer (circuit depth = 2, 1Q gates = 6, 2Q gates = 0, SWAPs = 0).
   - **Bit Ordering**: Qiskit little-endian convention `c5 c4 c3 c2 c1 c0` where rightmost bit `c0` corresponds to `q[0]` (`C096`) and leftmost bit `c5` corresponds to `q[5]` (`C062`).

## 4. How to Run
```bash
pip install -r requirements.txt
python3 main_pipeline.py
```
