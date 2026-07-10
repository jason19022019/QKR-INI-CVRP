# QKR-INI-CVRP

Code accompanying the manuscript

"A Two-Phase Hybrid Framework for the Capacitated Vehicle Routing Problem Using Quantum Kernel Clustering with Quantum Knapsack Repair"

# Quantum Knapsack Iterative Repair for Capacitated Clustering

Full pipeline implementing **Phase I – Quantum Kernel Clustering**, **Phase II – QAOA‑based Knapsack Repair (QKR‑INI)**, and **Phase III – Exact TSP Routing** for Capacitated Vehicle Routing Problems (CVRP).

This code was used to produce the results in the paper:  
*“Quantum Knapsack Iterative Repair for Capacitated Clustering”* (submitted).

---

## Features

- **Seven quantum feature maps** for kernel k‑means clustering (Amplitude, Angle, IQP, Z‑Map, ZZ‑Map, U3‑Single, U3‑TwoAngle)
- **QAOA‑based knapsack** solver (PennyLane) for cluster repair
- **QKR‑INI batch insertion** heuristic to re‑assign evicted customers
- **Exact TSP routing** via brute‑force (up to 9 customers) or nearest‑neighbour
- Automatic **LaTeX table** and three **publication‑quality figures** (PDF)
- Compatible with standard **Augerat et al. (1995)** VRP instance format

---

