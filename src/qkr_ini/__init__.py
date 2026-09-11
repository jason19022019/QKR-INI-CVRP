"""QKR-INI: Quantum Kernel Clustering + Quantum Knapsack Repair for CVRP.

A hybrid quantum-classical pipeline for the Capacitated Vehicle Routing
Problem: Phase I quantum-kernel clustering, Phase II QKR-INI capacity
repair (QAOA-based knapsack), Phase III TSP routing, and closed-loop
Or-opt/ALNS refinement.
"""
__version__ = "0.1.0"

from .data import VRPLoader
from .pipeline import main

__all__ = ["VRPLoader", "main", "__version__"]
