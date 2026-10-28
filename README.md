
# The Keller–Segel Model for Chemotaxis on Metapopulation Graphs and Graphons

This repository contains the numerical simulation codes.

The goal is to reproduce and analyze chemotaxis-driven spatial pattern formation in Keller–Segel models on finite metapopulation graphs and their continuum graphon limits.

## Contents

**Simulation notebooks and scripts**

Figures in the paper are generated using the Python and MATLAB simulation codes in this repository.

- `twopatch_four_panels.m`: Generates **Figure 1**.
- `bifurcation_graph.py`: Generates **Figure 2**.
- `dispersion_relation.ipynb`: Generates **Figure 3**.
- `bifurcation_graph.py` and `bifurcation_graph_small_world.py`: Generate **Figure 4**.
- `bifurcation_graph_new_reaction.py` and `bifurcation_graph_new_reaction_small_world.py`: Generate **Figure 5**.
- `run_network_ode.m` and `rhs_uniform_band_graph.m`: Generate **Figures 6 and 7**.
- `run_network_ode_limit_cycles.m` and `rhs_limit_cycles.m`: Generate **Figure 8**.
- `KS_simulation_ode15s.m` and `KS_simulation_ode15s_new_reaction.m`: Generate **Figures 9, 10, and 11**.
- `graphon_chi_ode15s.m` and `graphon_chi_ode15s_new_reaction.m`: Generate **Figure 12**.

## Requirements

The numerical simulations are implemented in:

- **MATLAB**, using built-in ODE solvers such as `ode45` and `ode15s`.
- **Python**, with numerical and plotting libraries.
- **Jupyter Notebook** for dispersion relation calculations and visualization.
