# Protein Network Analysis

A bioinformatics tool designed to parse, analyze, and detect clusters within protein-protein interaction networks. This project was developed as part of the Master of Bioinformatics curriculum.

## Features
- **Network Parsing:** Efficiently processes large-scale interaction data (e.g., YeastNet) with customizable confidence cutoffs.
- **Graph Topology:** Builds adjacency and degree matrices to compute the **Graph Laplacian**.
- **Connectivity Analysis:** Uses eigenvalues of the Laplacian matrix to determine the number of connected components in the network.
- **Functional Cluster Detection:** Implements clustering coefficient algorithms to identify highly interconnected protein complexes.

## Technical Stack
- **Language:** Python
- **Libraries:** NumPy, NetworkX (for visualization support), SciPy
- **Concepts:** Linear Algebra, Graph Theory, Proteomics

## Usage
The main analysis is performed via `analyze_network.py`. 
```python
# Example: Detecting clusters with a specific cutoff
protein_list, edges = parse_network("YeastNet.v3.txt", cutoff=4.0)
clusters = get_all_clusters(protein_list, edges)
