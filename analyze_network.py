import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Important note: throughout the code, cutoff 4.0 is used
def parse_network(filename: str, cutoff=1.0):
    # We only want each protein once, regardless of how many times it appears in the dataset
    unique_proteins = set()
    # Here, we again want to avoid duplicates
    edges = set()

    try:
        with open(filename, 'r') as f:
            for line in f:
                # Remove the white spaces and splits in the line; split also returns a list, so we can use index now
                parts = line.strip().split()

                #skip potential lines with not enough data, although all lines in the txt already seem to fulfill this condition
                if len(parts) < 3:
                    continue

                # We have 2 proteins in the file, 1 at index 0, an other one at index 2
                protein_a = parts[0]
                protein_b = parts[1]
                try:
                    # the score is on position 3, so index 2
                    score = float(parts[2])
                # skip the lines where the score is not a number
                except ValueError:
                    continue

                # Now we check if the given score fulfills our cutoff parameter
                if score >= cutoff:
                    # then we add the proteins to the set (our set makes sure it is unique)
                    unique_proteins.add(protein_a)
                    unique_proteins.add(protein_b)

                    # We add the edge in both directions, so from a to b and b to a:
                    # Inner parentheses make this a tuple
                    edges.add((protein_a, protein_b)) # A-->B
                    edges.add((protein_b, protein_a)) # B --> A
    except FileNotFoundError:
        print("Error: the file {filename} could not be found".format(filename=filename))
        return [], set()

    # sort the set, also convert to a list
    protein_list = sorted(list(unique_proteins))
    return protein_list, edges





def build_laplacian(protein_list: list, edges):
    # Determine amount of vertices to determine matrix dimensions
    n = len(protein_list)

    # Make an index map so we know what protein corresponds to what row / column
    protein_to_index = {protein: i for i, protein in enumerate(protein_list)}

    # Initialize adjacent matrix A and degree matrix M with all zeros
    A = np.zeros((n, n))
    D = np.zeros((n, n))

    # Populate A
    # if there is an edge between u and v, place a 1 on position (i, j)
    # loop through every edge; edges exist out of tuples of 2
    for u, v in edges:
        # check if the proteins are in the list
        if u in protein_to_index and v in protein_to_index:
            # Translate protein names into their numeric indices
            i = protein_to_index[u]
            j = protein_to_index[v]
            # Place a '1' on position i,j in matrix A: There is a connection between the protein on index i and the protein on index j
            A[i, j] = 1
            # Because edges are two way (a, b and b, a), the matrix will become symmetric

    # Populate matrix D
    # Degree of the node is sum of the row in matrix A; = how many neighbors does this protein have
    for i in range(n):
        degree = np.sum(A[i, :]) # we sum all 1's on row i
        D[i, i] = degree # place the nuber on the diagonal, so i, i

    # Calculate the laplacian: L = D - A
    L = D - A
    # For debugging purposes, you could add D and A to the return statement, this way letting you check the intermediate matrices
    return L


def connected_components(laplacian) -> int:
    # Calculate the eigenvalues
    # Since the laplacian is symmetric: more efficient to use np.linalg.eigvalsh instead of eigvals
    eigenvalues = np.linalg.eigvalsh(laplacian) # gives an array of floats

    # we can now count the zero's
    # We use floating point numbers; things like 0.0000000012 can occur due to rounding errors
    num_zeros = np.sum(np.abs(eigenvalues) < 1e-10)

    return int(num_zeros)

def get_neighborhood(protein: str, edges) -> set:
    # initiate the set, the protein itself is always in there
    neighborhood = {protein}

    # loop through all edges
    for u, v in edges:
        # if u is the searched protein, then v is a neighbor
        if u == protein:
            neighborhood.add(v)
        # if the searched protein is v, then u is a neighbor
        elif v == protein:
            neighborhood.add(u)
    return neighborhood


def detect_cluster(protein, edges):
    # Retrieve the neighborhood --> this returns a set including the protein itself
    neighborhood = get_neighborhood(protein, edges)

    k = len(neighborhood)

    # if k<2, we cannot divide by k(k-1)
    # a cluster of 1 protein has coefficient 0 by definition
    if k < 2:
        return 0.0, k

    # Count edges within the neighborhood
    # We need to know how many connections between the members of the neighborhood
    # edges are u, v and v, u; so we count 2 times

    edges_in_neighborhood = 0

    for u, v in edges:
        if u in neighborhood and v in neighborhood:
            edges_in_neighborhood += 1

    # calculate the clustering coefficient
    possible_edges_doubled = k * (k - 1)
    clustering_coefficient = edges_in_neighborhood / possible_edges_doubled

    return clustering_coefficient, k


def get_all_clusters(protein_list, edges):

    clusters = []

    # Loop through every protein in the network
    for protein in protein_list:
        coeff, size = detect_cluster(protein, edges)

        # Apply the filters from exercise 3.3
        if size >= 10 and coeff >= 0.75:
            # add to the list if it fulfills these conditions
            clusters.append((protein, size))

    # sort list on neighborhood size
    # descending as in the example
    # x[1] means we want to sort on the second element (index 1) of the tuple
    clusters.sort(key=lambda x: x[1], reverse=True)

    return clusters



# This magic if statement makes the code in the block only run when it is
# not imported as a module. You can run your functions here.

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    network_file = "YeastNet.v3.txt"
    confidence_cutoff = 4.0

    print(f"--- Starting Network Analysis (Cutoff: {confidence_cutoff}) ---")

    # 1. Parse the network
    protein_list, edges = parse_network(network_file, cutoff=confidence_cutoff)
    print(f"Total unique proteins: {len(protein_list)}")
    print(f"Total interactions (bidirectional): {len(edges)}")

    # 2. Laplacian & Components
    L = build_laplacian(protein_list, edges)
    num_components = connected_components(L)
    print(f"Number of connected components: {num_components}")

    # 3. Cluster Detection
    all_clusters = get_all_clusters(protein_list, edges)
    print(f"Significant clusters found (size >= 10, coeff >= 0.75): {len(all_clusters)}")

    # 4. Cluster Visualization
    if all_clusters:
        # Select the center protein of the largest cluster
        center_protein, top_size = all_clusters[0]
        print(f"Visualizing largest cluster center: {center_protein} with {top_size} neighbors")

        # Retrieve all proteins in this cluster
        cluster_nodes = get_neighborhood(center_protein, edges)

        # Build a NetworkX graph for this specific cluster
        G = nx.Graph()
        for u, v in edges:
            # Only add edges if BOTH proteins are within the cluster
            if u in cluster_nodes and v in cluster_nodes:
                G.add_edge(u, v)

        # Drawing the graph
        plt.figure(figsize=(10, 8))
        nx.draw(G, with_labels=True, node_color='lightblue', 
                edge_color='gray', node_size=800, font_size=9)
        plt.title(f"Protein Interaction Cluster: {center_protein} (Size: {top_size})")
        
        # Save and show
        plt.savefig('cluster_visualization2.svg', format='svg')
        print("Visualization saved as 'cluster_visualization2.svg'")
        plt.show()
    else:
        print("No significant clusters found to visualize.")

    print("--- Analysis Complete ---")


