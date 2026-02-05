# Packages you can use:
# * Everything in Python's standard library
# * NumPy
# * SciPy
# *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def my_name():
    # Replace this with your full name.
    return "Wout Simoen"

# Important note: throughout the assignment, cutoff 4.0 should be used; i'm not sure if this line was meant as indication to use default value for cutoff of 4.0
# because it was already defaulted at 1.0 in the python file, here I will leave it as 1.0 and manually change it to cutoff=4.0 for the rest of the assignment as requested by the pdf.
# The reason behind this is the example output blocks in the pdf. They do not pass a cutoff parameter, and with cutoff=4.0, the parse_network function gives the wrong output
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

    # now sort the set, also convert to a list because the output of protein_list should be a list as mentioned in the exercise
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
    #
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
    # note that in the numerator of requires 2 * |Ei| so this condition is now already fulfilled

    edges_in_neighborhood = 0

    for u, v in edges:
        if u in neighborhood and v in neighborhood:
            edges_in_neighborhood += 1

    # calclate the clustering coefficient
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

if __name__ == "__main__":
    print(my_name())


    # Question 1c: Count proteins with different cutoffs
    print("\n=== QUESTION 1c ===")
    proteins_no_cutoff, edges_no_cutoff = parse_network("YeastNet.v3.txt", 0.0)
    print(f"Total unique proteins (cutoff 0.0): {len(proteins_no_cutoff)}")

    protein_list, edges = parse_network("YeastNet.v3.txt", 4.0)
    print(f"Total unique proteins (cutoff 4.0): {len(protein_list)}")


    # Question 2: Graph Laplacian and Connected Components
    print("\n=== QUESTION 2 ===")
    L = build_laplacian(protein_list, edges)
    num_components = connected_components(L)
    print(f"Connected components (cutoff 4.0): {num_components}")


    # Question 3: Cluster Detection
    print("\n=== QUESTION 3 ===")

    # 3.1 & 3.2: Test with the example network from Toledo
    p_ex, e_ex = parse_network("example_network.tsv", 1.0) # override the default value for example network

    # 3.1: get_neighborhood
    neighborhood_E = get_neighborhood('E', e_ex)
    print(f"3.1 - Neighborhood of E: {neighborhood_E}")

    # 3.2: detect_cluster
    coeff_D, size_D = detect_cluster("D", e_ex)
    print(f"3.2 - Cluster D: coefficient={coeff_D}, size={size_D}")

    # 3.3: Find all clusters in YeastNet
    all_clusters = get_all_clusters(protein_list, edges)
    print(f"3.3 - Total clusters found: {len(all_clusters)}")
    print("All clusters: \n", all_clusters)

    # 3.3a: Check for PUP1 (YOR157C)
    print("\n--- Question 3.3a: PUP1 ---")
    target = "YOR157C"
    found = False
    for p, size in all_clusters:
        if p == target:
            print(f"{target} (PUP1) found in cluster with size {size}")
            found = True
            break
    if not found:
        print(f"{target} (PUP1) NOT found in cluster list")

    
    # Question 3.3b
    print("\n--- Question 3.3b: Largest cluster ---")
    if all_clusters:
        largest_protein, largest_size = all_clusters[0]
        print(f"Center: {largest_protein}, Size: {largest_size}")
        cluster_members = get_neighborhood(largest_protein, edges)
        print(f"Members: {cluster_members}")
