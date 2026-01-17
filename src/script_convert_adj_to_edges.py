import sys
import csv


def read_adjacency_matrix(filename):
    with open(filename, "r") as f:
        matrix = [list(map(int, line.strip().split())) for line in f if line.strip()]
    return matrix


def is_symmetric(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True


def adjacency_to_edgelist(matrix, symmetric):
    edges = []
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if matrix[i][j]:
                if symmetric and i < j:
                    edges.append((i, j))
                elif not symmetric:
                    edges.append((i, j))
    return edges


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python script_convert_adj_to_edges.py <adj_matrix.txt> <output_edges.csv>"
        )
        sys.exit(1)

    adj_file = sys.argv[1]
    out_file = sys.argv[2]

    matrix = read_adjacency_matrix(adj_file)
    symmetric = is_symmetric(matrix)
    edges = adjacency_to_edgelist(matrix, symmetric)

    with open(out_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for edge in edges:
            writer.writerow(edge)


if __name__ == "__main__":
    main()
