import sys
import numpy as np


def get_min_label_arr(input_filename):
    with open(input_filename, 'r') as file:
        first_line = file.readline().strip().split()
        N, M = map(int, first_line)  # N is the number of rows, M is the number of columns

        uf = []
        for i in range(N):
            uf_line = list(map(int, file.readline().strip().split()))
            uf.extend(uf_line)

    uf_array = np.array(uf, dtype=int).reshape(N, M)

    min_indices = {root: N * M for root in set(uf_array.flatten())}

    # Calculate the minimum index for each root node
    for i in range(N):
        for j in range(M):
            root = uf_array[i][j]
            if i * M + j < min_indices[root]:
                min_indices[root] = i * M + j
    # Create a label image based on the minimum indices
    label_image = np.zeros((N, M), dtype=int)
    for i in range(N):
        for j in range(M):
            label_image[i][j] = min_indices[uf_array[i][j]]
    return label_image


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py output_file output_file_ref")
        sys.exit(1)

    output_filename = sys.argv[1]
    ref_output_filename = sys.argv[2]

    label_arr = get_min_label_arr(output_filename)
    label_arr_ref = get_min_label_arr(ref_output_filename)

    if np.allclose(label_arr, label_arr_ref):
        print("Correct ({} refers ro {}).".format(output_filename, ref_output_filename))
    else:
        print("Incorrect ({} refers ro {}).".format(output_filename, ref_output_filename))

