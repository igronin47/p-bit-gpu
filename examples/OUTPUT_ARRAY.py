import numpy as np
from collections import defaultdict

import numpy as np
from collections import defaultdict

# Step 1: Read the data from the file
def read_arrays_from_file(file_name):
    with open(file_name, "r") as file:
        lines = file.readlines()

    arrays = []
    for line in lines:
        line = line.strip().replace('[', '').replace(']', '')
        array = tuple(float(x) for x in line.split())
        arrays.append(array)

    return arrays

# Step 2: Find the top 5 most frequent arrays
def find_top_frequent_arrays(arrays, top_n=5):
    frequency = defaultdict(int)

    # Count the frequency of each array
    for array in arrays:
        frequency[array] += 1

    # Sort arrays by frequency in descending order and select the top N
    top_arrays = sorted(frequency.items(), key=lambda item: item[1], reverse=True)[:top_n]

    return top_arrays

# Step 3: Load the arrays and find the top 5 most frequent ones
file_name = "comb_output_Blcksworld_sat.csv"  # Replace with your actual file name
arrays = read_arrays_from_file(file_name)
top_frequent_arrays = find_top_frequent_arrays(arrays, top_n=500)

# Step 4: Output the result
print("The top 5 most frequent arrays are:")
for i, (array, count) in enumerate(top_frequent_arrays, start=1):
    print(f"{i}. Array: {np.array(array)}, Count: {count}")

