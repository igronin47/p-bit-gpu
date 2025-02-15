array ="[010011000000101111100000000001110000100100000111101011100110]"


"""
# Remove the square brackets
cleaned_array = array.strip("[]")

# Get the first bit
first_bit = cleaned_array[8]

print(f"The first bit from the MSB side is: {first_bit}")
"""

# Remove the square brackets
cleaned_array = array.strip("[]")

# Print each bit with its index
for index, bit in enumerate(cleaned_array):
    print(f"Index {index}: {bit}")


# Extract values for each set based on specified indices
#first_set = [cleaned_array[i] for i in [30,32,24,16,8]]
fourth_set = [cleaned_array[i] for i in [29,25,22,32,30]]
fifth_set =  [cleaned_array[i] for i in [2,26,29]]

second_set = [cleaned_array[i] for i in [25,17,9,0]]
third_set = [cleaned_array[i] for i in [26,18,10,1]]


#first_set = [cleaned_array[i] for i in [1,7, 5, 3, 0]]
#second_set = [cleaned_array[i] for i in [9, 8, 6, 4,2]]



first_set = [cleaned_array[i] for i in [89,90,91,92,93,94,95,96]]




# Print the sets
print("Sum:", first_set)
print("A:", second_set)
print("B:", third_set)
print("FA:", fourth_set)
print("XOR_OUT:", fifth_set)
