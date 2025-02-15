import os

# Read data from the file
current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'output_4bit_ADDER_CSA.txt')

with open(file_path, 'r') as f:
    lines = f.readlines()

# Combine every set of 5 lines
combined_lines = []
for i in range(0, len(lines), 5):
    combined_line = ' '.join(line.strip() for line in lines[i:i+5])
    combined_lines.append(combined_line)

# Write combined lines to a new file
combined_file_path = os.path.join(current_dir, 'combined_output_4bit_ADDER_CSA1.txt')
with open(combined_file_path, 'w') as f:
    for line in combined_lines:
        f.write(line + '\n')

print(f"Combined data saved to {combined_file_path}")