'''
def format_matrix(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Strip any leading/trailing whitespace and split the line into elements
            elements = line.strip().split()
            # Join elements with commas
            formatted_line = ','.join(elements)
            # Write the formatted line to the output file
            outfile.write(formatted_line + '\n')

# Usage example
input_file = '4BITADDER_CSA.txt'  # Replace with your input file name
output_file = 'formatted_matrix.txt'  # Replace with your desired output file name


format_matrix(input_file, output_file)
'''


def format_matrix_with_brackets(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Strip any leading/trailing whitespace and split the line into elements
            elements = line.strip().split()
            # Join elements with commas
            formatted_line = ','.join(elements)
            # Add square brackets around the formatted line
            formatted_line_with_brackets = f'[{formatted_line}]'
            # Write the formatted line to the output file
            outfile.write(formatted_line_with_brackets + '\n')

# Usage example
input_file = 'hMATRIX.txt'  # Replace with your input file name
output_file = 'formatted_H_matrix1.txt'  # Replace with your desired output file name

format_matrix_with_brackets(input_file, output_file)
