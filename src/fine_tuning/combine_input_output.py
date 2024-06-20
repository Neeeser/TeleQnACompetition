import json

# Function to combine "input" and "output" keys
def combine_input_output(data):
    for entry in data:
        if 'input' in entry and 'output' in entry:
            entry['input'] = entry['input'] + entry['output']
            del entry['output']
    return data

# Load the JSON file
input_file = "prepared_train_data_with_rag.json"
output_file = "proper_train_data_with_rag.json"

with open(input_file, "r") as file:
    data = json.load(file)

# Combine "input" and "output" keys
combined_data = combine_input_output(data)

# Save the modified data to a new JSON file
with open(output_file, "w") as file:
    json.dump(combined_data, file, indent=4)

print(f"Combined data has been saved to {output_file}")
