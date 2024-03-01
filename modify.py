import json

# Load the JSON data
with open('mock_tea_data_embeddings.json', 'r') as f:
    data = json.load(f)

# Iterate over each item in the JSON array
for item in data:
    # Update the "embeddings" key to contain only the second element
    item['embeddings'] = item['embeddings'][1]

# Write the modified JSON data back to a new file
with open('modified_json_file.json', 'w') as f:
    json.dump(data, f, indent=4)
