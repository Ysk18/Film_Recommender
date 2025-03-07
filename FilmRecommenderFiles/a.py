import json

# Load the data from the original JSON file
with open('sample.json', 'r') as f:
    data = json.load(f)

# Filter the relevant fields ('reviewer', 'movie', 'rating')
filtered_data = [
    {
        'reviewer': entry['reviewer'],
        'movie': entry['movie'],
        'rating': entry['rating']
    }
    for entry in data
]
# Limit the data to 200,000 lines (approximately 50,000 reviews)
filtered_data = filtered_data[:1500]

# Write the filtered data to a new file 'filtered.json'
with open('filtered.json', 'w') as f:
    json.dump(filtered_data, f, indent=4)

print("Filtered data has been written to filtered.json")