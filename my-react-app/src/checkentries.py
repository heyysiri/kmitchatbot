import json

def count_pairs(json_object):
    if isinstance(json_object, dict):
        return sum(count_pairs(value) for value in json_object.values()) + len(json_object)
    elif isinstance(json_object, list):
        return sum(count_pairs(item) for item in json_object)
    else:
        return 0

def main(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    pair_count = count_pairs(data)
    print(f"The number of key-value pairs in the JSON file is: {pair_count}")

# Example usage
file_path = 'intents.json'
main(file_path)
