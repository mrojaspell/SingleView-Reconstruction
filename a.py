import json

# Load the JSON file
with open('notebook_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

with open('RojasPelliccia_Maximo_a4_p2.ipynb', 'w', encoding='utf-8') as ipynb_file:
    json.dump(data, ipynb_file, indent=2)