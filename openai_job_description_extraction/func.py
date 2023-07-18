import csv
import json

def append_dict_to_csv(data, filename):
    fieldnames = data[0].keys() if data else []

    with open(filename, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if file.tell() == 0:
            writer.writeheader()
        writer.writerows(data)

def process_string(input_string):
    input_string = input_string["json_string"]
    json_string = input_string.strip().replace('\n', '').replace('\t', '')
    json_string = json_string.replace('```json', '').replace('```', '')
    json_data = json.loads(json_string)
    return {"result": json_data}
