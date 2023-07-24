import csv
import json
import os


def process_string(input_string):
    if input_string is None:
        return {"result": [""]}
    input_string = input_string["json_string"]
    json_string = input_string.strip().replace("\n", "").replace("\t", "")
    json_string = json_string.replace("```json", "").replace("```", "")
    json_data = json.loads(json_string)
    return {"result": json_data}


def append_dict_to_csv(dict_data, output_file):
    # Check if the file exists (i.e., if it should have a header)
    file_exists = os.path.isfile(output_file)

    # Get the keys from the dictionary
    keys = dict_data.keys()

    # Open the output file and create a DictWriter
    with open(output_file, "a+", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)

        # Write the header row if the file didn't exist
        if not file_exists:
            writer.writeheader()

        # Write the data row
        writer.writerow(dict_data)
