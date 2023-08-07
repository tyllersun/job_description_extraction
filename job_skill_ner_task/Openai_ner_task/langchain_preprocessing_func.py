import csv
import json
import os
import pandas as pd


def read_distinct_lines(file_path, n=10000):
    unique_lines = set()
    df_list = []
    chunk_size = 1000

    # Use chunking to read the file in parts and avoid loading the entire file into memory
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            line = tuple(row)
            if line not in unique_lines:
                unique_lines.add(line)
                df_list.append(row)
            if len(unique_lines) >= n:
                break
        if len(unique_lines) >= n:
            break

    # Create a DataFrame from the list of unique rows
    df = pd.DataFrame(df_list)


    return df

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
