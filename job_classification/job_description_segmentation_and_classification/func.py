from datasets import Dataset
from transformers import AutoTokenizer
import random
from googletrans import Translator

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")


def process_list_of_lists(list_of_lists, col):
    result = []
    for sublist in list_of_lists:
        for string in sublist:
            string = string.replace('⁇', '')  # 移除 '⁇'
            string = string.replace('NULL', '')
            string = string.replace('nan', '')
            if col != "people_need":
                if len(string.strip()) >= 5 or len(string.strip().split(" "))>= 5:  # 檢查字數
                    result.append(string.strip())
            else:
                result.append(string.strip())
    return result

def dict_to_dataset(data):
    # Convert the dictionary into a list of
    # dictionaries where each entry is an example with the label and corresponding feature
    examples = [{'label': label, 'text': feature} for label, features in data.items() for feature in features]

    # Load the examples into a Hugging Face dataset
    dataset = Dataset.from_dict(
        {'label': [example['label'] for example in examples], 'text': [example['text'] for example in examples]}
    )

    # Split the dataset into train and test sets (70% train, 30% test)
    dataset = dataset.train_test_split(test_size=0.3)
    return dataset, examples

# Define the tokenize function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def dict_to_token_dataset(data):
    # Convert the dictionary into a list of dictionaries
    # where each entry is an example with the label and corresponding text
    examples = [{'label': label, 'text': text} for label, texts in data.items() for text in texts]

    # Load the examples into a Hugging Face dataset
    dataset = Dataset.from_dict(
        {'label': [example['label'] for example in examples], 'text': [example['text'] for example in examples]}
    )
    dataset = dataset.train_test_split(test_size=0.3)
    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Display the tokenized dataset
    return tokenized_dataset, examples


def convert_labels_to_int(example, label_map):
    # Convert the labels in the dataset to integers using the label map
    example['label'] = label_map[example['label']]
    return example

def convert_int_to_labels(example, inverse_label_map):
    example['label'] = inverse_label_map[example['label']]
    return example


def update_loc(loc):
    if loc < 0:
        loc -= 1
    else:
        loc += 1
    return loc


def count_valid_characters(string):
    count = 0
    for char in string:
        if char.isalnum():
            count += 1
    return count


def string_processing(string):
    string = string.replace('⁇', '')  # 移除 '⁇'
    string = string.replace('NULL', '')
    return string


def random_pick_from_list_and_return_value(lst, list_of_lists_dict, i, loc):
    lottery = random.choice(lst)
    if lottery == "none":
        return ""
    else:
        if len(list_of_lists_dict[lottery][i]) == 0:
            return random_pick_from_list_and_return_value(lst, list_of_lists_dict, i, loc)
        else:
            value = str(list_of_lists_dict[lottery][i][loc])
            while count_valid_characters(value) == 0:
                try:
                    value = list_of_lists_dict[lottery][i][update_loc(loc)]
                except:
                    return random_pick_from_list_and_return_value(lst, list_of_lists_dict, i, 0)
            if value == "nan":
                return random_pick_from_list_and_return_value(lst, list_of_lists_dict, i, 0)
            return value


def process_list_of_lists_neighbor(list_of_lists_dict, col, lottery_column):
    result = []
    list_of_lists = list_of_lists_dict[col]
    lottery_column.remove(col)
    for sublist in list_of_lists:
        for i in range(len(sublist)):
            string = str(sublist[i])
            if col != "people_need":
                if len(string.strip()) >= 5 or len(string.strip().split(" ")) >= 5:  # 檢查字數
                    string = string.strip()
                else:
                    continue
            else:
                if string != "nan":
                    string = string.strip()
                else:
                    continue

            if i == 0:
                string = random_pick_from_list_and_return_value(lottery_column, list_of_lists_dict, i,
                                                                -1) + " block " + string
            else:
                string = str(sublist[i - 1]) + " block " + string

            if i == len(sublist) - 1:
                string = string + " block " + \
                         random_pick_from_list_and_return_value(
                            lottery_column,
                            list_of_lists_dict,
                            i,
                            -1)
            else:
                string = string + str(sublist[i + 1])
            string = string.replace('⁇', '')  # 移除 '⁇'
            string = string.replace('NULL', '')
            result.append(string)
    return result


def add_trans_to_list(list_, string, lang, translator):
    try:
        list_.append(translator.translate(string, dest=lang).text)
    except:
        print(translator.translate(string, dest=lang).text)
    return list_


def process_list_of_lists_neighbor_trans(list_of_lists_dict, col, lottery_column):
    result = []
    list_of_lists = list_of_lists_dict[col]
    lottery_column.remove(col)
    for sublist in list_of_lists:
        for i in range(len(sublist)):
            string = str(sublist[i])
            if col != "people_need":
                if len(string.strip()) >= 5 or len(string.strip().split(" "))>= 5:  # 檢查字數
                    string = string.strip()
                else:
                    continue
            else:
                if string != "nan":
                    string = string.strip()
                else:
                    continue
            translate_string = []
            if col == "welfare" or col == "company_description" or col == "people_need":
                translator = Translator()
                translate_string = add_trans_to_list(translate_string, string, 'zh-tw', translator)
                translate_string = add_trans_to_list(translate_string, string, 'zh-cn', translator)
                translate_string = add_trans_to_list(translate_string, string, 'en', translator)
                translate_string = add_trans_to_list(translate_string, string, 'ja', translator)
            else:
                translate_string.append(string)
            for string in translate_string:
                if i == 0:
                    string = random_pick_from_list_and_return_value(lottery_column, list_of_lists_dict, i, -1) \
                             + " block " + string
                else:
                    string = str(sublist[i-1]) + " block " + string

                if i == len(sublist) -1:
                    string = string + " block " + \
                             random_pick_from_list_and_return_value(lottery_column, list_of_lists_dict, i, -1)
                else:
                    string = string + str(sublist[i+1])
                string = string.replace('⁇', '')  # 移除 '⁇'
                string = string.replace('NULL', '')
                result.append(string)
    return result


def generate_random_range(p=0.1):
    if random.random() > p:
        return ""

    options_people = ["人", "位", ""]
    if random.random() > 0.5:
        num1 = random.randint(1, 9)
        num2 = random.randint(num1 + 1, 10)

        options = ["至", "到", "～"]
        separator = random.choice(options)
        quant = random.choice(options_people)
        return f"{num1}{separator}{num2}{quant}"
    else:
        num1 = random.randint(1, 9)
        quant = random.choice(options_people)
        return f"{num1}{quant}"


def process_list_of_lists_neighbor_trans_create_people(list_of_lists_dict, col, lottery_column):
    result = []
    list_of_lists = list_of_lists_dict[col]
    lottery_column.remove(col)
    for sublist in list_of_lists:
        if (len(sublist) == 0 or sublist[0] == "nan") and col == "people_need":
            value = generate_random_range(0.4)
            if value != "":
                sublist[0] = value
        for i in range(len(sublist)):
            string = str(sublist[i])
            if col != "people_need":
                if len(string.strip()) >= 5 or len(string.strip().split(" ")) >= 5:  # 檢查字數
                    string = string.strip()
                else:
                    continue
            else:
                if string != "nan":
                    string = string.strip()
                else:
                    continue
            translate_string = []
            if col == "welfare" or col == "company_description" or col == "people_need":
                translator = Translator()
                translate_string = add_trans_to_list(translate_string, string, 'zh-tw', translator)
                translate_string = add_trans_to_list(translate_string, string, 'zh-cn', translator)
                translate_string = add_trans_to_list(translate_string, string, 'en', translator)
                translate_string = add_trans_to_list(translate_string, string, 'ja', translator)
            else:
                translate_string.append(string)
            for string in translate_string:
                if i == 0:
                    string = random_pick_from_list_and_return_value(lottery_column, list_of_lists_dict, i,
                                                                    -1) + "[ block ]" + string
                else:
                    string = str(sublist[i - 1]) + "[ block ]" + string

                if i == len(sublist) - 1:
                    string = string + "[ block ]" + \
                             random_pick_from_list_and_return_value(
                                lottery_column,
                                list_of_lists_dict,
                                i,
                                -1
                             )
                else:
                    string = string + str(sublist[i + 1])
                string = string.replace('⁇', '')  # 移除 '⁇'
                string = string.replace('NULL', '')
                result.append(string)
    return result
