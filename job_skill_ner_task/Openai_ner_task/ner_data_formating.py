import re


def get_interval_union(intervals):
    """
    This function merges overlapping intervals.

    :param intervals: A list of tuples each containing two numbers representing intervals.
    :return: A list of tuples with overlapping intervals merged.
    """
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    result = [sorted_intervals[0]]

    for interval in sorted_intervals[1:]:
        if interval[0] <= result[-1][1]:
            result[-1] = (result[-1][0], max(interval[1], result[-1][1]))
        else:
            result.append(interval)

    return result


def split_string_with_delimiters(string):
    """
    This function takes a string as an argument and returns a list of substrings.
    The input string is split at every occurrence of  " ", "-", "+", "/" or ".".
    These delimiters except " " are included as separate items in the resulting list.

    :param string: The input string that is to be split.
    :return: A list of substrings resulting from splitting the input string at the specified delimiters.
    """
    result = re.findall(r"[^-+/ .]+|[-+/.]", string)
    return result


def find_string_in_list(output_list, target_string):
    """
    This function finds the indices of occurrences of a target string within a given list.
    If the target string contains spaces, it's split into words, otherwise it's treated as a list of characters.
    The function checks the target string against the output list in three ways:
    1. As it is,
    2. Split by specified delimiters using the 'split_string_with_delimiters' function, and
    3. Split using the 'split_input_string' function.
    The function returns the first match found in the list.

    :param output_list: The list in which to find the target string. Can be nested. If nested, it is flattened first.
    :param target_string: The string to be found within the output list.
    :return: The start and end indices of the first occurrence of the target string within the output list.
             Returns -1 if the target string is not found.
    """
    # Flatten the output list if it is nested
    if isinstance(output_list[0], list):
        output_list = [item.lower() for sublist in output_list for item in sublist]
    output_list = [item.lower() for item in output_list]
    # Split the target string into words if it contains spaces, otherwise treat it as a list of characters
    target_list = (
        target_string.lower().split() if " " in target_string else list(target_string)
    )
    # Find the start and end index of the target_list in the output_list
    for i in range(len(output_list)):
        if output_list[i : i + len(target_list)] == target_list:
            return [j for j in range(i, i + len(target_list))]

    target_list = split_string_with_delimiters(target_string)
    target_list = [item.lower() for item in target_list]
    for i in range(len(output_list)):
        if output_list[i : i + len(target_list)] == target_list:
            return [j for j in range(i, i + len(target_list))]
    target_list = split_input_string(target_string)
    target_list = [item.lower() for item in target_list]
    for i in range(len(output_list)):
        if output_list[i : i + len(target_list)] == target_list:
            return [j for j in range(i, i + len(target_list))]
    return -1  # Target string not found in the output list


def find_substring_positions(string, list_of_substrings):
    """
    :param string:
    :param list_of_substrings:
    :return: substring start and end pos in string
    """
    positions = []
    for substring in list_of_substrings:
        position = find_string_in_list(string, substring)
        if position != -1:
            positions.append(position)
        else:
            print(f"substring:{substring}")
    return positions


def split_input_string(input_string):
    """
    :param input_string:
    :return: token list
    Explaination:
        If it's a Chinese character, each character will be an item in the token list.
        If it's an English word, it will use punctuation and spaces as dividing points,
        and each dividing point will be an item in the list.
    """
    output_list = []
    temp_word = ""
    for character in input_string:
        if re.match(r"[\u4e00-\u9fff]", character):  # Chinese character
            if temp_word:
                output_list.append(temp_word)
                temp_word = ""
            output_list.append(character)
        elif re.match(r"[a-zA-Z0-9]", character):  # English word
            temp_word += character
        else:  # Punctuation or space
            if temp_word:
                output_list.append(temp_word)
                temp_word = ""
            if character != " ":
                output_list.append(character)

    if temp_word:  # Append the last word if exists
        output_list.append(temp_word)

    return output_list


def indices_to_ner_tags(indices, num_of_tokens):
    # Start by labeling everything as 'O'
    ner_tags = ["O"] * num_of_tokens

    # For each set of indices, change the corresponding labels
    for idx_set in indices:
        for i, idx in enumerate(idx_set):
            if i == 0:
                ner_tags[idx] = "B-Skill"
            else:
                ner_tags[idx] = "I-Skill"
    return ner_tags


def get_ner_pair(test_sentence, output):
    """
    input: question send to openai chat, openai_chat output
    output: {"tokens": tokenized question , "ner_tags": ner label data}
    ## pipeline function
    """
    tokens = split_input_string(test_sentence)
    indices = find_substring_positions(tokens, output)
    return {"tokens": tokens, "ner_tags": indices_to_ner_tags(indices, len(tokens))}
