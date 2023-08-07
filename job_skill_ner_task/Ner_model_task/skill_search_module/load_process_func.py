import re


def read_skill_file(filepath, extra_skill=[]):
    with open(filepath, "r") as f:
        lines = f.readlines()

    skill_groups = []
    skill_count_map = dict()
    for line in lines:
        filter_skill = []
        # remove trailing newline character and split the line by tab character
        skills = line.strip("\n").split("\t")
        # remove any empty string resulted from split
        skills = [skill for skill in skills if skill]
        for skill in skills:
            if skill not in skill_count_map:
                skill_count_map[skill] = 1
                continue

            if skill_count_map[skill] >= 10:
                filter_skill.append(skill)
            else:
                skill_count_map[skill] += 1
        skill_groups.append(filter_skill)
    # 增加技能在這邊增加
    skill_groups.append(extra_skill)
    return skill_groups


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
