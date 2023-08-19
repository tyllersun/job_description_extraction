def remove_empty_elements(input_list):
    updated_list = [item for item in input_list if any(c.isalnum() for c in item.strip())]
    updated_list = [item.strip() for item in updated_list]
    return updated_list

def remove_common_elements(list1, list2):
    updated_list = []
    for item in list1:
        for word in list2:
            item = item.replace(word, '')
        if item.strip() != "":
            updated_list.append(item)
    return updated_list