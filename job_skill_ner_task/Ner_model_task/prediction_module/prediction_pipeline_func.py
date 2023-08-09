import torch
import re
from Ner_model_task.label_list import label_list


def cont_words_clean(predictions, processing_words, skill_search_system):
    for i in range(
        len(processing_words) - 1, 0, -1
    ):  # Iterate from the last element to the first
        if processing_words[i].startswith("##"):
            processing_words[i - 1] += processing_words[i].lstrip(
                "##"
            )  # Combine with the previous word
            processing_words[i] = "@"  # Empty the current word
            if predictions[i] == "B-Skill":
                predictions[i] = "I-Skill"
                predictions[i - 1] = "B-Skill"
            if skill_search_system.search([predictions[i-1]]) and skill_search_system.startsWith([predictions[i+1]]):
                predictions[i + 1] = "B-Skill"
    # Record the modified word and corresponding prediction
    return predictions, processing_words


def get_entities(tokens, ner_tags, not_skill_set):
    entities = []
    entity = []
    not_skill_set = set([s.lower() for s in not_skill_set])
    for i, (token, tag) in enumerate(zip(tokens, ner_tags)):
        if tag.startswith("B"):  # Beginning of entity
            if entity:  # If there is an existing entity, append it to the list
                entities.append(
                    "".join(entity)
                    if all("\u4e00" <= char <= "\u9fff" for char in "".join(entity))
                    else " ".join(entity)
                )
            entity = [token]  # Start a new entity
        elif tag.startswith("I"):  # Inside of entity
            if entity and entity[-1] == '':
                entities.append(
                    "".join(entity)
                    if all("\u4e00" <= char <= "\u9fff" for char in "".join(entity))
                    else " ".join(entity)
                )
                if ner_tags[i+1].startswith("I"):
                    ner_tags[i+1] = "B-Skill"
                entities = []
                continue
            if entity:  # If there is an existing entity, add token to it
                if entity[-1] == "@":
                    entity.pop()
                if token != "@":
                    entity.append(token)
            else:  # Else, start a new entity (this handles the case where a tag sequence starts with I)
                entity = [token]
        else:  # Outside of any entity
            if entity:  # If there is an existing entity, append it to the list
                entities.append(
                    "".join(entity)
                    if all("\u4e00" <= char <= "\u9fff" for char in "".join(entity))
                    else " ".join(entity)
                )
            entity = []

    # Append the last entity if it exists
    if entity:
        entities.append(
            "".join(entity)
            if all("\u4e00" <= char <= "\u9fff" for char in "".join(entity))
            else " ".join(entity)
        )
    final_entities = []
    for entity in entities:
        if len(entity) != 1 and entity != "":
            entity = entity.strip()
            prev = False
            str_ = ""
            for ch in entity:
                if prev and ch == " ":
                    prev = False
                    continue

                if '\u4e00' <= ch <= '\u9fff':
                    prev = True
                str_ += ch
            if str_.lower() not in not_skill_set:
                final_entities.append(str_)

    return final_entities


def entities_preprocess(tokens, ner_tags, credict, skill_tree):
    entities = []
    entity = []

    for i, (token, tag) in enumerate(zip(tokens, ner_tags)):
        if tag.startswith("B"):  # Beginning of entity
            if entity:  # If there is an existing entity, append it to the list
                entities.append(
                    "".join(entity)
                    if all("\u4e00" <= char <= "\u9fff" for char in "".join(entity))
                    else " ".join(entity)
                )
            entity = [token]  # Start a new entity

            # add tree data
            test_entity = entity

            k = 1
            while i + k <= len(tokens) and ner_tags[i + k].startswith("I"):
                if tokens[i + k] != "":
                    test_entity += [tokens[i + k]]
                k += 1
            if not skill_tree.search(test_entity):
                credict.add_dict(test_entity)

            while skill_tree.startsWith(test_entity):
                k += 1
                if skill_tree.search(test_entity):
                    break

                test_entity += tokens[i + k]

        elif tag.startswith("I"):  # Inside of entity
            pass

        else:  # Outside of any entity
            if entity:
                test_entity = entity + [tokens[i]]
                k = 0
                while skill_tree.startsWith(test_entity):
                    ner_tags[i + k] = "I-Skill"
                    k += 1
                    test_entity += [tokens[k]]

                # check if add this in tree
                # yes-> add I - > entity.append()
                # no -> if not in tree, add to tree, skill list -> end

                # check in tree
                # if all in tree -> mark B
                # if partial in tree -> while next in tree, end with find all or find_partial not true
            entity = []

    return tokens, ner_tags, credict, skill_tree

def output_processing(tokens, ner_tags, skill_tree):
    token_list = []
    i = 0
    while i < len(tokens):
        if len(token_list) == 0:
            start = i
        cur = i
        current_token = tokens[i]
        while cur + 1< len(tokens) and tokens[cur+1].startswith("##"):
            cur += 1
            current_token += tokens[cur].lstrip(
                "##"
            )
            i += 1
        token_list.append(current_token)
        if skill_tree.startsWith(token_list):
            if skill_tree.search(token_list):
                # 回朔標籤
                for tag_index in range(start, i+1):
                    if tag_index == start:
                        ner_tags[tag_index] = 'B-Skill'
                    else:
                        ner_tags[tag_index] = 'I-Skill'
        else:
            if len(token_list) > 1:
                i = i - len(token_list) + 1
            token_list = []

        i += 1

    return tokens, ner_tags

def chunk_and_overlap(sentence, chunk_len, overlap_len):
    tokens = re.split(r'[.,;!?。，；！？]', sentence)
    chunks = []
    if len(tokens) > chunk_len:
        chunks = [tokens[i:i+chunk_len] for i in range(0, len(tokens), chunk_len-overlap_len)]
        chunks = [' '.join(chunk) for chunk in chunks]
    else:
        chunks.append(sentence)
    return chunks

def prediction_pipeline(
    sentences, tokenizer, model, credit_system, skill_search_system, not_skill_list=[]
):
    chunk_len = 5 # Define the maximum length of each chunk (in words)
    overlap_len = 1  # Define the length of overlap between chunks (in words)

    chunks = chunk_and_overlap(sentences, chunk_len, overlap_len)
    entities_list = []
    not_same = False
    for chunk in chunks:
        tokenizer.truncation_side = "left"
        tokens = tokenizer([chunk], truncation=True, is_split_into_words=True)
        torch.tensor(tokens["input_ids"]).unsqueeze(0).size()
        # basic prediction
        predictions = model.forward(
            input_ids=torch.tensor(tokens["input_ids"]).unsqueeze(0),
            attention_mask=torch.tensor(tokens["attention_mask"]).unsqueeze(0),
        )
        predictions = torch.argmax(predictions.logits.squeeze(), axis=1)
        predictions = [label_list[i] for i in predictions]
        predictions_ = predictions.copy()
        words = tokenizer.batch_decode(tokens["input_ids"])
        processing_words = words.copy()

        # output processing and add to dict
        predictions_, processing_words = cont_words_clean(predictions_, processing_words, skill_search_system)
        words_, predictions_ = output_processing(words, predictions_, skill_search_system)
        words_, predictions_, credit_system, skill_search_system = entities_preprocess(
            words, predictions_, credit_system, skill_search_system
        )
        if predictions != predictions_:
            not_same = True
        entities_list = entities_list + get_entities(processing_words, predictions_, not_skill_list)


    # output smoothing
    return list(set(entities_list))
