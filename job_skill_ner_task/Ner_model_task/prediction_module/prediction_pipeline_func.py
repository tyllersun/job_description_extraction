import torch
from job_skill_ner_task.Ner_model_task.label_list import label_list


def cont_words_clean(predictions, processing_words):
    # have bug, ## 前有兩個會出問題
    for i in range(len(processing_words)-1, 0, -1):  # Iterate from the last element to the first
        if processing_words[i].startswith('##'):
            processing_words[i-1] += processing_words[i].lstrip('##')  # Combine with the previous word
            processing_words[i] = ''  # Empty the current word
            if predictions[i] == 'B-Skill':
                predictions[i] = 'I-Skill'
                predictions[i-1] = 'B-Skill'
    # Record the modified word and corresponding prediction
    return predictions, processing_words

def get_entities(tokens, ner_tags):
    entities = []
    entity = []

    for i, (token, tag) in enumerate(zip(tokens, ner_tags)):
        if tag.startswith('B'):  # Beginning of entity
            if entity:  # If there is an existing entity, append it to the list
                entities.append(''.join(entity) if all('\u4e00' <= char <= '\u9fff' for char in ''.join(entity)) else ' '.join(entity))
            entity = [token]  # Start a new entity
        elif tag.startswith('I'):  # Inside of entity
            if entity:  # If there is an existing entity, add token to it
                entity.append(token)
            else:  # Else, start a new entity (this handles the case where a tag sequence starts with I)
                entity = [token]
        else:  # Outside of any entity
            if entity:  # If there is an existing entity, append it to the list
                entities.append(''.join(entity) if all('\u4e00' <= char <= '\u9fff' for char in ''.join(entity)) else ' '.join(entity))
            entity = []

    # Append the last entity if it exists
    if entity:
        entities.append(''.join(entity) if all('\u4e00' <= char <= '\u9fff' for char in ''.join(entity)) else ' '.join(entity))

    return entities


def entities_preprocess(tokens, ner_tags, credict, skill_tree):
    entities = []
    entity = []

    for i, (token, tag) in enumerate(zip(tokens, ner_tags)):
        if tag.startswith('B'):  # Beginning of entity
            if entity:  # If there is an existing entity, append it to the list
                entities.append(''.join(entity) if all('\u4e00' <= char <= '\u9fff' for char in ''.join(entity)) else ' '.join(entity))
            entity = [token]  # Start a new entity

            # add tree data
            test_entity = entity

            k = 1
            while i + k <= len(tokens) and ner_tags[i + k].startswith("I"):
              if tokens[i+k] != "":
                test_entity += [tokens[i+k]]
              k+=1
            if not skill_tree.search(test_entity):
              credict.add_dict(test_entity)

            while skill_tree.startsWith(test_entity):
              k += 1
              if skill_tree.search(test_entity):
                break

              test_entity += tokens[i+k]

        elif tag.startswith('I'):  # Inside of entity
            pass

        else:  # Outside of any entity
            if entity:
                test_entity = entity + [tokens[i]]
                k = 0
                while skill_tree.startsWith(test_entity):
                    ner_tags[i+k] = "I-Skill"
                    k+=1
                    test_entity += [tokens[k]]

                # check if add this in tree
                # yes-> add I - > entity.append()
                # no -> if not in tree, add to tree, skill list -> end

                # check in tree
                  # if all in tree -> mark B
                  # if partial in tree -> while next in tree, end with find all or find_partial not true
            entity = []


    return tokens, ner_tags, credict, skill_tree

def prediction_pipeline(sentences, tokenizer, model, credit_system, skill_search_system):
    tokenizer.truncation_side='right'
    tokens = tokenizer([sentences], truncation=True, is_split_into_words=True)
    torch.tensor(tokens['input_ids']).unsqueeze(0).size()

    # basic prediction
    predictions = model.forward(input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0), attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0))
    predictions = torch.argmax(predictions.logits.squeeze(), axis=1)
    predictions = [label_list[i] for i in predictions]
    predictions_ = predictions.copy()
    words = tokenizer.batch_decode(tokens['input_ids'])
    processing_words = words.copy()

    # output processing and add to dict
    predictions_, processing_words = cont_words_clean(predictions_, processing_words)
    words_, predictions_, credit_system, skill_search_system = entities_preprocess(words, predictions_, credit_system, skill_search_system)
    not_same = False
    if predictions != predictions_:
      not_same = True
    # output smoothing
    return words, predictions_, credit_system, skill_search_system, get_entities(processing_words, predictions_), not_same