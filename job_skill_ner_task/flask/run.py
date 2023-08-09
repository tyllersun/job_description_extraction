import sys

sys.path.insert(0, "./")
from flask import Flask, jsonify, request
from transformers import AutoModelForTokenClassification, AutoTokenizer

from Ner_model_task.credit_system_module.data_structure_class import (
    credit_dict,
)
from Ner_model_task.label_list import label_list
from Ner_model_task.prediction_module.prediction_pipeline_func import (
    prediction_pipeline,
)
from Ner_model_task.skill_search_module.data_structure import Trie
from Ner_model_task.skill_search_module.load_process_func import (
    read_skill_file,
    split_input_string,
)

from Ner_model_task.google_sheet_data.func import (
    get_need_and_not_list,
    add_data_to_sheet,
)




tokenizer = AutoTokenizer.from_pretrained("./un-ner.model/")
model = AutoModelForTokenClassification.from_pretrained(
    "./un-ner.model/", num_labels=len(label_list)
)
global not_skill_list
global extra_skill_list
extra_skill_list, not_skill_list = get_need_and_not_list()

# load skill search system
skills_list_of_list = read_skill_file(
    "Openai_ner_task/skill_set/skill_dict.txt",
    extra_skill=extra_skill_list
)
global skill_search_system
skill_search_system = Trie()

for outer_list in skills_list_of_list:
    for inner_list in outer_list:
        skill_search_system.insert(split_input_string(inner_list))

# load credit system
global credit_system
credit_system = credit_dict(3)


app = Flask(__name__)
@app.route("/add_skill", methods=["POST"])
def add_skill(skill_search_system=skill_search_system):
    insert_values = request.get_json()

    for values in insert_values["add_skill_list"]:
        skill_search_system.insert(split_input_string(values))
    return jsonify({"Status": "success"})

@app.route("/remove_skill", methods=["POST"])
def removeSkill(not_skill_list=not_skill_list):
    insert_values = request.get_json()
    not_skill_list += insert_values["remove_skill_list"]
    return jsonify({"not_skill_list": not_skill_list})

@app.route("/predict", methods=["POST"])
def postInput(credit_system=credit_system, skill_search_system=skill_search_system):
    # 取得前端傳過來的數值
    insert_values = request.get_json()
    sentences = insert_values["sentences"]
    print(not_skill_list)
    predict_entities = prediction_pipeline(
        sentences, tokenizer, model, credit_system, skill_search_system, not_skill_list
    )
    add_data_to_sheet(sentences, predict_entities)

    return jsonify({"predict_entities": predict_entities})


if __name__ == "__main__":
    # do something here
    app.run(host="0.0.0.0", port=9874)
