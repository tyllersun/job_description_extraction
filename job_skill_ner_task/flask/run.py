import sys

from flask import Flask, jsonify, request
from transformers import AutoModelForTokenClassification, AutoTokenizer

from job_skill_ner_task.Ner_model_task.credit_system_module.data_structure_class import (
    credit_dict,
)
from job_skill_ner_task.Ner_model_task.label_list import label_list
from job_skill_ner_task.Ner_model_task.prediction_module.prediction_pipeline_func import (
    prediction_pipeline,
)
from job_skill_ner_task.Ner_model_task.skill_search_module.data_structure import Trie
from job_skill_ner_task.Ner_model_task.skill_search_module.load_process_func import (
    read_skill_file,
    split_input_string,
)

sys.path.insert(0, "./")


tokenizer = AutoTokenizer.from_pretrained("./un-ner.model/")
model = AutoModelForTokenClassification.from_pretrained(
    "./un-ner.model/", num_labels=len(label_list)
)

# load skill search system
skills_list_of_list = read_skill_file(
    "job_skill_ner_task/Openai_ner_task/skill_set/skill_dict.txt"
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


@app.route("/predict", methods=["POST"])
def postInput(credit_system=credit_system, skill_search_system=skill_search_system):
    # 取得前端傳過來的數值
    print("begin post")
    if request.is_json:
        print("is_json")
    else:
        print("not_json")
    insert_values = request.get_json()
    print(insert_values)
    sentences = insert_values["sentences"]
    print("in post")

    (
        tokenized_words,
        format_prediction,
        credit_system,
        skill_search_system,
        predict_entities,
        not_same,
    ) = prediction_pipeline(
        sentences, tokenizer, model, credit_system, skill_search_system
    )
    print("end_post")

    return jsonify({"predict_entities": predict_entities})


if __name__ == "__main__":
    # do something here
    app.run()
