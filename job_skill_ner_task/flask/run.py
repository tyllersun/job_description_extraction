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




tokenizer = AutoTokenizer.from_pretrained("./un-ner.model/")
model = AutoModelForTokenClassification.from_pretrained(
    "./un-ner.model/", num_labels=len(label_list)
)

# load skill search system
skills_list_of_list = read_skill_file(
    "Openai_ner_task/skill_set/skill_dict.txt"
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
    insert_values = request.get_json()
    sentences = insert_values["sentences"]

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

    return jsonify({"predict_entities": predict_entities})


if __name__ == "__main__":
    # do something here
    app.run(host="0.0.0.0", port=9874)
