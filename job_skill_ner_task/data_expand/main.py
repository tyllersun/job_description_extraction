import pandas as pd

import sqlite3
import sys

from transformers import AutoModelForTokenClassification, AutoTokenizer
sys.path.insert(0, "./job_skill_ner_task/")
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
from Openai_ner_task.ner_data_formating import get_ner_pair
from Openai_ner_task.langchain_preprocessing_func import append_dict_to_csv

# Connect to the SQLite database
# This will open a connection to the database file (or create it if it doesn't exist)
conn = sqlite3.connect('job_classification/openai_job_description_extraction/data/cakeresume.db')

# Create a cursor object
cur = conn.cursor()

# Execute a SQL query
cur.execute("SELECT requirements FROM cakeresume")

# Fetch all the rows
rows = cur.fetchall()
CR_requirement_list = []
for row in rows:
    CR_requirement_list.append(row)

# Close the connection
conn.close()
data_104 = pd.read_csv("job_classification/openai_job_description_extraction/data/104.csv")
jd_list_104 = list(data_104['工作內容'].unique())

meet_job_list = list(pd.read_csv("job_classification/openai_job_description_extraction/data/meet_job_2023-06-29.csv")['jd'])
whole_list = CR_requirement_list + jd_list_104 + meet_job_list


tokenizer = AutoTokenizer.from_pretrained("./job_skill_ner_task/un-ner.model/")
model = AutoModelForTokenClassification.from_pretrained(
    "./job_skill_ner_task/un-ner.model", num_labels=len(label_list)
)

not_skill_list = ['溝通', '活動', '執行', '設計','科技', 'to', '分析', 'open', '基本', "服務"]
extra_skill_list = ['Power BI',
                    'Data Driven',
                    '統計公式',
                    "電腦視覺",
                    "開源演算法", "肢體動作編輯", "機器學習", "open source", '演算法','專案管理']
# load skill search system
skills_list_of_list = read_skill_file(
    "./job_skill_ner_task/Openai_ner_task/skill_set/skill_dict.txt",
    extra_skill=extra_skill_list
)
skill_search_system = Trie()

for outer_list in skills_list_of_list:
    for inner_list in outer_list:
        skill_search_system.insert(split_input_string(inner_list))

# load credit system
credit_system = credit_dict(3)
whole_list = list(set(whole_list))
for i in range(len(whole_list)):
    sentences = whole_list[i]
    if i % 500 == 0:
        print(f"in {i}, total {len(whole_list)}")
    try:
        predict_entities = prediction_pipeline(sentences, tokenizer, model, credit_system, skill_search_system, not_skill_list)
        ner_pair = get_ner_pair(sentences, predict_entities)
        append_dict_to_csv(ner_pair, "./job_skill_ner_task/data_expand/data/ner_task.csv")
    except:
        pass