import sys
sys.path.insert(0, "./")

from job_skill_ner_task.Ner_model_task.label_list import label_list
from job_skill_ner_task.Ner_model_task.prediction_module.prediction_pipeline_func import prediction_pipeline
from job_skill_ner_task.Ner_model_task.skill_search_module.load_process_func import read_skill_file, split_input_string
from job_skill_ner_task.Ner_model_task.skill_search_module.data_structure import Trie
from job_skill_ner_task.Ner_model_task.credit_system_module.data_structure_class import credit_dict
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification


tokenizer = AutoTokenizer.from_pretrained('./un-ner.model/')
model = AutoModelForTokenClassification.from_pretrained('./un-ner.model/', num_labels=len(label_list))


# load skill search system
skills_list_of_list = read_skill_file("job_skill_ner_task/Openai_ner_task/skill_set/skill_dict.txt")
skill_search_system = Trie()

for outer_list in skills_list_of_list:
    for inner_list in outer_list:
        skill_search_system.insert(split_input_string(inner_list))


# load credit system
credit_system = credit_dict(3)

sentences = """
-負責前端技術研發工作
-參與、並協助定義前端研發工作流和開發規範
-建立前端框架核心功能元件，並能獨力解決對開發中遇到的技術問題
-可與後端工程師討論協作開發專案
-設計編寫API供後端工程師介接
-與後端工程師合作介接 RESTFul API
-熟悉 npx scss webpack. 懂得使用 material ui, core ui, element ui 尤佳.
-技術文件撰寫


【職務條件】
-具有 Git 版本控制相關知識
-可獨立或與團隊合作，工作態度積極、負責，能配合公司規定

【程式語言條件】
1. 熟悉 HTML / CSS / JavaScript
2. 熟悉前端框架 Vue (必要) /Vuex , Angular , Node.js
3. 會使用figma / sketch / zeplin 等工具尤佳
4. 有ui ux / 大型資料處理 / Google Map Api 經驗佳

"""

tokenized_words, format_prediction, credit_system, skill_search_system, predict_entities, not_same = \
    prediction_pipeline(
            sentences,
            tokenizer,
            model,
            credit_system,
            skill_search_system
    )

print(predict_entities)