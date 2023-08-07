import sys

from transformers import AutoModelForTokenClassification, AutoTokenizer
sys.path.insert(0, "./")
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

not_skill_list = ['溝通', '活動', '執行', '設計','科技', 'to', '分析', 'open', '基本', "服務", "基礎", '時間', '製作', '管理', '其他']
extra_skill_list = ['Power BI',
                    'Data Driven',
                    '統計公式',
                    "電腦視覺",
                    "開源演算法",
                    "肢體動作編輯",
                    "機器學習",
                    "open source", '演算法','專案管理', 'GA4','Big Data', 'Python']
# load skill search system
skills_list_of_list = read_skill_file(
    "Openai_ner_task/skill_set/skill_dict.txt",
    extra_skill=extra_skill_list
)
skill_search_system = Trie()

for outer_list in skills_list_of_list:
    for inner_list in outer_list:
        skill_search_system.insert(split_input_string(inner_list))

# load credit system
credit_system = credit_dict(3)

sentences = """
[About this job]
1.	Identify the business and technical demands of clients and implement the tools, methodologies and architectures to the various first-party data tracking requirements
2.	Explore & develop the advanced tracking data model prototype to improve first-party data product line by GA4, Firebase, GTM and BigQuery
3.	Develop, operate, and ETL for data collection, cleansing, processing, storage, and analytics on Web/App user behavioral raw data
4.	Develop data visualization dashboard to explore data insights and will participate in client meetings (if needed) and project executions to support the both business development and project delivery.
5.	Write documents for architecture design and implementation

[About you]
1.	At least 2 years of experience in processing data with Python and SQL-like script, such as BigQuery
2.	Experience in web analytics/tracking knowledges GA, GTM and Firebase
3.	Experience in data visualization tools Tableau, Data studio
4.	Expertise in hands-on ETL job design, components and modules development of data process
5.	Effective problem-solving, analytical, and writing well-organized reports
6.	Passionate at working in digital marketing industry and fast-moving startup environment
7.	Fluent in English or Japanese (both oral and written)

[More...]
1.	Knowledge of web technologies HTML, CSS, JavaScript and SEM/SEO is a big plus
2.	Strong cloud computing experience, such as GCP, AWS is a big plus
3.	Experience in building and operating large scale distributed systems or applications is a plus
4.	Expertise in Linux/Unix environments and familiar with shell scripting is a plus
"""

predict_entities = prediction_pipeline(sentences, tokenizer, model, credit_system, skill_search_system, not_skill_list)

print(predict_entities)
