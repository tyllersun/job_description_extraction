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
from Ner_model_task.google_sheet_data.func import (
    get_need_and_not_list,
    add_data_to_sheet,
)



tokenizer = AutoTokenizer.from_pretrained("./un-ner.model/")
model = AutoModelForTokenClassification.from_pretrained(
    "./un-ner.model/", num_labels=len(label_list)
)

# load skill provide by public
extra_skill_list, not_skill_list = get_need_and_not_list()

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
台大創創中心致力建構完整的新創與創新生態圈。創立於2013年，提供台灣新創加速器、孵化器輔導計畫，以六大輔導資源激發團隊成長：商業策略營運、企業合作專案、創業社群交流、新創人才媒合、關鍵資源介接、公關行銷曝光等，陪伴不同階段的團隊落實創意或創新構想、驗證為可行產品或服務方案；同時也提供中大型企業外部創新顧問服務，媒合並推進潛力新創與企業合作，以強化企業內部的新事業發展動能。
公司理念

我們的工作跟累積信任有關，以及為了維持這份信任，所需執行的各種大小任務。了解團隊、了解團隊的產業與產品、發展專長、準確介紹合適的諮詢業師、合作夥伴與創創校友、設計適合創辦人有效吸收的實戰課程、設計讓創業家們自然互動交流的場域、以及規劃展演，看著團隊苦練之後站上 Demo 舞台，在他們氣力放盡的時候拍拍肩膀跟他乾一瓶啤酒。
你說為什麼我們做這些事情？因為我們相信，幫助有潛力的創業團隊能夠對整體新創生態產生正向的影響。 如果你對於上述場景或理念感到興奮，你可能十分適合加入「協助新創團隊」的新創中介組織，我們在找你。
公司福利

〖在新創場域，我們希望你能擁有...〗
• 不受限的發揮空間，自主管理，並且能夠隨時表達自己的想法
• 融洽的工作氣氛，常與同事、新創團隊、導師們交流學習
• Free Coffee & Snacks，不定期員工聚餐
• 不定期Happy Hour、不定期創業團隊 / 創業人分享經驗
• 創創中心舉辦活動 / 講座，自由參加，累積創業知識、學習各項技能
〖正職員工〗
• 年終獎金1.5個月
• 週休二日/特休/年假
• 員工進修補助

"""

predict_entities = prediction_pipeline(sentences, tokenizer, model, credit_system, skill_search_system, not_skill_list)
add_data_to_sheet(sentences, predict_entities)
print(predict_entities)
