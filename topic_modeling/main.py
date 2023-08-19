import pandas as pd
from data_cleaning_func import remove_empty_elements, remove_common_elements
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import json

# strongly recommend using GPU to run this scripts

cake_resume_skill = list(pd.read_csv("job_classification/openai_job_description_extraction/data/cake_resume_tag.csv")["skill_tag"])
meet_job_skill = list(pd.read_csv("job_classification/openai_job_description_extraction/data/meet_job_2023-06-29_tag.csv")["skill_tag"])
bank104_job_skill = list(
        pd.read_csv("job_classification/openai_job_description_extraction/data/104_eng_tag.csv"
                    )["skill_tag"])

total_skill = meet_job_skill + bank104_job_skill + cake_resume_skill

# remove duplicate
total_skill = list(set(total_skill))

remove_list = ["[CLS]", " ", "", "主管交辦事項", "細心", "主管交辦事項"]
total_skill = [x for x in total_skill if not pd.isna(x)]
total_skill = remove_empty_elements(remove_common_elements(total_skill, remove_list))

# embedding

model = SentenceTransformer('intfloat/multilingual-e5-large')

def tokenize(text):
    words = text.split(",")
    return words

vectorizer = CountVectorizer(tokenizer=tokenize)




embeddings = model.encode(total_skill)

embedding_df = pd.DataFrame(embeddings)


# training
topic_model = BERTopic(embedding_model=model, verbose=True, vectorizer_model=vectorizer)
topics, probs = topic_model.fit_transform(total_skill, embeddings=embedding_df.values)


# prediction

cake_resume_csv = pd.read_csv("job_classification/openai_job_description_extraction/data/cake_resume_tag.csv")
cake_resume_csv["skill_tag"] = cake_resume_csv["skill_tag"].apply(lambda x: '' if pd.isna(x) else x)
prediction = topic_model.transform(cake_resume_csv["skill_tag"])
cake_resume_csv["topic_id"] = prediction[0]
cake_resume_csv["topic_confidence"] = prediction[1]
cake_resume_csv.to_csv("job_classification/openai_job_description_extraction/data/cake_resume_topic.csv")

bank104_csv = pd.read_csv("job_classification/openai_job_description_extraction/data/104_eng_tag.csv")
bank104_csv["skill_tag"] = bank104_csv["skill_tag"].apply(lambda x: '' if pd.isna(x) else x)
prediction = topic_model.transform(bank104_csv["skill_tag"])
bank104_csv["topic_id"] = prediction[0]

bank104_csv["topic_confidence"] = prediction[1]
bank104_csv.to_csv("job_classification/openai_job_description_extraction/data/bank104_csv_topic.csv")


meet_job_csv = pd.read_csv("job_classification/openai_job_description_extraction/data/meet_job_2023-06-29_tag.csv")
meet_job_csv["skill_tag"] = meet_job_csv["skill_tag"].apply(lambda x: '' if pd.isna(x) else x)
prediction = topic_model.transform(meet_job_csv["skill_tag"])
meet_job_csv["topic_id"] = prediction[0]
meet_job_csv["topic_confidence"] = prediction[1]
meet_job_csv.to_csv("job_classification/openai_job_description_extraction/data/meet_job_topic.csv")


topic_model.save("topic_modeling/topic_model_dir", serialization="safetensors", save_ctfidf=True, save_embedding_model=model)
topic_info_dict = dict()

topic_freq = topic_model.get_topic_freq()
all_topics_words = {topic_num: topic_model.get_topic(topic_num) for topic_num in topic_freq['Topic']}
for topic_num, words in all_topics_words.items():
    try:
        skills = [item[0] for item in words]
        scores = [item[1] for item in words]

        result = {
            "skill": skills,
            "score": scores
        }
        topic_info_dict[topic_num] = result
    except:
        print(topic_num)

def handle_float32(obj):
    """Handles float32 types during json serialization."""
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError(f"Object of type '{obj.__class__.__name__}' is not JSON serializable")


json_data = json.dumps(topic_info_dict, indent=4, default=handle_float32, ensure_ascii=False,)

# 將 JSON 寫入檔案
with open('topic_info.json', 'w') as json_file:
    json_file.write(json_data)




