import numpy as np
import pandas as pd
from func import (
    convert_labels_to_int,
    dict_to_dataset,
    dict_to_token_dataset,
    process_list_of_lists,
    process_list_of_lists_neighbor,
    process_list_of_lists_neighbor_trans,
    process_list_of_lists_neighbor_trans_create_people,
)
from punctuators.models import SBDModelONNX
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Instantiate this model
# This will download the ONNX and SPE models. To clean up, delete this model from your HF cache directory.

# import 斷句模型
m = SBDModelONNX.from_pretrained("sbd_multi_lang")

# 讀取由openai_job_description_extraction 產出的資料
job_parsed = pd.read_csv("./openai_job_description_extraction/data/job_parsed.csv")
job_parsed["job_requirement"] = job_parsed["job_requirement"].astype("str").fillna(
    ""
) + job_parsed["skill_nice_to_have"].astype("str").fillna("")

# 分類的欄位
classify_column = [
    "company_description",
    "people_need",
    "job_duty",
    "job_requirement",
    "welfare",
]
classification = dict()
for col in classify_column:
    result = m.infer(list(job_parsed[col].dropna().astype("str").unique()))
    classification[col] = process_list_of_lists(result, col)

# tokerizer載入
tokenizer = AutoTokenizer.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment"
)
dataset, example = dict_to_dataset(classification)
tokenized_dataset, examples = dict_to_token_dataset(classification)

# Create a label map
unique_labels = sorted(set(example["label"] for example in examples))
label_map = {label: i for i, label in enumerate(unique_labels)}

integer_labels_dataset = map(convert_labels_to_int, tokenized_dataset, label_map)

# new model with 5 label
# baseline model
model = AutoModelForSequenceClassification.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment", num_labels=5
)
training_args = TrainingArguments("test_trainer", use_mps_device=True)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=integer_labels_dataset["train"],
    eval_dataset=integer_labels_dataset["test"],
)
trainer.train()

eval_pred = trainer.predict(test_dataset=integer_labels_dataset["test"])

logits, labels = eval_pred[:2]
predictions = np.argmax(logits, axis=-1)
report = classification_report(y_true=labels, y_pred=predictions)
print("Base line model result")
print(report)

# Experiment 1: 加入上下文進行預測
classify_column = [
    "company_description",
    "people_need",
    "job_duty",
    "job_requirement",
    "welfare",
]
classification = dict()
result = dict()
for col in classify_column:
    result[col] = m.infer(list(job_parsed[col].astype("str")))
for col in classify_column:
    classification[col] = process_list_of_lists_neighbor(
        result, col, classify_column + ["none"]
    )

tokenized_dataset_, examples = dict_to_token_dataset(classification)
integer_labels_dataset_ = tokenized_dataset_.map(convert_labels_to_int)

# 重新載入模型
model = AutoModelForSequenceClassification.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment", num_labels=5
)
training_args = TrainingArguments("test_trainer", use_mps_device=True)

trainer_ = Trainer(
    model=model,
    args=training_args,
    train_dataset=integer_labels_dataset_["train"],
    eval_dataset=integer_labels_dataset_["test"],
)

trainer_.train()
eval_pred = trainer_.predict(test_dataset=integer_labels_dataset_["test"])

logits, labels = eval_pred[:2]
predictions = np.argmax(logits, axis=-1)
report = classification_report(y_true=labels, y_pred=predictions)
print("Experiment 1: Add previous and next context!!!")
print(report)

# Experiment 2: Add different language into training set to balance data
classify_column = [
    "company_description",
    "people_need",
    "job_duty",
    "job_requirement",
    "welfare",
]
classification = dict()
result = dict()
for col in classify_column:
    result[col] = m.infer(list(job_parsed[col].astype("str")))
for col in classify_column:
    classification[col] = process_list_of_lists_neighbor_trans(
        result, col, classify_column + ["none"]
    )

tokenized_dataset_tran, _ = dict_to_token_dataset(classification)
integer_labels_dataset_tran = tokenized_dataset_tran.map(convert_labels_to_int)

# reset model
model = AutoModelForSequenceClassification.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment", num_labels=5
)
training_args = TrainingArguments("test_trainer", use_mps_device=True)

trainer_ = Trainer(
    model=model,
    args=training_args,
    train_dataset=integer_labels_dataset_tran["train"],
    eval_dataset=integer_labels_dataset_tran["test"],
)

trainer_.train()

eval_pred = trainer_.predict(test_dataset=integer_labels_dataset_tran["test"])

logits, labels = eval_pred[:2]
predictions = np.argmax(logits, axis=-1)
report = classification_report(y_true=labels, y_pred=predictions)

print("Experiment 2: Add translate into training data to balance each categories!!!")
print(report)

# Experiment 3: 增加“需求人數”類別
classify_column = [
    "company_description",
    "people_need",
    "job_duty",
    "job_requirement",
    "welfare",
]
classification = dict()
result = dict()
for col in classify_column:
    result[col] = m.infer(list(job_parsed[col].astype("str")))
for col in classify_column:
    classification[col] = process_list_of_lists_neighbor_trans_create_people(
        result, col, classify_column + ["none"]
    )

tokenized_dataset_people, _ = dict_to_token_dataset(classification)
integer_labels_dataset_people = tokenized_dataset_people.map(convert_labels_to_int)

# reset model
model = AutoModelForSequenceClassification.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment", num_labels=5
)
training_args = TrainingArguments("test_trainer", use_mps_device=True)

trainer_ = Trainer(
    model=model,
    args=training_args,
    train_dataset=integer_labels_dataset_people["train"],
    eval_dataset=integer_labels_dataset_people["test"],
)

trainer_.train()

eval_pred = trainer_.predict(test_dataset=integer_labels_dataset_people["test"])

logits, labels = eval_pred[:2]
predictions = np.argmax(logits, axis=-1)
report = classification_report(y_true=labels, y_pred=predictions)
print("Experiment 3: Add extra case to number of required people !!! ")
print(report)
