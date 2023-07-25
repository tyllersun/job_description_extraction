import ast
import os

import numpy as np
import pandas as pd
from datasets import Dataset, load_metric
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from job_skill_ner_task.Ner_model_task.label_list import label_list


def tokenize_and_align_labels(examples):
    label_all_tokens = True
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        list(examples["tokens"]), truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == "0":
                label_ids.append(0)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(
                    label_encoding_dict[label[word_idx]] if label_all_tokens else -100
                )
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def get_all_tokens_and_ner_tags(directory):
    print(directory)
    print(os.listdir(directory))
    return (
        pd.concat(
            [
                get_tokens_and_ner_tags(os.path.join(directory, filename))
                for filename in os.listdir(directory)
            ]
        )
        .reset_index()
        .drop("index", axis=1)
    )


def get_tokens_and_ner_tags(filename):
    # Convert strings to lists
    data = pd.read_csv(filename)
    data["tokens"] = data["tokens"].apply(ast.literal_eval)
    data["ner_tags"] = data["ner_tags"].apply(ast.literal_eval)
    return data


def get_un_token_dataset(directory, training_percentage):
    # Get all tokens and NER tags
    df = get_all_tokens_and_ner_tags(directory)

    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=42)
    # Split the DataFrame into training and test DataFrames
    train_df, test_df = train_test_split(
        df, train_size=training_percentage, random_state=42
    )

    # Create Huggingface Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    return train_dataset, test_dataset


train_dataset, test_dataset = get_un_token_dataset(
    "job_skill_ner_task/Openai_ner_task/data", 0.7
)

label_encoding_dict = {"O": 0, "B-Skill": 1, "I-Skill": 2}
task = "ner"
model_checkpoint = "Babelscape/wikineural-multilingual-ner"
batch_size = 16

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, num_labels=len(label_list), ignore_mismatched_sizes=True
)

args = TrainingArguments(
    f"test-{task}",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.00001,
)

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load_metric("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    l = 0
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


trainer = Trainer(
    model,
    args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=test_tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()

trainer.save_model("un-ner.model")
