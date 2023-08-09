
# NER (Name Entity Recognition) Skill Extraction Model

"If a computer can achieve 90% accuracy in 1 hour, why should humans spend 10 hours to achieve 95% accuracy? Why not spend 1 hour to push 90% to 99%?"

-- --
[GitHub link](https://github.com/tyllersun/job_description_extraction/tree/main/job_skill_ner_task)
[Chinese Version](https://medium.com/@tylletsun/ner-name-entity-recognization-工作技能抽取模型-dcdfd1395cab)
## Content

### Introduction

### Dataset

* Source of the Dataset

* OpenAI NER Semi-labeling & Skillset Creation

* Preprocessing of NER data containing both Chinese and English 

### Model

* Selection of Embedding and Model 

* Preprocessing for Input

* Model Hyperparameter Tuning & MLflow

* Post-processing of Skillset System Output

* Model Expansion and Secondary Training

### Additional Features

* Expansion of Skillset Library 

* Exclusion of Output Skills 

* Data Feedback

### User Guide

* Launching API Service via Docker

* API Features

-- --

## Introduction

This article mainly documents the process of creating a job skill extraction model, the tools used, challenges encountered, and their corresponding solutions. It isn't a step-by-step code explanation article. There are two main reasons for choosing not to delve deeply into the code:

 1. If one truly wishes to understand the code, they can paste it into GPT for a clearer explanation, which will undoubtedly be better than mine.

 2. The code is quite complex, especially in the areas of data preprocessing and post-processing, which involve many intricate details. Discussing these would detract from the main focus of this article.

Similarly, for readability, deep learning frameworks like BERT and other technical details will be omitted from this article. Only the data input format and output results will be showcased.

To align this article with the traditional structure of deep learning projects, the arrangement and layout follow the flowchart below. However, since the model is only available as a Docker file and online Google Sheet for public use, this article won't cover aspects like online model monitoring or model iterations. Perhaps once the model officially goes live, that segment will be expanded upon.

![Overall Process (icons credit by flaticon)](https://cdn-images-1.medium.com/max/5100/1*DKXxjhYyD_GP602ky-b0lw.png)

-- --

## Dataset

### Source of the Dataset

The complete training dataset mainly comes from three sources: Meetjob, 104, and Cake Resume (Taiwan Job Posting Website). The data from 104 and Cake Resume was crawled by our partners in Xchange, while the Meetjob dataset was scraped by myself. All these datasets contain job listings and their related descriptions, including job title, salary, and job details. For this project, only the job description field is used.

***Number of Data Entries***

* Meetjob: Over 300 unique entries, mostly related to tech jobs.

* 104: Over 100,000 entries, but there's a high duplication rate (possibly fewer than 20,000 unique entries). It spans a variety of industries from catering, cosmetic medicine to marketing.

* CakeResume: Over 300 unique entries, mostly tech-related jobs.

### OpenAI NER Semi-labeling & Skillset Creation

Training a model for entity recognition requires labeled data, and currently, there's no identified self-supervised learning method for this. Additionally, obtaining labeled skill-related Chinese data is challenging and expensive. As such, we decided to use the OpenAI GPT3.5 API combined with Langchain for labeling.

However, generating NER-style labeled data from GPT3.5 posed the following challenges:

 1. It performs well with shorter sentences with fewer skills to label. But for longer sentences or more skills to label, the performance degrades significantly.

 2. The outputs for NER are typically complex, and there's a chance of missed words or even introducing new content. Moreover, generating such outputs is costlier.

To address these challenges, we forgo the conventional NER labeling method. Instead, we request GPT to return skill-related terms mentioned in the content. We then mark their corresponding positions using substring techniques.

*The following prompt is used to obtain the required data in a list format*:
```python
# Desired output format
  response_schemas = [
      ResponseSchema(name="skill", description="list of skill"),
  ]
  output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
  
  # One shot learning, prompt
  base_prompt = """
 範例：Question: 需會使用Adobe In Design 或相關軟體 Answer: {"skill": ["Adobe In Design"]}
 指令：把文章中提到的工作技能(skill)名稱(做NER用），按照順序放入"list of skill"中
  
  """
  
  # full prompt
  # question is the text that requires NER extraction.
  format_instructions = output_parser.get_format_instructions()
  prompt = PromptTemplate(
      template="{base_prompt} \n{format_instructions} \n{question}",
      input_variables=["question"],
      partial_variables={
          "base_prompt": base_prompt,
          "format_instructions": format_instructions,
      },
  )
```
Furthermore, each skill from the 'list_of_skill' is saved in the skillset library for subsequent analysis.

### Preprocessing NER Data with Mixed Chinese and English

There's abundant information available on handling NER for either Chinese or English. However, there isn't much on dealing with texts containing multiple languages like Chinese, English, or even Japanese. Unfortunately, our dataset often contains such multilingual data, sometimes even mixed within the same sentence. Different treatments are applied to Chinese and English during labeling (this approach can be extended to other languages).

Chinese characters are meaningful in isolation, so Chinese text is segmented character by character. English, on the other hand, often requires combinations of letters to convey meaning. Theoretically, segmenting by letter could also work, but for training convenience, English is segmented word by word.

![Labeling format: Unmarked is represented as ”0"](https://cdn-images-1.medium.com/max/3484/1*ukt7lCLWnPPkTDgLrhT1FQ.png)
-- --

## Model

### Selection of Embedding and Model

Training a language model from scratch demands significant computational resources and datasets. Furthermore, there are numerous pre-trained models available that offer superior performance than training one's own. Hence, this project leverages existing models and embeddings from Hugging Face.

We've chosen the [Babelscape/wikineural-multilingual-ner](https://huggingface.co/Babelscape/wikineural-multilingual-ner) from Hugging Face. This model was selected because it inherently supports multi-language recognition. Furthermore, since it's designed for the NER Task, training is more straightforward.

### Preprocessing the Input

Given that embeddings have text length constraints, and many job descriptions are quite lengthy, truncating is often necessary. To address this, the model divides the input into multiple chunks. Each chunk comprises several sentences. To maintain context continuity, there's an overlap between adjacent chunks. 

```python
# Using punctuation marks to segment sentences and allocate them to chunks
def chunk_and_overlap(sentence, chunk_len, overlap_len):
    tokens = re.split(r'[.,;!?。，；！？]', sentence)
    chunks = []
    if len(tokens) > chunk_len:
        chunks = [tokens[i:i+chunk_len] for i in range(0, len(tokens), chunk_len-overlap_len)]
        chunks = [' '.join(chunk) for chunk in chunks]
    else:
        chunks.append(sentence)
    return chunks
```

The above `chunk_and_overlap` function serves to partition the input text into manageable chunks. Using punctuation marks as delimiters ensures that each chunk mostly contains whole sentences, improving the model's ability to understand and generate contextually relevant outputs. The overlapping mechanism helps in preserving the contextual information that may be lost if each chunk were completely distinct.
   



### Hyperparameter Tuning and MLflow (Not available on Github)

During the training process, several parameters were adjusted. The approach for hyperparameter tuning started with grid search to narrow down the parameter space. After that, Bayesian Search was applied for further fine-tuning.

Parameters: 
- Learning rate
- Optimizer (and its adjustable parameters)
- Weight decay

The batch size mainly affects the training speed. Therefore, it was adjusted as high as possible, and 16 was eventually chosen.

To conveniently keep track of the results of each tuning, Mlflow on Dagshub was used.

![Appearance of MLflow on Dagshub](https://cdn-images-1.medium.com/max/5924/1*-_XbeK4d0mg1nRMgADh-6A.png)

### Skill Repository System and Post-processing

In addition to predictions made using deep learning, a traditional rule-based recognition mechanism was also integrated. This mechanism compares the output results. If there are skills in the input text that the deep learning model did not mark, the rule-based recognition system would assist in marking them. Furthermore, some skills might be identified by the model as skills, but they aren't in reality. Such falsely detected skills are automatically removed by the rule-based system.

***How was the rule-based recognition system established? It can be described based on additions and deletions:***

**Adding skills:**

1. Provided by OpenAI during tagging.
  
2. Skills automatically added to the system from a [google sheet](https://forms.gle/wJXBmRZwm1tmb7so7) (permanently added).
  
3. Added using an API function (temporarily added).

**Deleting skills:**

1. Skills removed via the [google sheet](https://forms.gle/wJXBmRZwm1tmb7so7) (permanently removed).

2. Removed using an API function.

***How to quickly verify if a skill exists in the Rule-based system?***

For speedy verification of skills in the rule-based system, the data structure used is a prefix Tree (Trie). In a traditional Trie, English is stored character by character. However, since the skills can contain both English and Chinese, Chinese characters are stored individually, while English is stored word by word. Moreover, to tackle issues with case sensitivity, everything in the Trie is stored in lowercase.
```python
# Trie Data structure
class TrieNode:
    def __init__(self):
        self.children = {}
        self.hasword = False  # only True if exist word,
        # apple in dict, but app not in, False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, list_of_word: list) -> None:
        cur = self.root
        for c in list_of_word:
            c = c.lower()
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.hasword = True

    def search(self, list_of_word: list) -> bool:
        cur = self.root
        for c in list_of_word:
            c = c.lower()
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return cur.hasword

    def startsWith(self, prefix_list: list) -> bool:
        cur = self.root
        for c in prefix_list:
            c = c.lower()
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return True
```


### Model Expansion and Retraining

In the initial round of training, only a portion of the dataset (Meetjob and over 1000 pieces of 104 data) was used to save on GPT3.5 usage. However, with the aid of post-processing, predictions were made on all the data using the model trained in the first round. After generating the final results using post-processing, these results can then be used for retraining the model, aiming for better performance.

With more feedback received through the Google sheet, future iterations of the model are anticipated. Users are encouraged to contribute to the Google sheet, making the model even more refined and immediately improving its prediction capabilities. Additionally, during usage, users' inputs and returned values will be collected to further optimize the model in subsequent iterations.

[Google Form Link](https://forms.gle/4tueefSY3CR7A9bx7)

---

Feedback mechanisms, such as the Google sheet, are critical in fine-tuning machine learning models, especially when applied in real-world scenarios. Such feedback loops allow for the model to continually improve, adapting to new data and evolving requirements. Given the mentioned strategy, this project can expect iterative refinement, leading to a model that is increasingly reliable and accurate over time.
-- --

## Additional Features

### Skill Repository Expansion

For the expansion of the skill repository, whether using the `Add_skill` API or filling out the Google sheet, skills are directly added to the Trie for checking their appearance in the article. The only difference between the two methods is that the `Add_skill` procedure will disappear upon completion without leaving a record. If the Google sheet method is used, it will be recorded in the Google sheet, benefiting all future users.

Conversely, if someone expands irrelevant data, it will also be included in the skill repository. Currently, no measures have been taken to address such situations.

### Skill Exclusion in Outputs

For skill exclusion, the Trie's data will not be directly modified. Instead, during the final data output, one last check will be made to ensure that the skills outputted do not appear in the exclusion list. The distinction between using the API and filling out the Google sheet is the same as with the skill repository expansion.

### Data Feedback

To allow the model to collect user feedback and subsequently optimize and iterate the model, a data feedback function has been added. The model will automatically gather user data and use the Google App Script to send it back to a Google sheet in the cloud. Only the user's input and output results will be collected. This data will solely be used for model training and research and will not be used for any commercial purposes.

---

Ensuring the privacy and security of user data is vital. It's good that there's a clear disclaimer that data will only be used for research and not for commercial purposes. It might also be useful to inform users beforehand about data collection and obtain their consent, to ensure compliance with data protection regulations.
```javascript
// google app script function
function doPost(e) {
  var data = JSON.parse(e.postData.contents);
  
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  sheet.appendRow([data.job_description, data.result]);
  
  return ContentService.createTextOutput(
    JSON.stringify({ 'status': 'success' })
  ).setMimeType(ContentService.MimeType.JSON);
}
```
-- --

## User Guide

### Launching the API Service with Docker

The NER extraction model is currently available for download on [Docker Hub](https://hub.docker.com/repository/docker/tyllersun/ner_skill_extraction_app/general). It encompasses the entire runtime environment and the model, making it convenient to download and run locally or deploy in a cloud environment. In addition to downloading from Docker Hub, it's recommended to use the following commands to run it locally:

```bash
    # Pull the Docker Image
    docker pull tyllersun/ner_skill_extraction_app
    
    # Run the Docker Container
    docker run -p 9874:9874 tyllersun/ner_skill_extraction_app
```

**Note:** Due to the Docker image containing the model and its parameters, it has a relatively large size (2.8GB). Therefore, downloading may take some time.

---

After initiating the service using Docker, you can access the API through your local host on port `9874`. Ensure that you have sufficient storage space before starting the download. If you face any issues during the setup, it's recommended to check Docker's logs or documentation for troubleshooting.

### 



### API Features

*Job Skill Extraction*
>  path: /predict, method: POST
  ```python
    # Input：欲進行預測的文章、段落...
    {
        "sentences": "CloudMile是一支年輕的創業團隊，與帶領市場的供應商一起，利用一流的人工智能、雲端和安全技術，協助客戶解決最關鍵的問題。我們也是亞洲領先的企業級人工智能，雲和安全解決方案提供商，並且為第一個獲得北亞基礎設施和機器學習專業知識的Google首要合作夥伴。 作為“ CloudMiler”，我們堅信敏捷、當責以及精進的價值觀，在這裡，我們和一群優秀的人才一起工作，也和公司一起快速成長。 工作內容： 1. 與機器學習工程師、雲端架構師、PM密切合作, 提供客戶高質量的雲端數據相關服務。 2. 負責協助客戶整合、設計、部署大數據處理流程與平臺。 必備技能/條件： - 3 年以上公有雲資料排程、處理、清洗、分析的實務經驗。 - 2 年以上資料處理常用的 Open Source 工具的實務經驗，eg. Spark, Airflow, hadoop - 2 年以上Python / Java or Golang軟體開發相關工作經驗 - 擅長解決問題、除錯、故障排除，提出問題解決方案。 加分條件 - 具備BigQuery、RDBMS、NoSQL使用經驗 - 具備爬蟲相關經驗 - 熟悉 Restful API - 具備Git 版控使用經驗 - 具備基本英文溝通能力"
    }

    # Output：技能清單
    {
        "predict_entities": [
            "解決方案",
            "API",
            "Restful",
            "英文",
            "雲端數據相關",
            "雲端架構",
            "Python",
            "Google",
            "機器學習",
            "爬蟲相關",
            "溝通能力",
            "CloudMiler",
            "Open Source",
            "大數據處理流程",
            "BigQuery",
            "Spark",
            "解決問題",
            "Airflow",
            "NoSQL",
            "CloudMile",
            "Git  版控",
            "爬蟲",
            "Golang",
            "RDBMS",
            "服務",
            "Java"
        ]
    }
```

Skill Repository Expansion
>  path: /add_skill, method: POST
```python
    # Input：欲加入的清單
    {
        "add_skill_list": ["人工智能", "深度學習"]
    }

    # Output
    {
        "Status": "success"
    }
```
Skill Exclusion
```python
    # Input：要排除的清單
    {
        "remove_skill_list": ["問題", "eq"]
    }

    # Output: 目前被排除的清單
    {
        "not_skill_list": [
            "溝通",
            "活動",
            "執行",
            "設計",
            "科技",
            "to",
            "分析",
            "open",
            "問題",
            "eq"
        ]
    }

```
