
# NER (Name Entity Recognization) 工作技能抽取模型

如果電腦能用1小時達到90%的準確率，人類何必用10小時達到95%的準確率。 何不花1小時讓90% -> 99% ?

-- --


## Content

### 序

### 資料集

* 資料集來源

* OpenAI NER 半標記、 技能庫創立

* 中文與英文同時存在的NER資料前處理

### 模型

* Embedding與模型的選定

* 輸入的前處理

* 模型調參( Hyperparameter tuning )與MLflow

* 技能庫系統輸出的後處理

* 模型擴充與二次訓練

### 額外功能

* 技能庫擴充

* 輸出技能排除

* 資料的回傳

### 使用說明

* Docker開啟API 服務

* API 功能

-- --

## 序



這篇文章主要是記錄製作工作技能抽取模型的過程、中間使用的工具及遇到問題和對應的處理方式，並不是一篇手把手講解程式碼的文章。這邊選擇不去細講程式碼主要原因有二

 1. 如果真的要理解程式碼可以自行把程式碼貼到GPT上，他講解的絕對比我好

 2. 程式碼十分複雜，尤其資料前處理和後處理上有過多細節要處理，程式碼因此也很繁雜，說明起來會讓這篇文章失去重點

同理，為方便閱讀，深度學習的框架如bert等知識點這篇文章也會全部跳過，僅展示輸入的資料樣態及輸出的結果。

為讓這篇文章的編排更加符合傳統在進行深度學習專案的架構，文章架構及編排如以下圖表展示，但由於模型僅提供docker檔和線上的google sheet供大家使用，因此並不會展示模型的上線監測與模型迭代的部分。或許未來模型正式上線使用，會額外補充這部分

![整體流程 (icons credit by flaticon)](https://cdn-images-1.medium.com/max/5100/1*DKXxjhYyD_GP602ky-b0lw.png)

-- --

## 資料集

### 資料集來源

完整的訓練資料集主要有三個來源，分別為Meetjob, 104, Cake Resume，其中104 和Cake Resume是由Xchange夥伴爬蟲取得，而Meetjob資料集則是由我爬取。這三個資料集的內容皆是職缺及職缺相關的敘述，裡面包函職缺名稱、薪資、工作說明…。但在這專案當中，唯一會用到的是工作說明的欄位。

***資料筆數***

* Meetjob: 300多筆不重複資料，多為科技相關職缺

* 104：10萬多筆資料，但重複的比重非常高（去重後可能不到2萬），從餐飲、醫美到行銷通通都有

* CakeResume: 300多筆不重複資料，也多半為科技業相關職缺



### OpenAI NER 半標記、 技能庫創立

由於進行實體辨識的模型訓練會需要有標記的資料，目前也沒有找到self-supervised learning的方法。此外，有標記的技能相關中文資料在市面上較難取得，且標記成本較高，因此決定使用OpenAI GPT3.5 之API進行並結合Langchain來進行標記。

然而若要讓GPT3.5產生和NER的輸入標記資料有相同格式遇到以下挑戰

 1. 在短句且技能相關須標記技能較少時表現良好，但如果句子較長或需標記技能較多時，效果有顯著的下降

 2. 由於NER的輸出較為複雜，通常會是整串文字，有發現到會有機率有漏字、甚至創造新內容的情況發生。除此之外，產生的整串文字所需的價格較高。

為解決以上問題，這邊放棄傳統的NER標記方式，改要求GPT回傳裡面提到的技能相關詞彙，之後再利用找substring的技術將對應的位置做標記。

*因此選用以下Prompt去把需要的資料進行List回傳*
```python
# 輸出格式要求
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
  # question 為要進行NER抽取的文章
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
除此之外，回傳的list_of_skill中的每個skill也會被儲存到技能庫當中，以方便後續的分析使用。



### 中文與英文同時存在的NER資料前處理

在網路上有找到非常多關於中文或英文NER處理的相關資料，但並沒有找到同時存在中文、英文甚至日文等其他國語言的相關處理方式。但很不幸的，這在本次的資料庫中經常出現，甚至有中英混合的情況發生，因此在進行標記時，有對中文及英文進行不同的處理（這種方法在不同語言間也能被處理）。

由於中文字只需幾個字便能有意思，因此中文的部分是以字為單位進行切分。而英文一個字母通常不具有意義，且需要經由很多個字母組成才具有意義，雖然理論上按字母做切分也要能有預測力，但為了方便訓練，英文是以單字作為最小區分單位。

![標記形式：沒標記的為”0"](https://cdn-images-1.medium.com/max/3484/1*ukt7lCLWnPPkTDgLrhT1FQ.png)

-- --

## 模型

### Embedding與模型的選定

由於語言模型要從頭開始訓練需要非常龐大的計算資源與資料集，且市面上已經有大量開源的語言預訓練模型可以使用，且效果比自己訓練好不少， 因此這邊是直接調用Hugging Face上面別人訓練好的模型和embedding。

這邊選用Hugging Face上的[Babelscape/wikineural-multilingual-ner](https://huggingface.co/Babelscape/wikineural-multilingual-ner)，選取的原因主要是他本身具有多語言辨識功能，且本身就是做NER Task的模型，訓練起來較為方便。



### 輸入的前處理

由於Embedding有文字的長度限制，但實際上很多職缺描述都非常的長，因此經常會有truncate的現象發生，為處理此問題，該模型會將輸入的語句切成不同的chunks，每個chunk皆是由數句話所組成，而為讓上下文有連貫，chunk與chunk之間會有部分的語句是重疊的。
```python
# 利用標點符號將句子切出，並分配到chunk中
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
   



### 模型調參( Hyperparameter tuning )與MLflow（Github 無此部分程式碼）

在訓練過程中，有去做調整的參數有以下幾個，超參數的方法主要是先以grid search先找到較小的參數空間，最後再以Baysian Search做最後的參數微調。

參數: learning rate, optimizer (及對應的可調控參數）, weight decay

batch size主要是影響訓練速度，因此儘可能地調高(後來選16)

最後，為方便記錄每次調參的結果，因此用Dagshub上的Mlflow來做紀錄

![Dagshub 上MLflow長相](https://cdn-images-1.medium.com/max/5924/1*-_XbeK4d0mg1nRMgADh-6A.png)



### 技能庫系統與輸出後處理

除了利用深度學習進行預測外，這邊也額外加入的傳統的rule-base的辨識機制。這項機制會將輸出的結果進行比對，如果在Input的文章中有技能是深度學習模型沒有標記的，則rule-base的辨識系統便會協助進行標記。此外，有些技能雖被模型判定為技能，但實際上並不是，在rule-base的系統當中也會自動地將這些技能在輸出時進行移除。

***那rule-based的辨識系統又是從何而來？分為增加和刪減來說明***

**增加的技能**

 1. 由OpenAI在標記時所提供

 2. 由填寫[google sheet](https://forms.gle/wJXBmRZwm1tmb7so7)會自動被加入系統當中 （永久加入）

 3. 利用API的功能進行加入（暫時加入）

**刪減的技能**

 1. 由填寫[google sheet](https://forms.gle/wJXBmRZwm1tmb7so7) 進行刪減 （永久刪減）

 2. 利用API的功能進行刪減

***如何快速比對是否有技能是在Rule-base system當中?***

為更加快速的比對是否存在技能是在rule-base system當中，在rule-base system 中是採用prefix Tree (Trie) 的數據結構來進行存處。在傳統的Trie當中，英文是用character來進行存處，但由於skill是英文中文混雜，因此這邊將中文的每個“字”作為單位，英文以每個單字為單位進行存放。此外，為處理大小寫無法找到的問題，在Trie中統一以小寫進行儲存。
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


### 模型擴充與二次訓練

在首輪訓練當中，為節省GPT3.5的使用量，因此只有使用部分的資料集來進行訓練(Meetjob及1000多筆104資料）。然而有後處理的幫助下，將所有資料用首輪訓練的模型進行預測，再利用後處理產生最終結果，這些結果便可以在重新訓練模型，達到更好的效果。

未來收到更多Google sheet的回饋後也預期再次進行下版本的迭代，歡迎大加填寫google sheet讓模型變得更完善，也可立即改善預測結果。另外，在使用過程中也會收集使用者的輸入與回傳值，以便進行後續的模型優化。

[Google表單連結](https://forms.gle/4tueefSY3CR7A9bx7)

-- --

## 額外功能

### 技能庫擴充

在技能庫的擴充上，不論是使用API 的Add_skill還是用填google sheet的方式，皆是直接將技能加入Trie中判斷是否在文章中有出現。兩者唯一的差別 在於Add_skill程序結束時便會消失，不會留下紀錄，而如果使用填表單的方式，則會記錄在Google sheet中，未來的使用者都能享有到擴充帶來的好處。

相對的，如果有人擴充不相關的資料，也會被放入技能庫當中，目前還未對此種情況做處理。



### 輸出技能排除

在技能的排除處理上，並不會直接去改變Trie中的直，反倒是在最後輸出資料時會在進行最後一次的檢查，確保輸出的技能沒有在排除表中出現。另外，調用API和填Google sheet的差別和技能庫擴充一樣。



### 資料的回傳

為了使模型能夠收集使用者的使用回饋，進行後續的模型優化與迭代，因此有加入了資料回傳功能， 模型會自動收集用戶的使用資料，利用google app script，回傳至雲端的google sheet表單。會收集的資料僅有使用者的傳入值與輸出結果，且該資料僅會作為模型訓練與研究使用，並不會用於任何商業用途。
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

## 使用說明

### Docker開啟API 服務

目前NER 提取模型已經在[Docker Hub ](https://hub.docker.com/repository/docker/tyllersun/ner_skill_extraction_app/general)公開開放下載，裏面涵蓋整體的運行環境及模型，方便下載到自己的本地進行運行或部署到雲端環境使用。除從Docker Hub下載外，比較推薦利用以下指令在本地端運行。
```bash
    # Docker Pull
    docker pull tyllersun/ner_skill_extraction_app
    # Docker Run
    docker run -p 9874:9874 tyllersun/ner_skill_extraction_app
```

PS: 由於Docker 涵蓋模型及其參數，故整體容量較大(2.8G)，會需要花較久的時間下載

### 



### API 功能

*工作技能擷取*
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

技能庫擴充
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
技能排除
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
