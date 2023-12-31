# Job Description Classification

-- --

## 欲解決問題
在一般職缺刊登網站中，裡面的職缺敘述不單單只包含一樣訊息，裡面可能還隱藏著公司介紹、工作內容、技能要求、福利等額外訊息。為方便去分析一個職缺的訊息，會傾向將職缺敘述進行分割，把每段話歸類到所屬的資訊類別去。此分類器主要就是要將各句話分類到所屬的資訊類別。

## 資料問題
* 資料是利用Scrapy到各大求職網站爬取下來，雖在Repro中只提供Meetjob的部分資料，但實際情況中也會有其他網站資料，因此模型除了需要再提供的資料分布中有良好的表現外，在其他網站資料中也能generalization。
* 由於資料是由Scrapy抓取下來，並不像一般在Kaggle上的資料已經有標籤可以去做訓練。此外，由於是依據每句話貼標籤，較為繁瑣。
* 由於爬取的是台灣的求職網站，因此在上面的語言多半以中文為主，但卻摻雜少量其他語言，因此模型亦須擴展到多語言皆可使用。這項要求除增加前處理的複雜程度，亦增加Tokenize的難度


## 模型流程、架構、結果
<img width="9136" alt="Flow of JD classification model" src="https://github.com/tyllersun/job_description_extraction/assets/27050741/90fc82ab-b9ae-402f-b14d-c35e2579081a">

### 訓練模型：
1. 載入由Scrapy 爬取的meetjob職缺網資料
2. 利用Langchain和Openai GPT3.5來進行文章中每段話的分類。
3. 由於Langchain是產生一整段屬於對應分類的話，長度過長，因此在會先做"斷句"，而斷句的標籤即為該文章對應的類別
4. 把斷句結果進行embedding並套入Bert預訓練分類模型中進行訓練得到Baseline Model

### 特徵工程：
雖然embedding本身即是一種特徵工程方法，但由於放入訓練的資料是經過斷句的結果，即代表前後句也有可能隱含某些特徵，但因為斷句而被truncate掉。除此之外，由於使用的是Multilanguage Bert，本身即具有在其他語言上推廣和理解的能力，因此嘗試將輸入翻譯為多種語言對模型的訓練會也幫助，也不會造成重複資料的問題，也可處理資料分布不均的問題。最後，由於其中一種預測類別“需求人數”的資料格式較為固定，因此可產生虛擬資料來幫助模型有效的學習此種Pattern。

由以上歸納以下三種實驗
1. 處理上下文遺漏：除將本身的字句加入訓練外，會額外題中上下文，中間加入"[Block]"來分割，若屬於首句或尾句，則隨機加入或不加入其他類別的尾句或首句
2. 1 + 增加各國語言翻譯：在較少的類別中，而外增加英文的資料進入訓練，以平衡資料
3. 1 + 2 + 增加"需求人數"類別的模擬資料

### 模型結果
見上方圖片


## 潛在問題與改良
1. 雖預測效果較高，但由於是將所有句子進行分類並整合，一句話分錯便會造成語意不順，因此有規劃利用Smoothing的技術去處理
2. 斷句會導致部分句子的可讀性降低，未來可改用傳統方式如標點符號進行切分
