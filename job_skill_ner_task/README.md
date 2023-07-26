# 技能提取系統 Skill Extraction System 
-- --
## 解決問題
在眾多的職缺分析等項目當中，常常會需要抓取職缺對應到的技能點。但實際上，大多數的技能參雜在`job description`當中，需要手動標記取出。此外，`job description`中除了有中文的敘述外，亦有英文，這讓分辨起來更加困難。因此，此`模型API`主要可以快速的將職缺中的`技能`擷取出來，當遇到未標示成功的技能，未來也將自我訓練。除此之外，此模型目前可用於中文及日文當中。

## 技術簡介
利用以`multi-language-bert`為基底的深度學習模型作為預訓練，並利用`OpenAI`的API進行`工作技能`的標記作為訓練資料(100多筆資料）進行微調來創建基底模型。除此之外，`OpenAI`標記資料的同時創立`技能查找系統`，並在`prediction`階段藉由`技能查找系統`加強標記效果。同時將`基底模型`預測出的`技能`且未在`技能查找系統`中出現的技能加入`candidate system`
當中，再藉由`candidate system`經過篩選加入`技能查找系統`。而經`技能查找系統`標記有誤的預測結果會再加入模型做做微調。最後時限`rule based`和`probability based`的模型整合及`semi-superivised learning`的效果。

## 使用方法
### Docker 指令
```
docker pull tyllersun/machine_learning_app
docker run -p 9874:9874 tyllersun/machine_learning_app
```
### Input
API網址為 `./predict`
```json
{
  "sentences": "【工作內容】 - 以 AI/ML 專業處理與解決 KKday 業務問題 - 執行機器學習專案，設計、建置將模型投入正式環境所需的架構 - 建立資料科學與機器學習所需基礎建設，落實 MLOps 精神與技術 - Survey AI/ML 相關新技術，以及 PoC 的建立與評估 【技術與經驗】 - 至少 3 年的機器學習相關開發工作經驗 - 熟悉資料科學、機器學習與深度學習的專業知識，能理解演算法的原理並選擇、建立合適的模型 - 熟悉 Python 語言，並有機器學習系統的開發經驗 - 具有整合第三方 AI 服務 API 的經驗 - 具有開發 Data or Model Serving API 服務的經驗 - 熟悉 SQL 與 Bigquery / PostgreSQL 的運用、操作 - 熟悉 GNU / Linux 系統與環境建置 - 具有 Git 相關版控工具實務經驗，與熟悉 Code Review 流程與要點 【態度與特質】 - 良好的團隊溝通能力，你將與 KKday 各種資料領域角色密切的合作 - 清晰的邏輯、獨立思考、解決問題的能力 - 對技術有熱忱，樂於增進自我、學習以及分享"
}
```
### Output
```json
{
    "predict_entities": [
        "Python",
        "Data",
        "Model Serving  API",
        "SQL",
        "Bigquery  ",
        "PostgreSQL  ",
        "GNU / Linux",
        "Git ",
        "Code Review"
    ]
}
```

