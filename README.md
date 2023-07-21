# Job Description Extraction

台灣有非常多的職缺招募平台，但裡面提供的job description會依據放職缺的每家公司有所不同，放的資訊、排版也都會有所差異。除此之外，在不同平台下，同公司也可能會有不同的職缺說明。
根據分析數個職缺網站發現，job description中通常會包含幾個部分
* 工作內容
* 公司介紹
* 需求人數
* 技能要求

這種編排方式或許可使一般讀者可在同個版面上獲取所有資訊，但對於分析資料來講，卻造成髒資料的問題，難以被分析。舉例來說，今天若想分析相似職缺所需要的技能有哪些，或如果想從A職缺轉職到B職缺需要哪些技能，以現有的job description難以去做分析。因此我們才需在此項目中將job description中的職缺抽取出來，以便後續的相關建模及分析。

本專案一共使用兩種方法來處理此問題。
1. classification (Job Classification)
2. Name entities recognization (Job Skill NER Task)

