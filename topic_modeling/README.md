# topic seed schema
* index: 紀錄比數及編號，實際上無使用 `(INT)`
* topic: topic key, 其他資料表利用此key和這張表的資料做串接 `(INT)`
* skill: 該topic中包含的技能(比較重要的幾個) `(LIST)`
* score: 對應前面的skill, 每個skill的重要度(數字越大越重要)，list的值和skill對應 `LIST`
* job_title: 依據前面的skill, 產生機率最到的五個職位，用","做分隔 `(string)`
