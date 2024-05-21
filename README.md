# Job Description Extraction

台灣有非常多的職缺招募平台，但裡面提供的job description會依據放職缺的每家公司有所不同，放的資訊、排版也都會有所差異。除此之外，在不同平台下，同公司也可能會有不同的職缺說明。
根據分析數個職缺網站發現，job description中通常會包含幾個部分
* 工作內容
* 公司介紹
* 需求人數
* 技能要求
* 福利

這種編排方式或許可使一般讀者可在同個版面上獲取所有資訊，但對於分析資料來講，卻造成髒資料的問題，難以被分析。舉例來說，今天若想分析相似職缺所需要的技能有哪些，或如果想從A職缺轉職到B職缺需要哪些技能，以現有的job description難以去做分析。因此我們才需在此項目中將job description中的職缺抽取出來，以便後續的相關建模及分析。

-- --
## 專案架構
本專案一共使用兩種方法來處理此問題。
1. classification (Job Classification)
2. Name entities recognization (Job Skill NER Task)
3. Topic Model

### Classification (Job Classification)

#### Input/Output
Input （某個職缺的JD):
```
 ◆ 與業務和設計團隊一起實施新功能並重新開發現有組件，以改善用戶體驗 ◆ 與UI/UX設計師溝通，並依照 Figma 實作頁面
◆ 當客戶或內部測試團隊提出問題時，解決新的和現有的錯誤積壓
◆ 不斷優化現有代碼庫，以獲得最佳性能 ◆ 通過與後端團隊合作，實施建議
◆ 使用 React.js 開發面向用戶的新功能 ◆ 構建可重用的組件和前端庫，以備將來使用
◆ 優化組件，以在大量支持網絡的設備和瀏覽器中，實現最佳性能 【必備條件】◆ 1 年至 3 年的前端開發經驗 ◆ 需熟悉 ES5/ES6、React.JS 前端框架，建構可複用的 React 組件，和設計及後端團隊協同分析並優化現有 Web 專案 ◆ 熟悉 React 的狀態管理流程並對數據管理方式有慨念 (例如 redux, context)
◆ 處理跨瀏覽器及跨裝置(RWD)的網頁|||◆ 熟悉使用版本控制工具, 擁有 Git 版本控制實務經驗 ◆ 熟悉 Restful API
◆ 基礎英文讀寫能力【加分項】◇ 曾使用 UI 框架 (Bootstrap, Ant Design, Tailwindcss, Material..etc) 完成需求、具備 UI 設計切版能力 ◇ 熟悉 Webpack 或其他模組打包工具 ◇
使用 TypeScript 撰寫代碼 ◇
有使用過 Javascript 測試框架的經驗，例如 Jest, React Testing Library, Cypress 等 ◇ 有使用過 Docker 與 CI/CD工具 ◇ 有開發 APP 經驗 ◇ 有前端效能優化經驗 ◇ 曾參與 HTML5 視播直播開發經驗 ◇ 有 Canvas/Pixi.js 的使用經驗 ◇ 瞭解網頁安全機制如 CORS, XSS, CSRF 等等💲具競爭力的薪資 ⏰彈性上下班時段 (9點-10點之間上班) 😴
優於勞基法的特休假 🎂生日假 🛌 不補班 週休二日、勞保健保、勞退提撥金、陪產假、產假、優於勞基法天數特休假、女性生理假、就業保險年終獎金/分紅、三節獎金/禮品、零食櫃咖啡吧、伙食津貼
```
Output (每句話所屬的類別，以下為減少篇幅，舉部分輸出結果)
```json
{
  "技能要求": [
    "◆ 基礎英文讀寫能力【加分項】",
    "◇ 曾使用 UI 框架 (Bootstrap, Ant Design, Tailwindcss, Material..etc) 完成需求、具備 UI 設計切版能力",
    "◇ 熟悉 Webpack或其他模組打包工具"
  ],
  "工作內容": [
    "◆ 與業務和設計團隊一起實施新功能並重新開發現有組件，以改善用戶體驗",
    "◆ 與UI/UX設計師溝通，並依照 Figma 實作頁面"
  ]
}

```

#### 實現方法

### Name entities recognization (Job Skill NER Task)







