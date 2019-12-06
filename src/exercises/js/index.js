const KEYPHRASES_LIST_ID = "keyphrases_list"
const ARTICLES_LIST_ID = "articles_list"

const ACTIVE_KP_CLASS = "list-group-item d-flex justify-content-between align-items-center active"
const KP_CLASS = "list-group-item d-flex justify-content-between align-items-center"

const WORD_CLOUD_ID = "word_cloud"
const WORD_CLOUD_WIDTH = 600
const WORD_CLOUD_HEIGHT = 600
const WORD_MAX_SIZE = 60
const WORD_MIN_SIZE = 20
const WORD_PADDING = 10

var articlesJson
var selectedKps = []


function setHeightOverflow(element, windowHeight) {
    element.style.height = `${windowHeight - 100}px`
    element.style.overflow = "auto"
}

function onloadHandler(articles, trends) {
    let windowHeight = window.innerHeight
    let keyphrasesList = document.getElementById(KEYPHRASES_LIST_ID)
    let articlesList = document.getElementById(ARTICLES_LIST_ID)

    setHeightOverflow(keyphrasesList, windowHeight)
    setHeightOverflow(articlesList, windowHeight)

    articlesJson = JSON.parse(articles)
    articlesJson.forEach(article => { addArticleElement(article) })

    createWordCloud(trends)
}

function selectedKP(element) {
    classStr = element.className
    kp = element.getAttribute("name")

    if (classStr.includes("active")) {
        element.className = KP_CLASS
        selectedKps.splice(selectedKps.indexOf(kp), 1)
    }

    else {
        element.className = ACTIVE_KP_CLASS
        selectedKps.push(kp)
    }

    updateArticlesList()
}

function resetArticlesList() {
    let articleListContainer = document.getElementById(ARTICLES_LIST_ID)
    while (articleListContainer.firstChild)
        articleListContainer.removeChild(articleListContainer.childNodes[0])
}

function addArticleElement(article) {
    let title = article.title
    let url = article.link
    let kps = article.kps
    let container = document.getElementById(ARTICLES_LIST_ID)

    let aElement = document.createElement("a")
    let dlElement = document.createElement("dl")
    let dtElement = document.createElement("dt")
    let ddElement = document.createElement("dd")
    let kpsText = document.createTextNode(kps)
    let aText = document.createTextNode(title)

    aElement.href = url;
    container.appendChild(dlElement)
    dlElement.appendChild(dtElement)
    dlElement.appendChild(ddElement)
    dtElement.appendChild(aElement)
    aElement.appendChild(aText)
    ddElement.appendChild(kpsText)
}

function updateArticlesList() {
    resetArticlesList()

    let newArticles = JSON.parse(JSON.stringify(articlesJson))
    selectedKps.forEach(selection => {
        newArticles = newArticles.filter(a => a.kps.includes(selection))
    })

    newArticles.forEach(article => { addArticleElement(article) })
}

function wordCloudLayout(words){
    let layout = d3.layout.cloud()
        .size([WORD_CLOUD_WIDTH, WORD_CLOUD_HEIGHT])
        .words(words)
        .padding(WORD_PADDING)
        .rotate(() => { return ~~(Math.random() * 2) * 90 })  // random rotation
        .fontSize((w) => { return w.size })      
        .on('end', drawWords)
    layout.start()

    function drawWords(words){
        d3.select("#" + WORD_CLOUD_ID)
            .attr('width', WORD_CLOUD_WIDTH)
            .attr('height', WORD_CLOUD_HEIGHT)
            .append("g")
            .attr("transform", "translate(" + layout.size()[0] / 2 + "," + layout.size()[1] / 2 + ")")
            .selectAll("text")
            .data(words)
            .enter().append("text")
            .style("font-size", (w) => { return w.size })
            .style("fill", "black")
            .attr("text-anchor", "middle")
            .attr("transform", (w) => {
                return "translate(" + [w.x, w.y] + ")rotate(" + w.rotate + ")"
            })
            .text((w) => { return w.text })
    }
}

function createWordCloud(trends) {
    // parse json object
    let jsonTrends = JSON.parse(trends)
    let prestiges = Object.values(jsonTrends)
    let keys = Object.keys(jsonTrends)

    // calculate word sizes
    let minPrestige = prestiges.reduce((prev, current) => {return Math.min(prev, current)})
    let maxPrestige = prestiges.reduce((prev, current) => {return Math.max(prev, current)})
    let diffPrest = maxPrestige - minPrestige
    let diffSize = WORD_MAX_SIZE - WORD_MIN_SIZE

    let wordSizes = []
    keys.forEach(key => {  
        let prestige = jsonTrends[key]
        let diffCurrentSize = diffSize * (maxPrestige - prestige) / diffPrest
        wordSizes.push({text: key, size: WORD_MAX_SIZE - diffCurrentSize})  
    })

    wordCloudLayout(wordSizes)
}
