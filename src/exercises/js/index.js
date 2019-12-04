const KEYPHRASES_LIST_ID = "keyphrases_list"
const ARTICLES_LIST_ID = "articles_list"

const ACTIVE_KP_CLASS = "list-group-item d-flex justify-content-between align-items-center active"
const KP_CLASS = "list-group-item d-flex justify-content-between align-items-center"

var articlesJson 
var selectedKps = []


function setHeightOverflow(element, windowHeight) {
    element.style.height = `${windowHeight}px`
    element.style.overflow = "auto"
}

function onloadHandler(articles) {
    let windowHeight = window.innerHeight
    let keyphrasesList = document.getElementById(KEYPHRASES_LIST_ID)
    let articlesList = document.getElementById(ARTICLES_LIST_ID)
    
    setHeightOverflow(keyphrasesList, windowHeight)
    setHeightOverflow(articlesList, windowHeight)
    
    articlesJson = JSON.parse(articles)
    articlesJson.forEach(article => {addArticleElement(article)})
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

function resetArticlesList(){
    let articleListContainer = document.getElementById(ARTICLES_LIST_ID)
    while(articleListContainer.firstChild) 
        articleListContainer.removeChild(articleListContainer.childNodes[0])
}

function addArticleElement(article){
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

function updateArticlesList(){
    resetArticlesList()

    let newArticles = JSON.parse(JSON.stringify(articlesJson))
    selectedKps.forEach(selection => {
        newArticles = newArticles.filter(a => a.kps.includes(selection))
    })

    newArticles.forEach(article => {addArticleElement(article)})
}