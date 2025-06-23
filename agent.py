from langgraph.graph import StateGraph, START, END
from typing import List, TypedDict, Annotated,Sequence,Dict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
import aiohttp
from langchain_core.documents import Document
import asyncio
import chromadb
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_community.chat_models import ChatFireworks
import os 
import re
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from markdownify import markdownify as md
import joblib


load_dotenv()

API = os.getenv('SERPAPI_API_KEY')
CRAWL=os.getenv("CRAWL")
classifier = joblib.load("weather_file_classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")

persist_directory=r'C:\Users\bunny\OneDrive\Documents\Programming\Python\langgraph\chromadb'



class AgentState(TypedDict):
    main_query:HumanMessage
    messages: Annotated[List[BaseMessage],add_messages]
    systemprompt:str
    queries:Dict[str,str]
    isdoc_loaded:bool
    vectorstore:bool
    file_query:str
    path:str


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)





llm2 = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

@tool
def basic_search(query: str) -> list:
    """Perform a basic web search using a lightweight search API or service.

    Args:
        query (str): The search term or user query for which relevant results are needed.

    Returns:
        list: A list of dictionaries containing basic search result metadata, such as:
            - title (str): The title of the web page.
            - link (str): The URL of the web page.
            - snippet (str): A short description or excerpt from the page content.

    Example:
        >>> results = basic_search("What is LangChain?")
        """
    params = {
        "q": query,
        "api_key": API,
        "engine": "google"
    }

    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code != 200:
        return f"Error: {response.status_code}"

    data = response.json()
    results = data.get("organic_results", [])

    if not results:
        return "No results found."

    formatted = []
    for result in results:
        title = result.get("title", "No title")
        link = result.get("link", "")
        snippet = result.get("snippet", "")
        formatted.append(f" {title}\n{snippet}\nURL: {link}\n\n")


    return formatted

def storing(content:str):
    try:
            doc=Document(page_content=content)
            vectorstore=Chroma.from_documents(
            documents=[doc],
            embedding_function=embeddings,
            persist_directory=persist_directory,
            collection_name="web_cache"
            )
            print(f"written into web_cache .")
    except Exception as e:
            print(f"failed to write due to {e}")

@tool
def deep_search(query:str)->str:
    """
Perform a deep web content extraction for a given query using advanced scraping or parsing methods.

Args:
    query (str): The search term or user query for which detailed content is required.

Returns:
    list: A list of dictionaries containing rich, full-text content and metadata from the most relevant sources, such as:
        - title (str): The title of the web page.
        - link (str): The URL of the web page.
        - content (str): The extracted main content or article body from the page.
        - source (str, optional): The source domain or provider of the page (if available).

Example:
    >>> results = deep_search("How does LangGraph work internally?")
"""
    print("started performing deepsearch")

    response=checknode(query)

    if response=="false" or "no relevant information" in response:
        print("response was false")
        scraped_content=deep(query)
        storing(scraped_content)
        return scraped_content
    
    else:
        print("response was true relevant info found ")
        return response

    

tools=[deep_search,basic_search]

llm1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).bind_tools(tools)



def classifieragent(state:AgentState)->AgentState:
    if state['isdoc_loaded'] is None:
        state['isdoc_loaded']=False
        state['vectorstore']=False
    
    parts = re.split(r'\b(?:and|also|,)\b', state['messages'][-1].content, flags=re.IGNORECASE)
    sub_queries=[p.strip() for p in parts if p.strip()]
    predictions=classifier.predict(sub_queries)
    decoded = label_encoder.inverse_transform(predictions)
    dicty={}
    for i in range(0,len(sub_queries)):
        dicty[sub_queries[i]]=decoded[i]
    state['queries']=dicty
    
    return state     

def checknode(q:str)->str:
    print("checking relevant info by rag")

    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name="web_cache")
    docs = collection.query(
    query_texts=[q],
    n_results=4
)
    response=llm1.invoke([SystemMessage(content="""
You are a helpful AI tasked with determining whether an answer is relevant to a given question.

Carefully read both the question and the answer. Consider whether the answer:
- Directly addresses the question,
- Provides useful or supporting information,
- Stays on topic.

Reply with only one of the following:
- "Yes" if the answer is clearly relevant to the question.
- "No" if the answer is unrelated, vague, or off-topic.

Be strict — say "Yes" only if the answer is truly useful or directly related.
""")]
+[HumanMessage(content=f"question: {q}\n\nanswer:\n{docs}")])
    
    if response.content=="No":
        return "I found no relevant information in the document."
    else:
        return docs   



def routing(state:AgentState)->str:

    last=state['messages']
    lastcontent=last[-1].content
    
    fallback_phrases = ["i don't know", "i am not sure", "unable to answer", "sorry"]
    print("Routing decided:")

    if hasattr(last[-1],"tool_calls") and last[-1].tool_calls:
        print("toolcall")    
        return 'tool_call'
    
    if "file" in lastcontent.lower() or "doc" in lastcontent.lower() or "document" in lastcontent.lower():
        return 'rag'
    
    if isinstance(last[-1], AIMessage) and any(phrase in lastcontent for phrase in fallback_phrases):
        print("fallback")
        return 'fallback'
    
    else:
        print("END")
        return 'general'
    



def rag(state:AgentState)-> AgentState:
    
    query=state['file_query']
    """
    Document was loaded already and use this tool to answer queries related to the document.
    use this tool for answering any query related to document or file or text file it implements retrieval augmented generation 
    and outputs the similar results
    """
    if state['vectorstore']==False:
        state['path'] = input("Enter file path: ")
        loader = TextLoader(state['path'])
        

        try:
            pages=loader.load()
            print(f"file has been loaded and has {len(pages)} pages.")
        except Exception as e:
            print(f"error while loading the file:{e}.")

        text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400
        )   

        pages_split=text_splitter.split_documents(pages)
        


        try:
            vectorstore=Chroma.from_documents(
            documents=pages_split,
            embedding_function=embeddings,
            persist_directory=persist_directory,
            collection_name="temporary_cache"
            )
            print(f"created ChromaDB vector store.")
        except Exception as e:
            print(f"vector store not created due to {e}")

        state['vectorstore']=True
        state['messages'].append(SystemMessage(content="your document has been successfully loaded you can start asking queries."))
        return state

    vectorstore = Chroma(
    collection_name="temporary_cache",
    persist_directory=persist_directory,
    embedding_function=embeddings
)

    
    retriever=vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":5})




    docs=retriever.invoke(query)
    if not docs:
        state['messages'].append(AIMessage(content="I found no relevant information in the provided document."))
        return state

    results=[]
    for i,doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown page")
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content.strip()
        results.append(f"[{i+1}] (Source: {source}, Page: {page})\n{content}")

    state['messages'].append(HumanMessage(content="\n\n".join(results)))
    return state


def agent(state:AgentState)->AgentState:
    print("Agent")

    query = list(state['queries'].keys())[0]
    print(query)

    toolclass = state['queries'][query]
    syspromp_content="you are a helpful assistant."
    if toolclass == "search":
        syspromp_content = f"""You received a sub query: '{query}'.
        classification suggests this is a 'basic-search' related query.
        Based on the query and your knowledge, determine if calling the 'basic-search tool' is appropriate."""

    elif toolclass == "file":
        syspromp_content = f"""You received a sub query: '{query}'.
        classification suggests this is a 'file' related query.recheck if its not file related answer appropriately
        Note that you will be provided with the file in the nextnode"""
        state["isdoc_loaded"]=True
        state['file_query']=query

    elif toolclass == "deepsearch":
        syspromp_content = f"""You received a sub query: '{query}'.
        classification suggests this is a 'deepsearch' related query and if user needs in depth intutions use it.
        Based on the query and your knowledge, determine if calling the 'deep-search tool' is appropriate."""
    else: 
        syspromp_content = f"""You received a sub query: '{query}'.
        classification suggests this is a 'basic' query, meaning no specific tool is needed.
        Based on the query and your knowledge,
        generate a tool call if required or if you can answer directly without a tool, provide a concise direct answer.
        """

    response=llm1.invoke([SystemMessage(content=syspromp_content)]+[HumanMessage(content=query)])
    state['messages'].append(response)
    del state['queries'][query]
    return state

def deep(query: str) -> str:
    """Search and answer recent news,weather ,articles etc.. that are unable to answered by the model"""
    params = {
        "q": query,
        "api_key": API,
        "engine": "google"
    }

    search_res = requests.get("https://serpapi.com/search", params=params)
    search_data = search_res.json()

    if 'organic_results' not in search_data:
        return "No search results found or error in API request."
    
    urls = [r["link"] for r in search_data["organic_results"]]
  
    
    print(urls)
    content=asyncio.run(main(urls[:2]))
    return "\n\n".join(content)


async def fetch(session,url,params=None):

    try:
        async with session.get(url,params=params) as response:
            print("extracting html")
            html=await response.text()
            text=md(html)
            print("returning markdown content of html\n")
            return text
    except Exception as e:
        print(f"Error while fetching {url}: {e}")
        return None

content=[]


async def main(urls:list):
    async with aiohttp.ClientSession() as session:
         
        tasks = []
        for url in urls:
             crawl_baseparams={
    "token":CRAWL,
     "url":url,
     "render":"true"
}
             tasks.append(fetch(session,"https://api.crawlbase.com/",crawl_baseparams))
        textlist= await asyncio.gather(*tasks)
        if textlist==[]:
            return "nothing fetched for the query"
        return textlist


def summarizer(state:AgentState)->AgentState:
    query=state['messages'][0].content
    summarized_version=llm1.invoke([SystemMessage(content=f"summarize the answer based on this main query and provide sources,links from where the information is taken if present and format well.:{query}")]+state['messages']).content
    state["messages"].append(AIMessage(content=summarized_version))
    return state


def loop(state:AgentState)->str:
    print(f"sub_queries remaining:{len(state['queries'])}")
    return state

def loopcheck(state:AgentState)->str:

    if len(state['queries'])==0:
        return "finish"
    else:
        return "continue"
    

graph=StateGraph(AgentState)
graph.add_node('classifier',classifieragent)
graph.add_node('Agent',agent)
graph.add_node('summarizer',summarizer)
toolnode=ToolNode(tools)
graph.add_node('ragnode',rag)
graph.add_node('check',checknode)
graph.add_node('tools',toolnode)
graph.add_node('loop',loop)
fallback_node = ToolNode([basic_search])
graph.add_node("basic-search", fallback_node)

#edges
graph.add_edge(START,'classifier')
graph.add_edge('classifier','Agent')
graph.add_conditional_edges(
    'Agent',
    routing,
    {
       'tool_call':'tools',
       'general':'loop',
       'fallback':'basic-search',
       'rag':'ragnode'
    }
)

graph.add_edge('ragnode','summarizer')

graph.add_edge('tools','summarizer')
graph.add_edge('summarizer','loop')
graph.add_conditional_edges(
    'loop',
    loopcheck,
    {
        "finish":END,
        "continue":'Agent'
    })
graph.add_edge('basic-search','loop')
model=graph.compile()


from IPython.display import Image
from pathlib import Path

# Get the image binary from the Mermaid graph
image_bytes = model.get_graph().draw_mermaid_png()

# Define the output file path
output_path = Path("agent.png")
# Write to local file
with open(output_path, "wb") as f:
    f.write(image_bytes)

print(f"Graph saved as {output_path.absolute()}")


# Initialize the state only once
state = {
    "messages": [],
    "queries": dict(),
    "systemprompt": "",
    "isdoc_loaded": None,
    "vectorstore": None,
    "file_query": "",
    "path": ""
}

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    state["messages"].append(HumanMessage(content=user_input))
    state = model.invoke(state)
    print("🤖:", state['messages'][-1].content)
