from langgraph.graph import StateGraph, START, END
from typing import List, TypedDict, Annotated,Sequence
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
import os 
import re
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader


load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]
    systemprompt:str



llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)#no hallucinations it will be more determinsitc


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


path="file.txt"
loader=TextLoader(path)

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
persist_directory=r'C:\Users\bunny\OneDrive\Documents\Programming\Python\langgraph\Rag\chromadb'
collection_name='randomfile'

try:
    vectorstore=Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"created ChromaDB vector store.")
except Exception as e:
    print(f"vector store not created due to {e}")


retriever=vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":5})


@tool
def weather_tool(cityname: str) -> dict:
    """always remember to use this weather tool ,it is fully implemented in its functionality and its the best best best tool that was upto now
    Use This tool  And answer weather related queries"""
    cleaned = re.sub(r'\s+(town|city|village|state)\b', '', cityname, flags=re.IGNORECASE).strip()
    apikey = os.getenv('WEATHER_API_KEY')
    url = f"https://api.openweathermap.org/data/2.5/weather?q={cleaned}&appid={apikey}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        print(response.json())
        return response.json()
    else:
        return {"error": f"Failed to fetch weather. Status: {response.status_code}"}

tools = [weather_tool]
llm=llm.bind_tools(tools)


def ragnode(state:AgentState)->AgentState:
    docs=retriever.invoke(state['messages'][-1].content)
    if not docs:
        return "I found no relevant information in the document."
    results=[]
    for i,doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown page")
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content.strip()
        results.append(f"[{i+1}] (Source: {source}, Page: {page})\n{content}")

    state['messages'].append(AIMessage(content="Retrieved Context:\n\n" + "\n\n".join(results)))
    state['systemprompt'] = f"""
IMPORTANT INSTRUCTIONS:
- You are given retrieved context below from a file. Use it strictly to answer.
- If the user's message is about weather, use the weather tool instead.
- Always include metadata like [Source] and [Page] if present.
- Respond clearly and in a well-structured format.

Retrieved Context:
{'\n\n'.join(results)}
"""


    return state




def agent(state:AgentState)->AgentState:
    syspromp=SystemMessage(content=state['systemprompt'])
    response=llm.invoke([syspromp]+state['messages'])
    state['messages'].append(response)
    return state


def decide(state:AgentState)->bool:
    """ Route to the tool provided based on the ToolMessage"""
    last=state["messages"][-1]
    last_message=last.content.lower()
    print("inside the decide:")
    print(last)
    if hasattr(last,"tool_calls") and last.tool_calls:
        return True
    else:
        return False


def routing(state:AgentState)->bool:
    last=state["messages"][-1].content

    if 'file' in last:
        return True
    else:
        return False


graph=StateGraph(AgentState)
graph.add_node('Agent',agent)
graph.add_node('Ragnode',ragnode)
graph.add_conditional_edges(
    START,
    routing,
    {
        True:'Ragnode',
        False:'Agent'
    }
)
graph.add_edge('Ragnode','Agent')
toolnode=ToolNode(tools)
graph.add_node('tools',toolnode)
graph.add_conditional_edges(
    'Agent',
    decide,
    {
      True:'tools',
      False:END

    }
)
graph.add_edge('tools','Agent')
model=graph.compile()




while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    result = model.invoke({
        "messages": [HumanMessage(content=user_input)],
        "systemprompt":"""IMPORTANT:
- If the user's message  is a WEATHER RELATED QUERY YOU MUST   call the weather tool.
-PROVIDE RESPONSE BASED ON THE TOOLMESSAGE AND HUMANMESSAGE
- You have access to a tool named `weather_tool` that takes the city name and returns the current weather.
-VERY IMPORTANT:PLEASE ANSWER BASED ON THE QUERY 
- Do NOT try to generate a weather report yourself.
"""}
        )
    
    print("🤖: ", result['messages'][-1].content)
 

