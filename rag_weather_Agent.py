from langgraph.graph import StateGraph, START, END
from typing import List, TypedDict, Annotated,Sequence,Dict
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
import ast
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
import joblib




load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage],add_messages]
    systemprompt:str
    queries:Dict[str,str]
    



classifier = joblib.load("weather_file_classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")
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
    """
    invoke this tool for subquery class="weather" and 
    This tool fetches weather reports for queries related to weather,humidity,rainfall etc..
    use this tool for a weather related query ?
    this was fully functional and it provides you the entire data by making an api call to OPENWEATHERAPI"""
    cleaned = re.sub(r'\s+(town|city|village|state)\b', '', cityname, flags=re.IGNORECASE).strip()
    apikey = os.getenv('WEATHER_API_KEY')
    url = f"https://api.openweathermap.org/data/2.5/weather?q={cleaned}&appid={apikey}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to fetch weather. Status: {response.status_code}"}




@tool
def file_tool(query:str)-> str:
    """
    invoke this tool for subquery class="file" 
    use this tool for answering any query related to document or file or text file it implements retrieval augmented generation 
    and outputs the similar results
    """
    docs=retriever.invoke(query)
    if not docs:
        return "I found no relevant information in the document."
    results=[]
    for i,doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown page")
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content.strip()
        results.append(f"[{i+1}] (Source: {source}, Page: {page})\n{content}")

    return "\n\n".join(results)

tools = [weather_tool,file_tool]
llm=llm.bind_tools(tools)






def classifieragent(state:AgentState)->AgentState:
    parts = re.split(r'\b(?:and|also|,)\b', state['messages'][-1].content, flags=re.IGNORECASE)
    sub_queries=[p.strip() for p in parts if p.strip()]
    predictions=classifier.predict(sub_queries)
    decoded = label_encoder.inverse_transform(predictions)
    dicty={}
    for i in range(0,len(sub_queries)):
        dicty[sub_queries[i]]=decoded[i]
    state['queries']=dicty
    
    return state     




def agent(state:AgentState)->AgentState:
    print(".....",state['queries'])
    if state['queries']=={}:
        return state
    query = list(state['queries'].keys())[0]
    print(query)
    toolclass = state['queries'][query]
    syspromp_content=""
    if toolclass == "weather":
        syspromp_content = f"""You received a sub query: '{query}'.
        A preliminary classification suggests this is a 'weather' related query.
        Based on the query and your knowledge, determine if calling the 'weather_tool' is appropriate.
        If it is, generate a tool call to 'weather_tool' with the correct city name extracted from the query.
        If not, or if you can answer directly without a tool, provide a concise direct answer."""
    elif toolclass == "file":
        syspromp_content = f"""You received a sub query: '{query}'.
        A preliminary classification suggests this is a 'file' related query.
        Based on the query and your knowledge, determine if calling the 'file_tool' is appropriate.
        If it is, generate a tool call to 'file_tool' with the full query as the argument.
        If not, or if you can answer directly without a tool, provide a concise direct answer."""
    else: 
        syspromp_content = f"""You received a sub query: '{query}'.
        A preliminary classification suggests this is a 'basic' query, meaning no specific tool is needed.
        Based on the query and your knowledge,
        If it is, generate a tool call to 'file_tool' or 'weather_tool' with the full query as the argument.
        If not, or if you can answer directly without a tool, provide a concise direct answer.
        """
        state["messages"].append(HumanMessage(content=syspromp_content+query))
        del state['queries'][query]
        return state

    response=llm.invoke([SystemMessage(content=syspromp_content)]+[HumanMessage(content=query)])
    state['messages'].append(response)
    del state['queries'][query]
    return state

def route(state:AgentState)->str:
    last=state['messages'][-1]
    if hasattr(last,"tool_calls") and last.tool_calls:
        return "tool"
    elif state['queries']!={}:
        return "back"
    else:
        return "call"

toolnode = ToolNode(tools)


def response(state:AgentState)->AgentState:
     state['systemprompt']="""You are a helpful assistant.
     Find the unanswered queries from the messages and provide answers by combining all the answers .
Utilise answer by subqueries and  analysing all the messages and answer must be relevant on the main query and note
When the previous message in the conversation is a tool response in JSON format (e.g., weather data), your task is:

- Parse and interpret the JSON content.
- Extract useful and relevant information.
- Summarize it in clear, concise, natural language.
- Do not return raw JSON to the user.
- Tailor your answer based on the original user query if available (e.g., city name).
- Avoid repeating the original question back.

When summarizing information from a 'file_tool' or other retrieved documents:
- **Analyze the content deeply:** Understand not just what is explicitly stated, but also what the text *implies* or *promises to cover*.
- **Elaborate based on the context:** If the document is an introduction, explain what the introduction *suggests* the full document is about regarding the user's query.
- **Directly answer the user's original query:** Synthesize information from all relevant messages in the conversation history (human messages, AI tool calls, tool responses).
- **Format for clarity:** Present the information in a well-structured, natural language response.
- **Crucially: Do NOT invent information.** If the provided text does not contain specific details for a part of the user's query, state that the information is not present in the provided context, rather than remaining silent or hallucinating.
- Do NOT return raw tool outputs (like JSON or direct document chunks) to the user.
- If there are multiple distinct parts to the user's original query (e.g., a factual question and a joke), address each part appropriately.

Always aim to provide informative answers.
"""
     syspromp=SystemMessage(content=state['systemprompt'])
     response=llm.invoke([syspromp]+state['messages'])
     state["messages"].append(response)
     return state


graph=StateGraph(AgentState)
graph.add_node('classifier',classifieragent)
graph.add_node('Agent',agent)
toolnode=ToolNode(tools)
graph.add_node('tools',toolnode)
graph.add_node('response',response)
graph.add_edge(START,'classifier')
graph.add_edge('classifier','Agent')

graph.add_conditional_edges(
    'Agent',
    route,
    {
        "tool":"tools",
        "call":"response",
        "back":"Agent"
    }
)
graph.add_edge('tools','Agent')
graph.add_edge('response',END)
model=graph.compile()



from IPython.display import Image
from pathlib import Path

# Get the image binary from the Mermaid graph
image_bytes = model.get_graph().draw_mermaid_png()

# Define the output file path
output_path = Path("rag_weather.png")
# Write to local file
with open(output_path, "wb") as f:
    f.write(image_bytes)

print(f"Graph saved as {output_path.absolute()}")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    result = model.invoke({
        "messages": [HumanMessage(content=user_input)],
        "systemprompt":"",
        "queries":dict()})
    print("list of messages::::",result['messages'])
    print("🤖: ", result['messages'][-1].content)
 

