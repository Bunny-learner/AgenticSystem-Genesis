# RAG-Weather-Agent 🌦️📄

This project is a conversational agent built using [LangGraph](https://github.com/langchain-ai/langgraph) that can:
- Retrieve answers from a file using RAG (Retrieval-Augmented Generation)
- Provide real-time weather data using OpenWeather API

---

## 🧰 Features
- 📄 **File-based QA** using LangChain + ChromaDB
- 🌦️ **Live Weather Retrieval** using a tool and OpenWeatherMap API
- 🤖 **Conversational Memory** managed through LangGraph state

---

## 📦 Installation

.
├── Agent.py                # Main script (LangGraph agent)
├── file.txt                # Your document for RAG
├── chromadb/               # Chroma vector store (persisted)
├── agent_workflow.png      # Visual of LangGraph workflow
├── .env                    # API keys
└── requirements.txt      # Python dependencies

### WEATHER_API_KEY=your_openweathermap_api_key
### GOOGLE_API_KEY=your_google_genai_api_key

The file.txt is loaded using LangChain's TextLoader. It contains summarized information about Marvel comics, the Transformers universe, cosmology, the Indian government, and various tax policies. You can replace this file with your own plain text document. The agent will then index its contents and answer questions based on the new context.
