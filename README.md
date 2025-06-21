# RAG-Weather-Agent 

This project is a conversational agent built using [LangGraph](https://github.com/langchain-ai/langgraph) that can:
- Retrieve answers from a file using RAG (Retrieval-Augmented Generation)
- Provide real-time weather data using OpenWeather API

---

##  Features

- 📄 **File-based QA** using LangChain + ChromaDB: The agent can answer questions by retrieving relevant information from a local text file (`file.txt`) which is indexed and stored in a ChromaDB vector store.
- 🌦️ **Live Weather Retrieval** using a tool and OpenWeatherMap API: Integrates with the OpenWeatherMap API to fetch and provide real-time weather data for specified locations.
- 🤖 **Conversational Memory** managed through LangGraph state: Maintains context across turns in a conversation, allowing for more natural and continuous interactions.

---
## Workflow

![agent_workflow](https://github.com/user-attachments/assets/3eef427a-10cd-4922-9ea2-b480c05aa044)


## Installations

To set up and run the RAG-Weather-Agent, follow these steps:

1.  **Project Structure:**

    ```
    .
    ├── Agent.py                # Main script (LangGraph agent)
    ├── file.txt                # Your document for RAG
    ├── chromadb/               # Chroma vector store (persisted)
    ├── agent_workflow.png      # Visual of LangGraph workflow
    ├── .env                    # API keys
    └── requirements.txt        # Python dependencies
    ```

2.  **Create and Configure `.env` file:**
    Create a file named `.env` in the root directory of your project and add your API keys:

    ```
    WEATHER_API_KEY=your_openweathermap_api_key
    GOOGLE_API_KEY=your_google_genai_api_key
    ```
    *Replace `your_openweathermap_api_key` with your actual API key from OpenWeatherMap.*
    *Replace `your_google_genai_api_key` with your actual API key for Google Generative AI.*

3.  **Install Dependencies:**
    Install the necessary Python packages listed in `requirements.txt` by running:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare `file.txt`:**
    The `file.txt` is loaded using LangChain's `TextLoader`. This file serves as your knowledge base for the RAG component. By default, it contains summarized information about Marvel comics, the Transformers universe, cosmology, the Indian government, and various tax policies. You can replace its content with any plain text document relevant to your needs. The agent will automatically index its contents to answer questions based on the new context.

---

## Usage

Once the installation and configuration are complete, you can run the main agent script:

```bash
python Agent.py
```
![note](https://img.shields.io/badge/Important-Note-red)
### does not remember previous converstations.

![note](https://img.shields.io/badge/Important-Note-blue)
#### kindly use file keyword when querying about the file.

#### In the terminal:
```bash
You: what is Goods and Services Tax?
You: give me humdity,visibility  and coordinates of  kadapa?
You: tell me about Captain America in the file?
