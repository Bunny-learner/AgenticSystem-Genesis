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
## Workflow for Agent.py

![agent_workflow](https://github.com/user-attachments/assets/3eef427a-10cd-4922-9ea2-b480c05aa044)

## Workflow for rag_weather_agent.py
![rag_weather](https://github.com/user-attachments/assets/dd281fb9-84dd-4943-ac00-cbb8ad4094a5)

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
You: tell me whether it rains in kadapa and what is the capital of delhi and is chicken nonveg or veg and calculate sum of first seven prime numbers and what is definetion of indirect taxes from the document?
Ai::  Based on the provided data:
* **Weather in Kadapa:** The weather in Kadapa is overcast with a temperature of 28.81°C and 59% humidity.

* **Capital of Delhi:**  The capital of Delhi is not found in the provided text.

* **Chicken:** Chicken is non-vegetarian.

* **Sum of first seven prime numbers:** The first seven prime numbers are 2, 3, 5, 7, 11, 13, and 17. Their sum is 58.

* **Definition of indirect taxes:**  The provided text gives a detailed explanation of the Goods and Services Tax (GST) in India, which is a significant indirect tax reform.  It replaced many central and state indirect taxes and operates under a dual GST model (CGST, SGST, IGST).  The document does not provide a general definition of indirect taxes beyond this specific example.
