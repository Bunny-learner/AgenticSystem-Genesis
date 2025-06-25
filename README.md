
# Agentic System

## 🧠 Capabilities

### 🧾 1. File-Based Question Answering (RAG)

- Loads and indexes any `.txt` file using LangChain’s `TextLoader`
- Splits content using `RecursiveCharacterTextSplitter`
- Embeds using **BAAI/bge-large-en-v1.5**
- Stored in ChromaDB (`temporary_cache` collection)
- Retrieves and responds to file-specific queries

### 🌐 2. Web Search

Two search modes, chosen by the ML classifier:

#### 🔹 Shallow Search (`shallow_search`)
- Uses [SerpAPI](https://serpapi.com/) to fetch metadata (title, snippet, URL)
- Fast, ideal for factual lookups

#### 🔹 Deep Search (`deep_search`)
- Scrapes full content using **Crawlbase API**
- Converts HTML to Markdown for readability
- Stores responses in **`web_cache`** vectorstore (ChromaDB)
- Future calls can retrieve without re-scraping

### 🧩 3. Sub-query Classification

- Uses a trained ML classifier (`joblib` + `label_encoder`)
- Splits compound queries like:
  
user-attachments/assets/dd281fb9-84dd-4943-ac00-cbb8ad4094a5)

## Installations



## Workflow
![agent](https://github.com/user-attachments/assets/d982758d-5f83-46bb-a7f8-f6e37d074ccc)



1.  **Project Structure:**

    ```
    .
    ├── Agent.py  # Main graph engine
    ├── chromadb/      # Stores web_cache & temporary_cache
    ├── file.txt     # Sample text file for file-based queries
    ├── agent_workflow.png     # Mermaid diagram of LangGraph
    ├── intent_classifier.pkl     # Trained classifier (joblib)
    ├── label_encoder.pkl # Class label encoder
    ├── .env # API keys
    ├── requirements.txt # All Python dependencies## Workflow for Agent.py

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

### Some examples of the answers system provided

```
You: hi
🤖: Hello! How can I help you today?
You: tell me some free gpu provider in cloud?
🤖: Here are some cloud providers that offer free GPU access, often with certain limitations:

*   **Google Colab**
    *   **Details:** A very popular choice, offering GPUs like NVIDIA K80/T4. Free sessions typically have a duration limit (e.g., 12 hours per session).    
    *   **Sources:**
        *   [https://zackproser.com/blog/cloud-gpu-services-jupyter-notebook-reviewed](https://zackproser.com/blog/cloud-gpu-services-jupyter-notebook-reviewed)
        *   [https://codesphere.com/articles/5-best-free-cloud-gpu-providers-for-hobbyists](https://codesphere.com/articles/5-best-free-cloud-gpu-providers-for-hobbyists)
        *   [https://www.geeksforgeeks.org/deep-learning/the-top-3-free-gpu-resources-to-train-deep-neural-network/](https://www.geeksforgeeks.org/deep-learning/the-top-3-free-gpu-resources-to-train-deep-neural-network/)
        *   [https://research.aimultiple.com/free-cloud-gpu/](https://research.aimultiple.com/free-cloud-gpu/)

*   **Kaggle**
    *   **Details:** Provides access to NVIDIA TESLA P100 GPUs, often used for data science competitions and personal projects.
    *   **Sources:**
        *   [https://codesphere.com/articles/5-best-free-cloud-gpu-providers-for-hobbyists](https://codesphere.com/articles/5-best-free-cloud-gpu-providers-for-hobbyists)
        *   [https://www.geeksforgeeks.org/deep-learning/the-top-3-free-gpu-resources-to-train-deep-neural-network/](https://www.geeksforgeeks.org/deep-learning/the-top-3-free-gpu-resources-to-train-deep-neural-network/)
        *   [https://research.aimultiple.com/free-cloud-gpu/](https://research.aimultiple.com/free-cloud-gpu/)

*   **Codesphere**
    *   **Details:** Listed as a free cloud GPU provider suitable for hobbyists.
    *   **Sources:**
        *   [https://codesphere.com/articles/5-best-free-cloud-gpu-providers-for-hobbyists](https://codesphere.com/articles/5-best-free-cloud-gpu-providers-for-hobbyists)
        *   [https://research.aimultiple.com/free-cloud-gpu/](https://research.aimultiple.com/free-cloud-gpu/)

*   **Gradient by Paperspace**
    *   **Details:** Another option mentioned for free cloud GPU access.
    *   **Sources:**
        *   [https://zackproser.com/blog/cloud-gpu-services-jupyter-notebook-reviewed](https://zackproser.com/blog/cloud-gpu-services-jupyter-notebook-review        *   [https://codesphere.com/articles/5-best-free-cloud-gpu-providers-for-hobbyists](https://codesphere.com/articles/5-best-free-cloud-gpu-providers-for-hobbyists)

*   **Microsoft Azure Notebooks (Azure ML)**
    *   **Details:** Identified as a free GPU resource for training deep neural networks.
    *   **Source:** [https://www.geeksforgeeks.org/deep-learning/the-top-3-free-gpu-resources-to-train-deep-neural-network/](https://www.geeksforgeeks.org/de

**Important Considerations:**
*   **Purpose:** These free services are generally best for learning, small projects, or light experimentation rather than heavy, continuous workloads.      
*   **Paid Tiers:** Most of these providers also offer paid tiers with more robust GPU options and fewer limitations.
You: thank you
🤖: You're welcome! Is there anything else I can help you with?
You: exit
