# SEO & Marketing Chatbot

This project is a Streamlit-based web application designed to answer queries related to SEO (Search Engine Optimization) and marketing. It leverages a combination of a large language model (LLM), vector search on uploaded PDFs, and web search capabilities to provide accurate and context-aware responses. The application uses an agentic workflow powered by LangGraph to orchestrate the query processing pipeline.

## Features
- **Query Processing**: Users can ask questions about SEO and marketing topics.
- **PDF Integration**: Upload PDFs to extract and index content for context-aware answers.
- **Web Search**: Fallback to web search for recent or unindexed information.
- **Chat History**: Maintains a record of user queries and responses.
- **Topic Validation**: Ensures queries are relevant to SEO/marketing before processing.

## Tools and Libraries Used
Below is a list of the key tools and libraries used in this project, along with the rationale for their inclusion:

1. **Streamlit** (`streamlit`)
   - **Purpose**: Provides the web-based user interface for the application.
   - **Why**: Streamlit is lightweight, easy to use, and allows rapid development of interactive web apps with minimal frontend coding. It supports real-time UI updates and file uploads, which are critical for the PDF upload feature.

2. **Python-dotenv** (`dotenv`)
   - **Purpose**: Loads environment variables from a `.env` file.
   - **Why**: Securely manages sensitive information like API keys (e.g., Groq API key) without hardcoding them in the source code.

3. **PyPDF2** (`PyPDF2`)
   - **Purpose**: Extracts text from uploaded PDF files.
   - **Why**: PyPDF2 is a reliable library for parsing PDF content, enabling the application to process user-uploaded documents for indexing and querying.

4. **LangChain** (`langchain`, `langchain_community`, `langchain_groq`)
   - **Purpose**: Provides tools for text splitting, embeddings, vector stores, and LLM integration.
   - **Why**:
     - `RecursiveCharacterTextSplitter`: Splits PDF text into manageable chunks for embedding, balancing context and performance.
     - `HuggingFaceEmbeddings`: Generates semantic embeddings for text chunks, enabling similarity-based retrieval.
     - `FAISS`: A fast vector store for indexing and searching embedded text, suitable for efficient similarity searches.
     - `ChatGroq`: Integrates with the Groq LLM for generating responses, offering high-performance inference.

5. **FAISS** (`faiss-cpu` via `langchain_community.vectorstores`)
   - **Purpose**: Stores and retrieves text embeddings for similarity search.
   - **Why**: FAISS is optimized for large-scale vector search, making it ideal for indexing PDF content and retrieving relevant chunks quickly.

6. **HuggingFace Embeddings** (`all-MiniLM-L6-v2` via `langchain_community.embeddings`)
   - **Purpose**: Converts text into dense vector representations.
   - **Why**: The `all-MiniLM-L6-v2` model is lightweight, fast, and provides good semantic embeddings for text, suitable for resource-constrained environments.

7. **Groq LLM** (`langchain_groq`)
   - **Purpose**: Powers the conversational AI for answering queries.
   - **Why**: The `mistral-saba-24b` model (via Groq) offers high-quality, context-aware responses with low latency, making it suitable for real-time query answering.

8. **LangGraph** (`langgraph`)
   - **Purpose**: Orchestrates the agentic workflow for query processing.
   - **Why**: LangGraph enables a modular, stateful workflow to handle complex logic (e.g., topic validation, LLM query, PDF search, web search) in a structured and extensible way.

9. **DuckDuckGo Search** (`duckduckgo_search`)
   - **Purpose**: Performs web searches for recent or unindexed information.
   - **Why**: DuckDuckGo provides a privacy-focused search API that is simple to integrate and effective for retrieving relevant web content as a fallback.

10. **Datetime** (`datetime`)
    - **Purpose**: Adds current date/time to responses for context.
    - **Why**: Ensures responses are time-aware, especially for queries requesting recent information or trends.

## Project Structure
- **Main Script**: The core application logic, including Streamlit UI, PDF processing, and LangGraph workflow.
- **FAISS Index**: Stored in `./agents/2-test/data/index.faiss` for persistent vector storage.
- **Environment Variables**: Loaded from a `.env` file (e.g., `GROQ_API_KEY`).

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure you have `requirements.txt` with the following:
   ```
   streamlit
   python-dotenv
   PyPDF2
   langchain
   langchain-community
   langchain-groq
   langgraph
   faiss-cpu
   sentence-transformers
   duckduckgo-search
   ```

3. **Set Up Environment Variables**:
   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=<your-groq-api-key>
   ```

4. **Run the Application**:
   ```bash
   streamlit run <script-name>.py
   ```

5. **Usage**:
   - Access the app in your browser (typically `http://localhost:8501`).
   - Upload a PDF via the sidebar to index its content.
   - Enter a query in the text input to get answers from the LLM, PDF, or web.

## How It Works
1. **PDF Processing**:
   - Users upload a PDF, which is processed using PyPDF2 to extract text.
   - Text is split into chunks using `RecursiveCharacterTextSplitter`.
   - Chunks are embedded using `HuggingFaceEmbeddings` and stored in a FAISS vector store.

2. **Query Workflow** (via LangGraph):
   - **Topic Validation**: Checks if the query is related to SEO/marketing.
   - **LLM Query**: If valid, queries the Groq LLM for a direct response.
   - **PDF Search**: If no LLM response or topic is invalid, searches the FAISS index for relevant PDF content.
   - **Web Search**: If no PDF response, performs a DuckDuckGo search for web content.
   - **Response Combination**: Combines responses, prioritizing LLM, then PDF, then web.

3. **UI**:
   - Streamlit provides an interactive interface for PDF uploads, query input, and chat history display.

## Limitations
- **PDF Quality**: Depends on the quality of text extraction from PDFs.
- **Web Search**: Limited to DuckDuckGo's API results, which may not always be comprehensive.
- **FAISS Index**: Requires sufficient disk space for large PDFs and may need periodic cleanup.

## Future Improvements
- Add support for multiple file formats (e.g., DOCX, TXT).
- Implement caching for web search results to improve performance.
- Enhance topic validation with more sophisticated NLP techniques.
- Add user authentication for personalized chat history.

## License
This project is licensed under the MIT License.