import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict, Optional
from duckduckgo_search import DDGS
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Groq model
llm = ChatGroq(model_name="mistral-saba-24b", api_key=os.getenv("GROQ_API_KEY"))

# Define state for LangGraph
class AgentState(TypedDict):
    query: str
    topic_valid: bool
    llm_response: Optional[str]
    pdf_response: Optional[str]
    web_response: Optional[str]
    current_date: str
    final_answer: str

# Define FAISS index folder
FAISS_INDEX_PATH = "./data/index.faiss"  #"faiss_index"


# SEO and Marketing keywords
seo_marketing_keywords = [
    "seo", "search engine", "marketing", "digital marketing", "advertising",
    "content marketing", "social media", "ppc", "keywords", "analytics"
]

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load existing FAISS index if available
def load_faiss_index():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            st.session_state.vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            #st.info("Loaded existing FAISS index from local folder.")
        except Exception as e:
            st.warning(f"Failed to load FAISS index: {e}. Starting with a new index.")
            st.session_state.vector_store = None
    else:
        st.session_state.vector_store = None

# Load FAISS index on app start
if st.session_state.vector_store is None:
    load_faiss_index()

# Streamlit app configuration
st.title("SEO & Marketing")
st.markdown("Ask about SEO/Marketing  or search the web for relevant SEO/Marketing topics.")

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload PDF")
    uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_pdf:
        with st.spinner("Processing PDF..."):
            # Extract text from PDF
            pdf_reader = PdfReader(uploaded_pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Add to FAISS index
            if st.session_state.vector_store is None:
                st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
            else:
                st.session_state.vector_store.add_texts(chunks)
            
            # Save FAISS index to local folder
            try:
                st.session_state.vector_store.save_local(FAISS_INDEX_PATH)
                st.success(f"PDF processed and FAISS index saved to {FAISS_INDEX_PATH}!")
            except Exception as e:
                st.error(f"Failed to save FAISS index: {e}")

# Define agentic workflow with LangGraph
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def get_current_date(state: AgentState) -> AgentState:
    """Get the current date for context-aware responses."""
    state["current_date"] = get_current_time()
    return state





def check_topic_and_news(state: AgentState) -> AgentState:
    """Check if the query is related to SEO/Marketing and if it asks for news."""
    query_lower = state["query"].lower()
    # Check for SEO/marketing topic
    state["topic_valid"] = any(keyword in query_lower for keyword in seo_marketing_keywords)
    # Check for news-related query
    news_keywords = ["latest", "news", "recent", "update", "trend"]
    state["is_news_query"] = any(keyword in query_lower for keyword in news_keywords) and state["topic_valid"]
    return state


def check_topic(state: AgentState) -> AgentState:
    """Check if the query is related to SEO or Marketing."""
    query_lower = state["query"].lower()
    state["topic_valid"] = any(keyword in query_lower for keyword in seo_marketing_keywords)
    return state

def query_llm(state: AgentState) -> AgentState:
    """Query the LLM if the topic is valid."""
    if state["topic_valid"]:
        prompt = PromptTemplate(
            input_variables=["query"],
            template="You are an expert in SEO and Marketing. Answer the following query concisely: {query}"
        )
        response = llm.invoke(prompt.format(query=state["query"]))
        state["llm_response"] = response.content
    else:
        state["llm_response"] = None
    return state

def query_pdf(state: AgentState) -> AgentState:
    """Query the vector store if LLM response is None or if explicitly requested."""
    if (not state["llm_response"] or not state["topic_valid"]) and st.session_state.vector_store:
        docs = st.session_state.vector_store.similarity_search(state["query"], k=3)
        if docs:
            context = "\n".join([doc.page_content for doc in docs])
            prompt = PromptTemplate(
                input_variables=["query", "context"],
                template="Based on the following context, answer the query: {query}\n\nContext: {context}"
            )
            response = llm.invoke(prompt.format(query=state["query"], context=context))
            state["pdf_response"] = response.content
        else:
            state["pdf_response"] = None
    else:
        state["pdf_response"] = None
    return state

def query_web(state: AgentState) -> AgentState:
    """Query DuckDuckGo if no LLM or PDF response and topic is valid."""
    if not state["llm_response"] and not state["pdf_response"] and state["topic_valid"]:
        try:
            with DDGS() as ddgs:
                # Perform search with SEO/marketing focus
                search_query = f"{state['query']} site:*.edu | site:*.org | site:*.gov SEO marketing"
                results = list(ddgs.text(search_query, max_results=3))
            
            if results:
                context = "\n".join([f"{r['title']}: {r['body']}" for r in results])
                prompt = PromptTemplate(
                    input_variables=["query", "context"],
                    template="Based on the following web search results, answer the query concisely: {query}\n\nWeb Results: {context}"
                )
                response = llm.invoke(prompt.format(query=state["query"], context=context))
                state["web_response"] = response.content
            else:
                state["web_response"] = None
        except Exception as e:
            state["web_response"] = None
            st.warning(f"Web search failed: {e}")
    else:
        state["web_response"] = None
    return state

def combine_responses(state: AgentState) -> AgentState:
    """Combine LLM, PDF, and web responses."""
    if state["llm_response"]:
        state["final_answer"] = state["llm_response"]
    elif state["pdf_response"]:
        state["final_answer"] = state["pdf_response"]
    elif state["web_response"]:
        state["final_answer"] = state["web_response"]
    else:
        state["final_answer"] = "Sorry, I couldn't find relevant information. Please try a different query or upload a PDF."
    return state

# Define LangGraph workflow
workflow = StateGraph(AgentState)
workflow.add_node("get_current_date", get_current_date)
workflow.add_node("check_topic_and_news", check_topic_and_news)
workflow.add_node("check_topic", check_topic)
workflow.add_node("query_llm", query_llm)
workflow.add_node("query_pdf", query_pdf)
workflow.add_node("query_web", query_web)
workflow.add_node("combine_responses", combine_responses)

# Define edges
workflow.set_entry_point("get_current_date")
#workflow.add_edge("get_current_date", "check_topic")
workflow.add_edge("get_current_date", "check_topic_and_news")
workflow.add_conditional_edges(
    "check_topic_and_news",
    lambda state: "query_web" if state["is_news_query"] else "query_llm",
    {
        "query_web": "query_web",
        "query_llm": "query_llm"
    }
)

#workflow.set_entry_point("check_topic")
workflow.add_edge("check_topic", "query_llm")
workflow.add_edge("query_llm", "query_pdf")
workflow.add_edge("query_pdf", "query_web")
workflow.add_edge("query_web", "combine_responses")
workflow.add_edge("combine_responses", END)

# Compile the graph
app = workflow.compile()

# Main query interface
st.header("Ask a Question")
query = st.text_input("Enter your query:", placeholder="E.g., What is SEO? or Ask about uploaded PDF content")
if st.button("Submit"):
    if query:
        with st.spinner("Processing your query..."):
            # Run the LangGraph workflow
            state = app.invoke({
                "query": query,
                "topic_valid": False,
                "llm_response": None,
                "pdf_response": None,
                "web_response": None,
                "final_answer": ""
            })
            response = state["final_answer"]
            
            # Update chat history
            st.session_state.chat_history.append({"query": query, "response": response})
            
            # Display response
            st.write("**Answer:**")
            st.write(response)
    else:
        st.warning("Please enter a query.")

# Display chat history
if st.session_state.chat_history:
    st.header("Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"**You:** {chat['query']}")
        st.write(f"**AI:** {chat['response']}")
        st.markdown("---")