import streamlit as st 
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv
import os 

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(api_key=GROQ_API_KEY, temperature=0.7)

# Load CSV
csv_path = "QA_fixed.csv"
if not os.path.exists(csv_path):
    st.error(f"Error: File '{csv_path}' not found!")
    st.stop()

try:
    loader = CSVLoader(file_path=csv_path, source_column='prompt', encoding='utf-8-sig')
    data = loader.load()
except Exception as e:
    st.error(f"Failed to load CSV: {str(e)}")
    st.stop()

# Ensure data is loaded
if not data:
    st.error("Error: No data found in the CSV file!")
    st.stop()

# Create Embeddings and FAISS Vector Database
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_db = FAISS.from_documents(data, embedding)
retriever = vector_db.as_retriever(score_threshold=0.7)

# System Prompt
system_prompt = """
Given the following context and a question, generate an answer based on this context only.
In the answer, try to provide as much text as possible from the "response" section in the source document without making much changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

CONTEXT
{context}

QUESTION
{question}
"""
PROMPT = PromptTemplate(template=system_prompt, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Create RetrievalQA Chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Streamlit UI
st.title("QA System")
question = st.text_area("Ask a question...")

if st.button("Get Answer"):
    if not question.strip():
        st.error("Please enter a question.")
    else:
        response = chain.invoke({"query": question}) 
        if response and "result" in response:
            st.success(response["result"])
        else:
            st.error("No response generated.")
