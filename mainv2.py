import streamlit as st
import os
import glob
import json
import logging
from datetime import datetime
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import pdfplumber

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is not set in the environment variables.")
    raise ValueError("GOOGLE_API_KEY is not set.")
genai.configure(api_key=GOOGLE_API_KEY)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurable parameters
BOOKS_FOLDER = "LLM_DB/"
DEFAULT_CHUNK_SIZE = 2028
DEFAULT_CHUNK_OVERLAP = 256
DEFAULT_PROMPT_TEMPLATE = """
As your Virtual Health Assistant, I am programmed to provide informed, precise, and clear answers based on medical knowledge extracted from the documents you've shared. Please be aware that while I aim to deliver accurate advice, my responses are strictly confined to the information from your documents and general medical principles.

**Note:** My capabilities are confined to interpreting and analyzing the medical information contained within your documents alongside established medical principles. My responses do not extend to generating programming language code, solving mathematical problems, answering any general questions, or addressing queries outside the specified medical domain. For any health issues beyond the scope of this information, it is imperative to consult with a qualified healthcare professional directly.

Now, let's focus on your health-related question, ensuring it pertains strictly to the medical context provided by you.

**Medical Context from Your Documents:**
{context}

**Your Question:**
{question}

**My Analysis and Recommendations:**

Please remember that while I offer insights based on the medical data shared, these are general guidelines. For personalized health advice, diagnoses, or treatments, it is crucial to consult a healthcare professional. Thank you for trusting me to assist with your medical inquiries.
"""

# Cache function to read PDF text
@st.cache_data
def get_pdf_text(pdf_folder):
    text = ""
    problematic_files = []
    for pdf_path in glob.glob(os.path.join(pdf_folder, '*.pdf')):
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
        except Exception as e:
            st.error(f"Error reading {pdf_path}: {str(e)}")
            problematic_files.append(pdf_path)
    if problematic_files:
        st.warning(f"Some files could not be processed properly: {problematic_files}")
    return text

# Function to check and update PDF text extraction based on changes
def get_updated_pdf_text(pdf_folder, embeddings_info_path):
    if os.path.exists(embeddings_info_path):
        with open(embeddings_info_path, 'r') as file:
            processed_files = json.load(file)
    else:
        processed_files = {}

    new_text = ""
    current_time = datetime.now().timestamp()
    for pdf_path in glob.glob(os.path.join(pdf_folder, '*.pdf')):
        modification_time = os.path.getmtime(pdf_path)
        if pdf_path not in processed_files or modification_time > processed_files[pdf_path]["last_processed_time"]:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
                new_text += text
                processed_files[pdf_path] = {"last_processed_time": current_time}
            except Exception as e:
                st.error(f"Error reading {pdf_path}: {str(e)}")

    with open(embeddings_info_path, 'w') as file:
        json.dump(processed_files, file)

    return new_text

# Cache function to split text into chunks
@st.cache_data
def get_text_chunks(text, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    if text:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(text)
        return chunks if chunks else []
    else:
        st.error("No text was extracted from the PDFs.")
        return []

# Function to get the latest modification time of PDFs
def get_latest_modification_time(pdf_folder):
    return max(os.path.getmtime(pdf_path) for pdf_path in glob.glob(os.path.join(pdf_folder, '*.pdf')))

# Function to check if embeddings need updating
def check_embeddings_update(pdf_folder, index_path="faiss_index/index.faiss"):
    embeddings_info_path = os.path.join("faiss_index", "embeddings_info.json")
    latest_modification_time = get_latest_modification_time(pdf_folder)
    
    if os.path.exists(embeddings_info_path):
        with open(embeddings_info_path, "r") as f:
            embeddings_info = json.load(f)
        last_processed_time = embeddings_info.get("last_processed_time", 0)
        
        if last_processed_time >= latest_modification_time and os.path.exists(index_path):
            return False
    return True

# Function to update embeddings info
def update_embeddings_info(pdf_folder):
    embeddings_info_path = os.path.join("faiss_index", "embeddings_info.json")
    latest_modification_time = get_latest_modification_time(pdf_folder)
    embeddings_info = {"last_processed_time": latest_modification_time}
    with open(embeddings_info_path, "w") as f:
        json.dump(embeddings_info, f)

# Cache function to get or update vector store
@st.cache_data
def get_vector_store(text_chunks, force_update=False):
    embeddings_info_path = os.path.join("faiss_index", "embeddings_info.json")
    if force_update or not os.path.exists(embeddings_info_path):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_texts(text_chunks, embeddings)
            if not os.path.exists("faiss_index"):
                os.makedirs("faiss_index")
            vector_store.save_local("faiss_index")
            update_embeddings_info(BOOKS_FOLDER)
            st.success("Embeddings created and saved successfully.")
            return vector_store
        except Exception as e:
            st.error(f"Error during embeddings creation: {str(e)}")
            return None
    else:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            st.success("Loaded existing embeddings.")
            return vector_store
        except Exception as e:
            st.error(f"Error loading existing embeddings: {str(e)}")
            return None

# Function to get the conversational chain
def get_conversational_chain():
    model = ChatGoogleGenerativeAI(model="gemini-1.0-pro-001", temperature=0.2, maxOutputTokens=1000, topK=6, topP=0.9)
    prompt = PromptTemplate(template=DEFAULT_PROMPT_TEMPLATE, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input and provide a response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index_path = "faiss_index/index.faiss"
    if os.path.exists(index_path):
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question})
        st.write("Reply: ", response["output_text"])
    else:
        st.error(f"Index file not found at {index_path}. Please ensure the index is created.")

# Main function for the Streamlit app
def main():
    st.set_page_config(page_title="Chat with Jillani SoftTech AI Bot 🤖", layout="wide")
    st.header("Chat with Jillani SoftTech AI Bot 🤖")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = ["Chatbot", "Dashboard", "Upload PDFs"]
    choice = st.sidebar.selectbox("Choose an option:", options)

    if choice == "Chatbot":
        st.subheader("Interactive Health Chatbot")
        user_question = st.text_input("Enter your health-related question here:")
        if user_question:
            user_input(user_question)
    elif choice == "Dashboard":
        st.subheader("Dashboard")
        st.markdown("### Processed Files")
        embeddings_info_path = "faiss_index/embeddings_info.json"
        if os.path.exists(embeddings_info_path):
            with open(embeddings_info_path, 'r') as file:
                processed_files = json.load(file)
                st.json(processed_files)
        else:
            st.write("No files processed yet.")

        st.markdown("### Embeddings Status")
        index_path = "faiss_index/index.faiss"
        if os.path.exists(index_path):
            st.success("Embeddings are up-to-date.")
        else:
            st.error("Embeddings need to be created.")
    elif choice == "Upload PDFs":
        st.subheader("Upload PDFs")
        uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
        if uploaded_files:
            for uploaded_file in uploaded_files:
                pdf_path = os.path.join(BOOKS_FOLDER, uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success("Uploaded files successfully. Click 'Process' to extract and update embeddings.")
        
        if st.button("Process"):
            with st.spinner("Processing..."):
                embeddings_info_path = "faiss_index/embeddings_info.json"
                new_text = get_updated_pdf_text(BOOKS_FOLDER, embeddings_info_path)
                if new_text:
                    text_chunks = get_text_chunks(new_text)
                    if text_chunks:
                        vector_store = get_vector_store(text_chunks, force_update=True)
                        if vector_store:
                            st.success("New PDFs processed and embeddings updated successfully. You can now ask questions!")
                    else:
                        st.error("Failed to process new text chunks.")
                else:
                    st.success("No new or updated PDFs to process. Using existing embeddings.")

    st.markdown(
        """
        <div style="text-align: center; margin-top: 50px;">
            <strong>This application is powered by <a href="https://pk.linkedin.com/in/jillanisofttech" target="_blank">Jillani SoftTech</a></strong>
            <br>
            <a href="https://pk.linkedin.com/in/jillanisofttech" target="_blank">LinkedIn</a> - 
            <a href="https://www.kaggle.com/jillanisofttech" target="_blank">Kaggle</a> - 
            <a href="https://jillanisofttech.medium.com/" target="_blank">Medium</a> - 
            <a href="https://github.com/MGJillaniMughal" target="_blank">GitHub</a> - 
            <a href="https://mgjillanimughal.github.io/" target="_blank">Portfolio</a>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
