import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS # just check documentation man so many changes😭 why not stick to one
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from streamlit_extras.add_vertical_space import add_vertical_space


# load the dotenv file
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key = GOOGLE_API_KEY)


def get_text_from_pdf(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding = embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are DocSage, the AI document helper for everyone who is wise enough to choose you. You are here to help people find answers to their questions from the documents they provide.
    Answer the question as detailed as possible from provided context, make sure to provide all the details. Sound like a wise sage full of wisdom and knowledge, helping a seeker attain knowledge like those medieval sages.
    Context: \n {context}\n
    Question: \n {question}\n

    Answer :
    """
    prompt = PromptTemplate(template = prompt_template, input_variables=["context", "question"])


    # the model name changes with release refer google api docs
    model = ChatGoogleGenerativeAI(model = 'gemini-1.5-pro', temperature=0.9)
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local(folder_path="./faiss_index/", embeddings=embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain.invoke(
        {"input_documents":docs, "question": user_question},
        return_only_outputs = True
    )

    print(response)
    st.write("Reply: ", response["output_text"])


#########################################################################################
#                                    UI
#########################################################################################


def main():
    # Set page configuration
    st.set_page_config(page_title="DocSage", page_icon=":crystal_ball:", layout="wide")
    
    # Custom CSS for better styling
    st.markdown(
        """
        <style>
        body {
            background-color: #1e1e2e;
            color: #ffffff;
        }
        .main-container {
            text-align: center;
        }
        .stTextInput>div>div>input {
            border-radius: 15px;
            border: 2px solid #6c5ce7;
            padding: 10px;
            background-color: #2d2d3a;
            color: white;
        }
        .stButton>button {
            background-color: #6c5ce7;
            color: white;
            border-radius: 12px;
            font-size: 16px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #a29bfe;
            color: black;
        }
        .sidebar .stButton>button {
            background-color: #fd79a8;
        }
        .sidebar .stButton>button:hover {
            background-color: #ff7675;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Header
    st.markdown("""<h1 style='text-align: center; color: #6c5ce7;'>🧙‍♂️ DocSage - AI Document Helper</h1>""", unsafe_allow_html=True)
    st.markdown("---")
    
    # Layout: Two columns (Main content | Sidebar)
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🔍 Ask your question")
        user_question = st.text_input("Hey answer seeker, ask me anything about the document you provided")
        
        if user_question:
            user_input(user_question)  # Process input
        
    with col2:
        st.sidebar.title("📂 Upload & Process")
        pdf_docs = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
        add_vertical_space(2)
        
        if st.sidebar.button("✨ Submit & Process"):
            with st.spinner("🔮 Processing..."):
                raw_text = get_text_from_pdf(pdf_docs)
                text_chunks = get_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("🎉 Processing Completed! Ready to answer your questions.")
    
if __name__ == "__main__":
    main()


###### old UI #######
'''def main():
    st.set_page_config(page_title="DocSage", page_icon="🧙")
    st.header("🧙‍ DocSage - AI Document Helper")

    user_question = st.text_input("Hey answer seeker, ask me anything about the document you provided")

    if user_question:
        user_input(user_question) # pass the questions

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_text_from_pdf(pdf_docs)
                text_chunks = get_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing Completed🔮")
'''