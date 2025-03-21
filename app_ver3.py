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
        pdf_reader = PdfReader(pdf) #pdf object containing pages to look from
        for page in pdf_reader.pages:
            text += page.extract_text() # storing all text in all pages in a single variable
    return text

def get_chunks(text): # create overlapping chunks of text which are small in size and can be run in parallel
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001") #embedding model basically converts those chunks into mathematical notation 
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

    print(response) #printing on the backend or terminal for debugging
    return response
    #st.write("Reply: ", response["output_text"])


#########################################################################################
#                                    UI
#########################################################################################


def main():
    # Set page configuration
    st.set_page_config(page_title="DocSage", page_icon=":crystal_ball:", layout="wide")

    # Sidebar for PDF upload
    st.sidebar.title("📂 Upload & Process")
    pdf_docs = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    add_vertical_space(2)
    if st.sidebar.button("✨ Submit & Process"):
        with st.spinner("🔮 Processing..."):
            raw_text = get_text_from_pdf(pdf_docs)
            text_chunks = get_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("🎉 Processing Completed! Ready to answer your questions.")

    # Custom CSS
    st.markdown(
        """
        <style>
        /* Body color, removing the default margin */
        body {
            margin: 0;
            padding: 0;
            background-color: #1e1e2e; /* Adjust as desired */
            color: #ffffff;
        }

        /* Container for chat messages (fixed height, scrollable) */
        .chat-container {
            height: 70vh; /* Adjust to taste */
            overflow-y: auto;
            margin-bottom: 120px; /* Space for the fixed input box */
            padding: 1rem;
        }

        /* Fixed text input at the bottom */
        .fixed-input {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: rgba(30, 30, 46, 0.9);
            padding: 1rem;
            z-index: 9999;
        }

        /* Style for the text input box itself */
        .fixed-input input {
            width: 100%;
            border-radius: 15px;
            border: 2px solid #6c5ce7;
            padding: 10px;
            background-color: rgba(45, 45, 58, 0.9);
            color: white;
            box-shadow: 0px 0px 15px rgba(108, 92, 231, 0.7);
        }

        /* Chat bubble styling */
        .user-message {
            text-align: right;
            background-color: #4a69bd;
            color: white;
            padding: 10px;
            border-radius: 10px;
            max-width: 90%;
            margin-left: auto;
            margin-right: 10px;
            margin-top: 10px;
        }
        .ai-message {
            text-align: left;
            background-color: #dcdde1;
            color: black;
            padding: 10px;
            border-radius: 10px;
            max-width: 90%;
            margin-right: auto;
            margin-left: 10px;
            margin-top: 10px;
        }

        /* GitHub repo link at bottom-right (optional) */
        .github-link {
            position: fixed;
            bottom: 60px; /* Above the input box */
            right: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Page header
    st.markdown(
        """
        <h1 style='text-align: center; color: #6c5ce7; margin-top: 1rem;'>
            🔮 DocSage - AI Document Helper
        </h1>
        <hr>
        """,
        unsafe_allow_html=True
    )

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat container (scrollable)
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # Display chat history
    for chat in st.session_state.chat_history:
        st.markdown(f"<div class='user-message'><strong>🧌 You:</strong><br>{chat['user']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='ai-message'><strong>🧙‍♂️ DocSage:</strong><br>{chat['response']}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Fixed input area at the bottom
    st.markdown("<div class='fixed-input'>", unsafe_allow_html=True)
    user_question = st.text_input("Ask me anything about your document:", key="chat_input")
    st.markdown("</div>", unsafe_allow_html=True)

    # If user typed something, process
    if user_question:
        response = user_input(user_question)  # Replace with your function
        st.session_state.chat_history.append({
            "user": user_question,
            "response": response["output_text"] if response else "No response found."
        })
        # Clear the text input after submission 
        # st.session_state["chat_input"] = ""

    # Optional: GitHub link at bottom-right
    st.markdown(
        """
        <div class="github-link">
            <a href="https://github.com/JoydeepMallick/DocSage" target="_blank">
                <img src="https://img.shields.io/badge/GitHub-Repo-%23181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Repo">
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()