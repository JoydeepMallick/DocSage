import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS # just check documentation man so many changes😭 why not stick to one
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from dotenv import load_dotenv


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
    Answer the question as detailed as possible from provided context, make sure to provide all the details.
    Context: \n {context}\n
    Question: \n {question}\n

    Answer :
    """
    prompt = PromptTemplate(template = prompt_template, input_variables=["context", "question"])
    # prompt = ChatPromptTemplate([
    #     ("system", "You are DocSage, the AI document helper for everyone who is wise enough to choose you. You are here to help people find answers to their questions from the documents they provide."),
    #     ("human", "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:")
    # ])


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
def main():
    st.set_page_config(page_title="DocSage", page_icon=":robot:")
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

if __name__ == "__main__":
    main()
