
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

import argparse

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
    Answer the question as detailed as possible from provided context, make sure to provide all the details.
    Context: \n {context}\n
    Question: \n {question}\n

    Answer :
    """
    model = ChatGoogleGenerativeAI(model = 'gemini-pro', temperature=0.9)
    prompt = PromptTemplate(template = prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain.invoke(
        {"input_documents":docs, "question": user_question},
        return_only_outputs = True
    )

    print(response)
    # st.write("Reply: ", response["output_text"])


def main():
    parser = argparse.ArgumentParser(description="DocSage - Chatbot Training on PDF Data")
    parser.add_argument("pdf_files", nargs="+", help="PDF file(s) to process")
    parser.add_argument("-q", "--question", help="Question to ask the chatbot")
    args = parser.parse_args()

    pdf_files = args.pdf_files
    user_question = args.question

    if not pdf_files:
        print("No PDF files provided. Please provide at least one PDF file.")
        return

    text = get_text_from_pdf(pdf_files)

    text_chunks = get_chunks(text)
    get_vector_store(text_chunks)

    if user_question:
        user_input(user_question)
    else:
        print("No question provided!!! Try again")
        return


if __name__ == "__main__":
    main()
