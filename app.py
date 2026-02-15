import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS # Vector datgabase wrapper for FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings # HuggingFaceEmbeddings wrapper for Hugging Face models
from langchain_text_splitters import RecursiveCharacterTextSplitter # Text splitter for splitting text into chunks
from pdfextractor import text_extractor_pdf

from pypdf import PdfReader
import streamlit as st

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Create the main page of the Streamlit app
st.title(':green[RAG Chatbot]')
tips = """Follow the steps to use this application:
* Upload your PDF document in the sidebar.
* Write your query and start chatting with the bot. """
st.subheader(tips)

# load the PDF file and extract text (from the sidebar)
st.sidebar.title(':orange[Upload your PDF document(pdf_only)]')
file_uploaded = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"]) # Restricted to pdf file type only

if file_uploaded:
    file_text = text_extractor_pdf(file_uploaded)
    
    #Step1: Configure the model
    key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=key)
    llm_model = genai.GenerativeModel("gemini-3-flash-preview")
    
    #Step2: Configure the embeddings
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    
    #Step3: Split the text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(file_text)
    
    #step4: Create the FAISS vector store
    vector_store = FAISS.from_texts(chunks, embedding_model)
    
    #Step5 Configure the retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    #Function to generate response from the model
    def generate_response(query: str) -> str:
        retrieved_docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        prompt = f"""
        You are helpful assistant for answering questions based on the following context, using RAG:
        {context}
        User query: {query}
        """
        
        #Generation
        content = llm_model.generate_content(prompt)
        return content.text if hasattr(content, "text") else content.candidates[0].content.parts[0].text # Return the generated response by the model
        # Return the generated text safely:
        # If the response object has a direct "text" attribute, use it.
        # Otherwise, extract the text from the nested candidates → content → parts structure
        # (fallback for models that return responses in a structured format).
        
     
    # Initialize chat if there is no history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Display the History
    for msg in st.session_state.history:
        if msg['role'] == 'user':
            st.write(f':green[User:] :blue[{msg["text"]}]')
        else:
            st.write(f':orange[Chatbot:] {msg["text"]}')

    # Input from the user (Using Streamlit Form)
    with st.form('Chat Form', clear_on_submit=True):
        user_input = st.text_input('Enter Your Text Here:')
        send = st.form_submit_button('Send')

    # Start the conversation and append the output and query in history
    if user_input and send:
        st.session_state.history.append({"role": 'user', "text": user_input})
        model_output = generate_response(user_input)
        st.session_state.history.append({'role': 'chatbot', 'text': model_output})
        st.rerun()    
    