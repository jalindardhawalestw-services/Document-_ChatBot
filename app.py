import streamlit as st
from chromadb import *
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter 
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain_community.vectorstores import Chroma

from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.embeddings import Embeddings
from PIL import Image

path_to_save = r"Chroma DB"


def main():
    
    persist_directory ='db'
    st.set_page_config(page_title="Ask your PDF", page_icon="📄")
    logo = Image.open("MicrosoftTeams-image (1).png")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("")
    with col2:
        st.image(logo, caption='STW Services')
    with col3:
        st.write("")
   
    # st.image(logo,caption='STW Services', use_column_width=False)
    st.header("Ask your PDF")
    pdf:any
    db:any
    pdf = st.file_uploader("upload your file",type="PDF")   
    query = st.text_input("Ask question")
       # Create a persistent client with the specified path
    client = chromadb.PersistentClient(path=path_to_save)
    client = chromadb.HttpClient(host='localhost', port=8000)
  
    collection = client.get_or_create_collection("AI-STW") 


      # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
         text += page.extract_text()     
   
        collection.upsert(documents=[text],metadatas=[{"Type":"Orchid data"}], ids=["id9"])
        ef = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
        class DefChromaEF(Embeddings):
         def __init__(self,ef):
            self.ef = ef
         def embed_documents(self,texts):
                return self.ef(texts)
         def embed_query(self, query):
            return self.ef([query])[0]
         
        db = Chroma(client=client, collection_name="AI-STW",embedding_function=DefChromaEF(ef))        
        retriever = db.as_retriever()
        llm = Ollama(model="llama2") 
        qa_chain = RetrievalQA.from_chain_type(llm =  llm,
                                                chain_type="stuff",
                                                retriever= retriever,
                                                return_source_documents=True) 
        llm_response = qa_chain(query)  
        response = process_llm_response(llm_response)
        st.write(response)
  
   

def result_processer(results):
   for result in results:
     print(result.page)
     return result.page_content

def process_llm_response(llm_response):
    return(llm_response['result'])
    # print('\n\nSources:')
    # for source in llm_response["source_documents"]:
    #     print(source.metadata['source'])

def preprocess_text(text):
    # Remove punctuation
    texts = re.sub(r'[^\w\s]', '', text)
    # Tokenize text
    tokens = word_tokenize(texts)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    # Convert tokens to lowercase
    tokens = [word.lower() for word in tokens]
    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text        

if __name__=='__main__':
    main()
