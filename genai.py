from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.llms import OCIGenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from fastembed import TextEmbedding


import oracledb
from oraclevs import OracleVS
import oraclevs
import sys
from sentence_transformers import CrossEncoder
from langchain_core.documents import BaseDocumentTransformer, Document
import array
import time
import oci
import streamlit as st
import os

def chunks_to_docs_wrapper(row: dict) -> Document:
    """
    Converts a row from a DataFrame into a Document object suitable for ingestion into Oracle Vector Store.
    - row (dict): A dictionary representing a row of data with keys for 'id', 'link', and 'text'.
    """
    metadata = {'id': str(row['id']), 'link': row['link']}
    return Document(page_content=row['text'], metadata=metadata)

def main():
    load_dotenv()
    username = os.getenv("username")
    password = os.getenv("password")
    dsn = os.getenv("dsn")
    COMPARTMENT_OCID = os.getenv("COMPARTMENT_OCID")


    st.set_page_config(page_title="ask question based on pdf")
    st.info("Oracle OCI GenAI and Oracle AI Vector Search")
    st.header(" Ask your question to get answers based on your pdf " )

    ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
    COMPARTMENT_OCID = os.getenv("COMPARTMENT_OCID")

    EMBED_MODEL="meta.llama-2-70b-chat"

    #upload the file
    pdf = st.file_uploader("upload your pdf",type="pdf")

    try:
       conn23c = oracledb.connect(user=username, password=password, dsn=dsn)
       print("Connection successful!")
    except Exception as e:
       print("Connection failed!")
       #sys.exit(1)


    #extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)

      text=""
      for page in pdf_reader.pages:
        text += page.extract_text()

      # split the text
      text_splitter = CharacterTextSplitter(separator="\n",chunk_size=1000,chunk_overlap=200,length_function=len)
      chunks = text_splitter.split_text(text)

      # Create documents using wrapper
      docs = [chunks_to_docs_wrapper({'id': page_num, 'link': f'Page {page_num}', 'text': text}) for page_num, text in enumerate(chunks)]

      s1time = time.time()

      #create knowledge base in Oracle.
      # Initialize model
      model_4db = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

      # Create vector store
      knowledge_base = OracleVS.from_documents(docs, model_4db, client=conn23c, table_name="MY_DEMO4", distance_strategy=DistanceStrategy.DOT_PRODUCT)
      #knowledge_base = OracleVS.from_documents(docs, model_4db, client=conn23c, table_name="MY_DEMO4", distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
      #knowledge_base = OracleVS.from_documents(docs, model_4db, client=conn23c, table_name="MY_DEMO4", distance_strategy=DistanceStrategy.COSINE)

      s2time =  time.time()

        # Create embeddings
        # Choice 1, Set the OCI GenAI LLM

      # set the LLM to get response
      # set docks to LLM and get answers
      llmOCI = OCIGenAI(
          model_id="meta.llama-2-70b-chat",
          service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
          compartment_id=COMPARTMENT_OCID,
          model_kwargs={"temperature": 0.7, "top_p": 0.75, "max_tokens": 2000},
          auth_type="API_KEY",
      )

      print("The LLM model you will use is meta.llama-2-70b-chat from OCI GenAI Service")

      #Take a moment to celebrate. You have successfully uploaded file, converted to text, split into chunks and embedded in knowledgebase.

      # ask a question
      user_question = st.text_input("Ask a question about your pdf")
      if user_question:
        s3time =  time.time()
        result_chunks=knowledge_base.similarity_search(user_question,5)
        s4time = time.time()
        # Define context and question dictionary
        template = """Answer the question based only on the  following context:
                   {context} Question: {question} """
        prompt = PromptTemplate.from_template(template)
        retriever = knowledge_base.as_retriever()
        context_and_question = {"context": retriever, "question": user_question}

        chain = (
          {"context": retriever, "question": RunnablePassthrough()}
             | prompt
             | llmOCI
             | StrOutputParser()
        )
        response = chain.invoke(user_question)

        print(user_question)
        s5time = time.time()
        st.write(response)
        print( f" vectorixing and inserting chunks duration: {round(s2time - s1time, 1)} sec.")
        st1 = " vectorizing and inserting chunks duration:  "+str(round(s2time - s1time, 1)) +"sec."
        st.caption( ':blue[' +st1+']' )
        print( f" search user_question and return chunks duration: {round(s4time - s3time, 1)} sec.")
        st1 = " :search user_question,vector search  and return chunks duration  "+str(round(s4time - s3time, 1)) +"sec."
        st.caption( ':blue[' +st1+']' )
        print( f" send user_question and ranked chunks to LLM and get answer duration: {round(s5time - s4time, 1)} sec.")
        st1 = "  send user_question and ranked chunks to LLM and get answer duration: "+str(round(s5time - s4time, 1)) +"sec."
        st.caption( ':blue[' +st1+']' )

if __name__ == '__main__':
    main()

