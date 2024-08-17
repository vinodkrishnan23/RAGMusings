import streamlit as st
import os
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from datasets import load_dataset
import pandas as pd
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(layout="wide")

client = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=st.secrets["OPENAI_API_KEY"])

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

mclient = MongoClient(st.secrets["MONGODB_URI"])
mcollection = mclient[st.secrets["DB_NAME"]][st.secrets["TRAVEL_COLLECTION_NAME"]]
ATLAS_VECTOR_SEARCH_INDEX_NAME = st.secrets["ATLAS_VECTOR_SEARCH_INDEX_NAME"]
chat_container, metadata_container = st.columns([1,1])

metadata_field_info = [
    AttributeInfo(
        name="bathrooms",
        description="Number of bathrooms in any accomodation or hotel or stay",
        type="integer or double",
    ),
    AttributeInfo(
        name="bedrooms",
        description="Number of bathrooms in any accomodation or hotel or stay",
        type="integer or double",
    ),
    AttributeInfo(
        name="security_deposit",
        description="It also known as security deposit, It is the Amount of security deposit in any accomodation or hotel or stay",
        type="integer or double",
    )
]
document_content_description = "Brief description of accomodation or hotel or stay"

vectorstore = MongoDBAtlasVectorSearch(
  collection=mcollection,
  embedding=embeddings,
  embedding_key='text_embeddings',
  text_key='description',
  index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
  relevance_score_fn="cosine",
)
llm = OpenAI(temperature=0)

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True
)

#relevant_docs=retriever.invoke(question)
relevant_docs=""

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with chat_container:
    st.subheader("Chat with Travelmate")
    if prompt := st.chat_input("Ask your Travelmate"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            relevant_docs=retriever.invoke(prompt)
            refined_documents=[]
            for doc in relevant_docs:
                document = Document(page_content=doc.page_content, metadata={"source": doc.metadata["listing_url"], "bathrooms": doc.metadata["bathrooms"], "bedrooms": doc.metadata["bedrooms"], "security_deposit": doc.metadata["security_deposit"]})
                refined_documents.append(document)
            prompt = PromptTemplate(
            input_variables = ["query","context"],
            template="""You are a friendly travel assistant and you suggest travellers accomodations in a cheerful manner
            Answer the question from a traveller:{query} by searching the following relevant documents :{context} and provide the best accomodation based on the requirements. Format in a readable fashion, 
            if you don't get any relevant documents from context then apologise and ONLY say you dont know or you dont have the answer""")
            document_chain = create_stuff_documents_chain(llm, prompt)
            #response = st.write_stream(document_chain.invoke({"query":prompt, "context":refined_documents}))
            response = st.write_stream(document_chain.stream({"query":prompt, "context":refined_documents}))
        st.session_state.messages.append({"role": "assistant", "content": response})

with metadata_container:
    if relevant_docs:
        st.subheader("Chunk Information")
        data = []
        for doc in relevant_docs:
            data.append({
                "source": doc.metadata['listing_url'],
                "bathrooms": doc.metadata['bathrooms'],
                "security deposit": doc.metadata['security_deposit'],
                "bedrooms": doc.metadata['bedrooms'],
                "name": doc.metadata['name'],
                "Description": doc.page_content
            })
        st.write(data)