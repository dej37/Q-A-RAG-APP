from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

def vectorize(docs,persist_directory):
    embeddings = MistralAIEmbeddings(model = 'mistral-embed')
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory = persist_directory )
    return vectorstore

def load_vectorstore(persist_directory):
    embeddings = MistralAIEmbeddings(model='mistral-embed')
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)