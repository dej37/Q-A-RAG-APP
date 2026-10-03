import streamlit as st
from pipeline import get_chain


@st.cache_resource
def load_chain():
    return get_chain()


chain = load_chain()

st.title("Document Q&A")

question = st.text_input("Ask a question about the document")

if st.button("Ask") and question:
    st.write(chain.invoke(question))