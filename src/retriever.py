def retrieve(vectorstore , k):
    retriever = vectorstore.as_retriever(search_kwargs = {"k": k})
    return retriever