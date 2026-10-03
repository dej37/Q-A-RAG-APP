from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

TEMPLATE = """Answer the question using only the context below.
Cite the page numbers you used, like [Page 3].
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}"""

def format_docs(docs):
    return "\n\n".join(f"[Page {d.metadata['page'] + 1}]\n{d.page_content}" for d in docs)


def build_chain(retriever , model):

    llm = ChatMistralAI(model = model , temperature = 0 )
    prompt = PromptTemplate.from_template(TEMPLATE)

    chain = ({ "context": retriever | format_docs , "question" : RunnablePassthrough() }
    | prompt
    | llm
    | StrOutputParser())
    return chain