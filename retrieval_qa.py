from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

llm = ChatOllama(model="llama2", temperature=0)
index = FAISS.load_local('./db', OllamaEmbeddings(),allow_dangerous_deserialization=True )
qa_prompt = ChatPromptTemplate.from_template('''
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
''')

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | qa_prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": index.as_retriever(), "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)
