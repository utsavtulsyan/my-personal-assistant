from langchain import hub
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

llm = ChatOllama(model="llama2", temperature=0)
rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
rephrase_prompt_chain = RunnableSequence(rephrase_prompt | llm | StrOutputParser())

def rephrased_prompt(query, history):
    return rephrase_prompt_chain.invoke({"input": query, "chat_history":history})
