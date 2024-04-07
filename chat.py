import streamlit as st
from langchain_community.chat_models.ollama import ChatOllama
from retrieval_qa import rag_chain_with_source
from rephrase_prompt import rephrased_prompt

st.title("My Jarvis")

client = ChatOllama(temperature=0, model='llama2')

def streamed_answer(stream):
    for item in stream:
        if 'answer' in item:
            yield item['answer']
        if 'context' in item:
            for doc in item['context']:
                yield 'source: ' + doc.metadata['source'] + ' \n\n'


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        contextual_query = rephrased_prompt(st.session_state.messages[-1]["content"], st.session_state.messages[:-2])
        stream = streamed_answer(rag_chain_with_source.stream(contextual_query))
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})