import streamlit as st
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI

st.title("Enterprise Intelligence Chatbot")

llm = ChatOpenAI(temperature=0)

conversation = ConversationChain(llm=llm)

query = st.text_input("Ask question")

if query:

    response = conversation.predict(input=query)

    st.write(response)