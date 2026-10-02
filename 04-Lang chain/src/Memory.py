from fastapi import FastAPI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

app = FastAPI()

llm = ChatOpenAI(temperature=0)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=100
)

conversation = ConversationChain(
    llm=llm,
    memory=memory
)

@app.get("/chat")
def chat(query: str):

    response = conversation.predict(input=query)

    return {"response": response}