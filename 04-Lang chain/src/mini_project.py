from langchain.agents import Tool, initialize_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI

embedding = HuggingFaceEmbeddings()

vectordb = Chroma(
    persist_directory="../data/chroma_db",
    embedding_function=embedding
)

search = DuckDuckGoSearchRun()

llm = OpenAI(temperature=0)

def hr_query(query):
    docs = vectordb.similarity_search(query, k=2)
    return docs[0].page_content

tools = [
    Tool(
        name="HR Database",
        func=hr_query,
        description="Internal HR knowledge"
    ),
    Tool(
        name="Market Search",
        func=search.run,
        description="Real-time market analysis"
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

query = input("Ask Question: ")

response = agent.run(query)

print("\nResponse:\n")
print(response)