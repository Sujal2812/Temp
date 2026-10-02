from langchain.agents import initialize_agent, Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.llms import OpenAI
from langchain.chains.llm_math.base import LLMMathChain

llm = OpenAI(temperature=0)

search = DuckDuckGoSearchRun()

math_chain = LLMMathChain.from_llm(llm=llm)

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Search internet"
    ),
    Tool(
        name="Calculator",
        func=math_chain.run,
        description="Math calculations"
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

response = agent.run(
    "What is India's GDP growth rate and calculate 15 percent of 25000?"
)

print(response)