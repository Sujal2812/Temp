from langchain.output_parsers import (
    PydanticOutputParser,
    OutputFixingParser
)

from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI

class MarketingOutput(BaseModel):
    title: str
    slogan: str
    audience: str

llm = ChatOpenAI(temperature=0)

parser = PydanticOutputParser(
    pydantic_object=MarketingOutput
)

fixing_parser = OutputFixingParser.from_llm(
    parser=parser,
    llm=llm
)

bad_output = """
{
"title": "AI Product"
}
"""

fixed = fixing_parser.parse(bad_output)

print(fixed)