from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List,Optional
from langchain_core.output_parsers import PydanticOutputParser


class Movies(BaseModel):
    summary: str
    cast: Optional[List[str]] = None
    title: Optional[str] = None
    year: Optional[int] = None
    rating: Optional[float] = None

parser = PydanticOutputParser(pydantic_object=Movies)


model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system","""extract movie information from the given paragraph and provide the output in the following format:
        {format_instructions}"""
    ),
    ("human", "{paragraph}")
])


paragraph = input("Enter the cinema paragraph: ")

final_prompt = prompt.invoke({
    "paragraph": paragraph,
    "format_instructions": parser.get_format_instructions()
})

response = model.invoke(final_prompt)

print("\n========== MOVIE INFORMATION ==========\n")
print(response.content)