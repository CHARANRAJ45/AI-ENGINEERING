from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
load_dotenv()

model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=1
)

response = model.invoke("write a poem on the virat kholi")

print(response.content)

