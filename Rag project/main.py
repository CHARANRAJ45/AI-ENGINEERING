from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)


model = ChatMistralAI(
    model="mistral-small-2506"
)
data = TextLoader("/workspaces/AI-ENGINEERING/docloader/book.txt")

docs = data.load()


prompt = ChatPromptTemplate.from_messages([
    (
        "system","""you are a helpful agent summeraize the above documnets and provide the output in the following format:"""
    ),
    ("human", "{docs}")
])

finalprompt = prompt.format_messages(docs=docs)

result=model.invoke(finalprompt)

print(result.content)