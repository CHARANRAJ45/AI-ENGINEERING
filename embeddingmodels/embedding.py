from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    dimensions=64
)

text = "What is machine learning?"

vector = embeddings.embed_query(text)

print(vector)
print(len(vector))