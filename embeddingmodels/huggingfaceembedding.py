from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model ="ai-sage/Giga-Embeddings-instruct-480M-0826"
    )
text = "What is machine learning?"

vector = embeddings.embed_query(text)

print(vector)
print(len(vector))