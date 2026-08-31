from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=0
)
data = TextLoader("docloader/book.txt")

docs=data.load()
chunks = text_splitter.split_documents(docs)

for i in chunks:
    print(i.page_content)
    print()
    print()
    print()