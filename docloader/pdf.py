from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
splitter = TokenTextSplitter(
    chunk_size=10,
    chunk_overlap=2
)




data = PyPDFLoader("docloader/deep-learning-and-neural-networks.pdf")

docs = data.load()
chunks = splitter.split_text(docs[0].page_content)
for i in chunks:
    print(i)
print(docs[0])