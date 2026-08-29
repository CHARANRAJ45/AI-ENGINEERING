from dotenv import load_dotenv


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-5.3-Flash",
    temperature=0.7,
)
model = ChatHuggingFace(llm=llm)
response = model.invoke("What is your name?")
print(response.content)