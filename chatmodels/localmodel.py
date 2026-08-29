from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="small-models-for-glam/detr-resnet-50_nls-chapbooks",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
)
model = ChatHuggingFace(llm=llm)
response = model.invoke("What is your aim of creator")
print(response.content)
