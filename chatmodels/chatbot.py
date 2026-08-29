from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
load_dotenv()

model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=1,max_tokens=1000
)
print("----------------welcome type 0 to exit-----------------")
print("choose the mode you want your responses in :")
print("1. Funny Roast Agent")
print("2. Serious Agent")
print("3. Helpful Agent")



choice = input("Enter your choice (1-3): ")

if choice == "1":
    mode="funny the roast agent based on user input"
elif choice == "2":
    mode="serious agent"
elif choice == "3":
    mode="helpful agent"
else:
    print("Invalid choice. Exiting.")
    exit()

messages=[
        SystemMessage(content=f"You are a {mode}")
        ]

while True:
    user_input=input("user: ")
    if user_input=="0":
        break
    messages.append(HumanMessage(content=user_input))
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("bot: ",response.content)
print(messages)