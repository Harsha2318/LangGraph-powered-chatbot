import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from typing import TypedDict

# Load API Key securely from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Ensure you're using the correct env variable

if not OPENAI_API_KEY:
    raise ValueError("OpenRouter API key not found! Set it in a .env file.")

# Define the Chat State
class ChatState(TypedDict):
    messages: list[dict]

# Initialize the OpenAI Chat Model with OpenRouter support
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    base_url="https://openrouter.ai/api/v1",  # Ensures OpenRouter is used
    default_headers={
        "HTTP-Referer": "https://your-site.com",  # Replace with your actual site URL
        "X-Title": "YourProjectName"  # Replace with your project's name
    }
)

# Define the chatbot function
def chat_node(state: ChatState) -> ChatState:
    messages = state["messages"]
    response = llm.invoke(messages)  # Ensure messages are formatted correctly
    
    # Append bot response to messages
    messages.append({"role": "assistant", "content": response.content})
    
    return {"messages": messages}

# Create the StateGraph
graph = StateGraph(ChatState)

# Add chatbot node to the graph
graph.add_node("chatbot", chat_node)
graph.set_entry_point("chatbot")

# Compile the chatbot
chatbot = graph.compile()

# Main loop for user interaction
state = {"messages": []}

while True:
    user_input = input("You: ")
    if user_input.lower() in ["bye", "end", "exit"]:
        print("Chat session closed")
        break

    state["messages"].append({"role": "user", "content": user_input})

    state = chatbot.invoke(state)

    print("Bot:", state["messages"][-1]["content"])
