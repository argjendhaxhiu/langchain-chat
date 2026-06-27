from dotenv import load_dotenv
load_dotenv()
import sqlite3
from typing import Literal
from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

class InputState(TypedDict):
    question: str
    messages: list

class OutputState(TypedDict):
    answer: str

class State(MessagesState):
    summary: str
    answer: str

FAQ = """
- Returns: You have 30 days to return any item with receipt.
- Shipping: Free shipping on orders over $50. Standard delivery 3-5 days.
- Payment: We accept Visa, Mastercard, PayPal.
- Account: Reset password via Settings > Security > Reset.
"""

llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

def check_question(state: State):
    keywords = ["return", "ship", "pay", "account", "password", "delivery", "visa", "mastercard"]
    last_msg = state["messages"][-1].content.lower()
    is_relevant = any(k in last_msg for k in keywords)
    return {"answer": "" if is_relevant else "OUT_OF_SCOPE"}

def route_question(state: State) -> Literal["chat", "blocked"]:
    if state.get("answer") == "OUT_OF_SCOPE":
        return "blocked"
    return "chat"

def call_model(state: State):
    summary = state.get("summary", "")
    system = f"Answer only from this FAQ:\n{FAQ}"
    if summary:
        system += f"\n\nConversation so far: {summary}"
    messages = [SystemMessage(content=system)] + state["messages"][-4:]
    response = llm.invoke(messages)
    return {"messages": response, "answer": response.content}

def summarize(state: State):
    summary = state.get("summary", "")
    prompt = "Extend this summary with new messages:\n" + summary if summary else "Summarize this conversation:"
    messages = state["messages"] + [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    delete = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete}

def should_continue(state: State) -> Literal["summarize", END]:
    if len(state["messages"]) > 6:
        return "summarize"
    return END

conn = sqlite3.connect("faq_memory.db", check_same_thread=False)
memory = SqliteSaver(conn)

builder = StateGraph(State, input_schema=InputState, output_schema=OutputState)
builder.add_node("check_question", check_question)
builder.add_node("blocked", lambda state: state)
builder.add_node("chat", call_model)
builder.add_node("summarize", summarize)

builder.add_edge(START, "check_question")
builder.add_conditional_edges("check_question", route_question)
builder.add_edge("blocked", END)
builder.add_conditional_edges("chat", should_continue)
builder.add_edge("summarize", END)

graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["blocked"]
)

config = {"configurable": {"thread_id": "user-1"}}

print("FAQ Chatbot (type 'quit' to exit)\n")
while True:
    q = input("You: ")
    if q == "quit":
        break

    first_token = True
    for token, metadata in graph.stream(
        {"messages": [HumanMessage(q)], "question": q},
        config,
        stream_mode="messages"
    ):
        if metadata["langgraph_node"] == "chat" and token.content:
            if first_token:
                print("Bot: ", end="", flush=True)
                first_token = False
            print(token.content, end="", flush=True)

    state = graph.get_state(config)
    if state.next == ("blocked",):
        correction = input("\nQuestion outside FAQ scope. Rephrase your question: ")
        graph.update_state(config, {"messages": [HumanMessage(correction)], "answer": ""})
        first_token = True
        for token, metadata in graph.stream(None, config, stream_mode="messages"):
            if metadata["langgraph_node"] == "chat" and token.content:
                if first_token:
                    print("Bot: ", end="", flush=True)
                    first_token = False
                print(token.content, end="", flush=True)

    print("\n")
