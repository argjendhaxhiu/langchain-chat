from dotenv import load_dotenv
load_dotenv()
import sqlite3
from typing import Literal
from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt

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
    if not is_relevant:
        interrupt("Question is outside FAQ scope")
    return {"answer": ""}

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
builder.add_node("chat", call_model)
builder.add_node("summarize", summarize)

builder.add_edge(START, "check_question")
builder.add_edge("check_question", "chat")
builder.add_conditional_edges("chat", should_continue)
builder.add_edge("summarize", END)

graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "user-1"}}

def stream_response(run_config, input_data=None):
    first_token = True
    for token, metadata in graph.stream(
        input_data,
        run_config,
        stream_mode="messages"
    ):
        if metadata["langgraph_node"] == "chat" and token.content:
            if first_token:
                print("Bot: ", end="", flush=True)
                first_token = False
            print(token.content, end="", flush=True)
    print("\n")

def show_history():
    all_states = list(graph.get_state_history(config))
    print(f"\n--- History: {len(all_states)} checkpoints ---")
    for i, state in enumerate(all_states):
        msgs = state.values.get("messages", [])
        summary = "yes" if state.values.get("summary") else "no"
        last = f'"{msgs[-1].content[:40]}"' if msgs else "empty"
        print(f"[{i}] msgs: {len(msgs)} | summary: {summary} | next: {state.next} | last: {last}")
    print("---\n")
    return all_states

SMALL_TALK = {"great", "thanks", "thank you", "ok", "okay", "sure", "got it",
              "perfect", "awesome", "cool", "nice", "good", "cheers", "thx", "ty"}

print("FAQ Chatbot (type 'quit' / 'history' / 'replay <n>' / 'fork <n>')\n")
while True:
    q = input("You: ")
    if q == "quit":
        break

    if q == "history":
        show_history()
        continue

    if q.startswith("replay "):
        index = int(q.split()[1])
        all_states = show_history()
        to_replay = all_states[index]
        if not to_replay.values.get("messages"):
            print("Cannot replay from an empty checkpoint. Pick one with messages > 0.\n")
            continue
        print(f"Replaying from checkpoint [{index}]...\n")
        stream_response(to_replay.config)
        continue

    if q.startswith("fork "):
        index = int(q.split()[1])
        all_states = list(graph.get_state_history(config))
        to_fork = all_states[index]
        last_human = next(m for m in reversed(to_fork.values["messages"]) if isinstance(m, HumanMessage))
        new_q = input(f"Original: '{last_human.content}'\nNew question: ")
        fork_config = graph.update_state(
            to_fork.config,
            {"messages": [HumanMessage(content=new_q, id=last_human.id)]}
        )
        print(f"Forking from checkpoint [{index}] with new question...\n")
        for event in graph.stream(None, fork_config, stream_mode="values"):
            msgs = event.get("messages", [])
            if msgs and hasattr(msgs[-1], "content") and msgs[-1].type == "ai":
                print(f"Bot: {msgs[-1].content}\n")
        continue

    if q.lower().strip() in SMALL_TALK:
        print("Bot: Glad I could help! Anything else you'd like to know?\n")
        continue

    stream_response(config, {"messages": [HumanMessage(q)], "question": q})

    state = graph.get_state(config)
    if state.tasks and state.tasks[0].interrupts:
        correction = input("Question outside FAQ scope. Rephrase your question: ")
        graph.update_state(config, {"messages": [HumanMessage(correction)]})
        stream_response(config)
