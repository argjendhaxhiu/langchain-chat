# FAQ Chatbot - LangGraph + Claude

A conversational FAQ chatbot built with LangGraph and Claude (Haiku). Built to explore state management, memory, human-in-the-loop patterns, and time travel with LangGraph.

## What it does

- Answers questions from a hardcoded FAQ (returns, shipping, payments, account)
- Streams responses token by token as Claude generates them
- Remembers conversation across sessions using SQLite
- Summarizes long conversations automatically to keep token costs low
- Pauses when questions are off-topic and asks the user to rephrase
- Lets you browse, replay, and fork past conversation checkpoints
- Validates inputs and guards against prompt injection

## Concepts covered

| Concept | Implementation |
|---------|---------------|
| State schema | `State`, `InputState`, `OutputState` |
| Reducers | `add_messages` via `MessagesState` |
| Multiple schemas | Input/output filtering |
| Message filtering | `state["messages"][-4:]` |
| Summarisation | `summarize` node with `RemoveMessage` |
| External memory | `SqliteSaver` persists to disk |
| Streaming | `stream_mode="messages"` token by token |
| Breakpoints | `interrupt()` for off-topic questions |
| Edit state | `graph.update_state()` on rephrase |
| Time travel | `get_state_history`, replay, fork |

## Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) - agent graph framework
- [Claude Haiku](https://www.anthropic.com/claude) - LLM
- SQLite - persistent memory
- Python 3.11+

## Setup

```bash
git clone https://github.com/argjendhaxhiu/langchain-chat
cd langchain-chat
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add your ANTHROPIC_API_KEY to .env
python chatbot.py
```

## Commands

```
history        - browse all saved checkpoints
replay <n>     - re-run the graph from checkpoint n
fork <n>       - replace the question at checkpoint n and re-run
quit           - exit
```

## Graph flow

```
START → check_question → chat → should_continue → END
              ↓ (off-topic)          ↓ (>6 messages)
          [interrupt]            summarize → END
```

## Security

Includes four layers of protection against prompt injection:
1. Input validation - blocks known injection phrases before hitting the graph
2. Hardened system prompt - explicit instructions Claude cannot be overridden
3. Output validation - flags suspicious response patterns
4. Summary poisoning protection - guards the summarization prompt and redacts injected false facts
