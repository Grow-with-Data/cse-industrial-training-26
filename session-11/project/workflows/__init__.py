"""ShopWise's LangGraph workflows, one folder each, all with the same anatomy:

    <workflow>/
        state.py    what the graph remembers while it runs (raw data, reducers)
        nodes.py    the steps: plain functions, state in, updates out
        graph.py    the wiring: one build_* function that returns it compiled
        __init__.py re-exports the build function, so callers write
                    `from workflows.inbox import build_inbox`

    inbox/            v7: triage -> split -> Send to desks -> compose
                      v8: + a conversation, a pause for a manager, memory
    corrective_rag/   v7: search (plain, fused or HyDE) -> grade -> generate,
                      or rewrite and search again

Two v8 modules sit beside the folders, because more than one caller uses them:

    memory.py         customer memory: the Store, and the recall / remember nodes
    events.py         the event vocabulary and stream_ticket(): what chat.py,
                      inbox.py and session 13's API are told while a turn runs

New in session 10. Before it, the project's only graph was the one create_agent
builds for you. Same anatomy in every folder, so a reader who has opened one
knows where everything is in the next (project-structure-standard.md, section 1).
"""
