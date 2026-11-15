"""ShopWise's LangGraph workflows, one folder each, all with the same anatomy:

    <workflow>/
        state.py    what the graph remembers while it runs (raw data, reducers)
        nodes/      the steps, one node per file (v10): plain functions, state in,
                    updates out; nodes/__init__.py re-exports them all
        graph.py    the wiring: one build_* function that returns it compiled
        __init__.py re-exports the build function, so callers write
                    `from shopwise.llm.workflows.inbox import build_inbox`

    inbox/            v7: triage -> split -> Send to desks -> compose
                      v8: + a conversation, a pause for a manager, memory
    corrective_rag/   v7: search (plain, fused or HyDE) -> grade -> generate,
                      or rewrite and search again
    supervisor/       v9: a supervisor routes each turn to specialists, by code
                      or by model; + handoffs.py
    refund/           v9: v8's refund flow (assess -> approval -> pay) as its
                      own graph, called by the refunds specialist
    specialists/      v9: orders, knowledge (a subgraph), refunds, [ADV] market,
                      one plain-function module each

inbox/ is v8's graph, kept and still runnable: session 12 measures the team
against it (evals/multiagent_compare.py).

Modules beside the folders, because more than one caller uses them:

    memory/           v8: customer memory: store.py (the Store) and nodes/ (recall,
                      remember). v10: was memory.py
    (events.py, the event vocabulary, moved to shopwise/utils/streaming/ in v10:
    it is what the API streams, not part of any one workflow)
    subagents.py      v9: pattern 1, the team as one agent whose tools are
                      specialists (for the comparison)
    skills.py         v9: load_skill, playbooks on demand (skills/<name>/SKILL.md)

New in session 10. Before it, the project's only graph was the one create_agent
builds for you. Same anatomy in every folder, so a reader who has opened one
knows where everything is in the next (project-structure-standard.md, section 1).
"""
