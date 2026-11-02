"""ShopWise's LangGraph workflows, one folder each, all with the same anatomy:

    <workflow>/
        state.py    what the graph remembers while it runs (raw data, reducers)
        nodes.py    the steps: plain functions, state in, updates out
        graph.py    the wiring: one build_* function that returns it compiled
        __init__.py re-exports the build function, so callers write
                    `from workflows.inbox import build_inbox`

    inbox/            v7: triage -> split -> Send to desks -> compose
    corrective_rag/   v7: search (plain, fused or HyDE) -> grade -> generate,
                      or rewrite and search again

New in session 10. Before it, the project's only graph was the one create_agent
builds for you. Same anatomy in every folder, so a reader who has opened one
knows where everything is in the next (project-structure-standard.md, section 1).
"""
