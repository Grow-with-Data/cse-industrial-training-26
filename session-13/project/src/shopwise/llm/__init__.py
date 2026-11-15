"""Everything that talks to a model, gathered in one place. New in v10.

    llm/models.py        build_model / make_model: the chat model and its fallback
    llm/middleware.py    the safety rails every agent runs inside (v4), ShopContext (v8)
    llm/tools.py         the records tools: orders, shipments, products, refunds
    llm/llm_data/        the mock backend's fixtures, loaded once
    llm/prompts/         one folder of YAML per feature; loader.get_prompt()
    llm/schemas/         LLM schemas: what a model is asked to return
    llm/kb/              the knowledge base: ingest, store, retrieval, RAG
    llm/workflows/       the LangGraph graphs: graph.py, state.py, nodes/ (one per file)
    llm/skills/          playbooks loaded on demand (SKILL.md)

The API never imports a node or a prompt. It calls services/, which call a
workflow. That keeps the HTTP layer ignorant of LangGraph, and the graph
ignorant of HTTP.
"""
