"""summarize: past SUMMARIZE_AFTER messages, older turns become one summary (v8)."""
from __future__ import annotations

from langchain.messages import RemoveMessage
from shopwise.core.config import settings
from shopwise.llm.models import stamp
from shopwise.llm.workflows.inbox.state import InboxState
from shopwise.llm.workflows.inbox.nodes._common import SUMMARIZE_PROMPT, _model, progress


def summarize(state: InboxState) -> dict:
    """Short-term memory inside the graph: keep a long thread inside a budget.

    Past SUMMARIZE_AFTER messages, every message but the newest SUMMARY_KEEP is
    folded into `summary` and deleted with RemoveMessage: add_messages, the
    reducer that appends, also deletes by id. Below the threshold this node
    makes no call and returns nothing.

    The thread's messages are only the customer's and ShopWise's, never tool
    calls (those stay inside the desks' own agents), so deleting from the front
    cannot orphan a tool result from the call that asked for it. A graph whose
    state does hold tool messages must cut at a turn boundary."""
    messages = state["messages"]
    if len(messages) <= settings.SUMMARIZE_AFTER:
        return {}
    old = messages[:-settings.SUMMARY_KEEP]
    progress("summarize", f"Summarizing {len(old)} older messages…")
    lines = "\n".join(f"{'Customer' if m.type == 'human' else 'ShopWise'}: {m.text}" for m in old)
    prompt = SUMMARIZE_PROMPT.format(summary=state.get("summary") or "(none)", messages=lines)
    summary = _model().invoke(prompt, config={"run_name": "summarize",
                                              **stamp("summarize_prompt", SUMMARIZE_PROMPT)})
    return {"summary": summary.text.strip(),
            "messages": [RemoveMessage(id=m.id) for m in old]}
