"""The manager's side: what is waiting for a decision, and sending one. New in v10.

    pending()                       every paused refund, across every thread
    resume(thread_id, decision)     answer one; the thread runs on and replies

WHO MAY RESUME
    In v9 the chat CLI resumed with role="manager" when you typed /approve. Here
    the resume endpoint does the same: the run gets a manager's context, built in
    this service and never taken from the request body. Until session 14 adds API
    keys, though, anyone who can reach the endpoint IS a manager. That is the
    limitation this version names out loud.

WHEN A THREAD HAS TWO REQUESTS
    A message about two orders can pause on two refunds at once. Each pause has
    an interrupt id. With one pending, a decision needs no id; with two, it must
    say which it answers, and LangGraph gets Command(resume={id: decision}).
"""
from __future__ import annotations

from langgraph.types import Command

from shopwise.api.v1.schemas.approvals import ApprovalList, Decision, PendingApproval
from shopwise.api.v1.schemas.messages import ApprovalRequest, MessageOut
from shopwise.core.exceptions import InvalidDecision, ThreadNotPaused
from shopwise.services.chat_service import ChatService
from shopwise.utils.streaming.events import paused_on


class ApprovalService:
    def __init__(self, chat: ChatService) -> None:
        self.chat = chat
        self.threads = chat.threads

    def pending(self) -> ApprovalList:
        items = []
        for row in self.threads.threads.all():
            for pending in paused_on(self.chat.graph, row.id):
                items.append(PendingApproval(
                    thread_id=row.id, customer_email=row.customer_email,
                    request=ApprovalRequest(interrupt_id=pending.id, **pending.value)))
        return ApprovalList(items=items)

    def resume(self, thread_id: str, decision: Decision) -> MessageOut:
        row = self.threads.row(thread_id)                          # 404
        waiting = paused_on(self.chat.graph, thread_id)
        if not waiting:
            raise ThreadNotPaused()                                # 409
        ids = [p.id for p in waiting]
        value = decision.as_resume_value()
        if decision.interrupt_id:
            if decision.interrupt_id not in ids:
                raise InvalidDecision(f"interrupt_id must be one of: {', '.join(ids)}")
            command = Command(resume={decision.interrupt_id: value})
        elif len(ids) > 1:
            raise InvalidDecision(f"This thread has {len(ids)} pending requests: "
                                  f"send interrupt_id, one of {', '.join(ids)}")
        else:
            command = Command(resume=value)

        manager = self.threads.customer_context(row.customer_email, role="manager")
        self.threads.claim(thread_id)                              # 409 if busy
        return self.chat.collect(thread_id, self.chat.run_turn(thread_id, command, manager))
