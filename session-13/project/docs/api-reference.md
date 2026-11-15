# ShopWise API reference (v1)

The contract a frontend or integration team builds against. **Hand-maintained**: it is not generated
from the OpenAPI schema, so it does not update itself. Changing any endpoint means editing this file in
**three** places — the [Endpoint index](#endpoint-index), the endpoint's own section, and the
[Provider and infrastructure summary](#provider-and-infrastructure-summary) — in the same pull
request. When this file and `src/shopwise` disagree, the code is right and this file is the bug.

- Base URL (local): `http://127.0.0.1:8000`
- Versioned routes live under `/v1`; health checks are unversioned.
- Machine-readable schema: `GET /openapi.json`, committed as `docs/openapi.json`.
- Interactive reference: `/scalar` (also `/docs`, `/redoc`).
- **Auth: none yet.** Anyone who can reach the API can act for any customer and approve any refund.
  Session 14 adds API keys.

## Conventions

| Topic | Rule |
| --- | --- |
| Request bodies | JSON, with `Content-Type: application/json` (a body without it is rejected with 422) |
| Ids | Thread ids are UUIDs minted by the server. Never chosen by a client |
| Times | ISO 8601, UTC (`2026-11-14T15:07:25.978925Z`) |
| Errors | `application/problem+json` (RFC 9457): `type`, `title`, `status`, `detail`, `instance`, `request_id`, and `errors[]` on 422. Branch on `type` |
| Request id | Every response has `X-Request-ID`; send your own to correlate with your logs |
| Pagination | Cursor: pass `next_cursor` back as `?cursor=`; `null` means last page |
| Retries | `POST /v1/threads/{id}/resume` requires `Idempotency-Key` |

## Endpoint index

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | [`/health`](#get-health) | Liveness |
| `GET` | [`/ready`](#get-ready) | Readiness |
| `POST` | [`/v1/threads`](#post-v1threads) | Create a thread |
| `GET` | [`/v1/threads`](#get-v1threads) | List threads |
| `GET` | [`/v1/threads/{thread_id}`](#get-v1threadsthread_id) | Get one thread |
| `GET` | [`/v1/threads/{thread_id}/messages`](#get-v1threadsthread_idmessages) | Thread history |
| `DELETE` | [`/v1/threads/{thread_id}`](#delete-v1threadsthread_id) | Delete a thread |
| `POST` | [`/v1/threads/{thread_id}/messages`](#post-v1threadsthread_idmessages) | Send, wait for the reply |
| `POST` | [`/v1/threads/{thread_id}/messages/stream`](#post-v1threadsthread_idmessagesstream) | Send, stream the reply (SSE) |
| `GET` | [`/v1/approvals`](#get-v1approvals) | Pending refund approvals |
| `POST` | [`/v1/threads/{thread_id}/resume`](#post-v1threadsthread_idresume) | Approve, edit or reject |

## Health

### `GET /health`

Liveness. Always `200 {"status": "ok"}` while the process answers. No model call.

### `GET /ready`

Readiness: graph built, API database answering, knowledge base folder present.

| Status | Body |
| --- | --- |
| 200 | `{"status": "ready", "checks": {"graph": true, "app_db": true, "knowledge_base": true}, "version": "10.0.0"}` |
| 503 | same shape, `"status": "not_ready"`, the failing check `false` |

## Threads

`ThreadOut`: `id` · `customer_email` · `title` (first message, ≤60 chars; `null` until one is sent) ·
`status` (`idle` · `running` · `awaiting_approval`) · `created_at` · `updated_at`.

### `POST /v1/threads`

| | |
| --- | --- |
| Body | `{"customer_email": "ayesha@example.com"}` |
| 201 | `ThreadOut`; `Location: /v1/threads/{id}` |
| 422 | `validation-error` (malformed email) · `customer-not-found` |

### `GET /v1/threads`

| | |
| --- | --- |
| Query | `limit` 1–50 (default 20) · `cursor` (from a previous page) |
| 200 | `{"items": [ThreadOut], "next_cursor": "…" \| null}`, newest activity first |
| 422 | `invalid-cursor` · `validation-error` |

### `GET /v1/threads/{thread_id}`

| | |
| --- | --- |
| 200 | `ThreadOut` |
| 404 | `thread-not-found` |

### `GET /v1/threads/{thread_id}/messages`

| | |
| --- | --- |
| 200 | `{"thread_id", "summary": str \| null, "messages": [{"role": "customer" \| "assistant", "text"}]}` |
| 404 | `thread-not-found` |

Use it to recover a reply after a stream was cut off: the run finishes and is saved either way.

### `DELETE /v1/threads/{thread_id}`

| | |
| --- | --- |
| 204 | Thread and its checkpoints removed (customer memories are kept) |
| 404 | `thread-not-found` |
| 409 | `thread-busy` (a turn is running) |

## Messages

`MessageIn`: `{"text": "…"}`, 1–4000 characters.

`MessageOut`: `thread_id` · `reply` (empty when paused) · `citations[]` (`label`, `source`, `section`) ·
`status` · `interrupt` (`ApprovalRequest` or `null`).

`ApprovalRequest`: `interrupt_id` · `order_id` · `amount` · `max_amount` · `recommendation` · `why` ·
`error` (why the previous decision was refused, or `null`).

### `POST /v1/threads/{thread_id}/messages`

| | |
| --- | --- |
| Body | `MessageIn` |
| 200 | `MessageOut`. A refund request ends with `status: "awaiting_approval"` and an `interrupt` |
| 404 | `thread-not-found` |
| 409 | `thread-busy` · `thread-awaiting-approval` |
| 422 | `validation-error` |
| 502 | `turn-failed` (the model or a dependency failed; `detail` is safe to show) |

### `POST /v1/threads/{thread_id}/messages/stream`

Same body and the same 404/409/422 checks, made **before** the stream opens. Then `200
text/event-stream`, one frame per event:

```
event: token
id: 3
data: {"type":"token","text":"Your order "}
```

| `event` | `data` | When |
| --- | --- | --- |
| `token` | `{"text"}` | a piece of the reply |
| `progress` | `{"node", "text"}` | a status line ("Asking orders…") |
| `tool` | `{"node", "name", "status": "started" \| "finished"}` | a tool call |
| `citation` | `{"label", "source", "section"}` | after the reply, one per source |
| `interrupt` | `{"id", "payload"}` | the turn paused for a manager (last event) |
| `done` | `{"thread_id", "reply"}` | the turn finished (last event) |
| `error` | `{"message"}` | the turn failed (last event); safe to show |

`id` counts up from 1 per stream. Replay from `Last-Event-ID` is **not** supported: after a
disconnect, fetch `GET …/messages`. A `: ping` comment arrives every 15 s on a quiet stream.

## Approvals

### `GET /v1/approvals`

| | |
| --- | --- |
| 200 | `{"items": [{"thread_id", "customer_email", "request": ApprovalRequest}]}` |

### `POST /v1/threads/{thread_id}/resume`

| | |
| --- | --- |
| Header | `Idempotency-Key: <uuid>` — new per decision, the same on every retry of it |
| Body | `{"type": "approve"}` · `{"type": "edit", "amount": 99.0, "reason": "…"}` · `{"type": "reject", "reason": "…"}`; add `interrupt_id` when the thread has two pending requests |
| 200 | `MessageOut` (the reply after the decision). A retry returns the stored response with `Idempotent-Replayed: true` |
| 400 | `idempotency-key-required` |
| 404 | `thread-not-found` |
| 409 | `thread-not-paused` · `thread-busy` · `idempotency-key-in-progress` |
| 422 | `validation-error` · `invalid-decision` · `idempotency-key-reused` |
| 502 | `turn-failed` |

An `edit` above `max_amount` is refused by the graph: the response is `awaiting_approval` again, with
`interrupt.error` saying why.

## Provider and infrastructure summary

| Endpoint | Model calls | Stores touched | Notes |
| --- | --- | --- | --- |
| `GET /health` | none | none | |
| `GET /ready` | none | API DB (ping), Qdrant folder (exists) | |
| `POST /v1/threads` | none | API DB | validates the customer against the fixtures |
| `GET /v1/threads` | none | API DB, checkpoints (status) | |
| `GET /v1/threads/{id}` | none | API DB, checkpoints (status) | |
| `GET /v1/threads/{id}/messages` | none | checkpoints | |
| `DELETE /v1/threads/{id}` | none | API DB, checkpoints | |
| `POST …/messages` | several per turn (OpenRouter chat; Gemini embeddings) | checkpoints, memory store, Qdrant, refund ledger | MCP server as a child process for records |
| `POST …/messages/stream` | same as above | same as above | SSE; run continues if the client leaves |
| `GET /v1/approvals` | none | API DB, checkpoints | |
| `POST …/resume` | several (the turn resumes) | as messages + idempotency keys | the only path that can issue a refund |
