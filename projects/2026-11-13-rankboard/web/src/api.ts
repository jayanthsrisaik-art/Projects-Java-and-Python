// Thin typed wrapper over the rankboard HTTP API.
import type { Card, Change, Snapshot } from "./board.js";

export class ApiError extends Error {
  constructor(public status: number, public body: any) {
    super(typeof body?.detail === "string" ? body.detail : `HTTP ${status}`);
  }
}

type Fetch = typeof fetch;

export class Api {
  constructor(private base = "", private fetchImpl: Fetch = fetch.bind(globalThis)) {}

  private async call<T>(method: string, path: string, body?: unknown, headers: Record<string, string> = {}): Promise<T> {
    const res = await this.fetchImpl(this.base + path, {
      method,
      headers: { ...(body !== undefined ? { "Content-Type": "application/json" } : {}), ...headers },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    if (res.status === 204) return undefined as T;
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new ApiError(res.status, data);
    return data as T;
  }

  boards() { return this.call<Array<{ id: string; name: string }>>("GET", "/api/boards"); }
  createBoard(name: string, columns?: string[]) {
    return this.call<Snapshot>("POST", "/api/boards", columns ? { name, columns } : { name });
  }
  snapshot(boardId: string) { return this.call<Snapshot>("GET", `/api/boards/${boardId}`); }
  changes(boardId: string, since: number) {
    return this.call<{ changes: Change[]; cursor: number; has_more: boolean }>(
      "GET", `/api/boards/${boardId}/changes?since=${since}`);
  }
  createCard(boardId: string, card: { id: string; column_id: string; title: string; after_id: string | null; before_id: string | null }) {
    return this.call<{ card: Card; placement: string }>("POST", `/api/boards/${boardId}/cards`, card);
  }
  moveCard(id: string, columnId: string, afterId: string | null, beforeId: string | null) {
    return this.call<{ card: Card; placement: string }>("POST", `/api/cards/${id}/move`,
      { column_id: columnId, after_id: afterId, before_id: beforeId });
  }
  editCard(id: string, version: number, title: string) {
    return this.call<Card>("PATCH", `/api/cards/${id}`, { title }, { "If-Match": `"${version}"` });
  }
  deleteCard(id: string, version: number) {
    return this.call<void>("DELETE", `/api/cards/${id}`, undefined, { "If-Match": `"${version}"` });
  }
}

/** Client-chosen card id: lets a retried create be recognised as the same card. */
export function newCardId(): string {
  const bytes = new Uint8Array(9);
  globalThis.crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}
