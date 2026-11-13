// Client-side board state: the server's copy (from snapshot + change feed)
// plus optimistic edits that have been sent but not yet acknowledged.
// Pure logic, no DOM, so it runs under node --test.

import { keyBetween } from "./fracindex.js";

export interface Card {
  id: string;
  board_id: string;
  column_id: string;
  title: string;
  description: string;
  rank: string;
  version: number;
}

export interface Column {
  id: string;
  name: string;
  position: number;
}

export interface Snapshot {
  board: { id: string; name: string };
  columns: Column[];
  cards: Card[];
  seq: number;
}

export interface Change {
  seq: number;
  op: "upsert" | "delete";
  card: Card | { id: string; column_id: string };
  reason: string;
}

export interface Placement {
  rank: string;
  resolved: "after" | "before" | "end" | "fallback_end";
}

/** Same rules as rankboard.store.resolve_drop; see its docstring. */
export function resolveDrop(
  column: Array<[string, string]>,
  afterId: string | null,
  beforeId: string | null,
  moving: string | null = null,
): Placement {
  if (moving !== null && (moving === afterId || moving === beforeId)) {
    throw new Error("a card cannot be placed relative to itself");
  }
  const cards = column.filter(([id]) => id !== moving);
  const pos = new Map(cards.map(([id], i) => [id, i]));
  if (afterId !== null && pos.has(afterId)) {
    const i = pos.get(afterId)!;
    const next = i + 1 < cards.length ? cards[i + 1][1] : null;
    return { rank: keyBetween(cards[i][1], next), resolved: "after" };
  }
  if (beforeId !== null && pos.has(beforeId)) {
    const i = pos.get(beforeId)!;
    const prev = i > 0 ? cards[i - 1][1] : null;
    return { rank: keyBetween(prev, cards[i][1]), resolved: "before" };
  }
  const last = cards.length ? cards[cards.length - 1][1] : null;
  const resolved = afterId === null && beforeId === null ? "end" : "fallback_end";
  return { rank: keyBetween(last, null), resolved };
}

type Pending = { token: number; card: Card | null }; // null = optimistic delete

export class BoardState {
  boardId = "";
  name = "";
  columns: Column[] = [];
  cursor = 0;
  private server = new Map<string, Card>();
  private pending = new Map<string, Pending>();
  private nextToken = 1;

  loadSnapshot(s: Snapshot): void {
    this.boardId = s.board.id;
    this.name = s.board.name;
    this.columns = [...s.columns].sort((a, b) => a.position - b.position);
    this.server = new Map(s.cards.map((c) => [c.id, c]));
    this.cursor = s.seq;
  }

  /** Apply a page from GET /changes. Entries at or below the cursor are skipped,
   *  so overlapping or repeated pages are harmless. */
  applyChanges(changes: Change[], cursor: number): void {
    for (const ch of changes) {
      if (ch.seq <= this.cursor) continue;
      if (ch.op === "delete") this.server.delete(ch.card.id);
      else this.upsertServer(ch.card as Card);
    }
    this.cursor = Math.max(this.cursor, cursor);
  }

  /** Accept a card from the server unless we already hold a newer version
   *  (a confirm can arrive before a poll that was generated earlier). */
  upsertServer(card: Card): void {
    const have = this.server.get(card.id);
    if (have && have.version > card.version) return;
    this.server.set(card.id, card);
  }

  /** Cards as the user should see them: server state with pending edits applied. */
  view(): Map<string, Card> {
    const out = new Map(this.server);
    for (const [id, p] of this.pending) {
      if (p.card === null) out.delete(id);
      else out.set(id, p.card);
    }
    return out;
  }

  cardsIn(columnId: string): Card[] {
    return [...this.view().values()]
      .filter((c) => c.column_id === columnId)
      .sort((a, b) => (a.rank < b.rank ? -1 : a.rank > b.rank ? 1 : a.id < b.id ? -1 : 1));
  }

  get(id: string): Card | undefined {
    return this.view().get(id);
  }

  private predict(columnId: string, afterId: string | null, beforeId: string | null, moving: string | null) {
    const col = this.cardsIn(columnId).map((c): [string, string] => [c.id, c.rank]);
    return resolveDrop(col, afterId, beforeId, moving);
  }

  /** Optimistically move a card; returns a token for confirm/rollback. */
  moveOptimistic(id: string, columnId: string, afterId: string | null, beforeId: string | null): number {
    const card = this.get(id);
    if (!card) throw new Error(`unknown card ${id}`);
    const { rank } = this.predict(columnId, afterId, beforeId, id);
    const token = this.nextToken++;
    this.pending.set(id, { token, card: { ...card, column_id: columnId, rank, version: card.version + 1 } });
    return token;
  }

  createOptimistic(id: string, columnId: string, title: string, afterId: string | null = null,
                   beforeId: string | null = null): number {
    const { rank } = this.predict(columnId, afterId, beforeId, null);
    const token = this.nextToken++;
    this.pending.set(id, {
      token,
      card: { id, board_id: this.boardId, column_id: columnId, title, description: "", rank, version: 1 },
    });
    return token;
  }

  editOptimistic(id: string, title: string): number {
    const card = this.get(id);
    if (!card) throw new Error(`unknown card ${id}`);
    const token = this.nextToken++;
    this.pending.set(id, { token, card: { ...card, title, version: card.version + 1 } });
    return token;
  }

  deleteOptimistic(id: string): number {
    const token = this.nextToken++;
    this.pending.set(id, { token, card: null });
    return token;
  }

  /** Server accepted the operation. Only clears the overlay if no newer
   *  optimistic operation on the same card has been made since. */
  confirm(id: string, token: number, serverCard: Card | null): void {
    if (serverCard) this.upsertServer(serverCard);
    else this.server.delete(id);
    if (this.pending.get(id)?.token === token) this.pending.delete(id);
  }

  /** Server rejected the operation: drop the overlay, optionally adopting the
   *  current card the server sent back (412 responses include it). */
  rollback(id: string, token: number, current?: Card): void {
    if (current) this.upsertServer(current);
    if (this.pending.get(id)?.token === token) this.pending.delete(id);
  }

  pendingCount(): number {
    return this.pending.size;
  }
}
