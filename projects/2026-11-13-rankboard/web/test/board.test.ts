import { test } from "node:test";
import assert from "node:assert/strict";
import { BoardState, resolveDrop, type Card, type Snapshot } from "../src/board.js";
import { vectors } from "./vectors.js";

test("resolveDrop matches Python placements", () => {
  assert.ok(vectors.placements.length > 40);
  for (const p of vectors.placements) {
    const got = resolveDrop(p.column, p.after, p.before, p.moving);
    assert.deepEqual(got, { rank: p.rank, resolved: p.resolved }, JSON.stringify(p));
  }
});

test("resolveDrop refuses self-anchors", () => {
  assert.throws(() => resolveDrop([["a", "a0"]], "a", null, "a"));
});

const card = (id: string, rank: string, column_id = "todo", version = 1): Card =>
  ({ id, board_id: "b", column_id, title: id, description: "", rank, version });

function snapshot(cards: Card[], seq = 10): Snapshot {
  return {
    board: { id: "b", name: "B" },
    columns: [{ id: "done", name: "Done", position: 1 }, { id: "todo", name: "Todo", position: 0 }],
    cards, seq,
  };
}

test("columns sorted by position and cards by rank", () => {
  const s = new BoardState();
  s.loadSnapshot(snapshot([card("b", "a1"), card("a", "a0")]));
  assert.deepEqual(s.columns.map((c) => c.id), ["todo", "done"]);
  assert.deepEqual(s.cardsIn("todo").map((c) => c.id), ["a", "b"]);
});

test("optimistic move predicts the server rank, then confirms", () => {
  const s = new BoardState();
  s.loadSnapshot(snapshot([card("a", "a0"), card("b", "a1"), card("c", "a2")]));
  const t = s.moveOptimistic("c", "todo", "a", "b");
  assert.deepEqual(s.cardsIn("todo").map((c) => c.id), ["a", "c", "b"]);
  assert.equal(s.get("c")!.rank, "a0V"); // same key the Python server computes
  s.confirm("c", t, card("c", "a0V", "todo", 2));
  assert.equal(s.pendingCount(), 0);
  assert.deepEqual(s.cardsIn("todo").map((c) => c.id), ["a", "c", "b"]);
});

test("rollback restores the server position", () => {
  const s = new BoardState();
  s.loadSnapshot(snapshot([card("a", "a0"), card("b", "a1")]));
  const t = s.moveOptimistic("a", "done", null, null);
  assert.deepEqual(s.cardsIn("done").map((c) => c.id), ["a"]);
  s.rollback("a", t);
  assert.deepEqual(s.cardsIn("todo").map((c) => c.id), ["a", "b"]);
});

test("a stale confirm does not clear a newer optimistic move", () => {
  const s = new BoardState();
  s.loadSnapshot(snapshot([card("a", "a0"), card("b", "a1")]));
  const t1 = s.moveOptimistic("a", "done", null, null);
  s.moveOptimistic("a", "todo", "b", null);
  s.confirm("a", t1, card("a", "a0", "done", 2));
  assert.equal(s.pendingCount(), 1);
  assert.deepEqual(s.cardsIn("todo").map((c) => c.id), ["b", "a"]);
});

test("feed entries are applied once and older versions never win", () => {
  const s = new BoardState();
  s.loadSnapshot(snapshot([card("a", "a0")], 10));
  s.applyChanges([
    { seq: 9, op: "delete", card: { id: "a", column_id: "todo" }, reason: "delete" }, // already in snapshot
    { seq: 11, op: "upsert", card: card("b", "a1"), reason: "create" },
  ], 11);
  assert.deepEqual(s.cardsIn("todo").map((c) => c.id), ["a", "b"]);
  s.upsertServer(card("b", "a2", "todo", 3));
  s.applyChanges([{ seq: 12, op: "upsert", card: card("b", "a1", "todo", 2), reason: "move" }], 12);
  assert.equal(s.get("b")!.version, 3);
  // a rebalance keeps the version but changes the rank: accepted
  s.applyChanges([{ seq: 13, op: "upsert", card: card("b", "a5", "todo", 3), reason: "rebalance" }], 13);
  assert.equal(s.get("b")!.rank, "a5");
  assert.equal(s.cursor, 13);
});

test("optimistic create and delete", () => {
  const s = new BoardState();
  s.loadSnapshot(snapshot([card("a", "a0")]));
  const t = s.createOptimistic("n", "todo", "new");
  assert.equal(s.get("n")!.rank, "a1");
  s.confirm("n", t, card("n", "a1"));
  const d = s.deleteOptimistic("a");
  assert.deepEqual(s.cardsIn("todo").map((c) => c.id), ["n"]);
  s.rollback("a", d);
  assert.deepEqual(s.cardsIn("todo").map((c) => c.id), ["a", "n"]);
});
