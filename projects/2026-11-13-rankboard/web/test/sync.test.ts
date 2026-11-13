import { test } from "node:test";
import assert from "node:assert/strict";
import { Api, ApiError } from "../src/api.js";
import { BoardState, type Card } from "../src/board.js";
import { pull } from "../src/sync.js";

const card = (id: string, rank: string): Card =>
  ({ id, board_id: "b", column_id: "todo", title: id, description: "", rank, version: 1 });

function fakeFetch(routes: Record<string, () => { status: number; body: unknown }>, log: string[]) {
  return (async (url: string, init?: RequestInit) => {
    log.push(`${init?.method ?? "GET"} ${url}`);
    const route = routes[url];
    if (!route) return new Response(JSON.stringify({ detail: "not found" }), { status: 404 });
    const { status, body } = route();
    return new Response(status === 204 ? null : JSON.stringify(body), { status });
  }) as typeof fetch;
}

function fresh(): BoardState {
  const s = new BoardState();
  s.loadSnapshot({ board: { id: "b", name: "B" }, columns: [{ id: "todo", name: "Todo", position: 0 }], cards: [], seq: 0 });
  return s;
}

test("pull follows has_more pages", async () => {
  const log: string[] = [];
  const api = new Api("", fakeFetch({
    "/api/boards/b/changes?since=0": () => ({ status: 200, body: { changes: [{ seq: 1, op: "upsert", card: card("a", "a0"), reason: "create" }], cursor: 1, has_more: true } }),
    "/api/boards/b/changes?since=1": () => ({ status: 200, body: { changes: [{ seq: 2, op: "upsert", card: card("b", "a1"), reason: "create" }], cursor: 2, has_more: false } }),
  }, log));
  const s = fresh();
  assert.equal(await pull(api, s), "ok");
  assert.deepEqual(s.cardsIn("todo").map((c) => c.id), ["a", "b"]);
  assert.equal(log.length, 2);
});

test("pull reloads the snapshot on 410 and keeps pending edits", async () => {
  const log: string[] = [];
  const api = new Api("", fakeFetch({
    "/api/boards/b/changes?since=0": () => ({ status: 410, body: { detail: "compacted", min_seq: 5 } }),
    "/api/boards/b": () => ({ status: 200, body: { board: { id: "b", name: "B" }, columns: [{ id: "todo", name: "Todo", position: 0 }], cards: [card("x", "a0")], seq: 9 } }),
  }, log));
  const s = fresh();
  s.createOptimistic("mine", "todo", "mine");
  assert.equal(await pull(api, s), "reloaded");
  assert.equal(s.cursor, 9);
  assert.deepEqual(s.cardsIn("todo").map((c) => c.id).sort(), ["mine", "x"]);
});

test("API errors carry status and server body", async () => {
  const api = new Api("", fakeFetch({
    "/api/cards/a": () => ({ status: 412, body: { detail: "card changed", current: card("a", "a0") } }),
  }, []));
  await assert.rejects(api.editCard("a", 1, "t"), (e: unknown) =>
    e instanceof ApiError && e.status === 412 && e.body.current.id === "a");
});
