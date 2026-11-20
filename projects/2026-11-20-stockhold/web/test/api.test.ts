import { test } from "node:test";
import assert from "node:assert/strict";
import { ApiClient, newIdempotencyKey, type FetchLike } from "../src/api.js";

interface Call {
  url: string;
  method: string;
  headers: Record<string, string>;
  body: unknown;
}

/**
 * A fake server that behaves like stockhold for POST /api/holds: it remembers
 * Idempotency-Keys and replays. `script` decides what happens to each call:
 * "ok" deliver the response, "lost" commit then throw (response lost),
 * "down" throw before committing, or a number to answer with that status.
 */
function fakeServer(script: Array<"ok" | "lost" | "down" | number>) {
  const calls: Call[] = [];
  const byKey = new Map<string, unknown>();
  let created = 0;
  const fetch: FetchLike = async (url, init) => {
    const headers = (init?.headers ?? {}) as Record<string, string>;
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ url, method: init?.method ?? "GET", headers, body });
    const step = script.shift() ?? "ok";
    if (step === "down") throw new TypeError("fetch failed");
    if (typeof step === "number") {
      return new Response(JSON.stringify({ error: "upstream" }), { status: step });
    }
    const key = headers["Idempotency-Key"];
    let replayed = false;
    let hold = byKey.get(key);
    if (hold) replayed = true;
    else {
      created += 1;
      hold = { id: `h_${created}`, sku: body.sku, qty: body.qty, status: "active", expires_at: 1120, server_time: 1000, order_id: null };
      byKey.set(key, hold);
    }
    if (step === "lost") throw new TypeError("connection reset");
    return new Response(JSON.stringify(hold), {
      status: 201,
      headers: replayed ? { "Idempotent-Replayed": "true" } : {},
    });
  };
  return { fetch, calls, created: () => created };
}

const noSleep = async () => {};

test("a lost response is retried with the same key and replays the original hold", async () => {
  const srv = fakeServer(["lost", "ok"]);
  const api = new ApiClient("ana", { fetch: srv.fetch, sleep: noSleep });
  const r = await api.reserve("VNYL-07", 1, "intent-1234-abcd");
  assert.equal(r.status, 201);
  assert.equal(r.replayed, true);
  assert.equal(r.attempts, 2);
  assert.equal(srv.created(), 1, "only one hold created");
  assert.deepEqual(srv.calls.map((c) => c.headers["Idempotency-Key"]), ["intent-1234-abcd", "intent-1234-abcd"]);
  assert.equal(srv.calls[0].headers["X-Customer-Id"], "ana");
});

test("5xx and 429 are retried with exponential backoff; business errors are not", async () => {
  const sleeps: number[] = [];
  const srv = fakeServer([503, 429, "ok"]);
  const api = new ApiClient("ana", { fetch: srv.fetch, sleep: async (ms) => void sleeps.push(ms), backoffMs: 100 });
  const r = await api.reserve("VNYL-07", 1, "intent-1234-abcd");
  assert.equal(r.status, 201);
  assert.deepEqual(sleeps, [100, 200]);

  const srv2 = fakeServer([409]);
  const api2 = new ApiClient("ana", { fetch: srv2.fetch, sleep: noSleep });
  const r2 = await api2.reserve("VNYL-07", 1, "intent-5678-abcd");
  assert.equal(r2.status, 409);
  assert.equal(srv2.calls.length, 1);
});

test("gives up with a network error after maxAttempts", async () => {
  const srv = fakeServer(["down", "down", "down"]);
  const api = new ApiClient("ana", { fetch: srv.fetch, sleep: noSleep, maxAttempts: 3 });
  const r = await api.reserve("VNYL-07", 1, "intent-1234-abcd");
  assert.equal(r.ok, false);
  assert.equal(r.body.error, "network");
  assert.equal(srv.calls.length, 3);
  assert.equal(srv.created(), 0);
});

test("server_time updates the clock offset", async () => {
  const srv = fakeServer(["ok"]);
  let t = 900_000; // browser ms; server says 1000 s = 1_000_000 ms
  const api = new ApiClient("ana", { fetch: srv.fetch, sleep: noSleep, now: () => (t += 100) });
  await api.reserve("VNYL-07", 1, "intent-1234-abcd");
  // sent at 900_100, received at 900_200 -> midpoint 900_150
  assert.equal(api.offsetMs, 1_000_000 - 900_150);
});

test("new keys are unique and match the server's key pattern", () => {
  const a = newIdempotencyKey();
  const b = newIdempotencyKey();
  assert.notEqual(a, b);
  assert.match(a, /^[A-Za-z0-9_:.-]{8,128}$/);
});
