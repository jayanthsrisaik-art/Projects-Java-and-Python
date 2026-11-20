import { test } from "node:test";
import assert from "node:assert/strict";
import { clockOffsetMs, explain, formatCountdown, formatPrice, remainingMs, stockBar, stockLabel } from "../src/hold.js";

test("clock offset uses the round-trip midpoint", () => {
  // browser clock is 90 s behind; request took 200 ms
  const sent = 1_000_000;
  const received = 1_000_200;
  const serverS = (1_000_100 + 90_000) / 1000;
  assert.equal(clockOffsetMs(serverS, sent, received), 90_000);
});

test("countdown follows the server clock, not the browser clock", () => {
  const hold = { expires_at: 2_000 }; // server seconds
  const browserNow = 1_880_000 - 90_000; // browser is 90 s behind the server
  assert.equal(remainingMs(hold, 90_000, browserNow), 120_000);
  assert.equal(remainingMs(hold, 0, browserNow), 210_000); // what a naive countdown would show
  assert.equal(remainingMs(hold, 90_000, browserNow + 500_000), 0); // never negative
});

test("formatCountdown rounds up partial seconds", () => {
  assert.equal(formatCountdown(120_000), "2:00");
  assert.equal(formatCountdown(59_001), "1:00");
  assert.equal(formatCountdown(9_000), "0:09");
  assert.equal(formatCountdown(0), "0:00");
});

test("formatPrice", () => {
  assert.equal(formatPrice(12900), "$129.00");
  assert.equal(formatPrice(5), "$0.05");
  assert.equal(formatPrice(-250), "-$2.50");
});

test("stock bar segments sum to 100", () => {
  const w = stockBar({ total: 25, sold: 5, held: 10 });
  assert.deepEqual(w, { sold: 20, held: 40, available: 40 });
  assert.deepEqual(stockBar({ total: 0, sold: 0, held: 0 }), { sold: 0, held: 0, available: 0 });
});

test("stock label explains held-but-not-sold stock", () => {
  assert.equal(stockLabel({ total: 3, sold: 1, held: 1, available: 1 }), "1 of 3 left");
  assert.match(stockLabel({ total: 3, sold: 1, held: 2, available: 0 }), /2 in carts/);
  assert.equal(stockLabel({ total: 3, sold: 3, held: 0, available: 0 }), "sold out");
});

test("explain maps server error codes", () => {
  assert.match(explain("hold_expired"), /expired/);
  assert.match(explain("payment_declined"), /hold is kept/);
  assert.equal(explain("weird"), "Something went wrong.");
  assert.equal(explain(undefined), "Something went wrong.");
});
