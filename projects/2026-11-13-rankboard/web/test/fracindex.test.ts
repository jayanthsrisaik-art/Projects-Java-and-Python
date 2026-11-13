import { test } from "node:test";
import assert from "node:assert/strict";
import { KeyError, keyBetween, keysSequence, validateKey } from "../src/fracindex.js";
import { vectors } from "./vectors.js";

test("matches every Python key vector", () => {
  assert.ok(vectors.between.length > 200);
  for (const [a, b, expected] of vectors.between) {
    assert.equal(keyBetween(a, b), expected, `keyBetween(${a}, ${b})`);
  }
});

test("rejects the same invalid inputs as Python", () => {
  for (const [a, b] of vectors.invalid) {
    assert.throws(() => keyBetween(a, b), KeyError, `keyBetween(${a}, ${b})`);
  }
});

test("sequence matches", () => {
  assert.deepEqual(keysSequence(10), vectors.sequence_10);
});

test("random insert histories stay sorted", () => {
  let seed = 42;
  const rnd = (n: number) => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed % n; };
  const keys: string[] = [];
  for (let i = 0; i < 500; i++) {
    const at = rnd(keys.length + 1);
    const k = keyBetween(at > 0 ? keys[at - 1] : null, at < keys.length ? keys[at] : null);
    validateKey(k);
    keys.splice(at, 0, k);
  }
  assert.deepEqual([...keys].sort(), keys);
  assert.equal(new Set(keys).size, keys.length);
});
