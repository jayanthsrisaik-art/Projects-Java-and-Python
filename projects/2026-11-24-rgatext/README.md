# rgatext: offline-first collaborative text that always converges

**Problem.** Notes, docs and comment boxes now get edited on a laptop, a phone and a tablet,
often offline. The obvious way to sync is to send each edit as "insert `s` at index `i`" or
"delete `n` chars at `i`" and replay it on the other devices. That works until two devices
edit at the same time. Then the indices point into *different* documents, and the copies
diverge with no error. A single network that delivers out of order or twice makes it worse.

`rgatext` implements a **Replicated Growable Array (RGA)** sequence CRDT for plain text, plus
the machinery a real sync layer needs around it: causal delivery buffering, idempotent
receive, version-vector delta sync, cursor anchors and op-log persistence. A fuzzer and an
independent reference model check that every replica ends with identical text, whatever the
order, duplication or batching of delivery.

Python 3.10+ standard library only. Tests use `pytest`.

## Architecture

```
 local edit (pos, text)                      remote ops (any order, maybe duplicated)
        │                                                   │
        ▼                                                   ▼
 Replica.insert / delete ──► ops ──► network ──► Replica.receive
        │   (assign Lamport id, origin = char to the left)  │ validate_op → dedupe by (rep, seq)
        │                                                   │ blocked?  ──► park under its missing
        ▼                                                   ▼               dependency; wake on arrival
                  _apply ─► _integrate (RGA placement in a chunked sequence)
                                       │
               text() / anchor() / resolve() / version() / ops_since(version)
```

| module | role |
|---|---|
| `rgatext/doc.py` | `Replica`: the CRDT, causal buffer, delta sync, anchors, rebuild from log |
| `rgatext/oracle.py` | independent tree-based RGA model used to cross-check results |
| `rgatext/naive.py` | baseline: index-replay sync (what people usually hand-roll) |
| `rgatext/sim.py` | JSON scenario runner (edits, syncs, lossy sends) and a randomized fuzzer |
| `rgatext/cli.py` | `simulate`, `fuzz`, `bench`, `merge` commands |

### The CRDT in one paragraph
Every character gets a unique id `(lamport, replica)` and remembers its **origin**, the
character it was typed after. Typing a run chains each character to the previous one.
To integrate an insert, start right after the origin and skip every node whose id is
**larger**: those are concurrent inserts at the same spot that win the tie, plus their
descendants, which always have even larger Lamport stamps. Insert at the first smaller id.
Deletes only set a tombstone, so a later insert can still reference a deleted neighbour.
Because placement depends only on ids, never on arrival order, all replicas agree.

### Operation wire format (JSON, one per line in logs)
```json
{"kind":"ins","rep":"alice","seq":3,"id":[7,"alice"],"origin":[6,"alice"],"ch":"x"}
{"kind":"del","rep":"bob","seq":1,"target":[7,"alice"]}
```
`rep`/`seq` (gap-free per replica) identify the operation and drive version vectors and
deduplication. `id`/`origin` order characters. `[0, ""]` is the document start.

## Setup
```bash
cd projects/2026-11-24-rgatext
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt      # only pytest; runtime is stdlib
python -m pytest -q
```

## Usage

### Library
```python
from rgatext import Replica

a, b = Replica("laptop"), Replica("phone")
a.insert(0, "Meet at noon.")
b.receive(a.ops_since(b.version()))          # delta sync

a.insert(8, "the cafe ")                     # both edit offline...
b.insert(8, "the park ")
b.delete(0, 5)
b.receive(a.ops_since(b.version()))          # ...then exchange deltas
a.receive(b.ops_since(a.version()))
assert a.text() == b.text() == "at the park the cafe noon."

cursor = a.anchor(3)                         # survives remote edits
a.resolve(cursor)                            # current index of that gap
```

### CLI
```bash
python -m rgatext simulate samples/concurrent_typing.json   # CRDT vs naive, exit 1 if CRDT diverged
python -m rgatext simulate samples/flaky_network.json       # shuffled / duplicated / partial sends
python -m rgatext merge samples/logs/alice.jsonl samples/logs/bob.jsonl [--out merged.jsonl]
python -m rgatext fuzz --runs 200 --replicas 4 --steps 400
python -m rgatext bench --chars 20000
```

Scenario steps: `{"edit": name, "insert": [pos, text]}`, `{"edit": name, "delete": [pos, len]}`,
`{"sync": [a, b]}` (two-way delta exchange), and
`{"send": {"from", "to", "shuffle_seed"?, "duplicate"?, "limit"?}}` (one-way, optionally
shuffled, doubled, or truncated to a prefix to model a dropped connection). Out-of-range
positions are clamped so the same script can drive both engines.

## Sample inputs and expected outputs
| input | expected output |
|---|---|
| `samples/concurrent_typing.json` | `samples/expected/concurrent_typing.json`: CRDT converges on `"at the park the cafe noon."`; naive ends with `"at the park the cafe noon."` vs `"at the pthe cafe ark noon."` |
| `samples/flaky_network.json` | `samples/expected/flaky_network.json`: all three CRDT replicas read `"TODO shopping: oatmilk, bread"`; the naive tablet reads `"TODO shoppoatggs, milk, bread"` |
| `samples/logs/alice.jsonl` + `bob.jsonl` | `samples/expected/merged.txt`: `Agenda: Q3 budget, hiring, roadmap` (either merge order) |

`tests/test_cli.py` checks that the CLI reproduces these files exactly.

## Measured results
Run on this machine (Python 3.11.18), with the commands above:

* **Fuzzing**, `fuzz --runs 200 --replicas 4 --steps 400`: the CRDT converged in **200/200**
  sessions, matched the independent oracle in **200/200**, and left 0 ops stuck in buffers.
  The naive index-replay baseline, on the same schedules, converged in **0/200**.
* **Exhaustive ordering**: all 720 delivery orders of a 6-op, 3-replica concurrent edit give
  the same text (`test_every_delivery_order_yields_the_same_text`).
* **Performance** (`bench`, append-heavy typing with every 10th insert mid-document; the
  remote side receives the ops in *reverse*, so everything is buffered first):

  | chars | local inserts | remote reversed receive |
  |---:|---:|---:|
  | 5,000 | 48 ms | 32 ms |
  | 20,000 | 293–366 ms | 166–238 ms |
  | 50,000 | 1.43 s | 0.66 s |

  The first version used a linked list with linear position lookups and took **15.9 s** for
  20,000 local inserts. The chunked sequence (below) brought that to about 0.3 s.
  Timings vary run to run; rerun `bench` locally.
* **Mutation check**: flipping the RGA tie-break (`>` to `<`) fails 41 of 72 tests. Disabling
  dependency buffering fails 27 of 72.

## Design decisions
* **RGA over OT.** Operational transformation needs a central server or tricky transform
  functions with known correctness pitfalls. RGA needs neither: ops commute once causally
  ready, so any peer-to-peer or store-and-forward topology works.
* **Two identities per op.** `(lamport, replica)` orders characters. `(replica, seq)`
  identifies operations. Gap-free `seq` makes version vectors exact and makes "what do you
  lack?" a cheap comparison. A Lamport-only scheme cannot tell a gap from a duplicate.
* **Causal buffer with indexed wakeups.** A blocked op is parked under exactly one missing
  dependency (previous seq from its replica, its origin, or its delete target). When that
  dependency is applied, only its waiters are retried. A reversed 50k-op stream still
  integrates in linear-ish time, with no rescanning of the whole buffer.
* **Chunked sequence instead of a linked list.** The document is a list of blocks (split at
  128 nodes), each keeping its visible-character count. Position lookups skip whole blocks.
  Placement scans from the origin exactly as in textbook RGA.
* **Independent oracle.** `oracle.py` rebuilds the causal tree and reads it off a pre-order
  walk with siblings sorted descending. It shares no placement code with `Replica`, so
  agreement between them is real evidence.
* **Fail loudly on corruption.** Malformed ops, ids reused across devices, and logs with
  missing dependencies raise `CRDTError` (CLI exit 2) instead of silently producing text.
* **Anchors as character ids.** A cursor is stored as "the gap after char X". If X is
  deleted, it falls back to the nearest visible char on the left, which matches what
  editors do.

## Limitations
* **Tombstones are never collected.** Deleted characters stay in memory and in the log.
  Safe GC needs every replica to acknowledge a causal-stability point, which is not
  implemented.
* **Character granularity.** Each character is one op of about 70 bytes of JSON. Real systems
  compress runs, e.g. `{"id":[7,"a"],"text":"hello"}` with id ranges. Python `str`
  characters are code points, so a grapheme cluster like an emoji with a skin-tone modifier
  is several characters and can be split by concurrent edits.
* **Interleaving anomalies remain.** RGA keeps a *run typed left-to-right* contiguous. Two
  users typing *backwards*, or edits made at the start of each other's runs, can still
  interleave. Fugue/YATA-style origin-right pointers would reduce this.
* **No transport, auth or rich text.** Sync is "call `ops_since` and `receive`". There is no
  networking, access control, formatting marks or undo. Replica ids must be unique per
  device. Reuse is detected only when it actually collides.
* **Index lookups are still O(n/B + B)**, not O(log n). That is fine for documents with tens
  of thousands of characters. A balanced tree with subtree counts would be needed for
  multi-megabyte texts.
* The naive baseline clamps out-of-range positions so it never crashes. A real index-replay
  system might instead error or corrupt data in other ways.
