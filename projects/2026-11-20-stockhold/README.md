# stockhold: flash-sale checkout that never oversells

A small full-stack checkout for limited drops (sneakers, signed records, group-buy kits): a
**FastAPI + SQLite** backend and a **TypeScript** storefront. Shoppers reserve stock, get a
countdown, and pay before the hold expires.

The tutorial version (read stock, then insert an order) works in a demo and fails the day the
drop goes live. stockhold handles the four places where that version breaks:

| failure | what goes wrong | what stockhold does |
|---|---|---|
| **concurrent checkouts** | two requests both see the last unit and both get it | the stock check and the hold insert run in one `BEGIN IMMEDIATE` transaction |
| **lost responses / retries / double-clicks** | the server committed but the client never heard, so the retry makes a second hold | `Idempotency-Key` per purchase intent; the key record commits in the same transaction as the hold, and retries replay the stored answer |
| **abandoned carts** | shoppers reserve and leave, and stock looks sold out while nothing sells | holds have a TTL; availability is computed from holds that haven't expired, so stock comes back the moment a hold expires, with no sweeper job |
| **late payment** | a hold expired, the unit was resold, and then the first shopper pays | confirm checks expiry inside the write lock and returns `410 Gone`; a guarded `UPDATE` and a `CHECK (sold <= total)` constraint back this up |

## Measured results

`python -m stockhold.bench --seeds 5` (full output in [`samples/bench_expected.md`](samples/bench_expected.md)).
These numbers come from that run. Experiment A uses real threads and real SQLite locks, so its numbers change from run to run; B and C are seeded and deterministic.

**A. Rush:** 200 shoppers, 32 threads, 50 units. The baseline is the same code with every statement in autocommit (check-then-act).

| strategy | added check→write delay | holds granted (mean) | oversold mean (max) | invariant violations |
|---|---|---|---|---|
| immediate (stockhold) | 0 ms | 50.0 | 0.0 (0) | 0 / 5 runs |
| check-then-act | 0 ms | 61.4 | 11.4 (14) | 5 / 5 runs |
| immediate (stockhold) | 1 ms | 50.0 | 0.0 (0) | 0 / 5 runs |
| check-then-act | 1 ms | 69.0 | 19.0 (23) | 5 / 5 runs |

The delay emulates app-server latency between reading and writing. Serializing has a cost: the
rush took 1.4 s with stockhold against 0.3 s for the baseline, mostly SQLite's busy-handler backoff
under heavy contention. Most of the baseline's speed comes from not waiting for the lock.

**B. Lost responses:** 80 shoppers, 60 units, limit 2. 30% of responses are lost after the
server committed, and clients retry up to 4 times.

| client | served | refused | duplicate holds | stranded units |
|---|---|---|---|---|
| retry with Idempotency-Key | 59.8 | 19.8 | 0.0 | 0.2 |
| retry without key | 45.4 | 34.2 | 11.4 | 14.6 |

Without keys, retries hoard stock. Some shoppers also hit their own per-customer limit with
holds they never saw. The 0.2 stranded units with keys come from shoppers whose 4 attempts all
lost their responses; those holds return when their TTL expires.

**C. Abandonment:** 250 shoppers over 30 min, 100 units. 40% of shoppers reserve and leave, and payers take 5–180 s.

| hold TTL | units sold | shoppers refused | payers whose hold expired first |
|---|---|---|---|
| none | 58.0 | 150.0 | 0.0 |
| 600 s | 92.8 | 94.6 | 0.0 |
| 120 s | 96.2 | 17.6 | 46.6 |

This is a trade-off, not a free win. A short TTL sells the most stock but takes the hold away
from slow payers. In this workload 600 s loses no payers and sells 93 of 100 units.

**Live HTTP check:** with uvicorn running, `python -m stockhold.loadcheck` sent 353 concurrent
requests (300 shoppers, 20% with a duplicate sent at the same moment) for 10 units of `KBD-65`.
Result: 10 holds created, 53 replays, 0 oversold, and the server's invariant check returned `ok`.

## Architecture

```
web/ (TypeScript, no framework)                 stockhold/ (Python)
  api.ts   ApiClient: one key per intent,  ──►   api.py    FastAPI routes, header validation,
           retries only network/5xx/429,           │       Idempotent-Replayed header
           clock offset from server_time           ▼
  hold.ts  countdown, stock bar, messages      store.py  Store: BEGIN IMMEDIATE per write,
  app.ts   DOM rendering, 2 s polling                    computed availability, TTL holds,
                                                          idempotency table, mock payments,
                                                          invariant checker
                                               naive.py  check-then-act baseline (same rules)
                                               bench.py  experiments A/B/C
                                               loadcheck.py  concurrent HTTP check vs a live server
                                               demo.py   scripted scenario → samples/
SQLite (WAL): products(total, sold) · holds(status, expires_at) · orders(hold_id UNIQUE) · idempotency(customer, key)
```

### API

All write calls need `X-Customer-Id`. This is a demo identity header; see Limitations.

| method | path | notes |
|---|---|---|
| GET | `/api/products`, `/api/products/{sku}` | `total, sold, held, available`, plus `server_time` |
| POST | `/api/holds` `{sku, qty}` | requires `Idempotency-Key` (8–128 chars). Returns 201 with the hold, 409 `sold_out` / `limit_exceeded`, 422 `idempotency_key_reused`, or 404 |
| GET | `/api/holds/{id}` | 404 for other customers' holds |
| POST | `/api/holds/{id}/confirm` `{payment_token}` | 201 with the order, 200 with the same order on repeat, 402 declined (the hold is kept), 410 expired, 409 released |
| POST | `/api/holds/{id}/release` | idempotent; 409 if already paid |
| GET | `/api/invariants` | `{ok, problems[]}` |

Test payment tokens: `tok_decline` is declined. Any other `tok_*` token is approved.

## Setup

Requires Python 3.11+ and Node 20+. No paid services and no network calls at runtime.

```bash
cd projects/2026-11-20-stockhold
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
(cd web && npm install && npm run build)

uvicorn --factory stockhold.api:app_factory --port 8000
# open http://127.0.0.1:8000. The catalog in samples/catalog.json is loaded into stockhold.db on first start.
# env: STOCKHOLD_DB=path.db  STOCKHOLD_HOLD_TTL=120
```

## Usage

```bash
python -m pytest -q                                   # 34 Python tests
(cd web && npm test)                                  # 12 TypeScript tests (node:test)
python -m stockhold.demo samples/scenario.json        # must match samples/scenario_expected.txt
python -m stockhold.bench --seeds 5                   # experiments A/B/C (~20 s)
python -m stockhold.loadcheck --url http://127.0.0.1:8000 --sku KBD-65   # needs a running server
```

The scenario replays a whole sale with a simulated clock. Excerpt:

```
t=  1 reserve ana  -> 201 (replayed) id=h_0001 status=active qty=1  # response was lost; client retries with the same key
t=  3 reserve ana  -> 422 error=idempotency_key_reused ...          # same key, different body
t= 20 confirm ben  -> 402 error=payment_declined ...                # declined card keeps the hold
t=126 confirm ben  -> 410 error=hold_expired ...                    # too late
t=134 invariants: ok
```

## Tests

* `tests/test_store.py`: limits, sold-out reporting, replay and key scoping, key reuse with a
  different body, replay of failed outcomes, expiry at exactly `expires_at`, resale before a
  late confirm, idempotent confirm and release, payment decline, customer isolation, the
  schema constraint, and an invariant checker that catches tampering.
* `tests/test_concurrency.py`: real threads. Shows no oversell with or without a widened race
  window, shows that the baseline *does* oversell, 8 concurrent confirms of one hold producing
  one order, and the B/C effects.
* `tests/test_api.py`: the full HTTP flow, the replay header, header validation, expiry,
  release, isolation, and the UI shell.
* `tests/test_samples.py`: the scenario output must match `samples/scenario_expected.txt`
  exactly.
* `web/test/`: retries reuse the key after a lost response, backoff covers only network
  errors/5xx/429, giving up, the clock offset, the countdown against a skewed browser clock,
  and formatting.

## Design decisions

* **Computed availability instead of a `reserved` counter.** A counter must be decremented
  when a hold expires, which needs a sweeper, and a crashed sweeper leaves stock locked.
  Computing availability from `expires_at > now` makes expiry instant and impossible to
  forget. An index on `(sku, status, expires_at)` keeps the sum cheap. Expired rows are marked
  lazily for display only.
* **Lock before checking.** `BEGIN IMMEDIATE` takes SQLite's single write lock before the
  first read, so no other writer can change stock between the check and the insert. On
  Postgres the equivalent is `SELECT … FOR UPDATE` on the product row, or a conditional
  `UPDATE … WHERE available >= qty`.
* **Idempotency keys are scoped per customer and fingerprinted.** A reused key with a
  different body is a client bug, so it gets 422 instead of a silent replay. Business failures
  (`409 sold_out`) are stored and replayed too. A retry of the same intent gets the same
  answer, and a new click is a new intent with a new key. Replays refresh `server_time` so the
  client's countdown stays right.
* **Confirm is naturally idempotent** (hold id → at most one order, `orders.hold_id UNIQUE`),
  so it needs no key. A declined payment keeps the hold so the shopper can try another card.
* **The countdown follows the server clock.** The client estimates an NTP-style offset from
  `server_time` and the round-trip midpoint, so a browser clock that is minutes off does not
  show a shopper time they don't have.
* **Defense in depth.** The guarded `UPDATE … WHERE sold + qty <= total`,
  `CHECK (sold <= total)` and `/api/invariants` are each enough to catch a regression on
  their own.

## Limitations

* **Single node, SQLite.** One write lock serializes every reservation across all SKUs. The
  measured cost is 1.4 s for 200 contended requests in this container. A real drop would use
  Postgres row locks per SKU, or a Redis/queue front door.
* **Payment is a local mock inside the transaction.** A real provider call must not hold the
  database lock. It would authorize outside the transaction with its own idempotency key, then
  capture in a second step, and that needs a reconciliation path this project does not have.
* **Identity is a header.** There is no authentication, so anyone can claim any customer id,
  and per-customer limits are only as strong as identity. Bot defense (queues, CAPTCHAs,
  device fingerprints) is out of scope.
* Idempotency records are never purged. Production would keep them for a fixed window, such as
  24 h.
* The UI polls every 2 s rather than pushing updates. After a reload it restores only active
  holds, and there is no order history page.
* Benchmark A depends on timing and the machine. B and C use simulated shoppers, not real
  traffic.
