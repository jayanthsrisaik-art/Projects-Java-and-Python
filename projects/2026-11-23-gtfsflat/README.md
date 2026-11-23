# gtfsflat: GTFS schedules flattened into stop events that happen at the right instant

A transit agency publishes its timetable as a [GTFS](https://gtfs.org/schedule/reference/) feed.
GTFS stores schedules in compressed form. A trip is a template. `calendar.txt` says which weekdays a
service runs, `calendar_dates.txt` adds and removes single dates, and `frequencies.txt` turns one
template into a trip every N minutes. Analytics, departure boards and on-time-performance joins need
the flat version: one row per scheduled stop event, with a real UTC timestamp.

Hand-written GTFS loaders usually get the flattening wrong in a few specific ways:

| shortcut | what goes wrong |
|---|---|
| ignore `calendar_dates.txt` | holiday timetables and one-off event services disappear or run on the wrong day |
| `25:10:00` read as `01:10` | in GTFS, after-midnight trips belong to the **previous** service day. Wrapping the hour puts them 24 h early |
| times read as wall clock | GTFS times count elapsed time from **noon minus 12 h** on the service day. On DST days that reference is 23:00 or 01:00, not midnight |
| frequencies ignored | headway-based routes run once (the template) instead of every 6 minutes |
| board filtered by `service_date = today` | at 00:40 the 24:40 trip is on *yesterday's* service day and vanishes from the board |

gtfsflat validates a feed with file:line diagnostics, expands it into a SQLite star schema with
set-based SQL, and serves departure-board queries over absolute instants. Every result is checked
against an independent row-at-a-time oracle. Each shortcut above is measured as a baseline.

## Architecture

```
GTFS dir ──► feed.py ──────────────► expand.py (SQLite) ─────────────────► board.py / cli.py
           load + validate          stage cleaned tables                   departures by instant
           Issue(file, line, code)  1 service_date  (calendar ∪ adds − removes, recursive date CTE)
           quarantine bad trips     2 trip_instance (per date; frequency starts via recursive CTE)
           interpolate blank times  3 stop_event    (template + offset, anchored at service_day_ref)
                                      service_day_ref: one "noon − 12h" epoch per date, computed in Python
                                      daily_route_summary view
oracle.py: per-date Python loops + datetime arithmetic (tests' oracle) and the shortcut baselines
synth.py:  seeded synthetic feed (holidays, event-only service, night owl 24:xx–27:xx, frequencies, blanks)
```

**Tables** (`stop_event` is the fact table, keyed by `(instance_id, stop_sequence)`):
`service_day_ref(service_date, ref_epoch)`, `service_date(service_id, service_date)`,
`trip_instance(instance_id, trip_id, service_date, offset_s, frequency_start_s, exact_times)`,
`stop_event(service_date, instance_id, trip_id, route_id, stop_sequence, stop_id, arrival_s, departure_s,
arrival_epoch, departure_epoch, is_first, is_last, interpolated, frequency_estimated)`, `dim_stop`, `dim_route`.
Instance ids look like `R1_WK_004@2026-11-02` or, for frequency trips, `F1_TEMPLATE@2026-11-02+07:12:00`.

## Setup

Python ≥ 3.10, standard library only (`sqlite3`, `zoneinfo`, `csv`). The tests need pytest.
On systems without an IANA tz database (e.g. Windows), also `pip install tzdata`.

```bash
cd projects/2026-11-23-gtfsflat
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. validate (exit 0 clean, 1 row errors, 2 fatal)
python -m gtfsflat validate samples/feed
python -m gtfsflat validate samples/broken_feed        # 9 errors, 1 warning, with file:line

# 2. build service dates [start, end] (idempotent; --strict refuses feeds with errors)
python -m gtfsflat build samples/feed --db sample.db --start 2026-10-30 --end 2026-11-03

# 3. query
python -m gtfsflat board --db sample.db --stop A --at 2026-11-01T00:00 --minutes 120
python -m gtfsflat summary --db sample.db --date 2026-11-01
python -m gtfsflat export --db sample.db --start 2026-10-30 --end 2026-11-03 --out events.csv

# synthetic feed + benchmark
python -m gtfsflat synth /tmp/synth_feed --seed 0
python scripts/bench.py --seeds 5 --queries 2000      # writes results/bench.{md,json}; ~35 s

pytest -q
```

Board output for the sample (expected files are in `samples/expected/`):

```
00:30:00-04:00  OWL  Harbor       trip OWL_LATE (service 2026-10-31)
01:30:00-04:00  OWL  Airport      trip OWL_EARLY (service 2026-11-01)
```

The first row is a `24:30:00` trip from Saturday's service. The second is a `00:30:00` trip on Sunday
2026-11-01, the fall-back day. That day's reference is 01:00 EDT, so `00:30:00` means 01:30 EDT, and the
same trip's `01:30:00` arrival is 01:30 **EST**, one hour later. The trip keeps its 60 minutes of travel time.

`--at` without an offset is read in the agency timezone. Ambiguous fall-back times resolve to the first
occurrence; pass an explicit offset (`2026-11-01T01:30-05:00`) for the second.

## Sample inputs and expected outputs

- `samples/feed/`: 3 routes, 4 trips covering an interpolated stop (shape distance), a `calendar_dates`
  swap (Mon 2026-11-02 runs weekend service), a `24:30:00` night trip, an early trip on the fall-back day
  and an `exact_times=1` frequency trip.
- `samples/broken_feed/`: an inverted calendar range, a duplicate exception, an unknown route, stop and trip,
  a malformed time, a trip that runs backwards in time, a trip with no end time, and a half-specified timepoint.
- `samples/expected/`: `events.csv` (36 events), `build.json`, the board above, `summary_2026-11-01.txt`,
  `validate_broken_feed.txt`. The CLI tests compare byte-for-byte against these files.

## Results (measured, `results/bench.md`)

Synthetic feed: 7 routes, 206 trips, 2,080 stop times. Two windows: **spring** 2026-03-01..14 (DST starts
03-08, event service 03-07) and **fall** 2026-10-25..12-05 (DST ends 11-01, Thanksgiving, event 11-28).
5 seeds each.

**Correctness.** The SQL engine matched the oracle on every event in all 10 runs: 104,730 spring + 306,855
fall events, 0 wrong / missing / extra. The SQL departure board matched the oracle's board on 20,000 / 20,000
random queries. The fall build (85 service dates, 61k events) takes about 0.23 s.

**Shortcut baselines**, as % of events that are not exactly right:

| variant | spring | fall |
|---|---|---|
| ignore `calendar_dates` | 0.29% (300 missing) | 0.41% (1,245 missing, 7,680 extra) |
| wrap hours ≥ 24 | 4.45% wrong time | 4.43% wrong time |
| wall-clock times | 0.24% (254 wrong; 233 on the DST day) | 0.08% (257 wrong; 154 on the DST day) |
| ignore frequencies | 21.20% | 21.46% |
| all four (naive loader) | 26.01% | 26.30% |

The wall-clock errors that fall outside the DST day come from trips on the *previous* service day that
run across the transition (e.g. `27:20:00` on 03-07). Elapsed time and wall-clock time diverge there too.

**Departure board** ("next 3 departures within 60 min", wrong answers). Night is 22:00–04:00 local.

| variant | spring night | spring day | fall night | fall day |
|---|---|---|---|---|
| gtfsflat | 0 / 2,448 | 0 / 7,552 | 0 / 2,543 | 0 / 7,457 |
| board filtered by service_date = today | 15.7% | 0.0% | 14.8% | 0.0% |
| wrap hours ≥ 24 | 3.7% | 0.0% | 2.8% | 0.0% |
| wall-clock times | 1.6% | 0.0% | 0.6% | 0.0% |
| ignore `calendar_dates` | 0.6% | 0.1% | 0.7% | 1.7% |
| ignore frequencies | 0.0% | 9.6% | 0.0% | 9.0% |
| naive loader (all four) | 3.6% | 9.7% | 3.0% | 10.6% |

Most errors are rare per event but land in specific hours: late nights, DST weekends and holidays.
Those are the hours when riders depend most on the board being right.

## Design decisions

- **Set-based SQL, timezone math in one small table.** SQLite has no IANA timezones. gtfsflat computes
  one "noon − 12 h" epoch per service date in Python (`service_day_ref`) and joins it in. All
  per-event work stays in SQL, and the DST logic lives in a single function that is tested on its own.
- **An independent oracle.** `oracle.py` loops per date and per trip and uses `datetime` arithmetic. The
  SQL engine uses recursive CTEs and integer epochs. Two implementations of the same rules that agree on
  400k events are stronger evidence than either alone. The same file holds the shortcut flags, so each
  baseline is "the oracle with one rule broken".
- **Quarantine whole trips, never half-load them.** One bad stop time drops the whole trip, with the
  reason and line number. `--strict` turns any error into a refusal to build.
- **Follow the spec on DST days.** Noon − 12 h is what the GTFS reference specifies. Some producers
  actually write wall-clock times. The `wallclock` baseline shows exactly how much that choice changes.
- **Boards query by instant and check coverage.** `required_service_dates` works out which service
  days can reach the query window: reference instant before the window end, and reference + the feed's
  max time after the window start. If any of those days isn't built, the board raises `CoverageError`
  instead of silently missing the 24:40 trip.
- **Idempotent, range-scoped rebuilds.** `expand` deletes and rebuilds only the requested dates in one
  transaction. Re-running produces identical rows, which a test checks.
- **Interpolation.** Blank times between timepoints are filled proportionally to `shape_dist_traveled`
  when it increases strictly, otherwise by stop count. Filled rows are flagged `interpolated=1`.

## Limitations

- Static GTFS only. No GTFS-Realtime, `transfers.txt`, `blocks`/`block_id` continuity, `stop_times`
  `pickup_type`/`drop_off_type`, `timepoint` column semantics, shapes or fares. The board treats
  every non-final stop as a departure.
- A single `agency_timezone` per feed; `stop_timezone` is not supported.
- `exact_times=0` trips are expanded at their nominal headway and flagged `frequency_estimated=1`. Real
  vehicles on those routes don't run to exact times.
- Feeds are loaded fully into memory. That is fine for city-sized feeds, but not tuned for national ones
  (tens of millions of stop times).
- The benchmark feed is synthetic. Seeds vary stop choice, run times and distances, but not the timetable
  structure, so event counts are identical across seeds. No real agency feed was tested in this run.
- The board's `--at` resolves ambiguous fall-back local times to their first occurrence unless an
  offset is given.
