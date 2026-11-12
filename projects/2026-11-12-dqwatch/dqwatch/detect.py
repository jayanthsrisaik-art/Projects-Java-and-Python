"""Sequential, look-ahead-free anomaly detection over daily metric series.

For day i the detector only reads days < i, and only the days it previously judged
"clean". The whole history is replayed in date order every run, so the result for a date
depends only on metrics up to that date (tested in test_detect.py).

Expected value
  seasonal      additive decomposition (in transformed units, i.e. multiplicative for counts)
                M_j  = mean of x over the centred week j-3..j+3 (all 7 days present and clean)
                S_i  = median over the last `weeks` weeks k of (x_{i-7k} - M_{i-7k})
                       -> weekday offset, independent of the level (needs data up to i-4 only)
                L_i  = median over clean j in the last `drift_days` of (x_j - S_j)
                       -> deseasonalized recent level; follows trends and level shifts
                expected = L_i + S_i
  non-seasonal  expected = median of clean x over the last `drift_days` days
Scale
  sigma = max(1.4826 * MAD(recent clean residuals), sigma_floor, today's sampling noise)
                sampling noise: sqrt(p(1-p)/n) for rates, sd/(mean*sqrt(n)) for log means;
                PSI must additionally exceed the chi-square(k-1) critical value at today's n
Decision
  alert when |x - expected| / sigma > z  AND  |x - expected| >= min_delta  (in allowed direction)
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import median

from .config import Config, DetectorParams, MetricSpec, TableConfig
from .profile import Series
from .stats import chi2_isf, z_tail

ALERT_STATUSES = ("alert", "missing", "level_shift")


@dataclass
class Result:
    table: str
    date: str
    metric: str
    value: float | None
    expected: float | None
    lower: float | None
    upper: float | None
    score: float | None
    status: str  # no_data | warmup | ok | alert | missing | level_shift
    reason: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("value", "expected", "lower", "upper", "score"):
            if d[k] is not None:
                d[k] = round(d[k], 6)
        return d


def _mad_sigma(xs: list[float]) -> float:
    m = median(xs)
    return 1.4826 * median(abs(x - m) for x in xs)


def _fwd(spec: MetricSpec, v: float) -> float:
    return math.log1p(max(v, 0.0)) if spec.transform == "log" else v


def _inv(spec: MetricSpec, x: float) -> float:
    return math.expm1(x) if spec.transform == "log" else x


def psi(today: dict[str, int], ref: dict[str, int]) -> float:
    """Population stability index with +0.5 smoothing so unseen categories stay finite."""
    cats = set(today) | set(ref)
    nt = sum(today.values()) + 0.5 * len(cats)
    nr = sum(ref.values()) + 0.5 * len(cats)
    total = 0.0
    for c in cats:
        p = (today.get(c, 0) + 0.5) / nt
        q = (ref.get(c, 0) + 0.5) / nr
        total += (p - q) * math.log(p / q)
    return total


def detect_series(table: str, metric: str, dates: list[str], values: list[float | None], ns: list[int],
                  spec: MetricSpec, params: DetectorParams,
                  categories: list[dict[str, int]] | None = None,
                  sds: list[float | None] | None = None) -> list[Result]:
    """Score one metric series.

    ns are today's sample sizes; sds the column standard deviations (for sampling='mean');
    for psi metrics pass per-day `categories` (values are then ignored).
    """
    is_psi = categories is not None
    seasonal = spec.seasonal and params.use_seasonal
    sampling = spec.sampling if params.use_sampling else None
    p_tail = z_tail(params.z)
    N = len(dates)
    X: list[float | None] = [None] * N  # transformed values
    clean = [False] * N
    resid: list[float | None] = [None] * N  # residuals of clean, scored points
    out: list[Result] = []
    run: list[int] = []  # consecutive alerts in one direction
    run_sign = 0

    def is_clean(j: int) -> bool:
        return j >= 0 and (clean[j] or not params.exclude_anomalies) and X[j] is not None

    def week_mean(j: int) -> float | None:
        if j - 3 < 0 or not all(is_clean(m) for m in range(j - 3, j + 4)):
            return None
        return sum(X[m] for m in range(j - 3, j + 4)) / 7.0

    def season(i: int) -> float | None:
        """Weekday offset for day i, from complete weeks that ended by day i-4."""
        offs = []
        for k in range(1, params.weeks + 1):
            j = i - 7 * k
            if is_clean(j):
                m = week_mean(j)
                if m is not None:
                    offs.append(X[j] - m)
        return median(offs) if len(offs) >= 2 else None

    for i in range(N):
        d = dates[i]
        raw = values[i]
        psi_crit = None  # smallest PSI that sampling noise alone would reach with prob p_tail
        if is_psi:
            today = categories[i]
            raw = None
            if today:
                ref: dict[str, int] = {}
                used = 0
                for j in range(max(0, i - params.psi_days), i):
                    if categories[j] and (clean[j] or not params.exclude_anomalies):
                        used += 1
                        for c, n in categories[j].items():
                            ref[c] = ref.get(c, 0) + n
                if used >= params.min_residuals:
                    raw = psi(today, ref)
                    # n_eff * PSI ~ chi2(k-1) when nothing changed (PSI ~ chi-square divergence)
                    n_t, n_r = sum(today.values()), sum(ref.values())
                    df = max(len(set(today) | set(ref)) - 1, 1)
                    psi_crit = chi2_isf(p_tail, df) * (1.0 / n_t + 1.0 / n_r)
                else:
                    # not enough reference yet: count the day as clean warm-up history
                    clean[i] = True
                    out.append(Result(table, d, metric, None, None, None, None, None, "warmup"))
                    run, run_sign = [], 0
                    continue

        if raw is None:
            out.append(Result(table, d, metric, None, None, None, None, None, "no_data"))
            run, run_sign = [], 0
            continue

        if metric == "row_count" and raw == 0:
            hist = [values[j] for j in range(max(0, i - 28), i) if is_clean(j)]
            if hist and median(hist) > 0:
                out.append(Result(table, d, metric, 0.0, median(hist), None, None, None, "missing",
                                  "partition is empty or absent"))
                run, run_sign = [], 0
                continue

        x = _fwd(spec, raw)
        X[i] = x

        # ---- expected value -------------------------------------------------------------
        expected = None
        if seasonal:
            s_i = season(i)
            if s_i is not None:
                deseason = []
                for j in range(max(0, i - params.drift_days), i):
                    if is_clean(j):
                        s_j = season(j)
                        if s_j is not None:
                            deseason.append(X[j] - s_j)
                if len(deseason) >= 3:
                    expected = median(deseason) + s_i
        else:
            hist = [X[j] for j in range(max(0, i - params.drift_days), i) if is_clean(j)]
            if len(hist) >= 3:
                expected = median(hist)

        recent = [resid[j] for j in range(max(0, i - params.scale_days), i)
                  if resid[j] is not None and (clean[j] or not params.exclude_anomalies)]
        if expected is None or len(recent) < params.min_residuals:
            clean[i] = True
            if expected is not None:
                resid[i] = x - expected
            out.append(Result(table, d, metric, raw, None if expected is None else _inv(spec, expected),
                              None, None, None, "warmup"))
            run, run_sign = [], 0
            continue

        # ---- scale ----------------------------------------------------------------------
        sigma = max(_mad_sigma(recent), spec.sigma_floor)
        n_i = ns[i] if ns else 0
        if sampling == "binomial" and n_i > 0:
            p = min(max(expected, 1.0 / n_i), 0.5)
            sigma = max(sigma, math.sqrt(p * (1 - p) / n_i))
        elif sampling == "mean" and n_i > 1 and sds and sds[i] is not None:
            se = sds[i] / math.sqrt(n_i)
            if spec.transform == "log":
                se = se / max(abs(raw), 1e-12)  # delta method: sd(log mean) ~ se / mean
            sigma = max(sigma, se)
        delta = x - expected
        score = delta / sigma
        half = max(params.z * sigma, spec.min_delta)
        lower = _inv(spec, expected - half) if spec.direction != "up" else None
        upper = _inv(spec, expected + half) if spec.direction != "down" else None
        noise_ok = True
        if sampling == "chi2" and psi_crit is not None:
            noise_ok = raw > psi_crit
            upper = max(upper, psi_crit) if upper is not None else psi_crit

        direction_ok = (spec.direction == "both" or (spec.direction == "up" and delta > 0)
                        or (spec.direction == "down" and delta < 0))
        is_alert = direction_ok and abs(score) > params.z and abs(delta) >= spec.min_delta and noise_ok

        if not is_alert:
            clean[i] = True
            resid[i] = delta
            run, run_sign = [], 0
            out.append(Result(table, d, metric, raw, _inv(spec, expected), lower, upper, score, "ok"))
            continue

        resid[i] = delta  # only read by later days if this point becomes clean (or exclusion is off)
        sign = 1 if delta > 0 else -1
        if sign == run_sign and run and run[-1] == i - 1:
            run.append(i)
        else:
            run, run_sign = [i], sign
        reason = (f"{'above' if sign > 0 else 'below'} expected {_inv(spec, expected):.4g} "
                  f"(score {score:+.1f})")
        if len(run) >= params.persist:
            # The change has persisted: accept it as the new normal so the baseline can adapt.
            for j in run:
                clean[j] = True
            out.append(Result(table, d, metric, raw, _inv(spec, expected), lower, upper, score, "level_shift",
                              f"level shift accepted after {len(run)} consecutive days; " + reason))
            run, run_sign = [], 0
        else:
            out.append(Result(table, d, metric, raw, _inv(spec, expected), lower, upper, score, "alert", reason))
    return out


def detect_table(series: Series, t: TableConfig, params: DetectorParams) -> list[Result]:
    results: list[Result] = []
    for m in t.metrics():
        spec = t.spec(m)
        cats = series.categories[m.split(":", 1)[1]] if m.startswith("psi:") else None
        results += detect_series(t.name, m, series.dates, series.values[m], series.n[m], spec, params,
                                 cats, series.sd[m])
    return results


def detect_all(conn, cfg: Config, until: str | None = None, params: DetectorParams | None = None) -> list[Result]:
    from .profile import load_series

    params = params or cfg.detector
    out: list[Result] = []
    for t in cfg.tables:
        out += detect_table(load_series(conn, t, until), t, params)
    return out


RESULTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS dq_results (
    table_name TEXT NOT NULL, partition_date TEXT NOT NULL, metric TEXT NOT NULL,
    value REAL, expected REAL, lower REAL, upper REAL, score REAL,
    status TEXT NOT NULL, reason TEXT NOT NULL,
    PRIMARY KEY (table_name, partition_date, metric)
);
"""


def save_results(conn, results: list[Result]) -> None:
    """Replace dq_results atomically (results are a pure function of dq_metrics)."""
    conn.executescript(RESULTS_SCHEMA)
    with conn:
        conn.execute("DELETE FROM dq_results")
        conn.executemany("INSERT INTO dq_results VALUES (?,?,?,?,?,?,?,?,?,?)",
                         [(r.table, r.date, r.metric, r.value, r.expected, r.lower, r.upper, r.score,
                           r.status, r.reason) for r in results])


def alerts(results: list[Result]) -> list[Result]:
    return [r for r in results if r.status in ALERT_STATUSES]
