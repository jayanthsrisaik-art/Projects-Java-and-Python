"""Monitor configuration: which tables/columns to profile and how each metric is judged."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def check_ident(name: str) -> str:
    """Identifiers are interpolated into SQL, so only plain names are accepted."""
    if not isinstance(name, str) or not _IDENT.match(name):
        raise ValueError(f"invalid SQL identifier: {name!r}")
    return name


@dataclass(frozen=True)
class MetricSpec:
    """How one metric series is scored.

    transform   'log' (log1p, for counts/amounts: multiplicative changes) or 'identity'
    seasonal    compare with the same weekday of previous weeks (plus a recent-drift correction)
    direction   'both' | 'up' | 'down' -- which deviations are incidents
    sampling    sampling-noise model for today's value given today's sample size:
                'binomial' (proportions), 'mean' (standard error of a mean), 'chi2' (PSI), or None
    min_delta   smallest |observed - expected| (in transformed units) worth alerting on
    sigma_floor lower bound on the residual scale (avoids infinite scores on flat series)
    """

    transform: str = "identity"
    seasonal: bool = False
    direction: str = "both"
    sampling: str | None = None
    min_delta: float = 0.0
    sigma_floor: float = 1e-9

    def __post_init__(self):
        if self.transform not in ("log", "identity"):
            raise ValueError(f"bad transform {self.transform!r}")
        if self.direction not in ("both", "up", "down"):
            raise ValueError(f"bad direction {self.direction!r}")
        if self.sampling not in (None, "binomial", "mean", "chi2"):
            raise ValueError(f"bad sampling {self.sampling!r}")


# Defaults per metric family. Keys are the metric-name prefix before ':'.
DEFAULT_SPECS = {
    "row_count": MetricSpec(transform="log", seasonal=True, direction="both", min_delta=0.10, sigma_floor=0.02),
    "dup_rate": MetricSpec(direction="up", sampling="binomial", min_delta=0.002, sigma_floor=1e-4),
    "null_rate": MetricSpec(direction="up", sampling="binomial", min_delta=0.01, sigma_floor=1e-3),
    "mean": MetricSpec(transform="log", direction="both", sampling="mean", min_delta=0.15, sigma_floor=0.01),
    "psi": MetricSpec(direction="up", sampling="chi2", min_delta=0.10, sigma_floor=0.005),
    "freshness_lag_h": MetricSpec(direction="up", min_delta=3.0, sigma_floor=0.25),
}


@dataclass(frozen=True)
class DetectorParams:
    z: float = 4.0  # alert when |score| exceeds this
    weeks: int = 4  # same-weekday history for seasonal metrics
    drift_days: int = 7  # recent window for the seasonal drift correction / non-seasonal level
    scale_days: int = 28  # residual window for the robust scale
    min_residuals: int = 7  # warm-up: need this many clean residuals before scoring
    persist: int = 4  # this many consecutive same-direction alerts = accepted level shift
    psi_days: int = 28  # pooled reference window for category distributions
    # alerted points do not feed later baselines. Off by default: with median/MAD baselines it
    # did not help on the dev seeds and slowed adaptation to genuine level shifts (see README).
    exclude_anomalies: bool = False
    use_sampling: bool = True  # sampling-aware noise floors (binomial / standard error / chi-square)
    use_seasonal: bool = True


@dataclass(frozen=True)
class StaticRules:
    """Hand-written thresholds, used only by the static-rules baseline."""

    row_count_wow: float = 0.30  # |today / same weekday last week - 1|
    null_rate_max: float = 0.01
    dup_rate_max: float = 0.0
    mean_wow: float = 0.50
    psi_max: float = 0.20  # vs the previous day's distribution
    freshness_max_h: float = 6.0


@dataclass(frozen=True)
class TableConfig:
    name: str
    partition_column: str
    key: str | None = None
    loaded_at: str | None = None
    null_checks: tuple[str, ...] = ()
    mean_checks: tuple[str, ...] = ()
    category_checks: tuple[str, ...] = ()
    static_rules: StaticRules = field(default_factory=StaticRules)
    overrides: dict = field(default_factory=dict)  # metric name -> partial MetricSpec dict

    def __post_init__(self):
        for n in (self.name, self.partition_column, self.key, self.loaded_at,
                  *self.null_checks, *self.mean_checks, *self.category_checks):
            if n is not None:
                check_ident(n)

    def metrics(self) -> list[str]:
        out = ["row_count"]
        if self.key:
            out.append("dup_rate")
        out += [f"null_rate:{c}" for c in self.null_checks]
        out += [f"mean:{c}" for c in self.mean_checks]
        out += [f"psi:{c}" for c in self.category_checks]
        if self.loaded_at:
            out.append("freshness_lag_h")
        return out

    def spec(self, metric: str) -> MetricSpec:
        base = DEFAULT_SPECS[metric.split(":", 1)[0]]
        if metric in self.overrides:
            base = replace(base, **self.overrides[metric])
        return base


@dataclass(frozen=True)
class Config:
    tables: tuple[TableConfig, ...]
    detector: DetectorParams = field(default_factory=DetectorParams)

    def table(self, name: str) -> TableConfig:
        for t in self.tables:
            if t.name == name:
                return t
        raise KeyError(name)


def config_from_dict(d: dict) -> Config:
    tables = []
    for t in d["tables"]:
        tables.append(TableConfig(
            name=t["name"],
            partition_column=t["partition_column"],
            key=t.get("key"),
            loaded_at=t.get("loaded_at"),
            null_checks=tuple(t.get("null_checks", ())),
            mean_checks=tuple(t.get("mean_checks", ())),
            category_checks=tuple(t.get("category_checks", ())),
            static_rules=StaticRules(**t.get("static_rules", {})),
            overrides=dict(t.get("overrides", {})),
        ))
    for t in tables:
        for m, o in t.overrides.items():
            if m not in t.metrics():
                raise ValueError(f"{t.name}: override for unknown metric {m!r}")
            t.spec(m)  # validates field names / values
    return Config(tables=tuple(tables), detector=DetectorParams(**d.get("detector", {})))


def load_config(path: str | Path) -> Config:
    return config_from_dict(json.loads(Path(path).read_text()))
