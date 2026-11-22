"""Shared constants. Values are documented in the README (Design decisions)."""

REGISTER_MAX = 100_000.0  # kWh; a 5-digit register (99999.99) wraps to 0 here
ROLLOVER_HIGH = 0.9 * REGISTER_MAX  # previous reading must be above this ...
ROLLOVER_LOW = 0.1 * REGISTER_MAX  # ... and the next one below this to count as a wrap
MAX_KW = 25.0  # highest plausible average household draw between two readings
TOL_KWH = 0.011  # register resolution is 0.01 kWh; allow one rounding step
LOOKAHEAD = 3  # readings inspected after a suspicious one to tell misread from reset
GAP_S = 2 * 3600  # an interval longer than this is a communication gap
HOUR = 3600

# Quality codes for hourly values (higher = less trustworthy). An hour takes the
# worst code of anything that contributed energy to it.
ACTUAL, ALLOCATED, ESTIMATED = 0, 1, 2
QUALITY_NAMES = {ACTUAL: "actual", ALLOCATED: "allocated", ESTIMATED: "estimated"}
