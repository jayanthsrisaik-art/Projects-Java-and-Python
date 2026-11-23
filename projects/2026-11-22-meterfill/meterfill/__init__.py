"""meterfill: smart-meter register readings -> trustworthy hourly usage.

Pipeline: ingest (idempotent) -> clean (duplicates, misreads, rollovers, meter
exchanges) -> intervals -> hourly allocation that conserves the metered energy
across communication gaps, using a learned hourly shape model.
"""

__version__ = "0.1.0"
