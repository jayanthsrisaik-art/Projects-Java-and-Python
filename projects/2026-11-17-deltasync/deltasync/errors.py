class FormatError(ValueError):
    """A serialized signature or delta is malformed (bad magic, truncated, trailing bytes...)."""


class DeltaError(ValueError):
    """A delta cannot be applied: out-of-range copy, wrong basis, or output hash mismatch."""
