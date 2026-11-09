"""sniffload: infer encoding, dialect, header and locale-aware column types of messy CSV files and load them."""
from .infer import Inference, Options, infer_bytes, infer_file
from .load import LoadResult, SchemaDriftError, parse, write_clean_csv, write_sqlite

__all__ = ["Inference", "Options", "infer_bytes", "infer_file", "LoadResult", "SchemaDriftError",
           "parse", "write_clean_csv", "write_sqlite"]
__version__ = "0.1.0"
