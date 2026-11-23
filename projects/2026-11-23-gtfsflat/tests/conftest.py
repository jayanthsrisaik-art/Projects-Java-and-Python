import os
import shutil

import pytest

HERE = os.path.dirname(__file__)
SAMPLES = os.path.join(HERE, "..", "samples")


@pytest.fixture
def sample_feed():
    return os.path.join(SAMPLES, "feed")


@pytest.fixture
def broken_feed():
    return os.path.join(SAMPLES, "broken_feed")


@pytest.fixture
def feed_copy(tmp_path, sample_feed):
    """A writable copy of the sample feed, for tests that mutate files."""
    dst = tmp_path / "feed"
    shutil.copytree(sample_feed, dst)
    return dst
