"""Send every file the tests generate to a temporary folder (config.NUTRISCAN_OUT) so a test run can never
overwrite real artifacts, processed data or reports. Must run before `nutriscan.config` is imported."""
import os
import tempfile

os.environ["NUTRISCAN_OUT"] = tempfile.mkdtemp(prefix="nutriscan_test_")

import pytest  # noqa: E402

from nutriscan import config  # noqa: E402


@pytest.fixture(scope="session")
def sample_df():
    from nutriscan.data import load_products
    if not config.SAMPLE_CSV.exists():
        pytest.skip("run training/make_sample_data.py first")
    return load_products(config.SAMPLE_CSV)
