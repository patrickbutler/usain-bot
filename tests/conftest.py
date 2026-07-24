from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mock_activities.json"


@pytest.fixture
def fixture_path() -> Path:
    return FIXTURE_PATH
