"""Root pytest configuration and fixtures."""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def dataset_dir() -> Path:
    p = PROJECT_ROOT / "samples" / "test_dataset"
    if not p.is_dir():
        p = PROJECT_ROOT / "bank_statement_test_dataset"
    return p


@pytest.fixture
def sample_pdf_path(dataset_dir: Path) -> Path:
    return dataset_dir / "text_based" / "01_text_bordered_debit_credit_carry_forward.pdf"

