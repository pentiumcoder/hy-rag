from __future__ import annotations

from pathlib import Path

import pytest

from hyrag.config import Settings
from tests.conftest import REPO_ROOT


def test_env_example_loads_as_shipped() -> None:
    """The README says to copy .env.example to .env; its empty values must mean "default"."""
    s = Settings(_env_file=REPO_ROOT / ".env.example")  # type: ignore[call-arg]
    defaults = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.judge_provider is None and s.generator_model is None and s.index_path is None
    assert s.index_dir == Path("data/index")
    assert s.abstain_threshold == defaults.abstain_threshold == 0.3
    assert s.api_cors_origins == ["http://localhost:5173", "http://127.0.0.1:5173"]


def test_every_setting_is_documented_in_env_example() -> None:
    documented = {
        line.split("=", 1)[0].removeprefix("HYRAG_").lower()
        for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if line.startswith("HYRAG_")
    }
    assert set(Settings.model_fields) - documented == set()


def test_index_path_overrides_the_default(tmp_path: Path) -> None:
    s = Settings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]
    assert s.index_dir == tmp_path / "index"
    moved = s.model_copy(update={"index_path": tmp_path / "elsewhere"})
    assert moved.index_dir == tmp_path / "elsewhere"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("chunk_overlap", 900, "chunk_overlap"),
        ("semantic_min_chars", 5000, "semantic_min_chars"),
        ("dense_weight", 0.0, "cannot both be zero"),
    ],
)
def test_inconsistent_settings_are_rejected(field: str, value: float, message: str) -> None:
    extra = {"sparse_weight": 0.0} if field == "dense_weight" else {}
    with pytest.raises(ValueError, match=message):
        Settings(**{field: value, **extra}, _env_file=None)  # type: ignore[arg-type]
