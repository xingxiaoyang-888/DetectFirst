import pytest

from defectfirst.commands import aggregate
from defectfirst.config import TrainConfig
from defectfirst.io import write_json


@pytest.mark.parametrize("variant,k", [("head_refit", 5), ("main", 1)])
def test_main_table_rejects_mixed_experimental_variants(tmp_path, variant, k):
    write_json(tmp_path / "result.json", {"variant": variant, "k": k})
    with pytest.raises(ValueError, match="cannot mix"):
        aggregate({"results": ["result.json"]}, tmp_path, tmp_path / "table")


def test_boolean_strings_cannot_enable_test_mode():
    with pytest.raises(ValueError, match="boolean"):
        TrainConfig.from_dict({"model": {"backbone": "tiny_test"}, "test_only": "false"})
