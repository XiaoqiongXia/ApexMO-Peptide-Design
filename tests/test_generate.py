import pandas as pd
import pytest

from amp_design.generate import allocate_length_distribution


def test_allocate_length_distribution_uses_largest_remainders() -> None:
    frame = pd.DataFrame({"length": [5, 5, 5, 6, 7, 7]})
    allocated = allocate_length_distribution(frame, total=10)
    assert len(allocated) == 10
    assert pd.Series(allocated).value_counts().sort_index().to_dict() == {
        5: 5,
        6: 2,
        7: 3,
    }


def test_allocate_length_distribution_validates_input() -> None:
    with pytest.raises(ValueError, match="positive"):
        allocate_length_distribution(pd.DataFrame({"length": [5]}), total=0)
    with pytest.raises(ValueError, match="length column"):
        allocate_length_distribution(pd.DataFrame({"sequence": ["AAAAA"]}), total=1)
    with pytest.raises(ValueError, match=r"\[5, 64\]"):
        allocate_length_distribution(pd.DataFrame({"length": [4]}), total=1)
