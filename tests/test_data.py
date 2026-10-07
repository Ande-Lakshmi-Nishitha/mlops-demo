import pytest
from sklearn.datasets import load_digits


@pytest.fixture(scope="session")
def digits_df():
    return load_digits(as_frame=True).frame


def test_data_schema(digits_df):
    assert digits_df.shape[1] == 65
    assert "target" in digits_df.columns


def test_no_missing_values(digits_df):
    assert not digits_df.isnull().any().any()


def test_labels_are_digits(digits_df):
    assert set(digits_df["target"].unique()) == set(range(10))


def test_pixel_range(digits_df):
    pixels = digits_df.drop(columns="target")
    assert pixels.min().min() >= 0
    assert pixels.max().max() <= 16
