import math
import sys
from pathlib import Path

import pytest

from spatialdata_io.readers.macsima import macsima


# The datasets should be downloaded and placed in the "data" directory;
@pytest.mark.skipif(sys.version_info < (3, 10), reason="Test requires Python 3.10 or higher")
@pytest.mark.parametrize(
    "dataset,expected",
    [
        ("Lung_adc_demo", "{'y': (0, 15460), 'x': (0, 13864)}"),
    ],
)
def test_image_size(dataset: str, expected: str) -> None:
    f = Path("./data") / dataset
    assert f.is_dir()
    sdata = macsima(f)
    from spatialdata import get_extent

    extent: dict[str, tuple[float, float]] = get_extent(sdata, exact=False)
    extent = {ax: (math.floor(extent[ax][0]), math.ceil(extent[ax][1])) for ax in extent}
    assert str(extent) == expected


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Test requires Python 3.10 or higher")
@pytest.mark.parametrize(
    "dataset,expected",
    [
        ("Lung_adc_demo", 116),
    ],
)
def test_total_channels(dataset: str, expected: int) -> None:
    f = Path("./data") / dataset
    assert f.is_dir()
    sdata = macsima(f)

    # get the number of channels
    channels: int = len(sdata.images[dataset]["scale0"]["c"])
    assert channels == expected


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Test requires Python 3.10 or higher")
@pytest.mark.parametrize(
    "dataset,expected",
    [
        ("Lung_adc_demo", ["R0 DAPI", "R1 CD68", "R1 CD163"]),
    ],
)
def test_channel_names(dataset: str, expected: list[str]) -> None:
    f = Path("./data") / dataset
    assert f.is_dir()
    sdata = macsima(f, c_subset=3)

    # get the channel names
    channels: int = sdata.images[dataset]["scale0"]["c"].values
    assert list(channels) == expected
