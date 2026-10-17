from pathlib import Path

import numpy as np
import pandas as pd
import readfcs
import tifffile

from tests._akoya_data import write_fcs, write_legacy_processor_tree, write_qptiff


def _page(tf: tifffile.TiffFile, index: int) -> tifffile.TiffPage:
    page = tf.pages[index]
    assert isinstance(page, tifffile.TiffPage)
    return page


def test_write_qptiff_is_qpi(tmp_path: Path) -> None:
    p = tmp_path / "x.qptiff"
    data = write_qptiff(p, biomarkers=["DAPI", "CD3", "CD8"], levels=2)
    with tifffile.TiffFile(p) as tf:
        assert tf.is_qpi
        s = tf.series[0]
        assert s.shape == (3, 256, 320)
        assert len(s.levels) == 2
        assert s.levels[1].shape == (3, 128, 160)
        desc = _page(tf, 0).description
        assert "<Biomarker>DAPI</Biomarker>" in desc
        assert 'encoding="utf-16"' in desc
        assert "<Biomarker>CD3</Biomarker>" in _page(tf, 1).description
    assert data.shape == (3, 256, 320)


def test_write_qptiff_rgb_and_stage_position(tmp_path: Path) -> None:
    p = tmp_path / "rgb.qptiff"
    data = write_qptiff(p, biomarkers=[None], rgb=True, stage_position_cm=(1.5, 2.5), label_page=False, levels=1)
    assert data.shape == (256, 320, 3)
    with tifffile.TiffFile(p) as tf:
        page = _page(tf, 0)
        x, y = (page.tags[t].value for t in ("XPosition", "YPosition"))
        assert x[0] / x[1] == 1.5
        assert y[0] / y[1] == 2.5


def test_write_fcs_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "x.fcs"
    df = pd.DataFrame({"cell_id": [1.0, 2.0], "cyc001_ch001": [0.5, 1.5]})
    write_fcs(p, df, marker_names={"cyc001_ch001": "DAPI"})
    fcs = readfcs.ReadFCS(p)
    # readfcs exposes events with positional integer columns; the $PnN names live in ``channels``
    assert list(fcs.channels["PnN"]) == list(df.columns)
    assert fcs.data.shape == df.shape
    assert "DAPI" in set(fcs.channels["PnS"])
    np.testing.assert_allclose(fcs.data.iloc[:, 1].to_numpy(), [0.5, 1.5])


def test_legacy_tree_per_marker(tmp_path: Path) -> None:
    info = write_legacy_processor_tree(tmp_path, markers=["DAPI", "CD3", "CD8", "CD4", "DAPI", "CD20", "CD68", "CD31"])
    assert (tmp_path / "processed_2020-01-01/stitched/reg001/reg001_cyc001_ch001_DAPI.tif").exists()
    assert (tmp_path / "processed_2020-01-01/segm/segm-1/fcs/compensated/reg001_compensated.fcs").exists()
    assert (tmp_path / "processed_2020-01-01/segm/segm-1/fcs/compensated/reg001_compensated.csv").exists()
    assert len(info["markers"]) == 8
    assert info["images"]["reg001"].shape == (8, 128, 160)
    assert len(info["cells"]) == 50


def test_legacy_tree_other_layouts(tmp_path: Path) -> None:
    info = write_legacy_processor_tree(tmp_path / "h", layout="hyperstack")
    with tifffile.TiffFile(tmp_path / "h/processed_2020-01-01/stitched/reg001_montage.tif") as tf:
        assert tf.series[0].shape == (2, 4, 128, 160)
    assert info["images"]["reg001"].shape == (2, 4, 128, 160)
    write_legacy_processor_tree(tmp_path / "f", layout="old_flat", fcs=False)
    assert (tmp_path / "f/exp_reg001.csv").exists()
    assert (tmp_path / "f/exp_reg001.tif").exists()
