import numpy as np
from PIL import Image

from defectfirst.controls.artifacts import load_binary, load_rgb
from defectfirst.controls.composition import compose
from defectfirst.fixtures import create_fixture
from defectfirst.io import read_jsonl


def test_saved_alpha_and_primitives_reproduce_all_pngs_exactly(tmp_path):
    create_fixture(tmp_path)
    group = read_jsonl(tmp_path / "groups.jsonl")[0]
    primitive = group["primitive"]
    normal = load_rgb(tmp_path / primitive["normal"])
    defect = load_rgb(tmp_path / primitive["defect"])
    g = load_binary(tmp_path / group["files"]["G"])
    valid = load_binary(tmp_path / group["files"]["valid"])
    with Image.open(tmp_path / group["files"]["alpha"]) as image:
        alpha = np.asarray(image).astype(np.float32) / 65535
    carriers = [normal] + [load_rgb(tmp_path / p) for p in primitive["shams"]]
    carriers = [np.where(valid[..., None], c, normal).astype(np.uint8) for c in carriers]
    reconstructed = compose(normal, defect, carriers, g, alpha)
    saved = np.array([[load_rgb(tmp_path / path) for path in state] for state in group["views"]])
    assert np.array_equal(reconstructed, saved)
