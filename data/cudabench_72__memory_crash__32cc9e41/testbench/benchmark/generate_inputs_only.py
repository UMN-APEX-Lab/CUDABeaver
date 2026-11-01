from pathlib import Path

SPECS = {
    "batchA.bin": 16 * 512 * 256,
    "batchB.bin": 16 * 256 * 128,
}

Path("data").mkdir(exist_ok=True)
for name, n_floats in SPECS.items():
    with (Path("data") / name).open("wb") as f:
        f.truncate(n_floats * 4)
