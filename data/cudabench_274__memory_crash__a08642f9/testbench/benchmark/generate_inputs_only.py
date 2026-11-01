from pathlib import Path

SPECS = {
    "matA.bin": 147456 * 64,
    "matB.bin": 64 * 256,
}

Path("data").mkdir(exist_ok=True)
for name, n_floats in SPECS.items():
    with (Path("data") / name).open("wb") as f:
        f.truncate(n_floats * 4)
