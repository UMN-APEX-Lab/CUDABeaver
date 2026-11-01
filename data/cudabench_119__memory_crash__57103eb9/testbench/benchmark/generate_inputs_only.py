from pathlib import Path

SPECS = {
    "matA.bin": 262144 * 4096,
    "matB.bin": 4096 * 2048,
}

Path("data").mkdir(exist_ok=True)
for name, n_floats in SPECS.items():
    with (Path("data") / name).open("wb") as f:
        f.truncate(n_floats * 4)
