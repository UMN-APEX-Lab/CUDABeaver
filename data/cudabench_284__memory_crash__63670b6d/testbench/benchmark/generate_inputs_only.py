from pathlib import Path

SPECS = {
    "conv_input.bin": 2048 * 2048,
    "conv_kernel.bin": 24 * 24,
}

Path("data").mkdir(exist_ok=True)
for name, n_floats in SPECS.items():
    with (Path("data") / name).open("wb") as f:
        f.truncate(n_floats * 4)
