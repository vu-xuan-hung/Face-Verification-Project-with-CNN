"""Re-run official PyTorch versus ONNX parity without downloading or training."""

import json
import runpy
from pathlib import Path

if __name__ == "__main__":
    setup = runpy.run_path(str(Path(__file__).with_name("setup-minifasnet.py")))
    print(json.dumps(setup["validate"](), indent=2))
