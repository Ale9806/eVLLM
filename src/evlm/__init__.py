# read version from installed package
from importlib.metadata import version
import sys
from pathlib import Path

FIXTURES_PATH = (Path(__file__).resolve().parent)
sys.path.append(FIXTURES_PATH)


__version__ = version("evlm")