import ast
import importlib
import pkgutil
from pathlib import Path

import grid_cells

PACKAGE_DIR = Path(grid_cells.__file__).parent


def test_every_module_imports():
    for info in pkgutil.iter_modules(grid_cells.__path__):
        importlib.import_module(f"grid_cells.{info.name}")


def test_no_main_blocks_in_library():
    offenders = []
    for path in PACKAGE_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) \
                    and node.left.id == "__name__":
                offenders.append(path.name)
    assert not offenders, f"library modules must not have __main__ blocks: {offenders}"
