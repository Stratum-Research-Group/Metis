"""Root conftest to un-shadow monorepo package directories and ensure clean unified pytest runs."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# If ROOT or cwd is in sys.path, remove it from the front so that subdirectories
# like /home/akarta/Projects/Metis/krisis do not shadow real packages as namespace packages
root_str = str(ROOT)
while root_str in sys.path:
    sys.path.remove(root_str)
while "" in sys.path:
    sys.path.remove("")

PACKAGE_PATHS = [
    ROOT / "kleroterion" / "src",
    ROOT / "dokime" / "src",
    ROOT / "stoma" / "src",
    ROOT / "hestia" / "src",
    ROOT / "dianoia",
    ROOT / "paideia",
    ROOT / "LTHCM",
    ROOT / "kybernetes",
    ROOT / "prosopon",
    ROOT / "phylax",
    ROOT / "krisis",
]

for p in PACKAGE_PATHS:
    p_str = str(p)
    if p_str in sys.path:
        sys.path.remove(p_str)
    sys.path.insert(0, p_str)

# Root at the end for top-level helper scripts / tests
sys.path.append(root_str)

# Evict any namespace packages that were shadowed by root folders without __file__
ALL_PACKAGES = [
    "kleroterion", "dokime", "stoma", "hestia", "dianoia",
    "paideia", "lthcm", "kybernetes", "prosopon", "phylax", "krisis"
]
for pkg in ALL_PACKAGES:
    if not getattr(sys.modules.get(pkg), "__file__", None):
        sys.modules.pop(pkg, None)
