import os
import sys
import tempfile
from pathlib import Path

# Make the package importable without an editable install.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Point any accidental settings load at a throwaway dir, never the repo.
_tmp = tempfile.mkdtemp(prefix="tgnb-test-")
os.environ.setdefault("DATA_DIR", _tmp)
os.environ.setdefault("DB_PATH", os.path.join(_tmp, "store.db"))
os.environ.setdefault("MEDIA_DIR", os.path.join(_tmp, "media"))
os.environ.setdefault("LOG_LEVEL", "CRITICAL")
