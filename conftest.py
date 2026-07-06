import os
import sys

# Make the repo root importable so tests can `import utils` / `import livesplat`
# regardless of where pytest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
