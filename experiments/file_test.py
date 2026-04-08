"""
# From Root
python experiments/file_test.py

# From experiments folder 1
cd experiments
python file_test.py

# From experiments folder 2
cd experiments; python file_test.py; cd ..
"""

from pathlib import Path
print("Path().resolve(), Path().resolve().parents, Path().resolve().parents[0], sep='\\n'")
print(Path().resolve(), Path().resolve().parents, Path().resolve().parents[0], sep='\n')
print("#"*50)
print("__file__, Path(__file__).resolve(), Path(__file__).resolve().parent.parent, sep='\\n'")
print(__file__, Path(__file__).resolve(), Path(__file__).resolve().parent.parent, sep="\n")
