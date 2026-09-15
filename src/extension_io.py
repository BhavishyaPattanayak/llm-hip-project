"""New-artifact IO: never replace a completed file."""
import json
from pathlib import Path
from src.data import write_tsv


def new_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write('\n')


def new_table(path, rows):
    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    write_tsv(path,rows)


def require_new(directory):
    directory=Path(directory)
    if directory.exists():
        raise FileExistsError(f'Refusing to replace existing artifacts: {directory}')
    directory.mkdir(parents=True)
    return directory
