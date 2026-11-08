"""Read old immutable experiment manifests after the local data reorganization."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MIGRATIONS={
    'data/review':'data/raw/review',
    'data/evaluation':'data/raw/evaluation',
    'data/features':'data/cache/features',
    'data/processed':'data/cache/processed',
    'data/samples':'data/cache/samples',
    'data/splits':'data/metadata/legacy_splits',
}


def resolve_data_path(value):
    path=Path(value)
    if not path.is_absolute():path=ROOT/path
    if path.exists():return path
    try:relative=path.relative_to(ROOT).as_posix()
    except ValueError:return path
    for old,new in MIGRATIONS.items():
        if relative==old or relative.startswith(old+'/'):
            return ROOT/(new+relative[len(old):])
    return path
