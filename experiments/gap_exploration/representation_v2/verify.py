"""Read-only verification of both original project and prior exploration."""
import hashlib,json,os
from pathlib import Path
from src.extension_io import new_json

ALLOWED=tuple(f'{base}/gap_exploration/representation_v2/' for base in ('src','experiments','tests','results'))


def verify():
    root=Path.cwd();manifest=json.loads((root/'results/gap_exploration/representation_v2/frozen_manifest.json').read_text())
    changed=[]
    for name,record in manifest.items():
        path=root/name
        if record['kind']=='symlink':
            if not path.is_symlink() or os.readlink(path)!=record['target']:changed.append(name)
        elif not path.is_file() or path.is_symlink():changed.append(name)
        else:
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            if h.hexdigest()!=record['sha256']:changed.append(name)
    unexpected=[]
    for directory,dirs,files in os.walk(root,followlinks=False):
        for name in files+[d for d in dirs if (Path(directory)/d).is_symlink()]:
            relative=str((Path(directory)/name).relative_to(root))
            if relative not in manifest and not relative.startswith(ALLOWED):unexpected.append(relative)
    return dict(entries_checked=len(manifest),changed_or_missing=changed,unexpected_new_files=unexpected,
                all_frozen_entries_unchanged=not changed and not unexpected)


if __name__=='__main__':
    report=verify();new_json('results/gap_exploration/representation_v2/final_integrity.json',report)
    print(json.dumps(report,indent=2))
    if not report['all_frozen_entries_unchanged']:raise SystemExit(1)
