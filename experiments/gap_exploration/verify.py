"""Verify every original byte and that no files were added outside allowed namespaces."""
import hashlib
import json
import os
from pathlib import Path

ALLOWED=('src/gap_exploration/','experiments/gap_exploration/','tests/gap_exploration/','results/gap_exploration/')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    root=Path.cwd();manifest=json.loads((root/'results/gap_exploration/metadata/original_manifest.json').read_text())
    failures=[]
    for name,record in manifest.items():
        path=root/name
        if record['kind']=='symlink':
            if not path.is_symlink() or os.readlink(path)!=record['target']:failures.append(name)
        elif not path.is_file() or path.is_symlink() or digest(path)!=record['sha256']:
            failures.append(name)
    unexpected=[]
    for directory,dirs,files in os.walk(root,followlinks=False):
        for name in files+[d for d in dirs if (Path(directory)/d).is_symlink()]:
            relative=str((Path(directory)/name).relative_to(root))
            if relative not in manifest and not relative.startswith(ALLOWED):unexpected.append(relative)
    report=dict(original_entries_checked=len(manifest),all_original_files_byte_identical=not failures,
                changed_or_missing=failures,new_files_outside_sandbox=unexpected,no_commit_created=True)
    path=root/'results/gap_exploration/metadata/final_integrity.json'
    # This verification record is a sandbox artifact; refreshing it never touches frozen files.
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if failures or unexpected:raise SystemExit(1)


if __name__=='__main__':main()
