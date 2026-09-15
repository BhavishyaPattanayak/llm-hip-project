"""Validated corpus IO and subject-averaged baseline preparation."""
import csv
import math
from collections import defaultdict
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[1] / "data/naturalstories"


def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return int(row["item"]), int(row["zone"])


def read_regions(corpus=CORPUS):
    rows = read_tsv(corpus / "naturalstories_RTS/all_stories.tok")
    rows = [dict(item=int(r["item"]), zone=int(r["zone"]), word=r["word"]) for r in rows]
    rows.sort(key=key)
    if len(rows) != 10256 or len({key(r) for r in rows}) != len(rows):
        raise ValueError("Expected 10,256 unique corpus regions")
    for item in range(1, 11):
        zones = [r["zone"] for r in rows if r["item"] == item]
        if not zones or zones != list(range(1, len(zones) + 1)):
            raise ValueError("Noncontiguous story zones")
    return rows


def validate_surprisals(rows, regions):
    reference = {key(r): r["word"] for r in regions}
    if len(rows) != len(reference) or len({key(r) for r in rows}) != len(rows):
        raise ValueError("Missing or duplicate model rows")
    for r in rows:
        if reference.get(key(r)) != r["word"]:
            raise ValueError(f"Model word mismatch: {key(r)}")
        value = float(r["surprisal"])
        if not (math.isnan(value) if key(r)[1] == 1 else math.isfinite(value) and value >= 0):
            raise ValueError(f"Invalid surprisal: {key(r)}")


def lexical_predictors(corpus, regions):
    # Use .word counts, whose multi-part forms may actually be phrase counts.
    entries = {}
    with (corpus / "freqs/freqs-1.tsv").open() as f:
        for code, order, token, count, denominator in csv.reader(f, delimiter="\t"):
            item, zone, part = code.split(".")
            if part in ("word", "whole"):
                entries[int(item), int(zone), part] = (token, int(count))
    result = {}
    for r in regions:
        k = key(r)
        whole = entries.get((*k, "whole"))
        lexical = entries.get((*k, "word"))
        matches = whole is not None and whole[0].replace(" ", "") == r["word"]
        available = bool(matches and lexical and lexical[0])
        result[k] = dict(word_length=len(r["word"]),
            frequency_form=lexical[0] if lexical else "",
            frequency_text_matches=bool(matches),
            frequency_is_multiword=bool(lexical and " " in lexical[0]),
            frequency_count=lexical[1] if available else "",
            log_frequency=math.log1p(lexical[1]) if available else "")
    return result


def build_analysis(corpus=CORPUS, model_paths=None):
    regions = read_regions(corpus)
    reference = {key(r): r["word"] for r in regions}
    totals = defaultdict(lambda: [0, 0.0])
    seen = set()
    with (corpus / "naturalstories_RTS/processed_RTs.tsv").open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            k = key(r)
            obs = (r["WorkerId"], *k)
            rt = float(r["RT"])
            if reference.get(k) != r["word"] or obs in seen or not math.isfinite(rt) or rt <= 0:
                raise ValueError(f"Invalid/duplicate RT observation: {obs}")
            seen.add(obs)
            totals[k][0] += 1
            totals[k][1] += rt
    if set(totals) != set(reference):
        raise ValueError("RT coverage differs from story tokens")
    predictors = lexical_predictors(corpus, regions)
    models = {}
    for name, path in (model_paths or {}).items():
        rows = read_tsv(path)
        validate_surprisals(rows, regions)
        models[name] = {key(r): float(r["surprisal"]) for r in rows}
    output = []
    previous = None
    columns = ["word_length", "log_frequency"] + [f"{m}_surprisal" for m in models]
    for r in regions:
        k = key(r)
        n, total = totals[k]
        row = dict(r, n_observations=n, mean_RT=total/n, **predictors[k])
        row.update({f"{m}_surprisal": values[k] for m, values in models.items()})
        for col in columns:
            row[f"prev_{col}"] = previous[col] if previous and previous["item"] == r["item"] else ""
        output.append(row)
        previous = row
    report = dict(rows=len(output), participant_observations=len(seen),
                  exclusions=0, models=list(models),
                  frequency_unavailable=sum(r["log_frequency"] == "" for r in output),
                  frequency_text_mismatches=sum(not r["frequency_text_matches"] for r in output))
    return output, report
