"""Read full local GECO DATA and validate both tokenizers without inference."""
from pathlib import Path
from transformers import AutoTokenizer
from collections import Counter
from src.geco import aggregate,workbook_rows,add_lexical,frequency_lookup,material_diagnostic
from src.surprisal import encode_regions
from src.representations import sha256
from src.extension_io import require_new,new_json,new_table


def main():
    out=Path('results/geco/processed')
    if out.exists():
        raise FileExistsError(out)
    source=Path('data/geco/MonolingualReadingData.xlsx')
    rows,report=aggregate(workbook_rows(source))
    rows=add_lexical(rows,frequency_lookup())
    report['frequency_missing_regions']=sum(r['log_frequency']=='' for r in rows)
    report['frequency_policy']='Local Natural Stories Google Books .word lookup, exact case, outer punctuation stripped, ln(1+count), unmatched missing'
    report['stimulus_source']='MonolingualReadingData.xlsx DATA; EnglishMaterial unused'
    report['source_sha256']=sha256(source)
    report['frequency_sha256']=sha256(Path('data/naturalstories/freqs/freqs-1.tsv'))
    report['alignment']={}
    for model in ('gpt2','distilgpt2'):
        tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
        multi=0
        for item in sorted({r['item'] for r in rows}):
            story=[r for r in rows if r['item']==item]
            _,owners=encode_regions([r['word'] for r in story],tok,allow_internal_whitespace=True)
            multi+=sum(n>1 for n in Counter(owners).values())
        report['alignment'][model]=dict(multi_bpe_regions=multi,failures=0,regions=len(rows))
    require_new(out)
    new_table(out/'regions.tsv',rows)
    new_json(out/'validation.json',report)
    new_json(out/'material_comparison.json',material_diagnostic(rows))
    print(report)


if __name__=='__main__':
    main()
