import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from sklearn.linear_model import Ridge
from src.geco import aggregate,add_lexical,lag_columns
from src.entropy import entropy_from_logits,extract_probabilities
from src.nested_regression import nested_predict,Design,ridge_path
from src.extension_io import new_json
from src.surprisal import encode_regions
from transformers import AutoTokenizer


class GecoTests(unittest.TestCase):
    def fixture(self):
        return [dict(PP_NR=pp,WORD_ID=f'{trial}-{zone}',PART=1,TRIAL=trial,WORD_ID_WITHIN_TRIAL=zone,
            WORD=word,WORD_SKIP=skip,WORD_GAZE_DURATION=gaze,WORD_TOTAL_READING_TIME=total)
            for pp,trial,zone,word,skip,gaze,total in [
                ('a',1,1,'Hello,',0,100,150),('b',1,1,'Hello,',1,'.',200),
                ('a',1,2,'long-bearded',1,80,90),('b',1,2,'long-bearded',0,120,160),
                ('a',2,1,'pouf!" ...he',1,'.','.'),('a',2,2,8,0,70,80)]]

    def test_reconstruction_means_skips_and_boundaries(self):
        rows,report=aggregate(reversed(self.fixture()))
        self.assertEqual(len(rows),4)
        self.assertEqual(rows[0]['mean_gaze'],100)
        self.assertEqual(rows[0]['mean_total'],175)
        self.assertEqual(rows[1]['mean_gaze'],100)
        self.assertEqual(rows[2]['mean_gaze'],'')
        self.assertEqual(rows[2]['word'],'pouf!" ...he')
        self.assertEqual(rows[3]['word'],'8')
        self.assertEqual(report['numeric_excel_word_regions'],1)
        self.assertEqual(report['multiword_regions'],1)
        lex=add_lexical(rows,{'Hello':10,'long-bearded':2})
        self.assertEqual(lex[2]['prev_word_length'],'')
        self.assertEqual(lex[1]['prev_word_length'],6)
        self.assertEqual(rows[0]['proportion_valid_gaze'],.5)

    def test_position_gaps_preserve_original_ids(self):
        rows=self.fixture()
        rows[-1]['WORD_ID_WITHIN_TRIAL']=4
        result,report=aggregate(rows)
        self.assertEqual(result[-1]['WORD_ID_WITHIN_TRIAL'],4)
        self.assertEqual(result[-1]['zone'],2)
        self.assertEqual(report['noncontiguous_trial_positions'][0]['missing_positions'],[2,3])

    def test_inconsistent_identity_and_duplicates_fail(self):
        for field,value in [('WORD','changed'),('PART',2),('TRIAL',3),('WORD_ID_WITHIN_TRIAL',3)]:
            rows=self.fixture(); rows[1][field]=value
            with self.assertRaises(ValueError): aggregate(rows)
        rows=self.fixture()
        with self.assertRaises(ValueError): aggregate(rows+rows[:1])

    def test_internal_space_punctuation_alignment(self):
        tok=AutoTokenizer.from_pretrained('gpt2',local_files_only=True)
        words=['The','pouf!" ...he','long-bearded','cat.']
        ids,owners=encode_regions(words,tok,allow_internal_whitespace=True)
        for i,w in enumerate(words):
            self.assertEqual(tok.decode([t for t,o in zip(ids,owners) if o==i]),(' ' if i else '')+w)
        self.assertGreater(owners.count(1),1)
        spaced=['The ', ' long-bearded', 'cat.']
        ids,owners=encode_regions(spaced,tok,allow_internal_whitespace=True)
        self.assertEqual(tok.decode(ids),' '.join(spaced))
        self.assertEqual(set(owners),{0,1,2})


class EntropyTests(unittest.TestCase):
    def test_uniform_and_first_bpe_causal_distribution(self):
        self.assertAlmostEqual(float(entropy_from_logits(torch.zeros(4))),2)
        class Tokenizer:
            def __call__(self,text,**kwargs):
                return dict(input_ids=[0,1,2,3],offset_mapping=[(0,1),(1,3),(3,4),(4,6)])
        class Model:
            config=SimpleNamespace(max_position_embeddings=4)
            def __call__(self,input_ids,use_cache=False):
                logits=torch.zeros((1,4,4))
                logits[0,1,2]=5
                logits[0,2,3]=2
                return SimpleNamespace(logits=logits)
        rows=[dict(item=1,zone=i+1,word=w) for i,w in enumerate(['a','bc','d'])]
        result=extract_probabilities(rows,Tokenizer(),Model(),'cpu',stride=2)
        self.assertTrue(np.isnan(result[0]['entropy_bits']))
        self.assertEqual(result[1]['entropy_bits'],2)
        expected=2+float(torch.nn.functional.cross_entropy(torch.tensor([[0.,0.,5.,0.]]),torch.tensor([2])))/np.log(2)
        self.assertAlmostEqual(result[1]['surprisal'],expected,places=6)
        self.assertGreaterEqual(result[2]['entropy_bits'],0)
        second=extract_probabilities(rows,Tokenizer(),Model(),'cpu',stride=2)
        self.assertTrue(np.isnan(second[0]['entropy_bits']))
        np.testing.assert_allclose([r['entropy_bits'] for r in result],[r['entropy_bits'] for r in second],equal_nan=True)


class NestedTests(unittest.TestCase):
    def test_svd_matches_sklearn_ridge(self):
        rng=np.random.default_rng(5); x=rng.normal(size=(50,5));y=rng.normal(size=50)
        coef,intercept=ridge_path(x,y,[.1,100])
        for i,alpha in enumerate([.1,100]):
            fit=Ridge(alpha=alpha).fit(x,y)
            np.testing.assert_allclose(coef[:,i],fit.coef_,atol=1e-12)
            self.assertAlmostEqual(intercept[i],fit.intercept_)

    def test_nested_and_spline_no_outer_leakage(self):
        rng=np.random.default_rng(10); x=rng.normal(size=(60,5)); y=rng.normal(size=60)
        folds=np.arange(60)%10
        for spline in (False,True):
            pred,audit=nested_predict(x,y,folds,2,spline,alphas=[.001,1,100])
            changed=x.copy(); changed[folds==0]+=1000
            other=y.copy(); other[folds==0]+=1000
            _,altered=nested_predict(changed,other,folds,2,spline,alphas=[.001,1,100])
            self.assertEqual(audit[0],altered[0])
            self.assertEqual(len(pred),60);self.assertTrue(np.isfinite(pred).all())
            for a in audit:
                test=set(a['outer_test_indices'])
                for inner in a['inner']:
                    self.assertFalse(test & set(inner['train_indices']))
                    self.assertFalse(test & set(inner['validation_indices']))
                self.assertIn(a['alpha'],[.001,1,100])
            again,_=nested_predict(x,y,folds,2,spline,alphas=[.001,1,100])
            np.testing.assert_array_equal(pred,again)
        design=Design(2,True).fit(x)
        z=design.transform(x)
        np.testing.assert_allclose(z[:,:2],(x[:,:2]-x[:,:2].mean(0))/x[:,:2].std(0))
        self.assertEqual(z.shape[1],2+3*5)

    def test_no_overwrite_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'result.json'
            new_json(p,{'kept':True});before=p.read_bytes()
            with self.assertRaises(FileExistsError):new_json(p,{'kept':False})
            self.assertEqual(p.read_bytes(),before)


class ExtensionIntegrationTests(unittest.TestCase):
    def test_geco_full_regression_path_on_synthetic_data(self):
        import os
        from src.data import write_tsv,read_tsv
        from src.representations import trajectory
        from src.extension_analysis import run_analysis
        rng=np.random.default_rng(19)
        regions=[];surprisal=[];entropy=[];representations=[]
        for item in range(1,13):
            for zone in range(1,13):
                r=dict(item=item,zone=zone,word=f'w{zone}',WORD_ID=f'{item}-{zone}',PART=1,TRIAL=item,
                       word_length=int(rng.integers(2,12)),log_frequency=float(rng.uniform(3,15)),
                       mean_gaze=float(rng.uniform(200,400)),mean_total=float(rng.uniform(300,500)))
                regions.append(r)
                surprisal.append(dict(r,surprisal=float('nan') if zone==1 else float(rng.uniform(2,10)),n_bpe=int(rng.integers(1,5))))
                entropy.append(dict(r,entropy_bits=float('nan') if zone==1 else float(rng.uniform(2,8))))
                representations.append(dict(r,**trajectory(np.cumsum(rng.normal(size=(7,8)),axis=0))))
        regions=lag_columns(regions,['word_length','log_frequency'])
        with tempfile.TemporaryDirectory() as tmp:
            old=os.getcwd()
            try:
                os.chdir(tmp)
                write_tsv('results/geco/processed/regions.tsv',regions)
                new_json('results/distilgpt2_surprisal.json',dict(model_revision='fixture'))
                for name,table in [('surprisal',surprisal),('entropy',entropy),('representations',representations)]:
                    path=Path(f'results/geco/{name}/distilgpt2.tsv');write_tsv(path,table)
                    new_json(path.with_suffix('.json'),dict(smoke_only=False,model_revision='fixture',rows=144))
                report=run_analysis('geco','distilgpt2',dv='gaze')
                self.assertEqual(report['n'],120)
                self.assertEqual(report['excluded'],24)
                path=Path('results/geco/extensions/gaze/distilgpt2')
                self.assertTrue((path/'fit_audits.json').exists())
                for predictions in path.glob('*_predictions.tsv'):
                    rows=read_tsv(predictions)
                    self.assertEqual(len(rows),120)
                    self.assertEqual(len({(r['item'],r['zone']) for r in rows}),120)
                with self.assertRaises(FileExistsError):run_analysis('geco','distilgpt2',dv='gaze')
            finally:
                os.chdir(old)


if __name__=='__main__': unittest.main()
