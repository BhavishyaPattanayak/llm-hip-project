import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from src.surprisal import encode_regions, context_windows
from src.representations import (trajectory, pool_states, thirds, extract_story,
    validate_features, feature_names)
from src.regression import make_folds, predictors, cross_validate
from src.representation_analysis import (bh_fdr, join_residuals, layer_correlations,
    residual_regressions, incremental_rt, confound_correlations)


class RepresentationTests(unittest.TestCase):
    def test_distances_and_thirds(self):
        h = np.array([[1.,0], [0,1], [-1,0], [-2,0]])
        values = trajectory(h)
        self.assertEqual([values[f'cosine_layer_{i}'] for i in range(1,4)], [1,1,0])
        np.testing.assert_allclose([values[f'relative_l2_layer_{i}'] for i in range(1,4)], [2**.5,2**.5,1])
        self.assertEqual(values['total_cosine'], 2)
        self.assertEqual(values['peak_layer_cosine'], 1)
        self.assertEqual(values['late_minus_early_cosine'], -1)
        self.assertEqual(thirds(12), {'early':[1,2,3,4], 'middle':[5,6,7,8], 'late':[9,10,11,12]})
        self.assertEqual(thirds(6), {'early':[1,2], 'middle':[3,4], 'late':[5,6]})

    def test_mean_pool_before_distances(self):
        states = np.arange(4*3*2).reshape(4,3,2)
        pooled = pool_states(states, [0,1,1], 2)
        np.testing.assert_array_equal(pooled[0], states[:,0])
        np.testing.assert_array_equal(pooled[1], (states[:,1]+states[:,2])/2)

    def test_exact_window_schedule(self):
        # Freeze the previously established algorithm independently of shared helper.
        for n in (2,8,9,11,22):
            expected=[]
            target=1
            while target<n:
                end=min(n,8 if target==1 else target+3)
                expected.append((max(0,end-8),end,target))
                target=end
            self.assertEqual(list(context_windows(n,8,3)), expected)

    def test_hidden_states_raw_blocks_and_cross_window_pooling(self):
        tok = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)
        words = ['The', 'long-bearded', 'cat.', 'extraordinarily', 'long-bearded', 'cat.']
        rows = [dict(item=1,zone=i+1,word=w) for i,w in enumerate(words)]
        ids, owners = encode_regions(words, tok)
        self.assertGreater(len(ids),8)
        for n_blocks in (12,6):
            torch.manual_seed(9)
            model = GPT2LMHeadModel(GPT2Config(n_layer=n_blocks,n_embd=12,n_head=3,
                n_positions=8,vocab_size=50257,attn_pdrop=0,resid_pdrop=0,embd_pdrop=0)).eval()
            actual = extract_story(rows,tok,model,'cpu',stride=3)
            all_states = np.zeros((n_blocks+1,len(ids),12))
            raw = []
            hooks = [block.register_forward_hook(lambda module,args,out: raw.append(
                out[0] if isinstance(out,tuple) else out)) for block in model.transformer.h]
            try:
                with torch.inference_mode():
                    for begin,end,target in context_windows(len(ids),8,3):
                        raw.clear()
                        out=model.transformer(torch.tensor([ids[begin:end]]),output_hidden_states=True,use_cache=False)
                        self.assertEqual(len(out.hidden_states),n_blocks+1)
                        self.assertEqual(len(raw),n_blocks)
                        # Native intermediate states equal preceding raw block outputs.
                        for i in range(1,n_blocks):
                            torch.testing.assert_close(out.hidden_states[i],raw[i-1])
                        first=0 if target==1 else target
                        local=first-begin
                        native=[out.hidden_states[0],*raw]
                        all_states[:,first:end]=np.stack([h[0,local:].numpy() for h in native])
            finally:
                for hook in hooks:
                    hook.remove()
            pooled=pool_states(all_states,owners,len(words))
            for i,row in enumerate(actual):
                expected=trajectory(pooled[i])
                for name,value in expected.items():
                    self.assertAlmostEqual(row[name],value,places=8)
            reference=[dict(r,n_bpe=owners.count(i)) for i,r in enumerate(rows)]
            validate_features(actual,reference,n_blocks)
            invalid=[dict(r) for r in actual]
            invalid[0]['cosine_layer_1']=float('nan')
            with self.assertRaises(ValueError):
                validate_features(invalid,reference,n_blocks)
            with self.assertRaises(ValueError):
                validate_features(actual+actual[:1],reference,n_blocks)


class RepresentationAnalysisTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(41)
        n=100
        self.folds=make_folds(n)
        x=rng.normal(size=(n,6))
        y=350+x@np.arange(1,7)+rng.normal(size=n)*5
        baseline,_=cross_validate(x,y,self.folds)
        self.features=[]
        self.residuals=[]
        for i in range(n):
            identity=dict(item=1,zone=i+3,word=f'w{i}')
            h=np.cumsum(rng.normal(size=(7,8)),axis=0)
            self.features.append(dict(identity,n_bpe=1+i%3,**trajectory(h)))
            self.residuals.append(dict(identity,observed_rt=y[i],predicted_rt=baseline[i],
                residual=y[i]-baseline[i],fold=int(self.folds[i]),surprisal=x[i,0],
                **dict(zip(predictors('gpt2'),x[i]))))
        self.rows=join_residuals(self.features[::-1],self.residuals,6)

    def test_bh_known_values(self):
        np.testing.assert_allclose(bh_fdr([.01,.04,.03,.8]),[.04,.053333333333,.053333333333,.8])

    def test_saved_folds_keys_and_residuals(self):
        self.assertEqual([r['fold'] for r in self.rows],self.folds.tolist())
        broken=[dict(r) for r in self.residuals]
        broken[0]['fold']=(broken[0]['fold']+1)%10
        with self.assertRaises(ValueError):
            join_residuals(self.features,broken,6)
        with self.assertRaises(ValueError):
            join_residuals(self.features+self.features[:1],self.residuals,6)
        broken=[dict(r) for r in self.features]
        broken[0]['word']='wrong'
        with self.assertRaises(ValueError):
            join_residuals(broken,self.residuals,6)

    def test_extended_no_leakage_and_analysis_outputs(self):
        report,output,audits=incremental_rt(self.rows,'gpt2')
        self.assertEqual(len(output),100)
        self.assertEqual([r['fold'] for r in output],self.folds.tolist())
        held=self.folds==0
        changed=[dict(r) for r in self.rows]
        for i in np.flatnonzero(held):
            changed[i]['early_mean_cosine']+=1000
        _,_,new_audits=incremental_rt(changed,'gpt2')
        self.assertEqual(audits['cosine']['folds'][0],new_audits['cosine']['folds'][0])
        self.assertEqual(len(layer_correlations(self.rows,6)),48)
        self.assertEqual(len(residual_regressions(self.rows)),16)
        self.assertEqual(len(confound_correlations(self.rows)),128)
        self.assertTrue(np.isfinite([r['cosine_predicted_rt'] for r in output]).all())
        self.assertAlmostEqual(report['cosine']['rmse'],np.sqrt(np.mean([r['cosine_residual']**2 for r in output])))

    def test_figures_in_temporary_directory_only(self):
        from experiments.analyze_representations import figures
        report,_,_=incremental_rt(self.rows,'gpt2')
        with tempfile.TemporaryDirectory() as temp:
            figures(self.rows,layer_correlations(self.rows,6),report,'synthetic_test',6,Path(temp))
            self.assertEqual(len(list(Path(temp).glob('*.png'))),5)
            self.assertEqual(len(list(Path(temp).glob('*.pdf'))),5)


if __name__ == '__main__':
    unittest.main()
