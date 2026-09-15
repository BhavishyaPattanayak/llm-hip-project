import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.data import write_tsv, read_tsv, key
from src.regression import predictors
from src.bpe_control import add_bpe_predictors, evaluate, run, snapshot


class BPEControlTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(55)
        self.trajectory=[]
        self.joined=[]
        for item in range(1,11):
            for zone in range(1,13):
                self.trajectory.append(dict(item=item,zone=zone,word=f'w{zone}',n_bpe=int(rng.integers(1,5))))
                if zone < 3:
                    continue
                x=rng.normal(size=6)
                row=dict(item=item,zone=zone,word=f'w{zone}',fold=(zone+item)%10,
                         observed_rt=350+x.sum()+rng.normal(),n_bpe=self.trajectory[-1]['n_bpe'])
                for model in ('gpt2','distilgpt2'):
                    row.update(dict(zip(predictors(model),x)))
                for measure in ('cosine','relative_l2'):
                    row.update({f'{third}_mean_{measure}':rng.random() for third in ('early','middle','late')})
                self.joined.append(row)
        self.residual=[dict(r) for r in self.joined]

    def test_actual_previous_word_including_excluded_regions(self):
        rows=add_bpe_predictors(self.joined[::-1],self.trajectory[::-1],self.residual)
        lookup={key(r):r for r in self.trajectory}
        for r in rows:
            self.assertEqual(r['prev_n_bpe'],lookup[int(r['item']),int(r['zone'])-1]['n_bpe'])
        initial=dict(self.joined[0],zone=1,n_bpe=self.trajectory[0]['n_bpe'])
        initial['word']='w1'
        result=add_bpe_predictors([initial],self.trajectory,[initial])
        self.assertTrue(np.isnan(result[0]['prev_n_bpe']))

    def test_saved_fold_reuse_and_identical_prediction_rows(self):
        rows=add_bpe_predictors(self.joined,self.trajectory,self.residual)
        _,a,_,_=evaluate(rows,'gpt2')
        _,b,_,_=evaluate(rows,'distilgpt2')
        expected=[(key(r),r['fold']) for r in sorted(self.residual,key=key)]
        self.assertEqual([(key(r),r['fold']) for r in a],expected)
        self.assertEqual([(key(r),r['fold']) for r in b],expected)
        bad=[dict(r) for r in self.joined]
        bad[0]['fold']=(bad[0]['fold']+1)%10
        with self.assertRaises(ValueError):
            add_bpe_predictors(bad,self.trajectory,self.residual)

    def test_training_only_scaling(self):
        rows=add_bpe_predictors(self.joined,self.trajectory,self.residual)
        _,pred,_,audit=evaluate(rows,'gpt2')
        changed=[dict(r) for r in rows]
        for r in changed:
            if r['fold']==0:
                r['n_bpe']+=10000
                r['early_mean_cosine']+=10000
        _,_,_,new=evaluate(changed,'gpt2')
        for variant in ('baseline','cosine','relative_l2'):
            self.assertEqual(audit[variant]['folds'][0],new[variant]['folds'][0])
        outcomes=[dict(r) for r in rows]
        for r in outcomes:
            if r['fold']==0:
                r['observed_rt']+=10000
        _,other,_,_=evaluate(outcomes,'gpt2')
        for old,new in zip(pred,other):
            if old['fold']==0:
                for variant in ('baseline','cosine','relative_l2'):
                    self.assertAlmostEqual(old[f'{variant}_predicted_rt'],new[f'{variant}_predicted_rt'])

    def test_outputs_preserved_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for model in ('gpt2','distilgpt2'):
                write_tsv(root/f'representations/analysis/{model}_joined.tsv',self.joined)
                write_tsv(root/f'representations/{model}_trajectories.tsv',self.trajectory)
                write_tsv(root/f'regression/{model}_heldout_residuals.tsv',self.residual)
            (root/'unrelated.txt').write_text('Keep exactly this text\n')
            before=snapshot(root)
            summary=run(root)
            after=snapshot(root)
            self.assertTrue(all(after[p]==v for p,v in before.items()))
            self.assertEqual(summary['gpt2']['change_in_n'],0)
            a=read_tsv(root/'representations/analysis/bpe_control/gpt2_predictions.tsv')
            self.assertEqual(len(a),100)
            with self.assertRaises(FileExistsError):
                run(root)
            self.assertEqual(snapshot(root),after)


if __name__ == '__main__':
    unittest.main()
