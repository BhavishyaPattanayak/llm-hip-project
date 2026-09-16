import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.gap_exploration.evaluation import evaluate,design_matrix
from src.gap_exploration.extraction import distribution_features
from src.gap_exploration.data import load_case,candidate_features
from src.extension_io import new_json


class GapTests(unittest.TestCase):
    def test_distribution_definitions(self):
        logits=torch.tensor([[0.,0.,0.,0.],[0.,1.,2.,3.]])
        features=distribution_features(logits,torch.tensor([2,0])).numpy()
        np.testing.assert_allclose(features[0],[np.log(2),1,0,.25],rtol=1e-6)
        self.assertAlmostEqual(features[1,0],np.log(5),places=6)
        self.assertGreater(features[1,2],0)

    def test_no_outer_leakage_in_selection_tuning_or_preprocessing(self):
        rng=np.random.default_rng(42);core=rng.normal(size=(70,3));strong=np.c_[core,rng.normal(size=(70,3))]
        y=rng.normal(size=70);folds=np.arange(70)%10
        families={'one':rng.normal(size=(70,2)),'two':rng.normal(size=(70,1))}
        prediction,audit=evaluate(core,strong,y,folds,families,True,[.01,10])
        other_core=core.copy();other_core[folds==0]+=1000
        other_strong=strong.copy();other_strong[folds==0]+=1000
        other_y=y.copy();other_y[folds==0]+=1000
        other_families={k:v.copy() for k,v in families.items()}
        for values in other_families.values():values[folds==0]+=1000
        _,other=evaluate(other_core,other_strong,other_y,folds,other_families,True,[.01,10])
        self.assertEqual(audit[0],other[0])
        for a in audit:
            for model in a['models'].values():
                if 'inner_splits' in model:
                    for split in model['inner_splits']:
                        self.assertFalse(set(a['test_indices'])&set(split['validation_indices']))
        self.assertTrue(all(len(v)==70 and np.isfinite(v).all() for v in prediction.values()))
        again,_=evaluate(core,strong,y,folds,families,True,[.01,10])
        for name in prediction:np.testing.assert_array_equal(prediction[name],again[name])

    def test_spline_placement_does_not_expand_new_candidates(self):
        core=np.ones((5,2));strong=np.c_[core,np.ones((5,3))*2];extra=np.ones((5,4))*3
        x,n,spline=design_matrix(core,strong,extra,True)
        self.assertEqual(n,6);self.assertTrue(spline)
        np.testing.assert_array_equal(x[:,-3:],strong[:,-3:])
        np.testing.assert_array_equal(x[:,2:6],extra)

    def test_real_sample_and_saved_folds(self):
        from src.data import read_tsv,key
        case=load_case('natural_stories','gpt2','rt')
        sample=read_tsv(Path(case['source'])/'sample.tsv')
        self.assertEqual([(key(r),int(r['fold'])) for r in case['rows']],[(key(r),int(r['fold'])) for r in sample])
        self.assertEqual(len(case['rows']),10216)
        self.assertIn('entropy_bits',case['strong_columns'])

    def test_actual_feature_alignment_and_no_row_drops(self):
        case=load_case('natural_stories','gpt2','rt')
        features=candidate_features(case)
        self.assertEqual(set(features),{'distribution_shape','extended_spillover','nonlinear_uncertainty','trajectory_shape'})
        for values in features.values():
            self.assertEqual(values.shape[0],len(case['rows']))
            self.assertTrue(np.isfinite(values).all())
        case['rows'][0]=dict(case['rows'][0],word='NOT THE ORIGINAL WORD')
        with self.assertRaises(ValueError):candidate_features(case)

    def test_frozen_source_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'frozen.json';new_json(file,{'fixed':True})
            digest=hashlib.sha256(file.read_bytes()).hexdigest()
            with self.assertRaises(FileExistsError):new_json(file,{'fixed':False})
            self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(),digest)
        # Audit every original code/test/documentation file now; full 3+GB audit is a separate final step.
        manifest=json.loads(Path('results/gap_exploration/metadata/original_manifest.json').read_text())
        for name,entry in manifest.items():
            if entry['kind']=='file' and (name.startswith(('src/','experiments/','tests/')) or name in ('README.md','requirements.txt')):
                self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(),entry['sha256'],name)


if __name__=='__main__':unittest.main()
