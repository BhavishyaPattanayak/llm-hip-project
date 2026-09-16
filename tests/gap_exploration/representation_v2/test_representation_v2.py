import unittest
import numpy as np
from src.gap_exploration.representation_v2.features import geometry,distances
from src.gap_exploration.representation_v2.evaluation import HiddenTransform,prepare,evaluate,qualify,ridge_predictions


class RepresentationV2Tests(unittest.TestCase):
    def test_directional_curvature_is_not_speed_change(self):
        straight=np.array([[[1+i,2.] for i in range(7)]])
        turning=straight.copy();turning[0,3:]=[[3.,3.],[3.,4.],[2.,4.],[1.,4.]]
        self.assertAlmostEqual(geometry(straight)['curvature'][0,0],0.)
        self.assertGreater(geometry(turning)['curvature'][0,0],.1)
        np.testing.assert_allclose(distances([3.,4.],[0.,5.]),[.2,np.sqrt(10)/5])

    def test_pca_and_reference_distributions_train_only(self):
        rng=np.random.default_rng(2026);h=rng.normal(size=(40,3,8));train=np.arange(30);test=np.arange(30,40)
        a,b,audit=prepare({'hidden':h},train,test,3)
        changed=h.copy();changed[test]+=1000
        c,d,other=prepare({'hidden':changed},train,test,3)
        self.assertEqual(audit,other)
        for name in a:np.testing.assert_array_equal(a[name],c[name])
        self.assertGreater(d['typicality'].mean(),b['typicality'].mean())
        transform=HiddenTransform(3).fit(h[train])
        np.testing.assert_allclose(transform.pcas[0].mean_,h[train,0].mean(0))
        self.assertTrue(all(p.n_samples_==30 for p in transform.pcas))

    def test_scaler_train_only(self):
        x=np.arange(40.).reshape(20,2);y=np.arange(20.)
        _,audit=ridge_predictions(x[:15],y[:15],x[15:]+1000,[1.])
        np.testing.assert_allclose(audit['mean'],x[:15].mean(0))
        np.testing.assert_allclose(audit['scale'],x[:15].std(0))

    def test_nested_selection_never_reads_outer_targets(self):
        rng=np.random.default_rng(2026);n=60
        base=rng.normal(size=(n,3));original=np.c_[base,rng.normal(size=(n,2))]
        y=rng.normal(size=n);folds=np.arange(n)%10
        features={'curvature':rng.normal(size=(n,2)),'hidden':rng.normal(size=(n,3,5))}
        p,a,_=evaluate(base,original,y,folds,features,alphas=[.1,10.],components=2)
        y2=y.copy();y2[folds==0]+=1000
        q,b,_=evaluate(base,original,y2,folds,features,alphas=[.1,10.],components=2)
        self.assertEqual(a[0],b[0])
        for name in p:np.testing.assert_array_equal(p[name][folds==0],q[name][folds==0])
        for fold in range(10):self.assertEqual(a[fold]['test_indices'],np.flatnonzero(folds==fold).tolist())
        for inner in a[0]['inner_pca']:self.assertTrue(all(p['n_samples']==43 for p in inner['pca']) or all(p['n_samples']==44 for p in inner['pca']))

    def test_combination_avoids_duplicate_layer_pcs(self):
        tuned={'baseline':{'loss':np.ones(5)*10},'layer_early':{'loss':np.ones(5)*9},
            'layer_combination':{'loss':np.ones(5)*8},'curvature':{'loss':np.array([5,12,12,12,5])}}
        accepted,best,combined=qualify(tuned,['layer_early','layer_combination','curvature'])
        self.assertEqual(best,'layer_combination');self.assertEqual(combined,['layer_combination'])

    def test_all_preexisting_files_unchanged(self):
        from experiments.gap_exploration.representation_v2.verify import verify
        report=verify()
        self.assertEqual(report['changed_or_missing'],[])
        self.assertEqual(report['unexpected_new_files'],[])

    def test_actual_sample_fold_and_history_alignment(self):
        from pathlib import Path
        from src.data import read_tsv,key
        from src.gap_exploration.representation_v2.data import load,ROOT
        for dataset in ('natural_stories','geco'):
            for model in ('gpt2','distilgpt2'):
                if not (ROOT/'features'/f'{dataset}_{model}.npz').exists():continue
                case,base,original,features=load(dataset,model)
                saved=read_tsv(Path(case['source'])/'sample.tsv')
                self.assertEqual([(key(r),int(r['fold'])) for r in case['rows']],[(key(r),int(r['fold'])) for r in saved])
                source={key(r):r for r in read_tsv(f'results/{model}_surprisal.tsv' if dataset=='natural_stories' else f'results/geco/surprisal/{model}.tsv')}
                for i,r in enumerate(case['rows']):
                    item,zone=key(r);previous=source[(item,zone-2)]
                    self.assertEqual(base[i,12],int(previous['n_bpe']))
                    value=float(previous['surprisal'])
                    self.assertEqual(base[i,14],value if np.isfinite(value) else 0.)
                self.assertEqual(len(features['hidden']),len(saved))
                self.assertTrue(np.isfinite(base).all())

    def test_extraction_ownership_raw_states_and_matched_positions(self):
        import re,torch
        from transformers import GPT2Config,GPT2LMHeadModel
        from src.gap_exploration.representation_v2.extraction import extract
        from src.representations import extract_story,trajectory
        class Tokenizer:
            eos_token_id=0
            def __init__(self):self.calls=0
            def __call__(self,text,**kwargs):
                self.calls+=1;spans=[];ids=[]
                for i,m in enumerate(re.finditer(r'\S+',text)):
                    a,b=m.span();mid=a+1
                    spans.extend([(a,mid),(mid,b)]);ids.extend([1+i%25,26+i%25])
                return {'input_ids':ids,'offset_mapping':spans}
        torch.manual_seed(2026)
        model=GPT2LMHeadModel(GPT2Config(vocab_size=64,n_positions=64,n_embd=12,n_layer=6,n_head=3,bos_token_id=0,eos_token_id=0)).eval()
        rows=[dict(item=1,zone=i+1,word='word') for i in range(41)];tok=Tokenizer()
        full=[];truncated=[]
        def inspect(module,args,kwargs):
            tokens=kwargs['input_ids'].cpu().numpy()
            if 'position_ids' not in kwargs:full.append(tokens[0])
            else:
                pos=kwargs['position_ids'].cpu().numpy();mask=kwargs['attention_mask'].cpu().numpy()
                for t,p,m in zip(tokens,pos,mask):
                    np.testing.assert_array_equal(t[m==1],full[-1][p[m==1]])
                truncated.append(len(tokens))
        handle=model.transformer.register_forward_pre_hook(inspect,with_kwargs=True)
        result,pooled=extract(rows,tok,model,'cpu',{(1,i) for i in range(3,42)},stride=16)
        handle.remove();self.assertEqual(tok.calls,1);self.assertTrue(truncated)
        reference=extract_story(rows,tok,model,'cpu',stride=16)
        for i,k in enumerate(result['keys']):
            for name,value in trajectory(pooled[i]).items():self.assertAlmostEqual(value,reference[k[1]-1][name],places=6)
        self.assertTrue(np.all(result['n_bpe']==2));self.assertEqual(len(result['keys']),39)


if __name__=='__main__':unittest.main()
