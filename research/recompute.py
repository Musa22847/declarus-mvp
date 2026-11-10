"""Offline recomputation from saved gold and predictions. No API calls."""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, random
from model_evaluation import evaluate_document
ROOT=Path(__file__).resolve().parent
B=20_000
SEED=20261107
KEYS=('true_positive','false_positive','false_negative')
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def aggregate(rows):
    total=Counter()
    for row in rows:total.update(row['metrics']['counts'])
    tp=total['true_positive'];fp=total['false_positive'];fn=total['false_negative']
    p=tp/(tp+fp) if tp+fp else 0.0;r=tp/(tp+fn) if tp+fn else 0.0
    bbox_n=sum(x['metrics']['bbox']['counts']['gold'] for x in rows)
    row_denominator=sum(max(x['metrics']['counts']['gold_items'],x['metrics']['counts']['predicted_items'],1) for x in rows)
    return {'documents':len(rows),'json_valid':sum(x['json_valid'] for x in rows),'counts':dict(total),
            'field_precision':p,'field_recall':r,'field_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
            'item_exact_match_rate':total['exact_items']/max(row_denominator,1),'item_denominator':row_denominator,
            'exact_documents':sum(x['metrics']['document_exact_match'] for x in rows),
            'document_exact_match_rate':sum(x['metrics']['document_exact_match'] for x in rows)/len(rows) if rows else 0,
            'mean_cer':sum(x['metrics']['cer'] for x in rows)/len(rows) if rows else None,
            'mean_wer':sum(x['metrics']['wer'] for x in rows)/len(rows) if rows else None,
            'bbox_mean_iou':sum((x['metrics']['bbox']['mean_iou'] or 0)*x['metrics']['bbox']['counts']['gold'] for x in rows)/bbox_n if bbox_n else None,
            'bbox_gold':bbox_n,'bbox_present':sum(x['metrics']['bbox']['counts']['predicted_for_gold'] for x in rows)}

def counts(row):
    return tuple(row['metrics']['counts'][k] for k in KEYS)

def f1(rows):
    tp, fp, fn = (sum(x[i] for x in rows) for i in range(3))
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

def percentile(values, q):
    a = sorted(values)
    pos = (len(a) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(a) - 1)
    return a[lo] + (a[hi] - a[lo]) * (pos - lo)

def check(name, baseline, treatment, groups=None):
    assert set(baseline) == set(treatment) and baseline
    ids = sorted(baseline)
    for d in ids:
        assert all(v >= 0 for v in baseline[d] + treatment[d])
        assert baseline[d][0] + baseline[d][2] == treatment[d][0] + treatment[d][2]
    strata = [ids] if groups is None else [[d for d in ids if groups[d] == g] for g in sorted(set(groups.values()))]
    rng = random.Random(SEED)
    deltas = []
    for _ in range(B):
        selected = [rng.choice(group) for group in strata for _ in group]
        deltas.append(100 * (f1([treatment[d] for d in selected]) - f1([baseline[d] for d in selected])))
    low, high = percentile(deltas, .025), percentile(deltas, .975)
    result = {'name': name, 'documents': len(ids), 'resampling': 'paired_documents' if groups is None else 'paired_documents_within_template',
              'baseline_f1_percent': 100*f1(list(baseline.values())), 'treatment_f1_percent': 100*f1(list(treatment.values())),
              'delta_f1_pp': 100*(f1(list(treatment.values()))-f1(list(baseline.values()))),
              'ci95_pp': [low, high], 'positive_effect_supported': low > 0,
              'degenerate_distribution': min(deltas) == max(deltas),
              'document_counts': [{'id': d, 'baseline_tp_fp_fn': baseline[d], 'treatment_tp_fp_fn': treatment[d]} for d in ids]}
    return result, deltas

def evaluate():
    manifest=read(ROOT/'data/manifest.json');alignment=read(ROOT/'few_shot_alignment.json')
    assert sha(ROOT/'model_evaluation.py')==alignment['evaluator_sha256']
    assert sha(ROOT.parent/alignment['prompt'])==alignment['prompt_sha256']
    for example in alignment['examples']:
        assert sha(ROOT.parent/example['image'])==example['image_sha256']
        assert sha(ROOT.parent/example['annotation'])==example['annotation_sha256']
    docs=manifest['documents'];assert len(docs)==50 and len({d['document_id'] for d in docs})==50
    for d in docs:
        assert sha(ROOT/d['image'])==d['image_sha256'] and sha(ROOT/d['gold'])==d['gold_sha256']
        assert sha(ROOT/d['ocr'])==d['ocr_sha256']
    summaries={};runs={};errors=[]
    for mode in ['ocr_only','ocr_image','ocr_image_3shot']:
        states={s['document_id']:s for s in read(ROOT/f'predictions/{mode}/attempts.json')}
        assert set(states)=={d['document_id'] for d in docs}
        rows=[]
        for d in docs:
            state=states[d['document_id']];path=ROOT/f"predictions/{mode}/{d['document_id']}.json"
            good=state['status']=='completed'
            if good:assert path.exists()
            metrics=evaluate_document(read(ROOT/d['gold']),read(path) if good else {})
            metrics['json_valid']=good
            rows.append({**d,**state,'json_valid':good,'metrics':metrics})
            errors.extend({'mode':mode,'document_id':d['document_id'],**e} for e in metrics['errors'])
        runs[mode]=rows;summaries[mode]=aggregate(rows)
    historical=read(ROOT/'reports/fewshot_comparison.json')
    image=read(ROOT/'reports/image_comparison.json')
    for mode,reference in [('ocr_only',image['summary']['ocr_only']),('ocr_image',historical['summary']['zero_shot']),('ocr_image_3shot',historical['summary']['few_shot_3'])]:
        assert summaries[mode]['counts']==reference['counts']
        assert abs(summaries[mode]['field_f1']-reference['field_f1'])<1e-12
    result={'summary':summaries,'runs':runs,'hypotheses':{
       'H1':{'baseline':'ocr_image','treatment':'ocr_image_3shot'},
       'H2':{'baseline':'ocr_only','treatment':'ocr_image'}},
       'failure_policy':'All 50 attempts included; unsuccessful prediction is empty extraction',
       'evaluator_sha256':sha(ROOT/'model_evaluation.py')}
    for h,pair in result['hypotheses'].items():
        pair['delta_f1_pp']=100*(summaries[pair['treatment']]['field_f1']-summaries[pair['baseline']]['field_f1'])
    save(ROOT/'reports/evaluation.json',result);save(ROOT/'reports/extraction_errors.json',errors)
    return result

def bootstrap(result):
    output=[]
    for h,pair in result['hypotheses'].items():
        a=result['runs'][pair['baseline']];b=result['runs'][pair['treatment']]
        base={x['document_id']:counts(x) for x in a};treated={x['document_id']:counts(x) for x in b}
        main,_=check(h+'_all_50',base,treated);output.append(main)
        valid=set.intersection(*[{x['document_id'] for x in rows if x['json_valid']} for rows in [a,b]])
        if valid:output.append(check(h+'_successful_pairs',{d:base[d] for d in valid},{d:treated[d] for d in valid})[0])
        groups={x['document_id']:x['template'] for x in a}
        output.append(check(h+'_stratified',base,treated,groups)[0])
    report={'bootstrap_replicates':B,'seed':SEED,'confidence_level':.95,
        'method':'paired document percentile bootstrap; pooled micro-F1 per replicate',
        'evaluation_sha256':sha(ROOT/'reports/evaluation.json'),'results':output}
    save(ROOT/'reports/bootstrap.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bootstrap',action='store_true');args=p.parse_args()
    result=evaluate()
    for mode,summary in result['summary'].items():print(mode,'F1:',round(100*summary['field_f1'],2))
    if args.bootstrap:
        for row in bootstrap(result)['results']:print(row['name'],row['delta_f1_pp'],row['ci95_pp'])
