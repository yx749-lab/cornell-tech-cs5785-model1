"""Train one predeclared word TF-IDF/logistic baseline; no tuning or test access."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/matplotlib_config'))
import hashlib, json, platform, time, warnings
from datetime import datetime
from zoneinfo import ZoneInfo
import importlib.metadata
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

RUN='model1_baseline_v1'
PARAMS={'tfidf':{'analyzer':'word','ngram_range':[1,2],'lowercase':True,
    'token_pattern':r'(?u)\b\w\w+\b','stop_words':None,'min_df':2,'max_df':1.0,
    'max_features':50000,'sublinear_tf':True,'norm':'l2','use_idf':True,'smooth_idf':True},
    'classifier':{'C':1.0,'l1_ratio':0.0,'solver':'lbfgs','max_iter':1000,'tol':1e-4,'random_state':42},
    'decision_threshold':0.5,'thread_limit':1}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def score(frame,probs):
    pred=(probs>=0.5).astype(int)
    return {'rows':len(frame),'auroc':float(roc_auc_score(frame.label,probs)),
        'accuracy_at_0_5':float(accuracy_score(frame.label,pred)),
        'f1_ai_at_0_5':float(f1_score(frame.label,pred)),
        'confusion_matrix_true_rows_predicted_columns':confusion_matrix(frame.label,pred,labels=[0,1]).tolist()}

def main():
    out=ROOT/'results'/RUN; model_path=ROOT/'models'/f'{RUN}.joblib'
    if (out/'metrics.json').exists() or model_path.exists():
        raise RuntimeError('Existing baseline preserved. Use a new run name for a new experiment.')
    split=json.loads((ROOT/'data/splits/split_config.json').read_text())
    assert split['split_id']=='sklearn_stratified_seed42_v2'
    for filename,record in split['files'].items():
        assert sha(ROOT/'data/splits'/filename)==record['sha256']
    assert sha(ROOT/'data/raw/train.csv')==split['source_train_sha256']
    train=pd.read_csv(ROOT/'data/splits/train_8000.csv',dtype={'id':str})
    val=pd.read_csv(ROOT/'data/splits/validation_2000.csv',dtype={'id':str})
    assert len(train)==8000 and len(val)==2000
    assert train.id.is_unique and val.id.is_unique and not set(train.id)&set(val.id)
    assert train.text.notna().all() and val.text.notna().all()
    tf_params={**PARAMS['tfidf'],'ngram_range':tuple(PARAMS['tfidf']['ngram_range'])}
    pipeline=Pipeline([('tfidf',TfidfVectorizer(**tf_params)),('classifier',LogisticRegression(**PARAMS['classifier']))])
    print('Fitting one pipeline on 8,000 training texts only...',flush=True)
    with threadpool_limits(limits=PARAMS['thread_limit']), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        start=time.perf_counter();pipeline.fit(train.text,train.label);fit_seconds=time.perf_counter()-start
        if any(issubclass(w.category,ConvergenceWarning) for w in caught):
            raise RuntimeError('Classifier did not converge: '+str([str(w.message) for w in caught]))
        assert pipeline.classes_.tolist()==[0,1]
        start=time.perf_counter();train_p=pipeline.predict_proba(train.text)[:,1];train_predict_seconds=time.perf_counter()-start
        start=time.perf_counter();val_p=pipeline.predict_proba(val.text)[:,1];val_predict_seconds=time.perf_counter()-start
    for probabilities in [train_p,val_p]:assert np.isfinite(probabilities).all() and ((probabilities>=0)&(probabilities<=1)).all()
    out.mkdir(parents=True,exist_ok=True)
    joblib.dump(pipeline,model_path,compress=3)
    loaded=joblib.load(model_path)
    with threadpool_limits(limits=1):reloaded_p=loaded.predict_proba(val.text)[:,1]
    assert np.allclose(reloaded_p,val_p,rtol=0,atol=1e-12)
    metrics={'run_id':RUN,'created_at':datetime.now(ZoneInfo('America/New_York')).isoformat(),
        'split_id':split['split_id'],'parameters':PARAMS,'train':score(train,train_p),'validation':score(val,val_p),
        'fit_seconds_including_tfidf':fit_seconds,'train_prediction_seconds':train_predict_seconds,
        'validation_prediction_seconds':val_predict_seconds,
        'feature_count':len(pipeline['tfidf'].vocabulary_),'solver_iterations':pipeline['classifier'].n_iter_.tolist(),
        'warnings':[str(w.message) for w in caught],
        'checks':{'split_hashes_verified':True,'raw_train_hash_verified':True,'disjoint_ids':True,
                  'training_only_pipeline_fit':True,'probabilities_valid':True,'saved_model_reload_predictions_match':True},
        'source_hashes':{'train_csv':split['source_train_sha256'],'split_config':sha(ROOT/'data/splits/split_config.json'),
                         'training_script':sha(Path(__file__)),'dependency_lock':sha(ROOT/'requirements.lock.txt')},
        'environment':{'python':platform.python_version(),'versions':{n:importlib.metadata.version(n) for n in ['scikit-learn','numpy','scipy','pandas','joblib','matplotlib']}},
        'model_path':str(model_path.relative_to(ROOT)),'test_data_used':False,'unlabeled_data_used':False,
        'hyperparameter_search_performed':False,'limitation':'One holdout split and one untuned baseline; no Kaggle score or calibration claim.'}
    for name,frame,probabilities in [('train',train,train_p),('validation',val,val_p)]:
        pd.DataFrame({'id':frame.id,'true_label':frame.label,'p_ai':probabilities,
            'predicted_label_at_0_5':(probabilities>=0.5).astype(int)}).to_csv(out/f'{name}_predictions.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    pd.DataFrame([{'partition':name,**{k:v for k,v in metrics[name].items() if not isinstance(v,list)}} for name in ['train','validation']]).to_csv(out/'metrics.csv',index=False)
    names=pipeline['tfidf'].get_feature_names_out();coef=pipeline['classifier'].coef_[0]
    indices=np.r_[np.argsort(coef)[:20],np.argsort(coef)[-20:][::-1]]
    pd.DataFrame({'feature':names[indices],'coefficient':coef[indices]}).to_csv(out/'selected_feature_weights.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.4),layout='constrained')
    RocCurveDisplay.from_predictions(val.label,val_p,ax=axes[0],name='Validation',plot_chance_level=True)
    axes[0].set_title('Model 1: validation ROC')
    ConfusionMatrixDisplay.from_predictions(val.label,(val_p>=0.5).astype(int),labels=[0,1],display_labels=['Human','AI'],ax=axes[1],colorbar=False,cmap='Blues')
    axes[1].set_title('Validation decisions (threshold = 0.5)')
    fig.savefig(out/'validation_summary.png',dpi=180);plt.close(fig)
    print(json.dumps(metrics,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
