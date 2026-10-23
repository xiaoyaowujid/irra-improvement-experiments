# -*- coding: utf-8 -*-
"""实验15：使用训练身份的完整图库AP效用监督同身份T2筛选器。

目的：检验仅看 (T1,T2,对应来源图A/B) 的轻量选择器，能否在未参与
选择器拟合的440个训练身份上接受有益T2并拒绝有害T2。

这是训练身份内部的已知同PID支持文本 TEACHER 诊断；不训练学生，不调用
Val/Test PID或Val/Test的原始标注，不改变原IRRA/缓存/任何源码。

训练标签（fit TRAIN PID 子集）: 将1760 fit PID分为四个互不重叠的440-PID图库，
分别计算全图库效用，保持与holdout 440-PID图库的候选身份规模一致。
对固定图库排除T1来源图A和T2来源图B，
比较T1与T1+T2的AP，delta>=0.01->beneficial;
             delta<=-0.01->harmful; otherwise neutral.
T2由原EXP12配对算法在每个身份内确定，在标签生成前确定。第三视角C仅在
holdout核对与事后统计使用，绝不用于选择器的输入。完全冻结融合beta。

评估：复现EXP12的2020组及EXP14全图库结果；只在1760 fit PID上训练
三分类线性Softmax，440 holdout PID仅评价；对照T1/All/EXP12原筛选/新筛选。
所有输出独立到“图文双向互补实验/实验15_..._日期时间”，只读原实验12。
仅依赖 numpy 和 Python 标准库（不额外import项目脚本）。

首跑：python irra_exp15_fullgallery_utility_selector.py
自检：python irra_exp15_fullgallery_utility_selector.py --self-test
续跑：python irra_exp15_fullgallery_utility_selector.py --resume
"""
from __future__ import annotations
import os
for _e in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_e] = '1'
import argparse
import csv
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
import zipfile
import numpy as np

PROJECT_ROOT = Path(r'/')
PROJECT_FOLDER = '图文双向互补实验'
EXP12_PREFIX = '实验12_最佳IRRA同身份文本互补迁移_'
EXP15_PREFIX = '实验15_全图库效用监督文本选择_'
EXPECTED_BEST_SHA = 'f3ab12028e3406546b836ed6a75c8fc5e02a42146c5da13af423e7f5cf0e7ea6'
EXPECTED_ANNOTATION_SHA = 'fc3d15542f91dd70e23028c2cb40171a8f41734a39afddb9505e352e396cabf0'
VERSION = 'exp15_full_gallery_utility_440PID_blocks_v2'
EPS_AP = 0.01  # 事前固定的AP有效改善/恶化界限，绝不在holdout上搜索
LABEL_NAMES = ('harmful', 'neutral', 'beneficial')
EXPECTED_TRIPLE = 2020
EXPECTED_OLD_ACCEPTED = 714
LOG_FILE = None


def log(s):
    print(s, flush=True)
    if LOG_FILE is not None:
        with LOG_FILE.open('a', encoding='utf-8') as f:
            f.write(str(s)+'\n')


def hash_file(p, chunk=4*1024*1024):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def hash_obj(v):
    return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def atomic_json(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.exp15_',suffix='.tmp',dir=p.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def atomic_csv(p,rows,columns):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.exp15_',suffix='.tmp',dir=p.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
            w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def atomic_npz(path,**values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.exp15_',suffix='.npz',dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(tmp,**values)
        with np.load(tmp,allow_pickle=False) as chk:
            for k in values:
                if k not in chk.files or chk[k].shape!=np.asarray(values[k]).shape:
                    raise IOError('Corrupt saved NPZ: '+k)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def norm(x):
    a=np.asarray(x,dtype=np.float32)
    return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)


def get_exp12(parent,explicit):
    if explicit:
        candidates=[Path(explicit).expanduser().resolve()]
    else:
        candidates=sorted(parent.glob(EXP12_PREFIX+'*'),reverse=True)
    required=['run_info.json','gate_model.json','internal_teacher_diagnostic.json',
              'preparation_complete.json','selected_train_pids.json','baseline_audit.json']
    for p in candidates:
        if p.is_dir() and all((p/name).is_file() for name in required):return p
    raise FileNotFoundError('找不到完整实验12目录，请用 --exp12-dir 指定原始实验12文件夹。只上传ZIP不含特征缓存。')


def audit_source(src,fast=False):
    run=json.loads((src/'run_info.json').read_text(encoding='utf-8'))
    opt=SimpleNamespace(**run['settings'])
    gate=json.loads((src/'gate_model.json').read_text(encoding='utf-8'))
    prior=json.loads((src/'internal_teacher_diagnostic.json').read_text(encoding='utf-8'))
    prep=json.loads((src/'preparation_complete.json').read_text(encoding='utf-8'))
    audit=json.loads((src/'baseline_audit.json').read_text(encoding='utf-8'))
    selected=json.loads((src/'selected_train_pids.json').read_text(encoding='utf-8'))
    if run.get('experiment')!='实验12_最佳IRRA同身份文本互补迁移':raise ValueError('不是原实验12')
    if selected!=opt.selected_pids or len(selected)!=2200 or len(opt.fit_pids)!=1760 or len(opt.holdout_pids)!=440:
        raise ValueError('实验12身份数量与清单错误')
    if set(opt.fit_pids)&set(opt.holdout_pids) or set(opt.fit_pids)|set(opt.holdout_pids)!=set(selected):
        raise ValueError('训练与holdout PID集合重叠或不完整')
    if prep.get('settings_hash')!=opt.settings_hash or not prep.get('support_selection_ready'):
        raise ValueError('EXP12准备报告与配置不一致')
    if not audit.get('passed') or audit.get('checkpoint_sha256')!=EXPECTED_BEST_SHA:
        raise ValueError('最佳IRRA基线校验未通过')
    if opt.fp_train['checkpoint_sha256']!=EXPECTED_BEST_SHA or opt.fp_val['checkpoint_sha256']!=EXPECTED_BEST_SHA:
        raise ValueError('训练/Validation缓存权重指纹不是最佳IRRA')
    if opt.fp_train['annotations_sha256']!=EXPECTED_ANNOTATION_SHA:
        raise ValueError('数据集标注指纹改变')
    if gate.get('label')!='same-pid-vs-similarity-matched-wrong-pid' or len(gate.get('weights',[]))!=9:
        raise ValueError('原EXP12 Gate不兼容')
    for p in (opt.checkpoint,opt.annotation_file,opt.config_file):
        if not Path(p).is_file():raise FileNotFoundError('源文件缺失 '+str(p))
    if not fast:
        for path,expected in [(opt.checkpoint,EXPECTED_BEST_SHA),
                              (opt.annotation_file,EXPECTED_ANNOTATION_SHA),
                              (opt.config_file,opt.fp_train['config_sha256'])]:
            if hash_file(path)!=expected:raise ValueError('源文件SHA-256不匹配：'+str(path))
    return opt,gate,prior


def all_selected_annotations(path,chosen):
    allow={int(v) for v in chosen}
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data,list):raise ValueError('训练标注不是列表')
    image_ids=[];text_ids=[];selected_images=[];selected_texts=[];all_image_count=0;all_text_count=0
    for rec in data:
        if rec.get('split')!='train':continue
        pid=int(rec['id']);imageidx=all_image_count;all_image_count+=1
        chosen_image=pid in allow
        if chosen_image:
            image_ids.append(imageidx)
            selected_images.append({'pid':pid,'file_path':rec['file_path']})
        if not rec.get('captions') or not isinstance(rec['captions'],list):
            raise ValueError('缺少训练描述')
        for caption in rec['captions']:
            if chosen_image:
                text_ids.append(all_text_count)
                selected_texts.append({'pid':pid,'img_idx':len(image_ids)-1,'text':caption})
            all_text_count+=1
    pos=np.asarray([r['img_idx'] for r in selected_texts],np.int32)
    return (all_image_count,all_text_count,image_ids,text_ids,selected_images,selected_texts,pos)


def cached_array(directory,prefix,fingerprint,count,dim=512):
    directory=Path(directory)
    f=directory/(prefix+'_features.npy');meta=directory/(prefix+'_cache_meta.json')
    if not f.is_file() or not meta.is_file():raise FileNotFoundError('缺少实验12特征 '+str(f))
    m=json.loads(meta.read_text(encoding='utf-8'))
    a=np.load(f,mmap_mode='r',allow_pickle=False)
    if m.get('fingerprint')!=fingerprint or m.get('done')!=count or m.get('shape')!=[count,dim] or a.shape!=(count,dim) or a.dtype!=np.float32:
        raise ValueError('特征缓存指纹/尺寸不符 '+str(f))
    return a


def load_pid_partition(opt,pids):
    ni,nt,image_ids,text_ids,imgs,texts,pos=all_selected_annotations(opt.annotation_file,opt.selected_pids)
    imgfp={'train_base':opt.fp_train,
           'selected_image_indices_sha256':hashlib.sha256(np.asarray(image_ids,np.int32).tobytes()).hexdigest(),
           'selected_image_count':len(image_ids),'format_version':1}
    # Order is exactly EXP12 read_annotations->get_subset. The train_text cache
    # contains ALL train captions; the train_image cache contains 2200 selected PID images.
    tx=cached_array(opt.train_text_cache_dir or opt.run_dir,'train_text',opt.fp_train,nt)
    vi=cached_array(opt.train_image_cache_dir or opt.run_dir,'train_image',imgfp,len(image_ids))
    allow={int(p) for p in pids}
    ti=np.asarray([i for i,r in enumerate(texts) if int(r['pid']) in allow],np.int32)
    ii=np.asarray([i for i,r in enumerate(imgs) if int(r['pid']) in allow],np.int32)
    loc={int(k):i for i,k in enumerate(ii)}
    pp=np.asarray([loc[int(pos[k])] for k in ti],np.int32)
    out_tx=[texts[int(i)] for i in ti];out_im=[imgs[int(i)] for i in ii]
    x=norm(tx[np.asarray(text_ids,np.int32)[ti]])
    v=norm(vi[ii])
    if set(int(z['pid']) for z in out_im)!=allow or set(int(z['pid']) for z in out_tx)!=allow:
        raise AssertionError('实际读取身份集合不正确')
    if not all(out_tx[i]['pid']==out_im[pp[i]]['pid'] for i in range(len(pp))):
        raise AssertionError('文本->图像身份映射不正确')
    return x,v,out_tx,out_im,pp


def construct_pairs(x,texts,pos,seed,wrong_pool=64):
    # Original EXP12 exact draw order (including wrong-PID draws) for replay.
    rng=np.random.default_rng(seed)
    bypid=defaultdict(list);byimage=defaultdict(list)
    pids=np.asarray([int(t['pid']) for t in texts],np.int64)
    for i,t in enumerate(texts):
        bypid[int(t['pid'])].append(i);byimage[int(pos[i])].append(i)
    supp=np.full(len(texts),-1,np.int32);wrong=np.full(len(texts),-1,np.int32)
    errors=np.full(len(texts),np.nan,np.float32)
    for i,t in enumerate(texts):
        candidates=[j for j in bypid[int(t['pid'])] if int(pos[j])!=int(pos[i])]
        if not candidates:continue
        supp[i]=int(candidates[int(rng.integers(0,len(candidates)))])
        target=float(x[i]@x[supp[i]])
        pool=rng.integers(0,len(x),size=wrong_pool*2)
        pool=pool[pids[pool]!=pids[i]][:wrong_pool]
        if not len(pool):raise RuntimeError('不同身份对照选择失败')
        s=x[pool]@x[i]
        k=int(np.argmin(np.abs(s-target)))
        wrong[i]=int(pool[k]);errors[i]=abs(float(s[k])-target)
    ok=supp>=0
    if np.any(pids[supp[ok]]!=pids[ok]) or np.any(pos[supp[ok]]==pos[ok]):
        raise AssertionError('T2不是跨图同身份')
    return supp,wrong,errors


def evidence_features(x,v,pos,idx,supp):
    t=x[idx];a=v[pos[idx]];u=x[supp];b=v[pos[supp]]
    prod=lambda l,r:np.einsum('ij,ij->i',l,r,optimize=True)
    tt=prod(t,u);ta=prod(t,a);tb=prod(t,b)
    ua=prod(u,a);ub=prod(u,b);ab=prod(a,b)
    return np.stack([tt,ta,tb,ua,ub,ab,ub-ua,tb-ta,ub-tb],axis=1).astype(np.float32)


def old_selection(feat,x,idx,supp,gate):
    mu=np.asarray(gate['mean'],np.float64);sd=np.asarray(gate['scale'],np.float64)
    w=np.asarray(gate['weights'],np.float64)
    z=np.clip((np.asarray(feat,dtype=np.float64)-mu)/sd,-10,10)
    prob=1/(1+np.exp(-np.clip(z@w+float(gate['bias']),-35,35)))
    novelty=1-np.einsum('ij,ij->i',x[idx],x[supp])
    return ((prob>=float(gate['threshold']))&
            (novelty>float(gate['novelty_threshold']))),prob,novelty


def choose_third(imgs,tx,pos,ids,supp):
    members=defaultdict(list)
    for i,v in enumerate(imgs):members[int(v['pid'])].append(i)
    qi=[];c=[]
    for i in ids:
        a=int(pos[i]);b=int(pos[supp[i]])
        other=[j for j in members[int(tx[i]['pid'])] if j!=a and j!=b]
        if other:
            qi.append(int(i));c.append(int(other[0]))
    return np.asarray(qi,np.int32),np.asarray(c,np.int32)


def produce_cases(x,v,tx,imgs,pos,seed,gate,third_only=False):
    supp,wrong,errors=construct_pairs(x,tx,pos,seed+1)
    valid=np.flatnonzero(supp>=0)
    idx,c=choose_third(imgs,tx,pos,valid,supp)
    if third_only and not len(idx):raise AssertionError('No triples')
    feat=evidence_features(x,v,pos,idx,supp[idx])
    old,old_prob,nov=old_selection(feat,x,idx,supp[idx],gate)
    imagep=np.asarray([int(im['pid']) for im in imgs],np.int64)
    pids=np.asarray([int(tx[i]['pid']) for i in idx],np.int64)
    a=pos[idx];b=pos[supp[idx]]
    if np.any(a==b) or np.any(a==c) or np.any(b==c):
        raise AssertionError('A B C must be independent')
    return {'idx':idx,'c':c,'supp':supp[idx],'wrong':wrong[idx],
            'feat':feat,'old':old,'old_prob':old_prob,'novelty':nov,
            'a':a,'b':b,'pids':pids,'image_pids':imagep,
            'match_error':errors[idx]}


def pair_replay(x,v,pack,beta,prior,all_valid_old_rate):
    # EXP12 source comparison (same formula and original T1 hard negative) with no
    # C in selector. Assert before new fitting or reports.
    idx=pack['idx'];s=pack['supp'];c=pack['c'];a=x[idx]
    fused=norm((1.-beta)*a+beta*x[s])
    pospid=pack['pids'];imagepid=pack['image_pids']
    mat=a@v.T
    mat[pospid[:,None]==imagepid[None,:]]=-np.inf
    neg=mat.argmax(1)
    vd=v[neg];vc=v[c]
    m0=np.einsum('ij,ij->i',a,vc)-np.einsum('ij,ij->i',a,vd)
    m1=np.einsum('ij,ij->i',fused,vc)-np.einsum('ij,ij->i',fused,vd)
    mf=np.where(pack['old'],m1,m0)
    ref=prior['holdout_teacher_third_view']
    if len(idx)!=EXPECTED_TRIPLE or int(pack['old'].sum())!=EXPECTED_OLD_ACCEPTED:
        raise AssertionError('实验12/13原样本数或接受数不一致')
    if int(ref['n_triple_view'])!=len(idx):raise AssertionError('第三视角样本数错误')
    for name,values in [('t1_only',m0),('all_same_pid',m1),('filtered_same_pid',mf)]:
        record=ref['methods'][name]
        vals={'third_view_pair_win_rate':float(np.mean(values>0)),
              'mean_margin':float(np.mean(values))}
        if name!='t1_only':
            delta=values-m0
            vals.update({'margin_improved_fraction':float(np.mean(delta>1e-7)),
                         'margin_harmed_fraction':float(np.mean(delta<-1e-7)),
                         'mean_margin_delta':float(np.mean(delta))})
        for k,val in vals.items():
            if abs(val-float(record[k]))>0.00003:
                raise AssertionError(f'EXP12指标回放不符 {name}.{k}: {val} vs {record[k]}')
    return {'matched':True,'n':len(idx),'original_accepted':int(pack['old'].sum())}


def ap_inp_and_r1(scores,pid,imagepid,a,b):
    """Return per-query actual gallery AP/INP/top-rank; A and B removed.

    Exactly same stable-sort semantics as EXP14; number of positives varies by PID.
    AP=0 if all positives vanished; this is prevented by >=3 image case check.
    """
    g=len(imagepid)
    if len(scores)!=g:raise ValueError('Gallery length error')
    if a==b:raise AssertionError('A/B must be distinct')
    eligible=(np.arange(g)!=a)&(np.arange(g)!=b)
    order=np.argsort(-scores,kind='stable')
    seq=order[eligible[order]]
    hit=imagepid[seq]==pid
    loc=np.flatnonzero(hit)+1
    if len(loc)<1 or len(seq)<2 or not np.any(~hit):
        raise AssertionError('Full gallery has invalid positive/negative split')
    ap=float(np.mean(np.arange(1,len(loc)+1,dtype=np.float64)/loc))
    inp=float(len(loc)/loc[-1]);top=int(loc[0])
    return ap,inp,top


def gallery_utility(x,v,pack,beta,batch=96):
    idx=pack['idx'];s=pack['supp'];a=pack['a'];b=pack['b'];qp=pack['pids'];gp=pack['image_pids']
    n=len(idx);ap0=np.empty(n,np.float64);ap1=np.empty(n,np.float64)
    inp0=np.empty(n,np.float64);inp1=np.empty(n,np.float64)
    rank0=np.empty(n,np.int32);rank1=np.empty(n,np.int32)
    for start in range(0,n,batch):
        sl=slice(start,min(n,start+batch))
        t=x[idx[sl]];mix=norm((1-beta)*t+beta*x[s[sl]])
        base=t@v.T; allscore=mix@v.T
        for j,k in enumerate(range(start,min(n,start+batch))):
            ap0[k],inp0[k],rank0[k]=ap_inp_and_r1(base[j],qp[k],gp,int(a[k]),int(b[k]))
            ap1[k],inp1[k],rank1[k]=ap_inp_and_r1(allscore[j],qp[k],gp,int(a[k]),int(b[k]))
    return {'ap0':ap0,'ap1':ap1,'inp0':inp0,'inp1':inp1,'r0':rank0,'r1':rank1,
            'delta':ap1-ap0}


def make_labels(delta,threshold=EPS_AP):
    y=np.ones(len(delta),np.int32)  # 1 neutral
    y[np.asarray(delta)<=-threshold]=0  # 0 harmful
    y[np.asarray(delta)>=threshold]=2  # 2 beneficial
    return y


def fit_selector(X,y,steps=1000):
    """Deterministic 3-class multinomial logistic regression; no extra packages.

    Per-class reweighting prevents neutral/beneficial majority from trivially
    masking harmful class. No threshold/hyperparameter selection on holdout.
    """
    X=np.asarray(X,np.float64);y=np.asarray(y,np.int64)
    if X.ndim!=2 or X.shape[1]!=9 or len(y)!=len(X):raise AssertionError('Gate input size wrong')
    cnt=np.bincount(y,minlength=3)
    if np.any(cnt<5):
        raise RuntimeError(f'Too few fit labels for balanced training: {cnt}. Stop rather than claim learned rejection.')
    mu=X.mean(0);sd=np.maximum(X.std(0),1e-3)
    Z=np.clip((X-mu)/sd,-10,10)
    w=np.zeros((9,3),np.float64);bias=np.zeros(3,np.float64)
    weights=(len(y)/(3.*cnt))[y]
    t=np.arange(len(y))
    for j in range(steps):
        logits=Z@w+bias;logits-=np.max(logits,axis=1,keepdims=True)
        probs=np.exp(logits);probs/=np.sum(probs,axis=1,keepdims=True)
        probs[t,y]-=1.
        diff=probs*(weights[:,None]/len(y))
        dw=Z.T@diff+0.03*w;db=np.sum(diff,axis=0)
        lr=0.15/(1+j/400)
        w-=lr*dw;bias-=lr*db
        if not (np.all(np.isfinite(w)) and np.all(np.isfinite(bias))):
            raise FloatingPointError('Selector NaN/Inf')
    return {'version':VERSION,'labels':list(LABEL_NAMES),'feature_mean':mu.tolist(),
            'feature_scale':sd.tolist(),'weights':w.tolist(),'bias':bias.tolist(),
            'class_counts_fit':cnt.tolist(),'training_steps':steps,
            'ap_label_threshold':EPS_AP,
            'decision':'accept iff beneficial has the largest predicted probability',
            'source':'9 source-only features T1 T2 A B; all-gallery AP used ONLY for fit labels'}


def predict_selector(gate,X):
    X=np.asarray(X,np.float64)
    mu=np.asarray(gate['feature_mean']);sd=np.asarray(gate['feature_scale'])
    w=np.asarray(gate['weights']);bias=np.asarray(gate['bias'])
    z=np.clip((X-mu)/sd,-10,10)
    logits=z@w+bias;logits-=np.max(logits,axis=1,keepdims=True)
    out=np.exp(logits);out/=out.sum(1,keepdims=True)
    return out,np.argmax(out,axis=1).astype(np.int32)


def metric_rows(util,pack,predicted,include_old=True):
    pids=pack['pids'];delta=util['delta'];gates={
       'T1_only':np.zeros(len(delta),bool),
       'All_T2':np.ones(len(delta),bool),
       'Original_EXP12':pack['old'].astype(bool),
       'EXP15_utility_selector':np.asarray(predicted,dtype=bool)}
    records=[]
    for name,accept in gates.items():
        ap=np.where(accept,util['ap1'],util['ap0'])
        inp=np.where(accept,util['inp1'],util['inp0'])
        rank=np.where(accept,util['r1'],util['r0'])
        records.append({'method':name,'n_cases':len(ap),'unique_pid':len(set(map(int,pids))),
                        'accepted_n':int(accept.sum()),'accepted_pct':100*float(np.mean(accept)),
                        'R1_pct':100*float(np.mean(rank<=1)),'R5_pct':100*float(np.mean(rank<=5)),
                        'R10_pct':100*float(np.mean(rank<=10)),
                        'mAP_pct':100*float(np.mean(ap)),'mINP_pct':100*float(np.mean(inp)),
                        'mean_delta_AP_from_T1_pp':100*float(np.mean(ap-util['ap0']))})
    return records,gates


def gate_diagnostics(gates,labels,util,pack):
    n=len(labels);positive=labels==2;negative=labels==0;neutral=labels==1
    rows=[]
    for name,acc in gates.items():
        rows.append({'method':name,'n':n,
          'beneficial_n':int(positive.sum()),'harmful_n':int(negative.sum()),
          'neutral_n':int(neutral.sum()),'accepted_n':int(acc.sum()),
          'beneficial_accept_recall_pct':100*float(np.mean(acc[positive])) if positive.any() else None,
          'harmful_reject_recall_pct':100*float(np.mean(~acc[negative])) if negative.any() else None,
          'accepted_beneficial_precision_pct':100*float(np.mean(positive[acc])) if acc.any() else None,
          'accepted_harmful_fraction_pct':100*float(np.mean(negative[acc])) if acc.any() else None,
          'neutral_accept_rate_pct':100*float(np.mean(acc[neutral])) if neutral.any() else None,
          'beneficial_missed':int((positive&~acc).sum()),
          'harmful_accepted':int((negative&acc).sum()),
          'net_fullgallery_delta_AP_pp':100*float(np.mean(np.where(acc,util['delta'],0.))),
          'total_delta_AP_pp':100*float(np.sum(np.where(acc,util['delta'],0.)))})
    return rows


def bootstrap_paired(util,pack,gates,reps=1200,seed=20261021):
    """Exploratory PID-cluster CI, not a fresh test set."""
    pid=pack['pids'];uniq=np.unique(pid)
    grouped=[np.flatnonzero(pid==p) for p in uniq]
    d=util['delta']
    cur=np.where(gates['EXP15_utility_selector'],d,0.)
    old=np.where(gates['Original_EXP12'],d,0.)
    allv=d
    # Baseline outcomes are per-row, so pairwise corrections can be resampled.
    diffs=[('new_minus_old',cur-old),('new_minus_all',cur-allv),('new_minus_T1',cur)]
    rng=np.random.default_rng(seed)
    result={}
    for name,vec in diffs:
        means=np.empty(reps,np.float64)
        # Aggregating PID sums/counts is faster, and preserves within-PID dependence.
        sums=np.asarray([vec[k].sum() for k in grouped],np.float64)
        count=np.asarray([len(k) for k in grouped],np.int32)
        for j in range(reps):
            z=rng.integers(0,len(uniq),size=len(uniq))
            means[j]=100*float(sums[z].sum()/count[z].sum())
        result[name]={'observed_pp':100*float(vec.mean()),
                      'exploratory_CI95_pp':np.quantile(means,[.025,.975]).tolist(),
                      'n_PID_clusters':len(uniq),'n_bootstrap':reps}
    return result


def make_detail_rows(util,pack,tx,ims,probs,labels,gates):
    rows=[]
    for k,i in enumerate(pack['idx']):
        s=int(pack['supp'][k]);r={
          'query_index_holdout':int(i),'support_index_holdout':s,
          'pid_train_only':int(pack['pids'][k]),
          'image_A':ims[int(pack['a'][k])]['file_path'],
          'image_B':ims[int(pack['b'][k])]['file_path'],
          'image_C':ims[int(pack['c'][k])]['file_path'],
          'T1':tx[int(i)]['text'],'T2':tx[s]['text'],
          'label_fullgallery_AP_utility':LABEL_NAMES[int(labels[k])],
          'delta_AP_pp':float(100*util['delta'][k]),
          'T1_AP_pct':float(100*util['ap0'][k]),'All_AP_pct':float(100*util['ap1'][k]),
          'T1_best_rank':int(util['r0'][k]),'All_best_rank':int(util['r1'][k]),
          'old_accepted':int(gates['Original_EXP12'][k]),
          'new_accepted':int(gates['EXP15_utility_selector'][k]),
          'new_prob_harmful':float(probs[k,0]),
          'new_prob_neutral':float(probs[k,1]),
          'new_prob_beneficial':float(probs[k,2]),
          'old_gate_prob':float(pack['old_prob'][k]),
          'old_novelty':float(pack['novelty'][k])}
        rows.append(r)
    return rows


def package(out):
    path=out/'发送给ChatGPT.zip';fd,tmp=tempfile.mkstemp(prefix='.exp15_',suffix='.zip',dir=out)
    os.close(fd)
    names=['实验15_目标与边界.md','run_info.json','experiment15_report.json',
           'selector_fit_summary.json','trained_selector.json','fit_learning_diagnostics.csv',
           'holdout_fullgallery_metrics.csv','holdout_accept_reject_metrics.csv',
           'all_holdout_cases.csv','example_review_cases.csv',
           'experiment_run.log',Path(__file__).name]
    try:
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name in names:
                if (out/name).is_file():z.write(out/name,arcname=name)
        with zipfile.ZipFile(tmp,'r') as z:
            if z.testzip() is not None:raise IOError('ZIP self-test failed')
            if 'experiment15_report.json' not in z.namelist():raise IOError('ZIP missing report')
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return path


def create_folder(parent,src,settings,resume):
    parent.mkdir(parents=True,exist_ok=True)
    identify={'version':VERSION,'source_exp12':str(src),'source_settings_hash':settings.settings_hash,
              'source_best_checkpoint_sha256':EXPECTED_BEST_SHA,'ap_cut':EPS_AP}
    if resume:
        for p in sorted(parent.glob(EXP15_PREFIX+'*'),reverse=True):
            if (p/'run_info.json').is_file():
                info=json.loads((p/'run_info.json').read_text(encoding='utf-8'))
                if info==identify:return p,True
        raise RuntimeError('没有找到可恢复的实验15，请首次运行时清空 --resume。')
    out=parent/(EXP15_PREFIX+datetime.now().strftime('%Y%m%d_%H%M%S'))
    if out.exists():raise FileExistsError('时间戳路径已存在: '+str(out))
    out.mkdir(parents=True)
    atomic_json(out/'run_info.json',identify)
    return out,False


def run(src,opt,old_gate,prior,out):
    stage=out/'fit_cases.npz';check=out/'fit_cases_meta.json'
    fp={'version':VERSION,'source_settings':opt.settings_hash,
        'fit_pids_sha':hash_obj(opt.fit_pids),'holdout_pids_sha':hash_obj(opt.holdout_pids),
        'teacher_beta':float(opt.teacher_beta),'support_seed':int(opt.support_seed),
        'ap_threshold':EPS_AP,'fit_gallery_blocks':4,'fit_PIDs_per_gallery':440}
    # Generate fit labels from fit identities only. No holdout PID labels/ranks
    # enter the fit stage or threshold selection.
    if stage.is_file() and check.is_file() and json.loads(check.read_text(encoding='utf-8'))==fp:
        with np.load(stage,allow_pickle=False) as z:
            fX=z['feat'];y=z['labels'];d=z['delta'];pids=z['pids']
        log(f'复用已经核验的拟合数据 {len(y)} 条')
    else:
        log('将1760 fit PID固定分为4个无重叠的440-PID图库，分别生成完整图库效用标签')
        shuffled=np.random.default_rng(20261021).permutation(np.asarray(opt.fit_pids,dtype=np.int64))
        blocks=np.array_split(shuffled,4)
        if any(len(block)!=440 for block in blocks):raise AssertionError('每个拟合图库必须恰好包含440 PID')
        feats=[];changes=[];identity=[]
        for block_i,block in enumerate(blocks,1):
            x,v,tx,imgs,pos=load_pid_partition(opt,block.tolist())
            fitpack=produce_cases(x,v,tx,imgs,pos,int(opt.support_seed),old_gate)
            fitutil=gallery_utility(x,v,fitpack,float(opt.teacher_beta))
            feats.append(fitpack['feat']);changes.append(fitutil['delta']);identity.append(fitpack['pids'])
            log('拟合图库 {}/4: PID={} 图像={} 查询={} labels={}'.format(
                block_i,len(block),len(imgs),len(fitpack['idx']),
                np.bincount(make_labels(fitutil['delta']),minlength=3).tolist()))
        fX=np.concatenate(feats,axis=0);d=np.concatenate(changes);pids=np.concatenate(identity)
        y=make_labels(d)
        if len(np.unique(pids))>1760:raise AssertionError('错误的拟合身份数')
        log('Fitting cases: %s, class counts: %s, four matched-size fit galleries.' %
            (len(y),np.bincount(y,minlength=3).tolist()))
        atomic_npz(stage,feat=fX,labels=y,delta=d,pids=pids)
        atomic_json(check,fp)
    if set(map(int,pids))-set(map(int,opt.fit_pids)):
        raise AssertionError('Fit label PID leakage')
    model_path=out/'trained_selector.json'
    if model_path.exists():
        gate=json.loads(model_path.read_text(encoding='utf-8'))
        if gate['version']!=VERSION or gate['class_counts_fit']!=np.bincount(y,minlength=3).tolist():
            raise ValueError('旧选择器和当前训练标签不兼容，禁止续跑')
        log('复用已有全图库监督选择器')
    else:
        gate=fit_selector(fX,y)
        atomic_json(model_path,gate)
    fitprobs,fitlabel=predict_selector(gate,fX)
    fselect=fitlabel==2
    training={'n_fit_cases':len(y),'n_fit_PID':int(len(np.unique(pids))),
              'fit_class_counts':dict(zip(LABEL_NAMES,map(int,np.bincount(y,minlength=3)))),
              'fit_accepted':int(fselect.sum()),'fit_harmful_rejected':int(((y==0)&~fselect).sum()),
              'fit_beneficial_accepted':int(((y==2)&fselect).sum()),
              'fit_positive_prob_mean':float(np.mean(fitprobs[:,2])),
              'WARNING':'TRAIN identity fitted metrics only; not unbiased test.'}
    atomic_json(out/'selector_fit_summary.json',training)
    fit_rows=[]
    for name,z in [('harmful',0),('neutral',1),('beneficial',2)]:
        sel=y==z
        fit_rows.append({'label':name,'count':int(sel.sum()),
          'selector_pred_beneficial_pct':100*float(np.mean(fselect[sel])) if sel.any() else None,
          'mean_true_delta_AP_pp':100*float(np.mean(d[sel])) if sel.any() else None})
    atomic_csv(out/'fit_learning_diagnostics.csv',fit_rows,list(fit_rows[0]))
    log('在440个独立holdout身份上，原样重建EXP12/13筛选与第三视角')
    hx,hv,ht,hi,hpos=load_pid_partition(opt,opt.holdout_pids)
    pack=produce_cases(hx,hv,ht,hi,hpos,int(opt.support_seed),old_gate,third_only=True)
    # Ensure EXP12 original accepted ratio for ALL valid captions, not only triple.
    hs,hw,_=construct_pairs(hx,ht,hpos,int(opt.support_seed)+1)
    valid=np.flatnonzero(hs>=0)
    allfeat=evidence_features(hx,hv,hpos,valid,hs[valid]);orig,_,_=old_selection(allfeat,hx,valid,hs[valid],old_gate)
    if abs(float(orig.mean())-float(prior['holdout']['true_accept_rate_proxy']))>0.00003:
        raise AssertionError('原EXP12完整holdout接受率不符')
    replay=pair_replay(hx,hv,pack,float(opt.teacher_beta),prior,float(orig.mean()))
    log(f'成功回放EXP12: {replay["n"]}个三视角样本，原筛选接受{replay["original_accepted"]}')
    if set(map(int,pack['pids']))-set(map(int,opt.holdout_pids)):
        raise AssertionError('Holdout identity error')
    if set(map(int,pids)) & set(map(int,pack['pids'])):
        raise AssertionError('Gate training identities overlap holdout')
    prob,pred=predict_selector(gate,pack['feat'])
    newacc=pred==2
    holdutil=gallery_utility(hx,hv,pack,float(opt.teacher_beta))
    gt=make_labels(holdutil['delta'])
    rows,choices=metric_rows(holdutil,pack,newacc)
    dec=gate_diagnostics(choices,gt,holdutil,pack)
    log('完整内部图库(每条查询排除A/B)结果：')
    for r in rows:
        log('  {} accepted {}/{} R1={:.4f} mAP={:.4f}'.format(
            r['method'],r['accepted_n'],r['n_cases'],r['R1_pct'],r['mAP_pct']))
    atomic_csv(out/'holdout_fullgallery_metrics.csv',rows,list(rows[0]))
    atomic_csv(out/'holdout_accept_reject_metrics.csv',dec,list(dec[0]))
    detail=make_detail_rows(holdutil,pack,ht,hi,prob,gt,choices)
    atomic_csv(out/'all_holdout_cases.csv',detail,list(detail[0]))
    review=sorted(detail,key=lambda r:(r['label_fullgallery_AP_utility']=='harmful',
                          r['old_accepted']!=r['new_accepted'],abs(r['delta_AP_pp'])),reverse=True)[:200]
    atomic_csv(out/'example_review_cases.csv',review,list(detail[0]))
    bstrap=bootstrap_paired(holdutil,pack,choices)
    report={
      'experiment':'15','question':'Can source-only selector learn beneficial/harmful same-PID text complementarity from fit-ID full-gallery retrieval utility?',
      'source_exp12':str(src),'best_IRRA_SHA256':EXPECTED_BEST_SHA,
      'train_PID_count':1760,'holdout_PID_count':440,
      'fitting_diagnostics':training,'exp12_replay':replay,
      'holdout_eligible_cases':len(pack['idx']),'holdout_eligible_PID':len(np.unique(pack['pids'])),
      'holdout_gallery_images':len(hi),
      'holdout_utility_label_counts':dict(zip(LABEL_NAMES,map(int,np.bincount(gt,minlength=3)))),
      'method_metrics':rows,'selector_decision_metrics':dec,
      'PID_cluster_bootstrap_exploratory':bstrap,
      'critical_limits':[
        'Teacher uses true TRAIN PID to construct T2. This is NOT a deployable Val/Test retrieval result.',
        'Holdout PID means not used for fitting EXP15 selector, but frozen IRRA originally trained on all TRAIN identities.',
        'Labels derive from full-gallery AP for known TRAIN PID, not semantic facts or visually verified true/false attributes.',
        'A/B excluded from each query ranking, C and other same-PID images retained as positives.',
        'Four non-overlapping 440-PID fit galleries match the holdout identity count; image counts may still differ.',
        'Score-beta, pair sampler, EXP12 original gate are frozen; C/gallery ranks never entered new selector inputs.',
        'Only 3-class 9-feature softmax is fitted; no student, IRRA or attribute model is trained.',
        'No guarantee teacher benefit transfers to single-text student with no T2 at inference.',
        'A finite positive AP delta may arise from text-embedding smoothing, not new identity semantics.',
        'Fit and holdout galleries each have 440 identities, but image count and composition vary; AP utility label prevalence may differ.',
        'This is internal teacher diagnostic and NOT official Validation R1, Test R1 or paper-ready improvement.'
      ]}
    atomic_json(out/'experiment15_report.json',report)
    (out/'实验15_目标与边界.md').write_text(
      '# 实验15：同身份文本效用筛选\n\n仅研究训练端T2筛选机制。\n\n'
      '1760个拟合PID分成四个独立440-PID图库（固定随机划分）；已知TRAIN PID仅用于取出跨图T2并计算完整图库AP提升/损害标签。'
      '保持原EXP12融合系数及跨图抽样；A/B来源图排除，C从不参与选择器输入。\n\n'
      '不改变IRRA/Adapter。440内部未见训练身份严格只用于最终评估。'
      'T1、All、原EXP12筛选、新监督筛选比较完整内部图库。\n\n'
      '有益/有害代表已知身份上的AP效用代理，不代表人工核验的真实属性；'
      '本实验不提供Val/Test单文本检索增益，也不研究Top16测试图库标注文本。\n',encoding='utf-8')
    shutil.copy2(__file__,out/Path(__file__).name)
    z=package(out)
    log('报告完成: '+str(z))
    return report


def self_test():
    """Synthetic end-to-end tests including annotation/cache, AP logic, classifier, I/O and ZIP."""
    x=norm(np.random.default_rng(1).normal(size=(18,8)).astype(np.float32))
    v=norm(np.random.default_rng(2).normal(size=(9,8)).astype(np.float32))
    pids=np.repeat([1,2,3],3)
    texts=[{'pid':int(pids[j//2]),'img_idx':j//2,'text':'caption'+str(j)} for j in range(18)]
    imgs=[{'pid':int(pids[j]),'file_path':f'img{j}.jpg'} for j in range(9)]
    pos=np.repeat(np.arange(9),2)
    gate={'mean':[0.]*9,'scale':[1.]*9,'weights':[0.]*9,'bias':2,
          'threshold':0.5,'novelty_threshold':-1.,'label':'same-pid-vs-similarity-matched-wrong-pid'}
    pack=produce_cases(x,v,texts,imgs,pos,123,gate)
    assert len(pack['idx'])==18 and len(set(pack['pids']))==3
    u=gallery_utility(x,v,pack,.3,batch=4)
    assert len(u['delta'])==18
    assert np.max(np.abs(u['delta']-(u['ap1']-u['ap0'])))<1e-9
    assert np.all((u['ap0']>=0)&(u['ap0']<=1))
    for k in range(18):
        assert pack['a'][k]!=pack['b'][k]!=pack['c'][k]
        assert pids[pack['a'][k]]==pids[pack['b'][k]]==pids[pack['c'][k]]
    # Hand-computed two positives and three negatives, A/B excluded.
    sc=np.array([.99,.98,.8,.7,.6,.5,.4],np.float32)
    gp=np.array([1,1,1,2,1,2,2],np.int64)
    ap,inp,r=ap_inp_and_r1(sc,1,gp,0,1)
    # Remaining: idx2 positive at rank1, idx3 negative rank2, idx4 positive rank3.
    assert abs(ap-(1+2/3)/2)<1e-12 and abs(inp-2/3)<1e-12 and r==1
    # Construct adequate 3-class supervision for logistic trainer.
    rng=np.random.default_rng(4)
    feats=np.vstack([rng.normal(loc=z,size=(18,9)) for z in [-2,0,2]])
    labels=np.repeat(np.arange(3),18)
    model=fit_selector(feats,labels,steps=200)
    prob,pred=predict_selector(model,feats)
    assert prob.shape==(54,3) and np.allclose(prob.sum(1),1,atol=1e-6)
    assert np.mean(pred==labels)>.7
    # End-to-end synthetic method/decision/cluster report generation.
    sel=pack['feat'][:,0]>0
    methods,gates=metric_rows(u,pack,sel)
    diagnostics=gate_diagnostics(gates,make_labels(u['delta']),u,pack)
    assert len(methods)==4 and len(diagnostics)==4
    for r in methods:
        assert 0<=r['R1_pct']<=100 and 0<=r['mAP_pct']<=100
    bootstrap=bootstrap_paired(u,pack,gates,reps=20,seed=2)
    assert 'new_minus_old' in bootstrap
    probs3=np.zeros((len(pack['idx']),3),dtype=np.float64)
    probs3[:,2]=sel.astype(np.float64);probs3[:,1]=~sel
    details=make_detail_rows(u,pack,texts,imgs,probs3,make_labels(u['delta']),gates)
    assert len(details)==len(pack['idx']) and 'T2' in details[0]
    # No PID input to train or predict functions.
    with tempfile.TemporaryDirectory() as d:
        out=Path(d)
        atomic_npz(out/'cache.npz',a=x,b=v)
        with np.load(out/'cache.npz',allow_pickle=False) as f:
            assert np.array_equal(f['a'],x)
        atomic_json(out/'trained_selector.json',model)
        atomic_csv(out/'holdout_fullgallery_metrics.csv',[{'method':'sample','R1':50}],['method','R1'])
        atomic_json(out/'experiment15_report.json',{'synthetic':True})
        (out/'实验15_目标与边界.md').write_text('自检',encoding='utf-8')
        (out/'experiment_run.log').write_text('self-test',encoding='utf-8')
        shutil.copy2(__file__,out/Path(__file__).name)
        path=package(out)
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None and 'experiment15_report.json' in z.namelist()
        # folder creation / --resume semantics and manifest compatibility
        parent=out/'runs';dummy=SimpleNamespace(settings_hash='synthetic')
        first,prev=create_folder(parent,out,dummy,False)
        second,recovered=create_folder(parent,out,dummy,True)
        assert first==second and not prev and recovered
    print('SELF-TEST PASS: pairing, A/B/C isolation, AP/mINP, 3-way classifier, safe storage, ZIP, resume.',flush=True)


def main():
    p=argparse.ArgumentParser(description='EXP15 full-gallery utility supervision for same-PID text support')
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--exp12-dir',default='',help='完整实验12输出目录（不是ZIP）')
    p.add_argument('--project-root',default=str(PROJECT_ROOT))
    p.add_argument('--skip-big-sha',action='store_true',help='仅调试用；正式实验不要使用')
    args=p.parse_args()
    if args.self_test:
        self_test();return
    root=Path(args.project_root).expanduser().resolve()
    folder=root/PROJECT_FOLDER
    src=get_exp12(folder,args.exp12_dir)
    opt,old_gate,prior=audit_source(src,fast=args.skip_big_sha)
    out,recovered=create_folder(folder,src,opt,args.resume)
    global LOG_FILE
    LOG_FILE=out/'experiment_run.log'
    log(f'EXP15 start; EXP12={src}; output={out}; resume={recovered}')
    if args.skip_big_sha:log('WARNING: Big checkpoint SHA skipped by explicit debugging flag')
    report_path=out/'experiment15_report.json'
    if recovered and report_path.exists():
        # Verify artifact completeness before considering experiment done.
        wanted=['holdout_fullgallery_metrics.csv','holdout_accept_reject_metrics.csv',
                'trained_selector.json','all_holdout_cases.csv','发送给ChatGPT.zip']
        if all((out/a).is_file() for a in wanted):
            with zipfile.ZipFile(out/'发送给ChatGPT.zip') as z:
                if z.testzip() is None:
                    log('Already finished; resume verified result files.');return
    run(src,opt,old_gate,prior,out)


if __name__=='__main__':
    main()
