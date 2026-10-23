# -*- coding: utf-8 -*-
"""实验20：旧视觉组不变，研究查询到真实身份组的多图多描述证据。

核心：分开验证图像组纯度和文本T1是否属于该组；仅训练身份原有描述可作支持来源。
训练身份内部进行；T1查询来源A、可取得描述来源B、评价图库C相互隔离。
支持B的captions仅属于训练集；官方Val/Test图库的人工描述不可用作补充。
四折fit身份校准查询归属模型及拒识阈值，440个holdout身份独立评估。
身份真值只允许训练fit模型与事后评估；不得参与holdout匹配决策。
只训练一个冻结特征上的轻量组归属评分器；不训练IRRA、学生或图像互补模型。
执行：先 --self-test，再无参数正式运行；已有本实验目录中断才使用 --resume。
依赖：仅numpy + Python标准库；绝不import其他实验Python脚本。
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
import argparse, csv, hashlib, json, shutil, tempfile, zipfile, sys, gzip
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT=Path(r'/')
PARENT='图文双向互补实验'
EXP12='实验12_最佳IRRA同身份文本互补迁移_'
EXP16='实验16_Top16多图与合法文本来源_'
EXP19='实验19_全图库双层身份关联与文本归属_'
EXP20='实验20_查询身份组归属与多描述证据_'
VERSION='exp20_v1_query_to_group_multi_evidence_controlled'
BEST_SHA='f3ab12028e3406546b836ed6a75c8fc5e02a42146c5da13af423e7f5cf0e7ea6'
ANNOTATION_SHA='fc3d15542f91dd70e23028c2cb40171a8f41734a39afddb9505e352e396cabf0'
CONFIG_SHA='ec2fae46ec4e0cce06372789afe1c877a5cebeddf73654781e11b0484fc30fac'
TOPK=16
BUDGETS=(1,2,4)  # 单条仅作对照，多条共同互补是核心
SEED=20261021

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''): h.update(data)
    return h.hexdigest()

def stable_json(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()

def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.exp19_',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def atomic_csv(path,rows,fields):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.exp19_',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
            writer.writeheader();writer.writerows(rows)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def find_exp12(parent,explicit):
    options=[Path(explicit).expanduser().resolve()] if explicit else sorted(parent.glob(EXP12+'*'),reverse=True)
    for p in options:
        if (p/'run_info.json').is_file() and (p/'selected_train_pids.json').is_file() and (p/'baseline_audit.json').is_file():
            return p
    raise FileNotFoundError('未找到完整实验12原始文件夹，请使用 --exp12-dir 指定。仅有发送给ChatGPT.zip 不含大型特征缓存。')

def cache_array(root,prefix,fp,n,d=512):
    root=Path(root)
    f=root/(prefix+'_features.npy');m=root/(prefix+'_cache_meta.json')
    if not f.is_file() or not m.is_file():
        raise FileNotFoundError('缺少实验12的只读特征缓存: '+str(f))
    meta=json.loads(m.read_text(encoding='utf-8'))
    arr=np.load(f,mmap_mode='r',allow_pickle=False)
    if meta.get('fingerprint')!=fp or meta.get('done')!=n or meta.get('shape')!=[n,d]:
        raise RuntimeError('特征缓存指纹或完成状态不一致: '+str(f))
    if arr.shape!=(n,d) or arr.dtype!=np.float32:
        raise RuntimeError('特征缓存尺寸或类型不符: '+str(f))
    return arr

def read_records(annfile,selected):
    """Keep original CUHK-PEDES ordering exactly; Val caption only becomes a query."""
    data=json.loads(Path(annfile).read_text(encoding='utf-8'))
    if not isinstance(data,list): raise RuntimeError('官方标注必须是图像列表')
    choose=set(map(int,selected));tr_ims=[];tr_txt=[];vi=[];vt=[]
    full_train_images=0;full_train_texts=0;chosen_img_orig=[];chosen_text_orig=[]
    for record in data:
        split=record.get('split');pid=int(record['id']);name=record['file_path']
        captions=record.get('captions',[])
        if not isinstance(captions,list) or not captions:raise RuntimeError('训练/验证标注缺少caption')
        if split=='train':
            global_i=full_train_images;full_train_images+=1
            eligible=pid in choose
            if eligible:
                local_i=len(tr_ims)
                chosen_img_orig.append(global_i)
                tr_ims.append({'pid':pid,'path':name})
            for c in captions:
                if eligible:
                    chosen_text_orig.append(full_train_texts)
                    tr_txt.append({'pid':pid,'img_idx':local_i,'text':c})
                full_train_texts+=1
        elif split=='val':
            local_i=len(vi)
            vi.append({'pid':pid,'path':name})
            for _ in captions:
                # Crucial: NO val-gallery caption text is placed into a lookup or output.
                # The cached frozen embeddings already exist and are only used as queries.
                vt.append({'pid':pid,'img_idx':local_i})
    if (len(vi),len(vt))!=(3078,6158):raise RuntimeError('Val size differs from correct baseline 3078/6158')
    if len(chosen_img_orig)!=len(tr_ims) or not tr_txt:raise RuntimeError('Selected train partition empty')
    return {'train_images':tr_ims,'train_texts':tr_txt,'val_images':vi,'val_queries':vt,
            'n_train_text':full_train_texts,'train_image_indices':chosen_img_orig,
            'train_text_indices':chosen_text_orig}

def source_audit(src,full_hash=True):
    run=json.loads((src/'run_info.json').read_text(encoding='utf-8'))
    settings=run['settings'];sel=json.loads((src/'selected_train_pids.json').read_text(encoding='utf-8'))
    au=json.loads((src/'baseline_audit.json').read_text(encoding='utf-8'))
    if run.get('experiment')!='实验12_最佳IRRA同身份文本互补迁移':raise RuntimeError('不是实验12原目录')
    if sel!=settings.get('selected_pids') or len(sel)!=2200:raise RuntimeError('训练身份清单不匹配')
    fit=set(map(int,settings['fit_pids']));hold=set(map(int,settings['holdout_pids']))
    if len(fit)!=1760 or len(hold)!=440 or fit&hold or fit|hold!=set(sel):
        raise RuntimeError('训练身份fit/holdout划分无效')
    if not au.get('passed') or au.get('checkpoint_sha256')!=BEST_SHA:
        raise RuntimeError('EXP12最佳IRRA基线核查未通过')
    if settings['fp_train']['checkpoint_sha256']!=BEST_SHA or settings['fp_val']['checkpoint_sha256']!=BEST_SHA:
        raise RuntimeError('缓存不是最佳Epoch58权重')
    if settings['fp_train']['annotations_sha256']!=ANNOTATION_SHA:
        raise RuntimeError('训练标注指纹不匹配')
    if settings['fp_val']['annotations_sha256']!=ANNOTATION_SHA:
        raise RuntimeError('Val标注指纹不匹配')
    if settings['fp_train']['config_sha256']!=CONFIG_SHA or settings['fp_val']['config_sha256']!=CONFIG_SHA:
        raise RuntimeError('续训最佳配置不匹配')
    for k in ('checkpoint','annotation_file','config_file'):
        if not Path(settings[k]).is_file():raise FileNotFoundError('原始文件缺失: '+settings[k])
    if full_hash:
        for k,expected in [('checkpoint',BEST_SHA),('annotation_file',ANNOTATION_SHA),('config_file',CONFIG_SHA)]:
            actual=sha256_file(settings[k])
            if actual!=expected:raise RuntimeError(f'哈希校验失败: {k} {actual}')
    return settings,sel

def load_features(src,settings,rec):
    valdir=Path(settings.get('val_cache_dir') or src)
    txtdir=Path(settings.get('train_text_cache_dir') or src)
    imgdir=Path(settings.get('train_image_cache_dir') or src)
    fp_img={'train_base':settings['fp_train'],
            'selected_image_indices_sha256':hashlib.sha256(np.asarray(rec['train_image_indices'],np.int32).tobytes()).hexdigest(),
            'selected_image_count':len(rec['train_image_indices']),'format_version':1}
    txfull=cache_array(txtdir,'train_text',settings['fp_train'],rec['n_train_text'])
    timg=cache_array(imgdir,'train_image',fp_img,len(rec['train_images']))
    vtxt=cache_array(valdir,'text',settings['fp_val'],len(rec['val_queries']))
    vimg=cache_array(valdir,'image',settings['fp_val'],len(rec['val_images']))
    # Only train captions are available as ground-truth support during training.
    tx=np.asarray(txfull[np.asarray(rec['train_text_indices'],np.int64)],dtype=np.float32)
    return tx,np.asarray(timg,dtype=np.float32),vtxt,vimg

def l2check(x,name):
    # Existing IRRA caches are normalized. Validate instead of silently altering.
    a=np.asarray(x[:min(256,len(x))])
    if not np.isfinite(a).all() or np.max(np.abs(np.linalg.norm(a,axis=1)-1))>2e-3:
        raise RuntimeError(f'{name}不是预期的L2归一化有效特征')

def rank_topk(x,v,k=TOPK,batch=96):
    """TopK exact sorted cosine; no truth PID input; index tie stabilized."""
    if k>len(v):raise ValueError('图库图像不足TopK')
    out=np.empty((len(x),k),dtype=np.int32)
    for a in range(0,len(x),batch):
        b=min(a+batch,len(x))
        score=np.asarray(x[a:b]@v.T,dtype=np.float32)
        # Stable full sort avoids k-edge tie indeterminacy.
        out[a:b]=np.argsort(-score,axis=1,kind='stable')[:,:k].astype(np.int32)
    return out

def split_train(train_tx,train_im,train_texts,train_images,fit,hold):
    pids=np.asarray([r['pid'] for r in train_images],np.int64)
    tpids=np.asarray([r['pid'] for r in train_texts],np.int64)
    ans=[]
    for chosen in (fit,hold):
        ti=np.flatnonzero(np.isin(tpids,list(chosen)))
        ii=np.flatnonzero(np.isin(pids,list(chosen)))
        if len(ti)==0 or len(ii)==0:raise RuntimeError('训练身份切分空')
        ans.append((train_tx[ti],train_im[ii],tpids[ti],pids[ii],ti,ii))
    return ans

def complete_link_clusters(indices,v,threshold):
    """Greedy complete-link grouping; only cosine from image vectors, no true PID."""
    n=len(indices)
    vectors=np.asarray(v[indices],dtype=np.float32)
    sim=vectors@vectors.T
    np.fill_diagonal(sim,1.0)
    edges=sorted(((float(sim[a,b]),a,b) for a in range(n) for b in range(a+1,n)),
                 key=lambda z:(-z[0],z[1],z[2]))
    groups=[[i] for i in range(n)]
    for score,a,b in edges:
        if score+1e-7 < threshold:break
        ga=next((g for g in groups if a in g),None)
        gb=next((g for g in groups if b in g),None)
        if ga is gb:continue
        if all(float(sim[u,w])+1e-7>=threshold for u in ga for w in gb):
            groups.remove(ga);groups.remove(gb)
            groups.append(sorted(ga+gb))
    return sorted(groups,key=lambda g:min(g))


def find_exp16(parent,explicit=''):
    paths=[Path(explicit).expanduser().resolve()] if explicit else sorted(parent.glob(EXP16+'*'),reverse=True)
    for p in paths:
        if (p/'run_info.json').is_file() and (p/'fit_threshold_calibration.json').is_file() and (p/'experiment16_report.json').is_file():
            return p
    raise FileNotFoundError('缺少实验16完整输出（视觉分组阈值只允许从拟合身份继承），可指定 --exp16-dir')


def audit_exp16(exp16,exp12,source_settings):
    i=json.loads((exp16/'run_info.json').read_text(encoding='utf-8'))
    c=json.loads((exp16/'fit_threshold_calibration.json').read_text(encoding='utf-8'))
    if Path(i.get('source_dir','')).resolve()!=exp12.resolve():
        raise RuntimeError('实验16和实验12的来源目录不匹配')
    if i.get('settings',{}).get('exp12_settings_hash')!=source_settings['settings_hash']:
        raise RuntimeError('实验16阈值对应不同的最佳IRRA特征/实验12设置')
    if not c.get('training_pid_only') or int(c.get('calibration_gallery_pids',0))!=1000:
        raise RuntimeError('实验16阈值未满足拟合身份独立校准要求')
    t=float(c['threshold'])
    if not np.isfinite(t):raise RuntimeError('阈值非有限值')
    return t,c


def find_exp15(parent,explicit=''):
    prefix='实验15_全图库效用监督文本选择_'
    paths=[Path(explicit).expanduser().resolve()] if explicit else sorted(parent.glob(prefix+'*'),reverse=True)
    for p in paths:
        if (p/'trained_selector.json').is_file() and (p/'experiment15_report.json').is_file() and (p/'run_info.json').is_file():
            return p
    return None


def optional_exp15_for_audit(exp15,exp12):
    """Prior selector expects hidden A; NEVER use its output in support selection."""
    if exp15 is None:return None
    meta=json.loads((exp15/'run_info.json').read_text(encoding='utf-8'))
    if Path(meta.get('source_exp12','')).resolve()!=exp12.resolve():
        raise RuntimeError('实验15选择器来自其他实验12目录')
    if meta.get('source_best_checkpoint_sha256')!=BEST_SHA:
        raise RuntimeError('实验15选择器最佳模型指纹不对')
    s=json.loads((exp15/'trained_selector.json').read_text(encoding='utf-8'))
    if (np.asarray(s['weights']).shape!=(9,3) or len(s['feature_mean'])!=9 or
        len(s['feature_scale'])!=9 or len(s['bias'])!=3):
        raise RuntimeError('实验15选择器形状不兼容')
    return s


def normalized(x):
    z=np.asarray(x,dtype=np.float32)
    return z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-12)


# ----------------- experiment 19: global image-group + T1-group matching -------------
GROUP_KNN=12
CANDIDATE_GROUPS=36
FIT_FOLDS=4
TARGET_PRECISION=.95
FIT_MAX_QUERY_PER_FOLD=420


def build_abc_subset(x,v,texts,images,chosen_pids,seed=SEED):
    """A source query / B supports / C untouched. PID is used ONLY for deterministic partition."""
    chosen=set(map(int,chosen_pids));bypid=defaultdict(list)
    for j,im in enumerate(images):
        if int(im['pid']) in chosen:bypid[int(im['pid'])].append(j)
    if set(bypid)!=chosen:raise RuntimeError('拟合或评估身份缺少原始图像')
    A=[];B=[];C=[]
    for pid in sorted(chosen):
        ids=bypid[pid]
        rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(pid)&0xffffffff]))
        ids=np.asarray(ids,dtype=np.int32)[rng.permutation(len(ids))]
        C.append(int(ids[0]))
        if len(ids)>=3:
            A.append(int(ids[1]));B.extend(int(t) for t in ids[2:])
        elif len(ids)==2:B.append(int(ids[1]))
    if not A or len(B)<16:raise RuntimeError('A/B分割不足以评价Top16')
    if set(A)&set(B) or set(A)&set(C) or set(B)&set(C):raise AssertionError('A/B/C有来源图像交叉')
    ai=set(A);bs=np.asarray(sorted(B),dtype=np.int32);cs=np.asarray(sorted(C),dtype=np.int32)
    bmap={int(j):i for i,j in enumerate(bs)}
    txidx=np.asarray([i for i,t in enumerate(texts) if int(t['img_idx']) in ai],np.int32)
    bsset=set(map(int,bs));bsidx=np.asarray([i for i,t in enumerate(texts) if int(t['img_idx']) in bsset],np.int32)
    t2b=np.asarray([bmap[int(texts[int(i)]['img_idx'])] for i in bsidx],np.int32)
    q_pid=np.asarray([int(texts[int(i)]['pid']) for i in txidx],np.int64)
    b_pid=np.asarray([int(images[int(i)]['pid']) for i in bs],np.int64)
    for tid,j in zip(bsidx,t2b):
        if int(texts[int(tid)]['pid'])!=int(b_pid[j]):raise RuntimeError('训练描述与来源图像PID不一致')
    return {'T1':normalized(x[txidx]),'TB':normalized(x[bsidx]),'VB':normalized(v[bs]),
            'query_pid_AUDIT':q_pid,'b_pid_AUDIT':b_pid,
            'query_text_idx':txidx,'B_text_idx':bsidx,'B_image_idx':bs,'C_image_idx':cs,
            't2b':t2b,'n_A_images':len(A),'n_C_images':len(C),'n_B_images':len(B),
            'n_source_query_pids':len(set(q_pid.tolist())),
            'source_captions':[texts[int(j)]['text'] for j in bsidx]}


def build_global_visual_graph(V,tau,mutual_k=GROUP_KNN):
    """All B images, no seed from query Top16. Mutual KNN edges + complete-link merge guard.

    O(N²) image cosine scores, but N is ~600-800 per 440-PID cohort; no dense text images.
    A tempting path via multiple intermediate impostors must not create unchecked chaining.
    """
    n=len(V);S=np.asarray(V@V.T,dtype=np.float32)
    np.fill_diagonal(S,-np.inf)
    knn=[]
    for i in range(n):
        inds=np.argsort(-S[i],kind='stable')[:min(mutual_k,n-1)]
        knn.append(set(map(int,inds)))
    edges=[]
    for i in range(n):
        for j in knn[i]:
            if j>i and i in knn[j] and float(S[i,j])>=tau:
                edges.append((float(S[i,j]),i,j))
    edges.sort(key=lambda z:(-z[0],z[1],z[2]))
    parent=list(range(n));members={i:[i] for i in range(n)}
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for sim,i,j in edges:
        a=find(i);b=find(j)
        if a==b:continue
        ma=members[a];mb=members[b]
        # complete-link is intentional: transitive chains do NOT automatically imply identity
        if float(np.min(S[np.ix_(ma,mb)]))<tau-1e-7:continue
        if len(ma)<len(mb) or (len(ma)==len(mb) and a>b):a,b=b,a;ma,mb=mb,ma
        parent[b]=a;members[a]=ma+mb;del members[b]
    groups=sorted((sorted(z) for z in members.values()),key=lambda x:x[0])
    lut=np.empty(n,dtype=np.int32)
    for gi,g in enumerate(groups):lut[g]=gi
    # Verify each resulting cluster respects the same threshold.
    for g in groups:
        if len(g)>1:
            sub=S[np.ix_(g,g)].copy();np.fill_diagonal(sub,1.)
            if float(sub.min())<tau-1e-6:raise AssertionError('visual-group complete-link violated')
    return groups,lut,S


def graph_audit(groups,pids):
    ntrue=defaultdict(int)
    for p in pids:ntrue[int(p)]+=1
    truepairs=sum(k*(k-1)//2 for k in ntrue.values())
    predictedpairs=sum(len(g)*(len(g)-1)//2 for g in groups)
    correctpairs=0;wronggroups=0;pure_multiview=0;ngmulti=0
    for g in groups:
        ct=defaultdict(int)
        for i in g:ct[int(pids[i])]+=1
        correctpairs+=sum(k*(k-1)//2 for k in ct.values())
        if len(ct)>1:wronggroups+=1
        if len(g)>1:
            ngmulti+=1
            if len(ct)==1:pure_multiview+=1
    return {'n_images':len(pids),'n_groups':len(groups),'n_multiview_groups':ngmulti,
            'n_pure_multiview_groups':pure_multiview,'n_impure_groups':wronggroups,
            'n_same_pid_pairs':truepairs,'n_proposed_pairs':predictedpairs,
            'n_correct_pairs':correctpairs,'pair_precision':correctpairs/predictedpairs if predictedpairs else 0.,
            'pair_recall':correctpairs/truepairs if truepairs else 0.,
            'whole_group_purity':(len(groups)-wronggroups)/len(groups) if groups else 0.}


def prepare_group_metadata(d,groups,vis_sim):
    # Group membership and training-caption indices are the only sources for features.
    text_of=[[] for _ in groups]
    b_to_group=np.full(len(d['VB']),-1,dtype=np.int32)
    for gi,g in enumerate(groups):
        for j in g:b_to_group[j]=gi
    for ti,bi in enumerate(d['t2b']):text_of[int(b_to_group[bi])].append(ti)
    if any(len(z)==0 for z in text_of):raise RuntimeError('B image has no training support caption')
    info=[]
    for gi,g in enumerate(groups):
        ti=np.asarray(text_of[gi],np.int32);images=np.asarray(g,np.int32)
        self_img_txt=np.einsum('nd,nd->n', d['TB'][ti],d['VB'][d['t2b'][ti]])
        sm=vis_sim[np.ix_(g,g)].copy()
        if len(g)>1:
            np.fill_diagonal(sm,np.nan)
            cohesion=float(np.nanmean(sm))
        else:cohesion=1.
        info.append({'images':images,'texts':ti,'cohesion':cohesion,
                     'img_txt_agreement':float(np.mean(self_img_txt))})
    return info,b_to_group


FEATURE_NAMES=('max_img','mean_img','top2_img','max_txt','mean_txt','top2_txt',
               'img_range','txt_range','log_sources','cohesion','img_caption_consistency',
               'group_in_Top16_frac')


def query_group_candidates(q,d,meta,b2group,max_groups=CANDIDATE_GROUPS):
    # Entire B support image and TRAIN caption library is searched, not only top16.
    si=np.asarray(d['VB']@q,dtype=np.float32)
    st=np.asarray(d['TB']@q,dtype=np.float32)
    top16=np.argsort(-si,kind='stable')[:min(TOPK,len(si))]
    top16set=set(map(int,top16))
    max_i=np.asarray([float(si[g['images']].max()) for g in meta],dtype=np.float32)
    max_t=np.asarray([float(st[g['texts']].max()) for g in meta],dtype=np.float32)
    # Candidate generation from *both* global vision and global train-caption similarity.
    # No PID/real source A/C/ground-truth ranking can enter this stage.
    cap=min(max_groups,len(meta))
    ord_i=np.argsort(-max_i,kind='stable')[:cap]
    ord_t=np.argsort(-max_t,kind='stable')[:cap]
    chosen=sorted(set(map(int,ord_i))|set(map(int,ord_t)))
    X=[]
    for gid in chosen:
        m=meta[gid];a=np.sort(si[m['images']]);b=np.sort(st[m['texts']])
        n_img=len(a)
        X.append([float(a[-1]),float(a.mean()),float(a[-min(2,len(a)):].mean()),
                  float(b[-1]),float(b.mean()),float(b[-min(2,len(b)):].mean()),
                  float(a[-1]-a[0]),float(b[-1]-b[0]),
                  float(np.log1p(n_img)),m['cohesion'],m['img_txt_agreement'],
                  sum(int(i) in top16set for i in m['images'])/n_img])
    return np.asarray(chosen,np.int32),np.asarray(X,np.float32),si,st,top16,b2group[int(top16[0])]


def feature_baseline(X):
    return .5*(X[:,0]+X[:,3])+.12*(X[:,1]+X[:,4])+.025*X[:,8]


def make_queries(d,meta,b2group,limit=None,training=False):
    qidx=list(range(len(d['T1'])))
    if training:
        # One A-description per query PID for generalization; remaining A captions don't disappear from holdout.
        first={}
        for qi,p in enumerate(d['query_pid_AUDIT']):
            if int(p) not in first:first[int(p)]=qi
        qidx=sorted(first.values())
        if limit is not None:qidx=qidx[:int(limit)]
    examples=[];records=[]
    for qi in qidx:
        gid,X,si,st,top16,top1group=query_group_candidates(d['T1'][qi],d,meta,b2group)
        qpid=int(d['query_pid_AUDIT'][qi])
        # IMPORTANT: ONLY offline fit labels / holdout audit use pids.
        true=np.asarray([bool(np.all(d['b_pid_AUDIT'][meta[int(gg)]['images']]==qpid)) for gg in gid],np.int32)
        group_true=np.asarray([int(np.all(d['b_pid_AUDIT'][m['images']]==qpid)) for m in meta],np.int32)
        records.append({'query_index':qi,'query_pid_AUDIT':qpid,'gids':gid,'X':X,'true':true,
                        'group_true':group_true,'top16':top16,'top1_group':int(top1group),
                        'top1_pid_correct_AUDIT':int(d['b_pid_AUDIT'][int(top16[0])]==qpid),
                        'has_same_pid_image_in_B_AUDIT':int(np.any(d['b_pid_AUDIT']==qpid)),
                        'true_B_img_beyond16_AUDIT':int(any(int(p)==qpid and i not in set(top16.tolist()) for i,p in enumerate(d['b_pid_AUDIT']))),
                        'positive_group_in_candidates_AUDIT':int(any(true))})
        if training and len(X):examples.append((X,true))
    return records,examples


def train_binary_linear(examples,rounds=370,lr=.12,decay=.035):
    X=np.concatenate([a for a,_ in examples],axis=0).astype(np.float64)
    y=np.concatenate([b for _,b in examples],axis=0).astype(np.float64)
    if not np.any(y==1) or not np.any(y==0):raise RuntimeError('组归属模型需要正负拟合样本')
    mu=X.mean(0);sd=X.std(0);sd=np.where(sd<1e-4,1.,sd)
    Z=np.clip((X-mu)/sd,-8,8)
    weights=np.where(y>0,.5/max(float(y.mean()),1e-3),.5/max(1.-float(y.mean()),1e-3))
    weights/=weights.mean()
    w=np.zeros(X.shape[1],np.float64);b=0.
    for it in range(rounds):
        z=np.clip(Z@w+b,-30,30);pr=1./(1.+np.exp(-z))
        err=(pr-y)*weights
        gw=Z.T@err/len(X)+decay*w
        gb=float(err.mean())
        step=lr*(.45+.55*(1.-it/rounds))
        w-=step*gw;b-=step*gb
    if not np.isfinite(w).all() or not np.isfinite(b):raise RuntimeError('拟合NaN')
    return {'weight':w.tolist(),'bias':float(b),'mean':mu.tolist(),'scale':sd.tolist(),
            'feature_names':list(FEATURE_NAMES),'n_fit_examples':int(len(y)),
            'n_fit_positive_groups':int(y.sum()),'n_fit_negative_groups':int(len(y)-y.sum())}


def group_score(X,model):
    if len(X)==0:return np.zeros(0,np.float32)
    mu=np.asarray(model['mean'],np.float64);scale=np.asarray(model['scale'],np.float64)
    w=np.asarray(model['weight'],np.float64)
    z=np.clip((np.clip((X-mu)/scale,-8,8)@w)+float(model['bias']),-30,30)
    return (1./(1.+np.exp(-z))).astype(np.float64)


def best_group_choice(record,model=None,baseline=False):
    if baseline:
        s=feature_baseline(record['X'])
    else:
        s=group_score(record['X'],model)
    if len(s)==0:raise RuntimeError('无候选身份组')
    b=int(np.argmax(s))
    return int(record['gids'][b]),float(s[b]),int(record['true'][b]),b


def threshold_from_crossfit(crossfit,target=TARGET_PRECISION):
    if not crossfit:raise RuntimeError('无校准记录')
    a=np.asarray([v['score'] for v in crossfit],np.float64)
    y=np.asarray([v['correct'] for v in crossfit],np.int32)
    # Precision is label information exclusively from disjoint FIT folds.
    candidates=sorted(set(map(float,a)),reverse=True)
    best=None
    for th in candidates:
        idx=a>=th-1e-12
        n=int(idx.sum());prec=float(y[idx].mean()) if n else 0.
        if prec+1e-12>=target:
            if best is None or n>best['n_accepted_cv']:
                best={'threshold':float(th),'n_accepted_cv':n,'cv_precision':prec,
                      'cv_coverage':n/len(y)}
    if best is None:
        # No false claim that precision target was met. Hard abstain all.
        return {'threshold':1.01,'n_accepted_cv':0,'cv_precision':None,'cv_coverage':0.,
                'target_met_on_fit':False,'desired_precision':float(target)}
    return {**best,'target_met_on_fit':True,'desired_precision':float(target)}



# ================= 实验20：固定实验19候选组，验证多视角-多描述共同证据 =================
# 这些附加特征完全从冻结T1、B图像与B训练描述得到；不读取身份、C图像或评估结果。
EXTRA_FEATURE_NAMES=(
    'joint_max_across_views', 'joint_mean_across_views',
    'joint_top2_mean_across_views', 'joint_second_view',
    'joint_view_disagreement', 'joint_support_fraction',
    'source_caption_per_view_mean', 'text_view_top2_mean',
    'relative_top_image', 'relative_top_caption',
    'log_caption_count', 'top2_image_supported_fraction'
)
ALL_FEATURE_NAMES=tuple(FEATURE_NAMES)+EXTRA_FEATURE_NAMES


def find_exp19_folder(parent, explicit=''):
    choices=[Path(explicit).expanduser().resolve()] if explicit else sorted(parent.glob(EXP19+'*'),reverse=True)
    for p in choices:
        if all((p/f).is_file() for f in ('run_info.json','experiment19_report.json',
            'fit_trained_group_matcher.json','fit_crossval_threshold.json','holdout_all_queries.csv')):
            return p
    raise FileNotFoundError('找不到实验19完整目录。请提供 --exp19-dir；只有上传ZIP不包含训练/图片大缓存。')


def audit_exp19_folder(exp19,exp12,exp16):
    run=json.loads((exp19/'run_info.json').read_text(encoding='utf-8'))
    report=json.loads((exp19/'experiment19_report.json').read_text(encoding='utf-8'))
    model=json.loads((exp19/'fit_trained_group_matcher.json').read_text(encoding='utf-8'))
    old_cut=json.loads((exp19/'fit_crossval_threshold.json').read_text(encoding='utf-8'))
    if run.get('version')!='exp19_v1_group_link_and_query_attribution_cvfit_only':
        raise RuntimeError('实验19版本与本次固定实验协议不匹配')
    if run.get('source_best_checkpoint_sha256')!=BEST_SHA:
        raise RuntimeError('实验19并非最佳Epoch58 IRRA')
    if Path(run['source_exp12']).resolve()!=exp12.resolve() or Path(run['source_exp16']).resolve()!=exp16.resolve():
        raise RuntimeError('实验12/16/19之间的数据来源目录不一致')
    if list(model.get('feature_names',[]))!=list(FEATURE_NAMES):
        raise RuntimeError('实验19旧模型的特征顺序不一致')
    if len(model['weight'])!=len(FEATURE_NAMES):raise RuntimeError('实验19旧模型维数错误')
    if report.get('holdout',{}).get('n_A_queries') not in (542,None):
        # Retain older field variations, exact per-case replay is compulsory regardless.
        raise RuntimeError('实验19的holdout查询规模不匹配')
    with (exp19/'holdout_all_queries.csv').open('r',encoding='utf-8-sig',newline='') as f:
        prior=list(csv.DictReader(f))
    if len(prior)!=542 or len({int(r['query_index']) for r in prior})!=542:
        raise RuntimeError('实验19逐查询记录不等于542条')
    return model,old_cut,prior,run,report


def expanded_evidence(X,chosen,si,st,d,meta):
    """每张B图像汇聚属于该图像的真实TRAIN描述，然后聚合不同图像的证据。

    注意：即使同一图片的两个描述相似，也只能算一个独立视觉来源；不将描述数量
    当作独立身份凭证。没有PID、查询来源A特征或第三视角C进入该函数。
    """
    if len(chosen)!=len(X):raise RuntimeError('候选组与旧特征数不符')
    view_caption=[[] for _ in range(len(d['VB']))]
    for tid,bi in enumerate(d['t2b']):view_caption[int(bi)].append(int(tid))
    if any(len(ids)<1 for ids in view_caption):raise RuntimeError('B图像没有配套训练描述')
    # 单张B图像相对T1的最大及平均文字证据
    tx_best=np.asarray([float(np.max(st[ids])) for ids in view_caption],np.float64)
    tx_avg=np.asarray([float(np.mean(st[ids])) for ids in view_caption],np.float64)
    view_joint=.5*np.asarray(si,np.float64)+.5*tx_best
    global_best_img=float(np.max(si));global_best_text=float(np.max(st))
    features=[]
    for gi in chosen:
        m=meta[int(gi)];ims=np.asarray(m['images'],np.int32)
        jj=np.sort(view_joint[ims]);tm=np.sort(tx_best[ims]);iscore=np.sort(si[ims])
        jtop=float(jj[-1]);jsnd=float(jj[-2]) if len(jj)>1 else jtop
        ttop=float(tm[-1]);tsnd=float(tm[-2]) if len(tm)>1 else ttop
        # 固定的0.04和0.08只是特征尺度，不是身份选择阈值。
        features.append([
            jtop, float(jj.mean()), float(jj[-min(2,len(jj)):].mean()),
            jsnd, jtop-float(jj[0]), float(np.mean(jj>=jtop-.04)),
            float(np.mean(tx_avg[ims])),.5*(ttop+tsnd),
            float(iscore[-1]-global_best_img),float(tm[-1]-global_best_text),
            float(np.log1p(len(m['texts']))),float(np.mean(iscore>=global_best_img-.08)),
        ])
    z=np.asarray(features,np.float32)
    if z.shape!=(len(X),len(EXTRA_FEATURE_NAMES)) or not np.isfinite(z).all():
        raise RuntimeError('新特征形状或有限性失败')
    return np.concatenate([np.asarray(X,np.float32),z],axis=1)


def queries_with_extended_evidence(d,meta,b2group,training=False):
    first={}
    if training:
        for qi,p in enumerate(d['query_pid_AUDIT']):
            if int(p) not in first:first[int(p)]=qi
        qi_list=sorted(first.values())
    else:qi_list=list(range(len(d['T1'])))
    result=[]
    for qi in qi_list:
        gid,X,si,st,top16,top1group=query_group_candidates(d['T1'][qi],d,meta,b2group)
        xe=expanded_evidence(X,gid,si,st,d,meta)
        qpid=int(d['query_pid_AUDIT'][qi])  # only supervised FIT label / HOLDOUT offline audit
        truth=np.asarray([int(np.all(d['b_pid_AUDIT'][meta[int(g)]['images']]==qpid)) for g in gid],np.int8)
        result.append({'query_index':int(qi),'query_pid_AUDIT':qpid,
                       'gids':gid,'X':np.asarray(X,np.float32),'XE':xe,'true':truth,
                       'top16':top16,'top1_group':int(top1group),
                       'top1_correct_PID_AUDIT':int(d['b_pid_AUDIT'][int(top16[0])]==qpid),
                       'positive_group_in_candidates_AUDIT':int(np.any(truth))})
    return result


def fit_extended_model(records):
    ex=[(r['XE'],r['true']) for r in records]
    model=train_binary_linear(ex)
    model['feature_names']=list(ALL_FEATURE_NAMES)
    return model


def choose_rank(rec,score):
    score=np.asarray(score,np.float64)
    if len(score)!=len(rec['gids']) or not np.isfinite(score).all():
        raise RuntimeError('分数和候选组不匹配或非有限')
    idx=np.argsort(-score,kind='stable')
    win=int(idx[0]);positive=np.flatnonzero(rec['true']>0)
    ranks=np.empty(len(score),np.int32);ranks[idx]=np.arange(1,len(score)+1)
    return {'best_pos':win,'gid':int(rec['gids'][win]),'score':float(score[win]),
            'correct':int(rec['true'][win]),
            'true_group_rank':int(np.min(ranks[positive])) if len(positive)>0 else 0,
            'top_candidate_group_ids':[int(rec['gids'][v]) for v in idx[:5]],
            'top_candidate_scores':[float(score[v]) for v in idx[:5]]}


def gzip_atomic_write(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.exp20_',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:
                z.write(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8'))
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def fold_cache_save(path,fi,pids,threshold,records):
    payload={'version':VERSION,'fold':int(fi),'pid_digest':stable_json(list(map(int,pids))),
             'visual_tau':float(threshold),'feature_names':list(ALL_FEATURE_NAMES),
             'records':[{'query_index':r['query_index'],'gids':r['gids'].tolist(),
                         'X':r['X'].tolist(),'XE':r['XE'].tolist(),'true':r['true'].tolist()}
                        for r in records]}
    gzip_atomic_write(path,payload)


def fold_cache_load(path,fi,pids,threshold):
    with gzip.open(path,'rt',encoding='utf-8') as f:p=json.load(f)
    if (p.get('version')!=VERSION or int(p.get('fold',-1))!=fi or
        p.get('pid_digest')!=stable_json(list(map(int,pids))) or
        abs(float(p.get('visual_tau',-100))-threshold)>1e-8 or
        p.get('feature_names')!=list(ALL_FEATURE_NAMES)):
        raise RuntimeError('实验20拟合折缓存版本/身份/特征阈值不一致')
    rows=[]
    for r in p['records']:
        z={'query_index':int(r['query_index']),'gids':np.asarray(r['gids'],np.int32),
           'X':np.asarray(r['X'],np.float32),'XE':np.asarray(r['XE'],np.float32),
           'true':np.asarray(r['true'],np.int8)}
        if (len(z['gids'])!=len(z['true']) or z['X'].shape!=(len(z['gids']),len(FEATURE_NAMES))
            or z['XE'].shape!=(len(z['gids']),len(ALL_FEATURE_NAMES))):
            raise RuntimeError('实验20缓存形状错误')
        rows.append(z)
    return rows


def atomic_jsonl20(path,rows):
    path=Path(path);fd,tmp=tempfile.mkstemp(prefix='.exp20_',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def read_jsonl20(path):
    with open(path,'r',encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]


def log20(folder,message):
    print(message,flush=True)
    with (folder/'experiment_run.log').open('a',encoding='utf-8') as f:
        f.write(datetime.now().isoformat(timespec='seconds')+' '+message+'\n')


def fit_4fold20(out,tx,vi,texts,images,fit_ids,visual_tau,resume=True):
    folds=[list(map(int,v)) for v in np.array_split(np.asarray(sorted(fit_ids),np.int64),FIT_FOLDS)]
    cached=[];diags=[]
    for fi,pids in enumerate(folds):
        file=out/f'fit_fold_{fi:02d}_multi_evidence.json.gz'
        if resume and file.is_file():
            rr=fold_cache_load(file,fi,pids,visual_tau)
            log20(out,f'续跑：拟合折 {fi+1}/{FIT_FOLDS} 缓存有效，{len(rr)} queries')
        else:
            d=build_abc_subset(tx,vi,texts,images,pids)
            gs,lut,sim=build_global_visual_graph(d['VB'],visual_tau)
            gm,lut2=prepare_group_metadata(d,gs,sim)
            if not np.array_equal(lut,lut2):raise RuntimeError('拟合折视觉组映射错误')
            rr=queries_with_extended_evidence(d,gm,lut,training=True)
            fold_cache_save(file,fi,pids,visual_tau,rr)
            log20(out,f'拟合折 {fi+1}/{FIT_FOLDS}: PID={len(pids)}, queries={len(rr)}, groups={len(gs)}')
        if len(rr)<10 or not any(np.any(r['true']) for r in rr):
            raise RuntimeError('拟合折缺少足够正例')
        cached.append(rr)
        diags.append({'fold':fi,'pid_count':len(pids),'n_train_queries':len(rr),
                      'n_with_true_group':sum(int(np.any(r['true'])) for r in rr)})
    allfold=[r for fold in cached for r in fold]
    cross=[]
    for fi,records in enumerate(cached):
        model=fit_extended_model([r for fj,sub in enumerate(cached) if fj!=fi for r in sub])
        for r in records:
            choice=choose_rank(r,group_score(r['XE'],model))
            cross.append({'fold':fi,'query_index':int(r['query_index']),
                          'score':choice['score'],'correct':choice['correct'],
                          'true_group_rank':choice['true_group_rank']})
    cut90=threshold_from_crossfit(cross,.90)
    cut95=threshold_from_crossfit(cross,.95)
    fullmodel=fit_extended_model(allfold)
    atomic_json(out/'new_group_matcher.json',fullmodel)
    atomic_json(out/'fit_crossval_thresholds.json',{'precision90':cut90,'precision95':cut95})
    atomic_csv(out/'fit_fold_diagnostics.csv',diags,list(diags[0]))
    atomic_csv(out/'fit_crossval_decisions.csv',cross,list(cross[0]))
    return fullmodel,{'precision90':cut90,'precision95':cut95},diags


def validate_exp19_replay(records,old_model,old_rows,d):
    if len(records)!=len(old_rows):raise RuntimeError('实验19逐查询数量与本次重建不一致')
    errors=[]
    for r,previous in zip(records,old_rows):
        qi=r['query_index']
        if qi!=int(previous['query_index']):
            raise RuntimeError('实验19/20查询排列不一致')
        old=choose_rank(r,group_score(r['X'],old_model))
        if (int(previous['group_fit_all_source_group_id'])!=old['gid'] or
            abs(float(previous['group_fit_all_score'])-old['score'])>2e-5 or
            int(previous['n_candidate_groups'])!=len(r['gids']) or
            int(previous['top1_B_image_correct_PID_AUDIT'])!=int(r['top1_correct_PID_AUDIT']) or
            int(previous['query_text_train_index'])!=int(d['query_text_idx'][qi]) or
            int(previous['query_pid_AUDIT_ONLY'])!=int(r['query_pid_AUDIT'])):
            errors.append({'query':qi,'old_gid':previous['group_fit_all_source_group_id'],
                           'replay_gid':old['gid'],'old_score':previous['group_fit_all_score'],
                           'replay_score':old['score']})
    if errors:
        raise RuntimeError('实验19真实结果未能逐条复现，已停止避免虚假比较: '+str(errors[:3]))
    return {'replayed_queries':len(records),'original_model_selected_groups_identical':True,
            'score_max_absolute_tolerance':2e-5}


def produce_holdout20(out,d,meta,b2group,old_model,new_model,cutoffs,old_rows,
                      chunk_size=32):
    allrec=queries_with_extended_evidence(d,meta,b2group,training=False)
    # 任何方法比较前，强制精确复现实验19旧选择和旧评分。
    audit=validate_exp19_replay(allrec,old_model,old_rows,d)
    rows=[]
    for st in range(0,len(allrec),chunk_size):
        en=min(st+chunk_size,len(allrec));cache=out/f'holdout_part_{st:05d}_{en:05d}.jsonl'
        if cache.is_file():
            part=read_jsonl20(cache)
            if len(part)!=en-st or [r['query_index'] for r in part]!=list(range(st,en)):
                raise RuntimeError('实验20续跑段索引不连续，请删掉不完整段文件再继续')
            log20(out,f'续跑：holdout 查询 {st}:{en}')
        else:
            part=[]
            for r in allrec[st:en]:
                old=choose_rank(r,group_score(r['X'],old_model))
                new=choose_rank(r,group_score(r['XE'],new_model))
                qi=r['query_index'];p=int(r['query_pid_AUDIT']);old_gid=old['gid'];new_gid=new['gid']
                old_meta=meta[old_gid];new_meta=meta[new_gid]
                top1bad=int(not r['top1_correct_PID_AUDIT'])
                single_score=float(d['VB'][r['top16'][0]]@d['T1'][qi])
                # True groups are a POST-HOC reference only. They play no role in group selection.
                row={'query_index':qi,'query_train_text_index':int(d['query_text_idx'][qi]),
                    'query_pid_AUDIT_ONLY':p,'query_text_TRAIN':str(d['source_query_captions'][qi]) if 'source_query_captions' in d else '',
                    'n_candidate_groups':len(r['gids']),'correct_group_available_AUDIT':int(np.any(r['true'])),
                    'top1_correct_AUDIT':int(r['top1_correct_PID_AUDIT']),
                    'top1_original_B_image_index':int(d['B_image_idx'][int(r['top16'][0])]),
                    'top1_single_score':single_score,
                    'old_correct_AUDIT':old['correct'],'new_correct_AUDIT':new['correct'],
                    'old_selected_gid':old_gid,'new_selected_gid':new_gid,
                    'old_probability':old['score'],'new_probability':new['score'],
                    'old_true_group_best_rank_AUDIT':old['true_group_rank'],
                    'new_true_group_best_rank_AUDIT':new['true_group_rank'],
                    'old_selected_group_images':len(old_meta['images']),
                    'new_selected_group_images':len(new_meta['images']),
                    'old_selected_group_captions':len(old_meta['texts']),
                    'new_selected_group_captions':len(new_meta['texts']),
                    'old_selected_group_original_images':json.dumps([int(d['B_image_idx'][j]) for j in old_meta['images']]),
                    'new_selected_group_original_images':json.dumps([int(d['B_image_idx'][j]) for j in new_meta['images']]),
                    'new_selected_group_original_captions':json.dumps([int(d['B_text_idx'][j]) for j in new_meta['texts']]),
                    'old_top5_candidate_gids':json.dumps(old['top_candidate_group_ids']),
                    'new_top5_candidate_gids':json.dumps(new['top_candidate_group_ids']),
                    'new_top5_scores':json.dumps([round(z,6) for z in new['top_candidate_scores']]),
                    'top1_error_fixed_old':int(top1bad and old['correct']),
                    'top1_error_fixed_new':int(top1bad and new['correct']),
                    'top1_correct_broken_old':int(not top1bad and not old['correct']),
                    'top1_correct_broken_new':int(not top1bad and not new['correct']),
                    'old_wrong_new_right':int(not old['correct'] and new['correct']),
                    'old_right_new_wrong':int(old['correct'] and not new['correct'])}
                for target in ('precision90','precision95'):
                    th=cutoffs[target]['threshold']
                    accept=int(new['score']>=th-1e-12 and th<=1.)
                    row['new_'+target+'_accepted']=accept
                    row['new_'+target+'_correct_if_accepted_AUDIT']=int(accept and new['correct'])
                    row['new_'+target+'_top1_error_fixed']=int(accept and top1bad and new['correct'])
                    row['new_'+target+'_top1_correct_broken']=int(accept and not top1bad and not new['correct'])
                # Group B descriptions are permitted TRAIN captions; useful for manual case review.
                row['new_selected_group_train_captions']=json.dumps(
                    [d['source_captions'][int(j)] for j in new_meta['texts'][:8]],ensure_ascii=False)
                part.append(row)
            atomic_jsonl20(cache,part)
            log20(out,f'holdout processed {en}/{len(allrec)}')
        rows.extend(part)
    if len(rows)!=len(allrec):raise RuntimeError('段文件恢复总量不一致')
    return rows,audit,allrec


def describe_methods20(rows):
    n=len(rows)
    methods=[('single_top1','top1_correct_AUDIT',None),
             ('experiment19_old_no_reject','old_correct_AUDIT',None),
             ('experiment20_multi_evidence_no_reject','new_correct_AUDIT',None),
             ('experiment20_multi_evidence_90cv','new_precision90_correct_if_accepted_AUDIT','new_precision90_accepted'),
             ('experiment20_multi_evidence_95cv','new_precision95_correct_if_accepted_AUDIT','new_precision95_accepted')]
    output=[]
    for name,col,acc in methods:
        accepted=[r for r in rows if acc is None or r[acc]]
        correct=sum(int(r[col]) for r in accepted)
        nwrong=sum(1 for r in rows if not r['top1_correct_AUDIT'])
        fixed=sum(1 for r in accepted if not r['top1_correct_AUDIT'] and r[col])
        broken=sum(1 for r in accepted if r['top1_correct_AUDIT'] and not r[col])
        output.append({'method':name,'n_queries':n,'n_accepted':len(accepted),
           'coverage':len(accepted)/n,'correct_identity_source_per_query':correct/n,
           'precision_when_accepted':correct/len(accepted) if accepted else None,
           'n_top1_wrong':nwrong,'top1_wrong_fixed':fixed,'top1_right_broken':broken,
           'net_correct_vs_top1_when_unaccepted_fallback':fixed-broken,
           'n_accepted_with_multiple_source_images':sum(int(r['new_selected_group_images']>=2) for r in accepted) if name.startswith('experiment20') else None})
    return output


def rank_diagnostic20(rows):
    report=[]
    for cohort, rr in [('all',rows),('top1_wrong',[r for r in rows if not r['top1_correct_AUDIT']]),
                       ('top1_right',[r for r in rows if r['top1_correct_AUDIT']])]:
        for meth in ('old','new'):
            rk=[int(r[meth+'_true_group_best_rank_AUDIT']) for r in rr]
            n=len(rr)
            report.append({'cohort':cohort,'ranker':meth,'queries':n,
               'n_missing_true_group':sum(x==0 for x in rk),
               'n_true_group_rank1':sum(x==1 for x in rk),
               'n_true_group_rank_le3':sum(1<=x<=3 for x in rk),
               'n_true_group_rank_le5':sum(1<=x<=5 for x in rk),
               'n_true_group_rank_le10':sum(1<=x<=10 for x in rk),
               'n_true_group_rank_le20':sum(1<=x<=20 for x in rk)})
    return report


def cluster_bootstrap20(rows,reps=1600,seed=20261022):
    bypid=defaultdict(list)
    for r in rows:bypid[int(r['query_pid_AUDIT_ONLY'])].append(r)
    pids=list(sorted(bypid));rng=np.random.default_rng(seed)
    measurements={'new_minus_old_accuracy':[], 'new_minus_top1_accuracy':[],
                  'new_minus_old_top1_error_repairs':[],
                  'new_minus_old_top1_correct_harms':[]}
    for _ in range(reps):
        sample=[r for j in rng.integers(len(pids),size=len(pids)) for r in bypid[pids[int(j)]]]
        inv=1./len(sample)
        measurements['new_minus_old_accuracy'].append(inv*sum(r['new_correct_AUDIT']-r['old_correct_AUDIT'] for r in sample))
        measurements['new_minus_top1_accuracy'].append(inv*sum(r['new_correct_AUDIT']-r['top1_correct_AUDIT'] for r in sample))
        measurements['new_minus_old_top1_error_repairs'].append(inv*sum(r['top1_error_fixed_new']-r['top1_error_fixed_old'] for r in sample))
        measurements['new_minus_old_top1_correct_harms'].append(inv*sum(r['top1_correct_broken_new']-r['top1_correct_broken_old'] for r in sample))
    answer={}
    for key,val in measurements.items():
        v=np.asarray(val,np.float64)
        answer[key]={'pid_cluster_95ci_percentage_points':[float(x*100) for x in np.quantile(v,[.025,.975])]}
    return answer


def inspection_cases20(rows,d,meta,source_train_texts,source_train_images,limit=130):
    scores=[]
    for r in rows:
        # Prioritize: old-new reversals, original Top1 errors, positive group remaining rank2/3.
        importance=(300*int(r['old_wrong_new_right'])+250*int(r['old_right_new_wrong'])+
                   100*int(not r['top1_correct_AUDIT'])+
                   50*int(r['correct_group_available_AUDIT'] and r['new_true_group_best_rank_AUDIT'] in (2,3)))
        scores.append((importance,r))
    chosen=[r for _,r in sorted(scores,key=lambda p:(-p[0],p[1]['query_index']))[:limit]]
    result=[]
    for r in chosen:
        gi=int(r['new_selected_gid']);m=meta[gi]
        qidx=int(r['query_train_text_index']);orig_text=str(source_train_texts[qidx]['text'])
        def describe(gg):
            if gg<0:return ''
            x=meta[gg];return ' || '.join(str(d['source_captions'][int(i)]) for i in x['texts'][:6])
        result.append({'query_index':r['query_index'],'query_PID_AUDIT':r['query_pid_AUDIT_ONLY'],
            'T1':orig_text,'top1_correct':r['top1_correct_AUDIT'],
            'old_correct':r['old_correct_AUDIT'],'new_correct':r['new_correct_AUDIT'],
            'old_true_rank':r['old_true_group_best_rank_AUDIT'],
            'new_true_rank':r['new_true_group_best_rank_AUDIT'],
            'old_group_descriptions_TRAIN':describe(int(r['old_selected_gid'])),
            'new_group_descriptions_TRAIN':describe(gi),
            'new_group_image_names_TRAIN':json.dumps([source_train_images[int(d['B_image_idx'][i])]['path']
                                                       for i in m['images']],ensure_ascii=False)})
    return result


def write_reports20(out,rows,audit,fit_diags,cutoffs,graphstats,oldcut,exp19,visualtau,d,meta,src_texts,src_images,self_test=False):
    method=describe_methods20(rows);ranks=rank_diagnostic20(rows)
    boot=cluster_bootstrap20(rows,400 if self_test else 1600)
    manual=inspection_cases20(rows,d,meta,src_texts,src_images)
    group_same=sum(int(r['old_selected_gid']==r['new_selected_gid']) for r in rows)
    comparison={'n_queries':len(rows),'n_query_pid':len(set(r['query_pid_AUDIT_ONLY'] for r in rows)),
        'n_top1_wrong':sum(not r['top1_correct_AUDIT'] for r in rows),
        'n_positive_in_candidate_pool':sum(r['correct_group_available_AUDIT'] for r in rows),
        'old_and_new_same_selected_group':group_same,
        'old_wrong_new_right':sum(r['old_wrong_new_right'] for r in rows),
        'old_right_new_wrong':sum(r['old_right_new_wrong'] for r in rows),
        'new_fixes_top1_errors':sum(r['top1_error_fixed_new'] for r in rows),
        'new_breaks_top1_correct':sum(r['top1_correct_broken_new'] for r in rows),
        'old_fixes_top1_errors':sum(r['top1_error_fixed_old'] for r in rows),
        'old_breaks_top1_correct':sum(r['top1_correct_broken_old'] for r in rows),
        'graph_statistics':graphstats,'old_exp19_cv_threshold':oldcut,
        'exp19_source':str(exp19),'exp16_image_similarity_threshold_fit_only':float(visualtau)}
    atomic_csv(out/'holdout_all_query_group_ranks.csv',rows,list(rows[0]))
    atomic_csv(out/'holdout_method_summary.csv',method,list(method[0]))
    atomic_csv(out/'holdout_true_group_rank_summary.csv',ranks,list(ranks[0]))
    atomic_csv(out/'人工核查案例.csv',manual,list(manual[0]))
    atomic_json(out/'bootstrap_pid.json',boot)
    report={'version':VERSION,'exp19_replay':audit,'fit_folds':fit_diags,
      'new_90_95_cv_thresholds':cutoffs,'comparison':comparison,
      'methods':method,'true_group_rank':ranks,'bootstrap':boot,
      'only_train_image_captions_for_support':True,
      'holdout_PID_only_for_posthoc_audit':True,
      'A_B_C_independent_images':True,
      'no_new_visual_group_clustering':True,
      'same_exp19_group_candidates':True,
      'no_val_gallery_captions':True,
      'does_not_train_IRRA_or_text_fusion':True,
      'interpretation_warnings':[
        'Only first hard problem (query attribution to same-ID B image group) is tested. No multi-text utility filtering or fusion.',
        'Extra features jointly aggregate TRAIN captions by distinct source images; group matching still needs legal gallery descriptions.',
        'Train-holdout PIDs were previously seen by frozen IRRA; not unseen-ID encoder evaluation.',
        'Candidate recall and fixed B graph may already be high; new classifier can fail or reduce accuracy.',
        'Precision conditioned on abstention is NOT coverage or Top1-error correction.',
        'No official CUHK-PEDES Validation improvement is claimed.']}
    atomic_json(out/'experiment20_report.json',report)
    (out/'实验20_目标与边界.md').write_text(
      '# 实验20：查询归属而非额外文本的筛选/融合\n\n'
      '继承实验19的视觉分组和候选池，逐查询复现旧模型的组编号和分数，若不一致则停止。\n'
      '同样的1,760训练拟合PID四折、多组证据模型；另外440训练身份仅评估。\n'
      '新增候选组内多张B图像及其训练标注文本的联合证据，独立视觉来源为基本单位。\n'
      '计算完整候选组的正确身份组排名及Top1错误修复/破坏、拒绝覆盖率、PID Bootstrap。\n'
      '所有B文本来自Train，A为Train查询，C独立未用于拟合和选择；Val图库描述不可用。\n'
      '绝不把组内多条描述等同于第二技术难点的信息过滤，也不声称有融合收益。\n',encoding='utf-8')
    return report


def package20(out):
    files=['run_info.json','irra_exp20_query_group_multi_evidence.py',
           'new_group_matcher.json','fit_crossval_thresholds.json',
           'fit_fold_diagnostics.csv','fit_crossval_decisions.csv',
           'holdout_all_query_group_ranks.csv','holdout_method_summary.csv',
           'holdout_true_group_rank_summary.csv','人工核查案例.csv',
           'bootstrap_pid.json','experiment20_report.json','experiment_run.log','实验20_目标与边界.md']
    missing=[f for f in files if not (out/f).is_file()]
    if missing:raise RuntimeError('ZIP必须文件缺失: '+str(missing))
    dest=out/'发送给ChatGPT.zip';tmp=out/'._发送给ChatGPT.zip'
    try:
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
            for p in files:z.write(out/p,arcname=p)
        with zipfile.ZipFile(tmp) as z:
            if z.testzip() is not None or set(z.namelist())!=set(files):
                raise RuntimeError('报告ZIP完整性检查失败')
        os.replace(tmp,dest)
    finally:
        if tmp.exists():tmp.unlink()
    return dest


def choose_out20(parent,exp12,exp16,exp19,settings,vis_tau,resume):
    parent.mkdir(parents=True,exist_ok=True)
    signature=stable_json({'version':VERSION,'source12':str(exp12.resolve()),
       'source16':str(exp16.resolve()),'source19':str(exp19.resolve()),
       'settings_hash':settings['settings_hash'],'image_threshold':vis_tau,
       'features':list(ALL_FEATURE_NAMES),'script_sha256':sha256_file(Path(__file__).resolve())})
    if resume:
        found=[]
        for p in sorted(parent.glob(EXP20+'*'),reverse=True):
            try:r=json.loads((p/'run_info.json').read_text(encoding='utf-8'))
            except (ValueError,OSError):continue
            if r.get('fingerprint')==signature:found.append(p)
        if not found:raise FileNotFoundError('--resume没找到版本、数据、源码完全匹配的实验20目录')
        return found[0],True
    now=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path=parent/(EXP20+now);path.mkdir(exist_ok=False)
    atomic_json(path/'run_info.json',{'fingerprint':signature,'version':VERSION,
       'source_exp12':str(exp12),'source_exp16':str(exp16),'source_exp19':str(exp19),
       'best_checkpoint_sha256':BEST_SHA,'supports_only_from_TRAIN_captions':True,
       'no_validation_caption_support':True,'group_candidates_fixed_from_exp19':True})
    return path,False


def run20():
    p=argparse.ArgumentParser(description='实验20：实验19组复现 + 查询身份归属多图多描述证据诊断')
    p.add_argument('--project-root',default=str(ROOT))
    p.add_argument('--exp12-dir',default='');p.add_argument('--exp16-dir',default='');p.add_argument('--exp19-dir',default='')
    p.add_argument('--resume',action='store_true');p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:self_test20();return
    root=Path(a.project_root).expanduser().resolve();parent=root/PARENT
    src12=find_exp12(parent,a.exp12_dir)
    settings,selected=source_audit(src12,full_hash=True)
    src16=find_exp16(parent,a.exp16_dir)
    visu_tau,cal=audit_exp16(src16,src12,settings)
    src19=find_exp19_folder(parent,a.exp19_dir)
    old_model,old_cut,old_rows,_,prior_report=audit_exp19_folder(src19,src12,src16)
    records=read_records(settings['annotation_file'],selected)
    tx,vi,vtx,vim=load_features(src12,settings,records)
    for ar,n in ((tx,'Train文本'),(vi,'Train图像'),(vtx,'Val查询'),(vim,'Val图像')):l2check(ar,n)
    if set(settings['fit_pids'])&set(settings['holdout_pids']):raise RuntimeError('拟合和holdout身份重叠')
    out,resumed=choose_out20(parent,src12,src16,src19,settings,visu_tau,a.resume)
    shutil.copy2(Path(__file__).resolve(),out/'irra_exp20_query_group_multi_evidence.py')
    if resumed and (out/'发送给ChatGPT.zip').is_file() and (out/'experiment20_report.json').is_file():
        with zipfile.ZipFile(out/'发送给ChatGPT.zip') as z:
            if z.testzip() is None and 'experiment20_report.json' in z.namelist():
                print('实验20已有完整报告（未重复运算）:',out);return
    log20(out,'实验20开始：最佳IRRA与实验19严格复现，使用训练身份原始合法描述')
    model,cuts,fitdiag=fit_4fold20(out,tx,vi,records['train_texts'],records['train_images'],settings['fit_pids'],visu_tau)
    # Only after all model fitting and threshold selection has finished, open unseen internal holdout.
    d=build_abc_subset(tx,vi,records['train_texts'],records['train_images'],settings['holdout_pids'])
    gs,lut,sim=build_global_visual_graph(d['VB'],visu_tau)
    meta,lut2=prepare_group_metadata(d,gs,sim)
    if not np.array_equal(lut,lut2):raise RuntimeError('holdout视觉组的索引不一致')
    gs_audit=graph_audit(gs,d['b_pid_AUDIT'])
    prior_graph=prior_report['holdout']['global_visual_graph']
    for name in ('n_images','n_groups','n_multiview_groups','n_pure_multiview_groups',
                 'n_impure_groups','n_same_pid_pairs','n_proposed_pairs','n_correct_pairs'):
        if int(gs_audit[name])!=int(prior_graph[name]):
            raise RuntimeError(f'实验19原始视觉图谱统计未重现: {name}: {gs_audit[name]} vs {prior_graph[name]}')
    if len(d['T1'])!=542 or len(d['VB'])!=637 or len(d['TB'])!=1274:
        raise RuntimeError('实验19 A/B查询和描述集合规模不一致：禁止比较')
    # Source captions and T1 need to be human-readable for case inspection, but are NEVER ground-truth matching labels.
    d['source_query_captions']=[records['train_texts'][int(z)]['text'] for z in d['query_text_idx']]
    rows,replay,allrecs=produce_holdout20(out,d,meta,lut,old_model,model,cuts,old_rows)
    rep=write_reports20(out,rows,replay,fitdiag,cuts,gs_audit,old_cut,src19,visu_tau,
                        d,meta,records['train_texts'],records['train_images'])
    log20(out,f'完成。实验19逐条复现{replay["replayed_queries"]}条；new正确={sum(r["new_correct_AUDIT"] for r in rows)}/{len(rows)}')
    dest=package20(out)
    print('实验20输出目录:',out,'\n请上传:',dest,flush=True)


def self_test20():
    """模拟训练-隔离-多源聚合-旧模型复现-报告-恢复的完整闭环。"""
    rng=np.random.default_rng(20020)
    nfit=48;nhold=8;ids=list(range(1000,1000+nfit+nhold));images=[];texts=[];VI=[];TX=[]
    for pid in ids:
        center=normalized(rng.normal(size=(1,48)))[0]
        for img_no in range(4):
            idx=len(images);images.append({'pid':pid,'path':f'train/{pid}/image{img_no}.jpg'})
            VI.append(normalized((center+.035*rng.normal(size=48))[None,:])[0])
            for ti in range(2):
                texts.append({'pid':pid,'img_idx':idx,'text':f'PID{pid} different viewpoint {img_no} caption {ti}'})
                TX.append(normalized((center+.06*rng.normal(size=48))[None,:])[0])
    VI=np.asarray(VI,np.float32);TX=np.asarray(TX,np.float32)
    d=build_abc_subset(TX,VI,texts,images,ids[-nhold:]);gs,lut,S=build_global_visual_graph(d['VB'],.83)
    meta,lut2=prepare_group_metadata(d,gs,S);assert np.array_equal(lut,lut2)
    rr=queries_with_extended_evidence(d,meta,lut)
    assert len(rr)==2*nhold and all(len(r['gids'])>=1 for r in rr)
    assert all(r['XE'].shape[1]==len(ALL_FEATURE_NAMES) for r in rr)
    assert not any(np.isnan(r['XE']).any() for r in rr)
    assert all(np.any(r['true']) for r in rr)
    # A concrete single image group must not have magical independent second-view evidence.
    single_idx=next((i for i,z in enumerate(meta) if len(z['images'])==1),None)
    if single_idx is not None:
        q=rr[0];gid=np.flatnonzero(q['gids']==single_idx)
        if len(gid):assert q['XE'][gid[0],len(FEATURE_NAMES)+3]==q['XE'][gid[0],len(FEATURE_NAMES)+0]
    with tempfile.TemporaryDirectory() as temp:
        out=Path(temp)
        fitted,cuts,diags=fit_4fold20(out,TX,VI,texts,images,ids[:nfit],.83)
        f2,c2,diags2=fit_4fold20(out,TX,VI,texts,images,ids[:nfit],.83)
        assert cuts==c2 and diags==diags2 and fitted==f2,'四折续跑改变输出'
        base_records=[]
        old_model=fit_extended_model([{'XE':np.asarray(r['X'],np.float32),'true':r['true']} for r in rr])
        old_model['feature_names']=list(FEATURE_NAMES)
        for r in rr:
            co=choose_rank(r,group_score(r['X'],old_model))
            base_records.append({'query_index':r['query_index'],'group_fit_all_source_group_id':co['gid'],
                 'group_fit_all_score':co['score'],'n_candidate_groups':len(r['gids']),
                 'top1_B_image_correct_PID_AUDIT':r['top1_correct_PID_AUDIT'],
                 'query_text_train_index':int(d['query_text_idx'][r['query_index']]),
                 'query_pid_AUDIT_ONLY':r['query_pid_AUDIT']})
        d['source_query_captions']=[texts[int(z)]['text'] for z in d['query_text_idx']]
        rows,audit,_=produce_holdout20(out,d,meta,lut,old_model,fitted,cuts,base_records,chunk_size=5)
        rows2,audit2,_=produce_holdout20(out,d,meta,lut,old_model,fitted,cuts,base_records,chunk_size=5)
        assert rows==rows2 and audit==audit2 and audit['replayed_queries']==2*nhold
        atomic_json(out/'run_info.json',{'self_test':True})
        report=write_reports20(out,rows,audit,diags,cuts,graph_audit(gs,d['b_pid_AUDIT']),
                               {'threshold':.99},Path('synthetic_exp19'),.83,d,meta,texts,images,self_test=True)
        shutil.copy2(Path(__file__).resolve(),out/'irra_exp20_query_group_multi_evidence.py')
        z=package20(out)
        with zipfile.ZipFile(z) as f:
            assert f.testzip() is None and len(f.namelist())==14
        assert len(list(csv.DictReader((out/'holdout_all_query_group_ranks.csv').open(encoding='utf-8-sig',newline=''))))==2*nhold
        # The validation utility never infers 'same PID' solely from a single edge.
        vv=normalized(np.asarray([[1,0,0],[.9,.43589,0],[.62,.78,0]],np.float32))
        bad,_,_=build_global_visual_graph(vv,.80,3)
        assert not any(len(g)==3 for g in bad)
    print('SELF-TEST PASS: independent train/holdout, fixed visual graph & candidates, multi-view + caption evidence, old-score exact replay, crossfit, 90/95 abstain, CSV/ZIP, chunk & fold resume.',flush=True)


if __name__=='__main__':
    run20()
