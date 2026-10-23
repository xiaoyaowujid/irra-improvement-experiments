# -*- coding: utf-8 -*-
"""实验19：训练身份内部，无PID全图库身份图谱构建与查询身份组归属。

核心：分开验证图像组纯度和文本T1是否属于该组；仅训练身份原有描述可作支持来源。
训练身份内部进行；T1查询来源A、可取得字幕来源B、评价图库C相互隔离。
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
VERSION='exp19_v1_group_link_and_query_attribution_cvfit_only'
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


def summarize_methods(rows):
    result=[]
    for name in ('single_top1','group_static','group_fit_all','group_fit_with_abstention'):
        accepted=[r for r in rows if bool(r[name+'_accepted'])]
        correct=sum(r[name+'_correct_PID_AUDIT'] for r in accepted)
        mult=[r for r in accepted if r[name+'_n_images']>=2]
        correct_mult=sum(r[name+'_correct_PID_AUDIT'] for r in mult)
        result.append({'method':name,'n_queries':len(rows),'n_accepted':len(accepted),
            'accept_coverage':len(accepted)/len(rows),'identity_precision_when_accepted':correct/len(accepted) if accepted else None,
            'correct_identity_support_recall_per_query':correct/len(rows),
            'n_accepted_with_2plus_images':len(mult),
            'multiview_identity_precision':correct_mult/len(mult) if mult else None,
            'n_accepted_groups_with_B_image_outside_top16':sum(bool(r[name+'_any_group_member_beyond16']) for r in accepted),
            'n_accepted_correct_identity_with_B_image_outside_top16':sum(bool(r[name+'_any_group_member_beyond16']) and bool(r[name+'_correct_PID_AUDIT']) for r in accepted),
            'n_queries_with_2plus_training_captions_from_2plus_images':sum(bool(r[name+'_can_supply_2plus_distinct_view_captions']) for r in accepted)})
    return result


def evaluate_holdout(d,groups,meta,bmap,fit_model,cutoff,records=None):
    if records is None:records,_=make_queries(d,meta,bmap,training=False)
    rows=[]
    for rec in records:
        qi=rec['query_index'];qpid=int(rec['query_pid_AUDIT'])
        rt={'query_index':int(qi),'query_text_train_index':int(d['query_text_idx'][qi]),
            'query_pid_AUDIT_ONLY':qpid,
            'top1_B_image_correct_PID_AUDIT':int(rec['top1_pid_correct_AUDIT']),
            'has_true_B_source_AUDIT':int(rec['has_same_pid_image_in_B_AUDIT']),
            'true_B_source_beyond_Top16_AUDIT':int(rec['true_B_img_beyond16_AUDIT']),
            'true_group_in_global_B_AUDIT':int(np.any(rec['group_true'])),
            'true_group_in_candidate_pool_AUDIT':int(rec['positive_group_in_candidates_AUDIT']),
            'n_candidate_groups':len(rec['gids'])}
        gi0=int(rec['top1_group'])
        # Baseline single image: do not imply the image set has >1 image.
        answers={'single_top1':(gi0,float(d['VB'][rec['top16'][0]]@d['T1'][qi]),
                                 int(rec['top1_pid_correct_AUDIT']),[int(rec['top16'][0])]),
                 'group_static':(*best_group_choice(rec,baseline=True)[:3],None),
                 'group_fit_all':(*best_group_choice(rec,fit_model)[:3],None),
                 'group_fit_with_abstention':(*best_group_choice(rec,fit_model)[:3],None)}
        for method,(gid,score,correct,forced_images) in answers.items():
            is_accepted=(method!='group_fit_with_abstention' or score>=cutoff-1e-12)
            g=list(map(int,meta[gid]['images'])) if forced_images is None else forced_images
            texts_ids=meta[gid]['texts'] if forced_images is None else np.flatnonzero(d['t2b']==forced_images[0])
            text_local=list(map(int,texts_ids));image_indices=[int(d['B_image_idx'][j]) for j in g]
            top16set=set(map(int,rec['top16']))
            outside=any(j not in top16set for j in g)
            # For abstained queries, source is empty, not a false success.
            rt.update({method+'_accepted':int(is_accepted),method+'_score':float(score),
                method+'_correct_PID_AUDIT':int(correct) if is_accepted else 0,
                method+'_source_group_id':int(gid) if is_accepted else -1,
                method+'_n_images':len(g) if is_accepted else 0,
                method+'_n_support_captions':len(text_local) if is_accepted else 0,
                method+'_can_supply_2plus_distinct_view_captions':int(is_accepted and len(g)>=2 and len(text_local)>=2),
                method+'_any_group_member_beyond16':int(is_accepted and outside),
                method+'_local_support_image_indices':json.dumps(g,ensure_ascii=False) if is_accepted else '[]',
                method+'_original_support_image_indices':json.dumps(image_indices,ensure_ascii=False) if is_accepted else '[]',
                method+'_original_support_text_indices':json.dumps([int(d['B_text_idx'][t]) for t in text_local],ensure_ascii=False) if is_accepted else '[]'})
        rt['error_category_OFFLINE_ONLY']='source_matched' if rt['group_fit_with_abstention_correct_PID_AUDIT'] else (
          'abstained' if not rt['group_fit_with_abstention_accepted'] else 'pure_or_mixed_wrong_query_identity')
        rows.append(rt)
    return rows


def bootstrap_identity_query(rows,reps=1200,seed=2306):
    pids=sorted(set(r['query_pid_AUDIT_ONLY'] for r in rows))
    idx={p:[i for i,r in enumerate(rows) if r['query_pid_AUDIT_ONLY']==p] for p in pids}
    if not pids:return {}
    rng=np.random.default_rng(seed);result={}
    for meth in ('single_top1','group_static','group_fit_all','group_fit_with_abstention'):
        samples=[]
        for _ in range(reps):
            chosen=rng.integers(0,len(pids),size=len(pids));indices=[j for z in chosen for j in idx[pids[int(z)]]]
            n=len(indices);correct=sum(rows[j][meth+'_correct_PID_AUDIT'] for j in indices)
            samples.append(correct/n)
        vals=np.quantile(samples,[.025,.975]);result[meth]={'correct_support_per_query':float(np.mean([r[meth+'_correct_PID_AUDIT'] for r in rows])),
             'pid_cluster_95ci':[float(v) for v in vals]}
    return result


def atomic_jsonl19(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.exp19_',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def read_jsonl19(path,n=None):
    with open(path,'r',encoding='utf-8') as f:rows=[json.loads(s) for s in f if s.strip()]
    if n is not None and len(rows)!=n:raise RuntimeError('恢复块长度不一致 '+str(path))
    return rows



def save_fit_fold19(path,rec,train,fold_id,pids):
    # Compact gzip-JSON cache: no PID labels used in production inference; fit data only.
    payload={'version':VERSION,'fold':fold_id,'fit_pids_hash':stable_json(sorted(map(int,pids))),
             'records':[{'query_index':int(r['query_index']),
               'gids':r['gids'].tolist(),'X':r['X'].tolist(),'true':r['true'].tolist(),
               'positive_group_in_candidates_AUDIT':int(r['positive_group_in_candidates_AUDIT'])} for r in rec]}
    path=Path(path);fd,tmp=tempfile.mkstemp(prefix='.exp19_',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as gz:
                gz.write(json.dumps(payload,ensure_ascii=False,allow_nan=False).encode('utf-8'))
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def load_fit_fold19(path,fold_id,pids):
    with gzip.open(path,'rt',encoding='utf-8') as f:payload=json.load(f)
    if payload.get('version')!=VERSION or payload.get('fold')!=fold_id or payload.get('fit_pids_hash')!=stable_json(sorted(map(int,pids))):
        raise RuntimeError('无法恢复：拟合折缓存指纹无效 '+str(path))
    rec=[];tr=[]
    for r in payload['records']:
        q={'query_index':int(r['query_index']),'gids':np.asarray(r['gids'],np.int32),
           'X':np.asarray(r['X'],np.float32),'true':np.asarray(r['true'],np.int32),
           'positive_group_in_candidates_AUDIT':int(r['positive_group_in_candidates_AUDIT'])}
        if q['X'].shape!=(len(q['gids']),len(FEATURE_NAMES)) or len(q['true'])!=len(q['gids']):
            raise RuntimeError('拟合折特征缓存形状错误')
        rec.append(q);tr.append((q['X'],q['true']))
    return rec,tr

def log_stage19(out,message):
    with (Path(out)/'experiment_run.log').open('a',encoding='utf-8') as f:
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+' '+str(message)+'\n')
    print(message,flush=True)


def dataset_fold(d,threshold):
    groups,lut,S=build_global_visual_graph(d['VB'],threshold)
    meta,lut2=prepare_group_metadata(d,groups,S)
    if not np.array_equal(lut,lut2):raise RuntimeError('组索引映射错误')
    return groups,meta,lut,graph_audit(groups,d['b_pid_AUDIT'])


def do_crossfit_and_holdout(out,tx,vi,texts,images,settings,threshold,
                            n_folds=FIT_FOLDS,target=TARGET_PRECISION, self_test=False):
    fit=sorted(map(int,settings['fit_pids']));hold=sorted(map(int,settings['holdout_pids']))
    if set(fit)&set(hold):raise RuntimeError('fit/holdout PID overlap')
    folds=[list(map(int,x)) for x in np.array_split(np.asarray(fit,np.int64),n_folds)]
    foldsets=[];folds_diag=[]
    for fi,ids in enumerate(folds):
        cache=out/f'fit_fold_{fi:02d}_features.json.gz'
        statcache=out/f'fit_fold_{fi:02d}_audit.json'
        if cache.is_file() and statcache.is_file():
            rec,train=load_fit_fold19(cache,fi,ids)
            diag=json.loads(statcache.read_text(encoding='utf-8'))
            if diag.get('fold')!=fi or diag.get('n_PID')!=len(ids):
                raise RuntimeError('拟合折统计元信息不匹配')
            log_stage19(out,f' RESUME FIT fold={fi+1}/{n_folds}, queries={len(rec)}')
        else:
            d=build_abc_subset(tx,vi,texts,images,ids)
            gs,meta,lut,gdiag=dataset_fold(d,threshold)
            rec,train=make_queries(d,meta,lut,limit=FIT_MAX_QUERY_PER_FOLD,training=True)
            if not train:raise RuntimeError('fit训练候选为空')
            diag={'fold':fi,'n_PID':len(ids),'n_support_B_images':len(d['VB']),
                           'n_fit_A_queries':len(rec),'n_fit_positive_in_candidates':sum(r['positive_group_in_candidates_AUDIT'] for r in rec),
                           **gdiag}
            save_fit_fold19(cache,rec,train,fi,ids)
            atomic_json(statcache,diag)
            log_stage19(out,f' FIT fold={fi+1}/{n_folds} PID={len(ids)} queries={len(rec)} groups={len(gs)}')
        if not train:raise RuntimeError('恢复的拟合折不含样本')
        foldsets.append((rec,train));folds_diag.append(diag)
    crossfit=[]
    for fi,(rec,_) in enumerate(foldsets):
        others=[data for fj,(_,tr) in enumerate(foldsets) if fj!=fi for data in tr]
        model=train_binary_linear(others)
        for r in rec:
            gid,s,c,_=best_group_choice(r,model)
            crossfit.append({'fold':fi,'query_index':r['query_index'],
                             'score':s,'correct':c,'selected_group':gid,
                             'positive_group_in_candidate_pool_AUDIT':int(r['positive_group_in_candidates_AUDIT'])})
    cutoff=threshold_from_crossfit(crossfit,target)
    model=train_binary_linear([data for _,training in foldsets for data in training])
    # Holdout data is read only after model and rejection threshold are final.
    d=build_abc_subset(tx,vi,texts,images,hold)
    gs,meta,lut,gdiag=dataset_fold(d,threshold)
    allrecs,_=make_queries(d,meta,lut,training=False)
    rows=[]
    for st in range(0,len(allrecs),32):
        ed=min(st+32,len(allrecs));piece=out/f'holdout_cases_{st:05d}_{ed:05d}.jsonl'
        if piece.is_file():
            chunk=read_jsonl19(piece,ed-st)
            log_stage19(out,f' RESUME HOLDOUT {st}:{ed}')
        else:
            chunk=evaluate_holdout(d,gs,meta,lut,model,cutoff['threshold'],allrecs[st:ed])
            atomic_jsonl19(piece,chunk)
            log_stage19(out,f' HOLDOUT processed {ed}/{len(allrecs)}')
        rows.extend(chunk)
    if [r['query_index'] for r in rows]!=list(range(len(d['T1']))):
        raise RuntimeError('holdout query restore order mismatch')
    # Report stand-alone controls and exact multi-caption provenance per selected group.
    methods=summarize_methods(rows)
    atomic_json(out/'fit_trained_group_matcher.json',model)
    atomic_json(out/'fit_crossval_threshold.json',cutoff)
    atomic_csv(out/'fit_fold_diagnostics.csv',folds_diag,list(folds_diag[0]))
    atomic_csv(out/'fit_crossval_decisions.csv',crossfit,list(crossfit[0]))
    atomic_csv(out/'holdout_all_queries.csv',rows,list(rows[0]))
    atomic_csv(out/'holdout_group_matching_metrics.csv',methods,list(methods[0]))
    group_rows=[]
    for gi,g in enumerate(gs):
        pidset=set(int(d['b_pid_AUDIT'][j]) for j in g)
        group_rows.append({'group_id':int(gi),'n_B_images':len(g),
          'n_B_captions':len(meta[gi]['texts']),'n_true_PID_AUDIT':len(pidset),
          'is_pure_AUDIT':int(len(pidset)==1),
          'image_original_train_indices':json.dumps([int(d['B_image_idx'][j]) for j in g]),
          'caption_original_train_indices':json.dumps([int(d['B_text_idx'][j]) for j in meta[gi]['texts']])})
    atomic_csv(out/'holdout_all_visual_groups.csv',group_rows,list(group_rows[0]))
    boot=bootstrap_identity_query(rows,400 if self_test else 1200)
    atomic_json(out/'bootstrap_pid.json',boot)
    fit_stats={'n_fit_PID':len(fit),'n_folds':n_folds,'fit_fold_graph':folds_diag,
               'crossfit_queries':len(crossfit),'CV_target_precision':target,
               'threshold_calibration':cutoff}
    hold_stats={'n_hold_PID':len(hold),'n_B_images':len(d['VB']),'n_B_train_captions':len(d['TB']),
                'n_C_independent_images_unused_for_decisions':len(d['C_image_idx']),
                'n_A_query_images':d['n_A_images'],'n_query_texts':len(rows),
                'global_visual_graph':gdiag,
                'n_queries_correct_B_image_outside_Top16':int(sum(r['true_B_source_beyond_Top16_AUDIT'] for r in rows)),
                'n_queries_true_group_in_global_B':int(sum(r['true_group_in_global_B_AUDIT'] for r in rows)),
                'n_queries_true_group_in_candidate_pool':int(sum(r['true_group_in_candidate_pool_AUDIT'] for r in rows)),
                'n_queries_no_correct_B_source':int(sum(not r['has_true_B_source_AUDIT'] for r in rows)),
                'n_queries_selected_wrong_group':int(sum(r['group_fit_with_abstention_accepted'] and not r['group_fit_with_abstention_correct_PID_AUDIT'] for r in rows)),
                'n_queries_abstained':int(sum(not r['group_fit_with_abstention_accepted'] for r in rows))}
    return {'fit':fit_stats,'holdout':hold_stats,'methods':methods,'bootstrap':boot}


def choose_outfolder(parent,src,ex16,settings,resume):
    parent.mkdir(parents=True,exist_ok=True)
    fp=stable_json({'version':VERSION,'source_exp12':str(src.resolve()),'source_exp16':str(ex16.resolve()),
                    'settings_hash':settings['settings_hash'],'target_precision':TARGET_PRECISION,
                    'graph_k':GROUP_KNN,'candidate_groups':CANDIDATE_GROUPS})
    old=sorted(parent.glob(EXP19+'*'),reverse=True)
    if resume:
        matches=[]
        for p in old:
            try:r=json.loads((p/'run_info.json').read_text(encoding='utf-8'))
            except (OSError,ValueError):continue
            if r.get('fingerprint')==fp:matches.append(p)
        if not matches:raise FileNotFoundError('--resume找不到与本次源码/数据设置完全匹配的实验19输出目录')
        return matches[0],True
    now=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    p=parent/(EXP19+now);p.mkdir(parents=True,exist_ok=False)
    atomic_json(p/'run_info.json',{'version':VERSION,'fingerprint':fp,'source_exp12':str(src),
       'source_exp16':str(ex16),'source_best_checkpoint_sha256':BEST_SHA,
       'calibration_only_from_fit_pids':True,'holdout_no_PID_matching':True,
       'candidate_caption_source':'Train original captions only',
       'official_validation_caption_support':False})
    return p,False


def package19(out):
    name='发送给ChatGPT.zip'
    files=['run_info.json','irra_exp19_global_identity_group_attribution.py',
      'fit_trained_group_matcher.json','fit_crossval_threshold.json',
      'fit_fold_diagnostics.csv','fit_crossval_decisions.csv','holdout_all_queries.csv',
      'holdout_group_matching_metrics.csv','holdout_all_visual_groups.csv',
      'bootstrap_pid.json','experiment19_report.json','experiment_run.log','实验19_目标与边界.md']
    for f in files:
        if not (out/f).is_file():raise RuntimeError('报告文件缺失:'+f)
    tmp=out/('._'+name)
    try:
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
            for f in files:z.write(out/f,arcname=f)
        with zipfile.ZipFile(tmp) as z:
            if z.testzip() is not None or set(z.namelist())!=set(files):raise RuntimeError('ZIP完整性核验失败')
        os.replace(tmp,out/name)
    finally:
        if tmp.exists():tmp.unlink()
    return out/name


def main19():
    ap=argparse.ArgumentParser(description='实验19：全图库无标签身份关联/查询组归属，两种错误分开评估')
    ap.add_argument('--project-root',default=str(ROOT));ap.add_argument('--exp12-dir',default='')
    ap.add_argument('--exp16-dir',default='');ap.add_argument('--resume',action='store_true')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:self_test19();return
    root=Path(args.project_root).expanduser().resolve();parent=root/PARENT
    src=find_exp12(parent,args.exp12_dir)
    settings,selected=source_audit(src,full_hash=True)
    ex16=find_exp16(parent,args.exp16_dir)
    threshold,cal=audit_exp16(ex16,src,settings)
    rec=read_records(settings['annotation_file'],selected)
    tx,vi,vtx,vim=load_features(src,settings,rec)
    for x,n in ((tx,'train text'),(vi,'train image'),(vtx,'Val query'),(vim,'Val image')):l2check(x,n)
    valimgpid=np.asarray([a['pid'] for a in rec['val_images']],np.int64)
    valtxtpid=np.asarray([a['pid'] for a in rec['val_queries']],np.int64)
    if set(valimgpid)&set(settings['selected_pids']):raise RuntimeError('Train/Val身份不隔离')
    val_r1=float(100*np.mean(valimgpid[rank_topk(vtx,vim,k=1)[:,0]]==valtxtpid))
    if abs(val_r1-66.43390655517578)>.03:raise RuntimeError('最佳IRRA官方Val基线核验失败')
    out,resumed=choose_outfolder(parent,src,ex16,settings,args.resume)
    shutil.copy2(Path(__file__).resolve(),out/Path(__file__).name)
    if resumed and (out/'发送给ChatGPT.zip').exists() and (out/'experiment19_report.json').exists():
        with zipfile.ZipFile(out/'发送给ChatGPT.zip') as z:
            if z.testzip() is None and 'experiment19_report.json' in z.namelist():
                print('实验19完整报告已存在：',out,flush=True);return
    log_stage19(out,f'实验19开始：最佳IRRA官方Val R1={val_r1:.6f}、fit视觉阈值={threshold:.6f}')
    result=do_crossfit_and_holdout(out,tx,vi,rec['train_texts'],rec['train_images'],settings,threshold)
    report={'version':VERSION,'best_checkpoint_sha256':BEST_SHA,'official_best_val_R1_verify_only':val_r1,
       'source_exp12':str(src),'source_exp16':str(ex16),'exp16_threshold_fit_metadata':cal,
       'no_IRRA_training':True,'only_lightweight_group_attribution_model_is_fitted':True,
       'train_only_training_captions_used_as_B_sources':True,'val_test_gallery_caption_support':False,
       'groups_found_globally_not_seeded_at_Top16':True,'Top16_only_auxiliary_diagnostic':True,
       'feature_names':list(FEATURE_NAMES),'target_fit_cv_precision':TARGET_PRECISION,
       'fit_and_holdout_pid_disjoint':True,'A_B_C_image_disjoint':True,
       'warnings':['These numbers concern provenance discovery, NOT real text set fusion or official Validation R1.',
         'Frozen best IRRA already trained on TRAIN holdout identities. This tests matcher generalization, not the encoder\'s unseen-ID retrieval.',
         'A visual group may be pure while belonging to the WRONG query identity.',
         'Holdout PID is used ONLY for post-hoc scoring; teacher labels used ONLY among disjoint FIT PIDs.',
         'The group-linking threshold is inherited from EXP16, not tuned on holdout.',
         'The query-group selector uses original TRAIN image captions, which are unavailable as Val gallery captions under the standard protocol.',
         'CV-selected abstention cutoff is not a guarantee of precision on unseen holdout identities.',
         'Group features built from whole B gallery, no test C image nor label used in decisions.',
         'No multi-text information selection, fusion, or image-feature fusion is claimed in this experiment.'],
       **result}
    atomic_json(out/'experiment19_report.json',report)
    (out/'实验19_目标与边界.md').write_text(
      '# 实验19：寻找同身份多文本集合的可靠来源（不做文本融合）\n\n'
      '目标：图像成组纯度和文本T1到身份组归属分别评价。全图库成组，无须先进入Top16。\n'
      'Train 1760 PID分4折训练并crossfit校准拒识阈值；440 PID独立评估归属模型。\n'
      '每个身份Train图像C独立保留但绝不用于选组或模型拟合；A给T1，B给图像+已有训练描述。\n'
      '真实PID只用于离线拟合标签和事后核查；holdout候选生成、选组不读取真实PID。\n'
      '输出一个身份组可提供的多张图像和所有不同TRAIN描述，不强制只选一条T2。\n'
      'Val/Test的人工图库文字不得被模型用于推理；本实验未得到正式Validation检索提升。\n'
      '本实验不包含Huffman树、不包含最短路径算法；这些仅作后续方法讨论。\n',
      encoding='utf-8')
    dest=package19(out)
    print('实验19完成。输出：',out,'\n上传文件：',dest,flush=True)


def self_test19():
    """Test formulas, source-leakage, fit-CV, abstention, exporter and resume self-consistency."""
    rng=np.random.default_rng(1942)
    # Trap: A->B->C similarity >tau on each adjacent edge must NOT force A,C merged.
    vv=normalized(np.asarray([[1,0,0], [.90,.43589,0],[.62,.78,0],[0,0,1]],np.float32))
    gg,_,_=build_global_visual_graph(vv,.80,mutual_k=3)
    assert not any(set([0,1,2]).issubset(g) for g in gg),gg
    # Synthetic actual query IDs remain unseen to score-maker.
    D=32;all_pids=list(range(1000,1056));fit=all_pids[:48];hold=all_pids[48:]
    images=[];texts=[];V=[];T=[]
    for p in all_pids:
        base=normalized(rng.normal(size=(1,D)))[0]
        for j in range(4):
            idx=len(images);images.append({'pid':p,'path':f'train/{p}_{j}.jpg'})
            V.append(normalized((base+.04*rng.normal(size=D))[None,:])[0])
            for c in range(2):
                texts.append({'pid':p,'img_idx':idx,'text':f'train PID{p} image {j} caption {c}'})
                T.append(normalized((base+.08*rng.normal(size=D))[None,:])[0])
    V=np.asarray(V,np.float32);T=np.asarray(T,np.float32)
    d=build_abc_subset(T,V,texts,images,hold)
    assert len(d['C_image_idx'])==8 and len(d['VB'])==16 and len(d['T1'])==16
    assert set(map(int,d['C_image_idx']))&set(map(int,d['B_image_idx']))==set()
    group,meta,idx,stats=dataset_fold(d,.8)
    assert len(group)>0 and stats['pair_precision']==1.
    rec,_=make_queries(d,meta,idx)
    assert all(r['positive_group_in_candidates_AUDIT'] for r in rec)
    # Exhaustive check: graph always built over ALL B, Top16 only recorded per-query.
    with tempfile.TemporaryDirectory() as t:
        out=Path(t);settings={'fit_pids':fit,'holdout_pids':hold}
        a=do_crossfit_and_holdout(out,T,V,texts,images,settings,.8,n_folds=4,target=.7,self_test=True)
        b=do_crossfit_and_holdout(out,T,V,texts,images,settings,.8,n_folds=4,target=.7,self_test=True)
        assert a==b,'resume/repeat changed results'
        rows=list(csv.DictReader((out/'holdout_all_queries.csv').open(encoding='utf-8-sig',newline='')))
        assert len(rows)==16
        assert all(int(r['has_true_B_source_AUDIT'])==1 for r in rows)
        assert all(int(r['true_group_in_global_B_AUDIT'])==1 for r in rows)
        assert all(int(r['group_fit_all_correct_PID_AUDIT'])==1 for r in rows)
        assert all(int(r['group_fit_all_n_support_captions'])>=2 for r in rows)
        assert all(int(r['group_fit_all_n_images'])>=2 for r in rows)
        assert a['holdout']['n_C_independent_images_unused_for_decisions']==8
        atomic_json(out/'run_info.json',{'self_test':True})
        atomic_json(out/'experiment19_report.json',{'self_test':True,'results':a['methods']})
        (out/'实验19_目标与边界.md').write_text('Self test',encoding='utf-8')
        shutil.copy2(Path(__file__),out/Path(__file__).name)
        pack=package19(out)
        with zipfile.ZipFile(pack) as z:assert z.testzip() is None and 'holdout_all_queries.csv' in z.namelist()
        # No stale cache leak from file handles at temp dir removal on Windows.
    print('SELF-TEST PASS: global graph, non-transitive merges, CV calibration, correct PID audit, disjoint A/B/C, no Val caption, CSV, ZIP, repeat',flush=True)

if __name__=='__main__':
    main19()
