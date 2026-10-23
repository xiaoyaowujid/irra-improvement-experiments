# -*- coding: utf-8 -*-
"""实验22：非线性多文本集合与跨候选组自注意力竞争。

继承实验19冻结缓存与视觉组，严格复现542个查询，再对照两种新模型。
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
import sys

# Windows + Conda: avoid loading a NumPy/MKL OpenMP runtime before PyTorch's.
# Sequential MKL disables MKL's additional threading runtime; all other original
# thread limits below are kept unchanged. Never set KMP_DUPLICATE_LIB_OK.
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')


def _omp_diagnose_before_imports():
    """Diagnose runtime conflicts without crashing this parent Python process.

    All import combinations run in independent child processes. On Windows an
    OpenMP #15 fatal error can abort a child, but the main diagnostic survives.
    This mode reads the environment, installs or deletes nothing.
    """
    import subprocess
    from pathlib import Path
    env = os.environ.copy()
    cases = (
        ('torch only', 'import torch; print("torch", torch.__version__)'),
        ('numpy only', 'import numpy as np; print("numpy", np.__version__)'),
        ('torch then numpy', 'import torch; import numpy as np; print("both OK")'),
        ('numpy then torch', 'import numpy as np; import torch; print("both OK")'),
    )
    print('=== OpenMP / Intel MKL diagnostic (independent processes) ===', flush=True)
    print('Python executable:', sys.executable, flush=True)
    print('MKL_THREADING_LAYER:', env.get('MKL_THREADING_LAYER'), flush=True)
    for name, command in cases:
        try:
            p = subprocess.run([sys.executable, '-c', command], env=env,
                               capture_output=True, text=True, timeout=45,
                               errors='replace')
            print(f'[{name}] returncode={p.returncode}', flush=True)
            if p.stdout.strip(): print(' stdout:', p.stdout.strip()[-1000:], flush=True)
            if p.stderr.strip(): print(' stderr:', p.stderr.strip()[-1800:], flush=True)
        except subprocess.TimeoutExpired:
            print(f'[{name}] timeout after 45 sec', flush=True)
    locations = [
        Path(sys.prefix) / 'Library' / 'bin',
        Path(sys.prefix) / 'Lib' / 'site-packages' / 'torch' / 'lib',
        Path(sys.prefix) / 'Lib' / 'site-packages' / 'numpy.libs',
        Path(sys.prefix) / 'Lib' / 'site-packages' / 'numpy' / '.libs',
        Path(sys.base_prefix) / 'Library' / 'bin',
    ]
    locations += [Path(p) for p in os.environ.get('PATH', '').split(os.pathsep) if p]
    found = set()
    for d in locations:
        p = d / 'libiomp5md.dll'
        if p.is_file(): found.add(str(p.resolve()))
    print('Detected libiomp5md.dll paths:', flush=True)
    for p in sorted(found): print(' ', p, flush=True)
    if not found: print(' No copy found in usual DLL paths.', flush=True)
    print('If either combined import fails, copy this entire output for investigation.', flush=True)


if '--omp-diagnose' in sys.argv:
    _omp_diagnose_before_imports()
    raise SystemExit(0)

import argparse, csv, hashlib, json, shutil, tempfile, zipfile, gzip
from collections import defaultdict
from datetime import datetime
from pathlib import Path
# Loading PyTorch first is deliberate; NumPy follows the PyTorch runtime.
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

ROOT=Path(r'/')
PARENT='图文双向互补实验'
EXP12='实验12_最佳IRRA同身份文本互补迁移_'
EXP16='实验16_Top16多图与合法文本来源_'
EXP19='实验19_全图库双层身份关联与文本归属_'
EXP21='实验21_查询级图组竞争与多描述集合证据_'
EXP22='实验22_非线性跨组证据竞争与多文本来源_'
VERSION='exp22_v1_nonlinear_set_vs_crossgroup'
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



def dataset_fold(d,threshold):
    groups,lut,S=build_global_visual_graph(d['VB'],threshold)
    meta,lut2=prepare_group_metadata(d,groups,S)
    if not np.array_equal(lut,lut2):raise RuntimeError('组索引映射错误')
    return groups,meta,lut,graph_audit(groups,d['b_pid_AUDIT'])


# ----------------- Experiment 21: query-conditioned set competition -----------------
# Everything above this line is bundled from the already-used EXP19 source, so this is
# a standalone script: no imports of other experiment scripts, no new Python dependencies.
SET_FEATURE_NAMES=(
    'is_singleton','has_two_distinct_images','joint_best','joint_mean',
    'joint_second_if_exists','joint_min_if_exists','joint_top_two_min_if_exists',
    'distinct_view_support_fraction','distinct_view_support_gap',
    'text_embedding_diversity','img_caption_joint_agreement',
    'joint_competitor_gap','text_competitor_gap','image_competitor_gap',
    'n_distinct_captions_log'
)
BASE_FEATURE_NAMES=tuple(FEATURE_NAMES)+('base_competitor_gap',)
FULL_FEATURE_NAMES=BASE_FEATURE_NAMES+SET_FEATURE_NAMES
EXP19_VERSION='exp19_v1_group_link_and_query_attribution_cvfit_only'
CHUNK_SIZE=32


def find_exp19_folder(parent, explicit=''):
    choices=[Path(explicit).expanduser().resolve()] if explicit else sorted(parent.glob(EXP19+'*'),reverse=True)
    for p in choices:
        if all((p/f).is_file() for f in (
            'run_info.json','experiment19_report.json','fit_trained_group_matcher.json',
            'holdout_all_queries.csv')):return p
    raise FileNotFoundError('找不到实验19完整目录，请用 --exp19-dir 指定；上传ZIP不包含特征缓存。')


def audit_exp19(exp19,src12,src16):
    run=json.loads((exp19/'run_info.json').read_text(encoding='utf-8'))
    rep=json.loads((exp19/'experiment19_report.json').read_text(encoding='utf-8'))
    model=json.loads((exp19/'fit_trained_group_matcher.json').read_text(encoding='utf-8'))
    if run.get('version')!=EXP19_VERSION or run.get('source_best_checkpoint_sha256')!=BEST_SHA:
        raise RuntimeError('实验19版本/最佳模型权重不符')
    if Path(run['source_exp12']).resolve()!=src12.resolve() or Path(run['source_exp16']).resolve()!=src16.resolve():
        raise RuntimeError('实验19使用了不同的实验12/16来源')
    if model.get('feature_names')!=list(FEATURE_NAMES):raise RuntimeError('旧模型特征顺序不符')
    with (exp19/'holdout_all_queries.csv').open('r',encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=542 or sorted(int(r['query_index']) for r in rows)!=list(range(542)):
        raise RuntimeError('实验19必须提供完整542条原始holdout逐查询')
    return model,rep,{int(r['query_index']):r for r in rows}


def sig_one_group(q,meta_g,d,si,st):
    """One identity candidate -> evidence from distinct B images, not repeated captions.

    A singleton has zero second-view evidence and explicit singleton flags. Train IDs are
    NOT inspected here. True PID, query A image and validation/C features are NOT inputs.
    """
    imgs=list(map(int,meta_g['images']))
    tis=list(map(int,meta_g['texts']))
    feats=[]; unique_text=[]; diag=[]
    for im in imgs:
        ti=[t for t in tis if int(d['t2b'][t])==im]
        if not ti:raise RuntimeError('支持图像没有原始训练文本')
        imgscore=float(si[im]); tv=np.asarray(st[ti],np.float64)
        best_t=float(np.max(tv)); mean_t=float(np.mean(tv))
        joint=.5*imgscore+.5*best_t
        feats.append(joint)
        # Text2image ground compatibility is computed FROM support image, not A/C.
        diag.append(float(np.max(d['TB'][ti]@d['VB'][im])))
        unique_text.extend(ti)
    arr=np.sort(np.asarray(feats,np.float64))[::-1]
    n=len(arr);multi=int(n>=2)
    best=float(arr[0]); second=float(arr[1]) if multi else 0.
    true_fraction=float(np.sum(arr[1:]>=best-.08)/max(n-1,1)) if multi else 0.
    emb=d['TB'][np.asarray(sorted(set(unique_text)),np.int32)]
    if len(emb)>1:
        sim=np.asarray(emb@emb.T,np.float32);r,c=np.triu_indices(len(emb),1)
        diversity=float(np.mean(1.-sim[r,c]))
    else:diversity=0.
    return [float(n==1),float(multi),best,float(arr.mean()),second,
            float(arr[-1]) if multi else 0.,min(best,second) if multi else 0.,
            true_fraction,float(best-second) if multi else 0.,diversity,
            float(np.mean(diag)),0.,0.,0.,float(np.log1p(len(set(unique_text))))]


def extract_query_features(rec,q,d,meta,old_model):
    """Append set-level multi-image/text features to SAME experiment19 candidate pool."""
    gids=np.asarray(rec['gids'],np.int32);old=np.asarray(rec['X'],np.float32)
    if old.shape!=(len(gids),len(FEATURE_NAMES)) or len(gids)==0:
        raise RuntimeError('旧候选组特征维数/顺序错误')
    # This vector search is ID-free. It uses support B image and TRAIN caption embeddings.
    si=np.asarray(d['VB']@q,np.float32)
    st=np.asarray(d['TB']@q,np.float32)
    extras=np.asarray([sig_one_group(q,meta[int(g)],d,si,st) for g in gids],np.float32)
    # Competing group contrasts, not the score of an isolated group.
    for a,b in ((11,2),(12,3),(13,0)):
        m=float(np.max(extras[:,b] if b==2 else old[:,b]))
        val=extras[:,b] if b==2 else old[:,b]
        extras[:,a]=val-m
    # Do not use EXP19 fit-model logits here. EXP19 was trained on all fit PIDs,
    # including the would-be CV fold: using its predictions as CV input would
    # silently leak supervision into cross-validation and cutoff calibration.
    base_static=feature_baseline(old)
    gap=(base_static-np.max(base_static)).astype(np.float32)
    base=np.column_stack([old,gap]).astype(np.float32)
    full=np.column_stack([base,extras]).astype(np.float32)
    if not np.isfinite(full).all():raise RuntimeError('集合特征含NaN/Inf')
    for pos in range(len(gids)):
        if extras[pos,0]==1:
            assert extras[pos,1]==0 and extras[pos,4]==0 and extras[pos,6]==0 and extras[pos,7]==0
    return base,full


def make_fit_records(d,meta,lut,old_model,training):
    rec,_=make_queries(d,meta,lut,training=training)
    result=[]
    for r in rec:
        qi=int(r['query_index'])
        b,full=extract_query_features(r,d['T1'][qi],d,meta,old_model)
        result.append({'query_index':qi,'gids':r['gids'].tolist(),
                       'base':b.tolist(),'full':full.tolist(),
                       'y':r['true'].tolist(),
                       'positive_in_pool_AUDIT':int(r['positive_group_in_candidates_AUDIT']),
                       'top1_correct_AUDIT':int(r['top1_pid_correct_AUDIT'])})
    return result


def records_to_numpy(records,variant):
    key='base' if variant=='listwise_base' else 'full'
    expected=len(BASE_FEATURE_NAMES) if key=='base' else len(FULL_FEATURE_NAMES)
    out=[]
    for r in records:
        X=np.asarray(r[key],np.float32);y=np.asarray(r['y'],np.float32)
        if X.ndim!=2 or X.shape[1]!=expected or X.shape[0]!=len(y):
            raise RuntimeError('拟合缓存损坏: feature schema不符')
        out.append((X,y))
    return out


def fit_listwise(records,variant,epochs=150,lr=.028,reg=.025,seed=129):
    """Listwise query-group softmax: each query votes ONCE, hard rivals compete jointly.

    Only fit-identity PID supervision is used to construct y; no holdout identity label
    participates in fitting, thresholding, feature assembly, or candidate selection.
    """
    inp=records_to_numpy(records,variant)
    train=[(x,y) for x,y in inp if np.any(y>0) and np.any(y<=0)]
    if len(train)<5:raise RuntimeError(f'listwise有效训练查询太少: {len(train)}')
    nfeat=train[0][0].shape[1]
    concat=np.concatenate([x for x,_ in train],0).astype(np.float64)
    mu=concat.mean(0);sd=concat.std(0);sd=np.where(sd<1e-4,1.,sd)
    maxk=max(len(x) for x,_ in train);N=len(train)
    P=np.zeros((N,maxk,nfeat),np.float32);Y=np.zeros((N,maxk),np.float32)
    mask=np.zeros((N,maxk),bool)
    for j,(x,y) in enumerate(train):
        k=len(x);P[j,:k]=np.clip((x-mu)/sd,-6,6)
        mask[j,:k]=True;Y[j,:k]=y/max(1.,float(y.sum()))
    w=np.zeros(nfeat,np.float64);m=np.zeros_like(w);v=np.zeros_like(w)
    # Stable deterministic Adam, 128 queries per update. No PyTorch, no GPU required.
    rg=np.random.default_rng(seed);t=0
    for epoch in range(epochs):
        for ix in np.array_split(rg.permutation(N),max(1,int(np.ceil(N/128)))):
            z=np.einsum('nkd,d->nk',P[ix],w,optimize=True)
            z=np.where(mask[ix],z,-1e9)
            z-=np.max(z,axis=1,keepdims=True)
            ex=np.exp(z);prob=ex/np.maximum(ex.sum(1,keepdims=True),1e-30)
            diff=(prob-Y[ix])/len(ix)
            grad=np.einsum('nkd,nk->d',P[ix],diff,optimize=True)+reg*w
            if not np.isfinite(grad).all():raise RuntimeError('listwise拟合数值溢出')
            grad=np.clip(grad,-6,6)
            t+=1;m=.9*m+.1*grad;v=.999*v+.001*grad*grad
            w-=lr*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-7)
    return {'variant':variant,'weights':w.tolist(),'mean':mu.tolist(),'scale':sd.tolist(),
      'feature_names':list(BASE_FEATURE_NAMES if variant=='listwise_base' else FULL_FEATURE_NAMES),
      'n_fit_queries':N,'n_fit_candidate_rows':int(sum(len(x) for x,_ in train)),
      'epochs':epochs,'lr':lr,'l2':reg,'seed':seed,'query_level_listwise':True}


def choose_listwise(r,model):
    key='base' if model['variant']=='listwise_base' else 'full'
    arr=np.asarray(r[key],np.float64);d=len(model['feature_names'])
    if arr.ndim!=2 or arr.shape[1]!=d:raise RuntimeError('推理特征与拟合模型不符')
    z=(np.clip((arr-np.asarray(model['mean']))/np.asarray(model['scale']),-6,6)
       @ np.asarray(model['weights']))
    z=np.clip(z,-100,100);z-=np.max(z)
    p=np.exp(z);p/=np.maximum(np.sum(p),1e-12)
    winner=int(np.argmax(p))
    return int(r['gids'][winner]),float(p[winner]),winner,p


def fit_cutoff(decisions,score_name,target):
    vals=np.asarray([r[score_name] for r in decisions],np.float64)
    correct=np.asarray([r[score_name+'_correct_AUDIT'] for r in decisions],np.int32)
    best=None
    for th in np.unique(vals)[::-1]:
        keep=vals>=th-1e-12;n=int(keep.sum())
        if n>0 and float(correct[keep].mean())+1e-12>=target:
            if best is None or n>best['n_cv_accepted']:
                best={'threshold':float(th),'n_cv_accepted':n,
                      'cv_precision':float(correct[keep].mean()),
                      'cv_coverage':n/len(vals),'target_met':True}
    return best if best else {'threshold':2.0,'n_cv_accepted':0,'cv_precision':None,
                               'cv_coverage':0.,'target_met':False}


def save_gz_json21(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.exp21_',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as fo:
            with gzip.GzipFile(filename='',fileobj=fo,mode='wb',mtime=0) as z:
                z.write(json.dumps(obj,ensure_ascii=False,allow_nan=False).encode('utf-8'))
            fo.flush();os.fsync(fo.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def load_gz_json21(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)


def log21(out,msg):
    with (Path(out)/'experiment_run.log').open('a',encoding='utf-8') as f:
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+' '+str(msg)+'\n')
    print(msg,flush=True)


def prepare_fit_folds21(out,tx,vi,texts,images,fit_ids,tau,old_model,nfold=4):
    groups=[list(map(int,x)) for x in np.array_split(np.asarray(sorted(fit_ids),np.int64),nfold)]
    folds=[];diags=[]
    for fi,pids in enumerate(groups):
        path=out/f'fit_fold_{fi:02d}_query_sets.json.gz'
        expected_hash=stable_json({'pids':pids,'version':VERSION,'tau':tau,'features':list(FULL_FEATURE_NAMES),'source_hash':sha256_file(Path(__file__))})
        if path.is_file():
            payload=load_gz_json21(path)
            if payload.get('fingerprint')!=expected_hash or payload.get('fold')!=fi:
                raise RuntimeError('断点续跑折缓存与身份/视觉阈值/特征协议不一致')
            rec=payload['records'];stat=payload['audit']
            log21(out,f'续跑：已复用拟合折{fi+1}/{nfold}，查询{len(rec)}')
        else:
            d=build_abc_subset(tx,vi,texts,images,pids)
            gs,meta,lut,stat=dataset_fold(d,tau)
            rec=make_fit_records(d,meta,lut,old_model,training=True)
            stat={**stat,'fit_fold':fi,'n_PID':len(pids),'n_queries':len(rec),
                  'n_true_candidate_available':int(sum(r['positive_in_pool_AUDIT'] for r in rec))}
            save_gz_json21(path,{'version':VERSION,'fold':fi,
                                  'fingerprint':expected_hash,'records':rec,'audit':stat})
            log21(out,f'拟合折{fi+1}/{nfold}：{len(pids)} PID、{len(rec)}查询、{len(gs)}组')
        folds.append(rec);diags.append(stat)
    return folds,diags


def save_fold_model21(path,model,other_folds,variant):
    atomic_json(path,{'model':model,'version':VERSION,'training_folds':sorted(other_folds),
                       'variant':variant})


def load_fold_model21(path,other_folds,variant):
    j=json.loads(Path(path).read_text(encoding='utf-8'))
    if j.get('version')!=VERSION or j.get('training_folds')!=sorted(other_folds) or j.get('variant')!=variant:
        raise RuntimeError('跨折训练恢复的模型不符')
    return j['model']


def crossfit_models21(out,folds,epochs=150):
    cv=[]
    for fi,recs in enumerate(folds):
        modelset={}
        for variant in ('listwise_base','listwise_set'):
            file=out/f'cv_{variant}_leaveout_{fi:02d}.json'
            fold_ids=[k for k in range(len(folds)) if k!=fi]
            if file.is_file():model=load_fold_model21(file,fold_ids,variant)
            else:
                train=[r for k in fold_ids for r in folds[k]]
                model=fit_listwise(train,variant,epochs=epochs,seed=217+fi)
                save_fold_model21(file,model,fold_ids,variant)
            modelset[variant]=model
        for r in recs:
            row={'fold':fi,'query_index':int(r['query_index']),
                 'positive_in_pool_AUDIT':int(r['positive_in_pool_AUDIT'])}
            for name,model in modelset.items():
                gid,p,idx,_=choose_listwise(r,model)
                row[name]=p;row[name+'_correct_AUDIT']=int(bool(r['y'][idx]));
                row[name+'_group_id']=gid
            cv.append(row)
    cuts={}
    for name in ('listwise_base','listwise_set'):
        cuts[name]={str(int(100*p)):fit_cutoff(cv,name,p) for p in (.90,.95)}
    final={}
    for name in ('listwise_base','listwise_set'):
        f=out/(name+'_final_model.json')
        if f.is_file():
            model=json.loads(f.read_text(encoding='utf-8'))
            if model.get('variant')!=name or model.get('feature_names')!=(list(BASE_FEATURE_NAMES) if name=='listwise_base' else list(FULL_FEATURE_NAMES)):
                raise RuntimeError('断点恢复全量拟合模型不符')
        else:
            model=fit_listwise([r for subset in folds for r in subset],name,epochs=epochs,seed=991)
            atomic_json(f,model)
        final[name]=model
    return final,cuts,cv


def evaluate_holdout21(d,meta,all_old,old_model,models,cuts,prior=None):
    res=[];replay_max=0.
    for rec in all_old:
        qi=int(rec['query_index']);qpid=int(rec['query_pid_AUDIT'])
        gid0=int(rec['top1_group']); top1ok=int(rec['top1_pid_correct_AUDIT'])
        # Complete exact EXP19 old-query replay; optional for small synthetic self-test.
        gid19,prob19,_,_=best_group_choice(rec,old_model)
        if prior is not None:
            pr=prior[qi]
            if gid19!=int(pr['group_fit_all_source_group_id']):
                raise RuntimeError(f'实验19组ID未重建: query={qi}')
            delta=abs(prob19-float(pr['group_fit_all_score']))
            replay_max=max(replay_max,delta)
            if delta>1e-6 or top1ok!=int(pr['top1_B_image_correct_PID_AUDIT']):
                raise RuntimeError(f'实验19得分或Top1正确性未严格复现: query={qi}, delta={delta}')
        b,full=extract_query_features(rec,d['T1'][qi],d,meta,old_model)
        r={'base':b,'full':full,'gids':rec['gids'],'y':rec['true']}
        local={}
        for name,model in models.items():
            gid,conf,ix,p=choose_listwise(r,model)
            local[name]=(gid,conf,ix,p)
        correct19=int(rec['true'][np.flatnonzero(rec['gids']==gid19)[0]])
        row={'query_index':qi,'query_train_text_index':int(d['query_text_idx'][qi]),
             'query_pid_AUDIT_ONLY':qpid,'top1_correct_AUDIT':top1ok,
             'old19_correct_AUDIT':correct19,'old19_gid':gid19,'old19_score':prob19,
             'positive_group_in_pool_AUDIT':int(rec['positive_group_in_candidates_AUDIT']),
             'n_candidate_groups':len(rec['gids']),'true_support_outside_top16_AUDIT':int(rec['true_B_img_beyond16_AUDIT'])}
        for name,(gid,conf,ix,p) in local.items():
            images=list(map(int,meta[gid]['images']))
            captions=list(map(int,meta[gid]['texts']))
            good=int(bool(rec['true'][ix]))
            prior_ranks=np.argsort(-p,kind='stable')
            true_rank=min((int(np.flatnonzero(prior_ranks==j)[0])+1 for j,a in enumerate(rec['true']) if a),default=-1)
            row.update({name+'_gid':gid,name+'_confidence':conf,name+'_correct_AUDIT':good,
                 name+'_n_images':len(images),name+'_n_captions':len(captions),
                 name+'_true_group_best_rank_AUDIT':true_rank,
                 name+'_top5_group_ids':json.dumps([int(rec['gids'][j]) for j in prior_ranks[:5]]),
                 name+'_top5_confidences':json.dumps([round(float(p[j]),5) for j in prior_ranks[:5]]),
                 name+'_original_B_image_indices':json.dumps([int(d['B_image_idx'][j]) for j in images]),
                 name+'_original_B_text_indices':json.dumps([int(d['B_text_idx'][j]) for j in captions]),
                 name+'_top1_error_fixed_AUDIT':int(not top1ok and good),
                 name+'_top1_correct_broken_AUDIT':int(top1ok and not good),
                 name+'_gain_over_19_AUDIT':good-correct19})
            for target in ('90','95'):
                th=float(cuts[name][target]['threshold']);accept=int(conf>=th-1e-12)
                row[name+f'_accept_{target}']=accept
                row[name+f'_accepted_correct_{target}_AUDIT']=accept*good
                row[name+f'_fallback_top1_correct_{target}_AUDIT']=good if accept else top1ok
                row[name+f'_repair_{target}_AUDIT']=int(not top1ok and good and accept)
                row[name+f'_break_{target}_AUDIT']=int(top1ok and not good and accept)
        row['old19_top1_error_fixed_AUDIT']=int(not top1ok and correct19)
        row['old19_top1_correct_broken_AUDIT']=int(top1ok and not correct19)
        res.append(row)
    if prior is not None and len(res)!=len(prior):raise RuntimeError('实验19核对行数不同')
    return res,replay_max


def summarise21(rows,cuts):
    methods=[]
    for name in ('top1','old19','listwise_base','listwise_set'):
        correct=[int(r['top1_correct_AUDIT'] if name=='top1' else r['old19_correct_AUDIT'] if name=='old19' else r[name+'_correct_AUDIT']) for r in rows]
        repairs=sum(int((not r['top1_correct_AUDIT']) and a) for r,a in zip(rows,correct))
        harms=sum(int(r['top1_correct_AUDIT'] and not a) for r,a in zip(rows,correct))
        x={'method':name,'n_queries':len(rows),'n_correct_source':sum(correct),
           'source_accuracy':float(np.mean(correct)),'repairs_from_top1':repairs,
           'harms_to_top1':harms,'net_repairs':repairs-harms}
        if name.startswith('listwise'):
            x['chosen_multiview_count']=sum(int(r[name+'_n_images']>=2) for r in rows)
            x['chosen_correct_multiview_count']=sum(int(r[name+'_n_images']>=2 and r[name+'_correct_AUDIT']) for r in rows)
            for target in ('90','95'):
                accepted=[r for r in rows if r[name+'_accept_'+target]]
                x['accepted_'+target]=len(accepted)
                x['accepted_precision_'+target]=sum(r[name+'_accepted_correct_'+target+'_AUDIT'] for r in accepted)/len(accepted) if accepted else None
                x['fallback_accuracy_'+target]=float(np.mean([r[name+'_fallback_top1_correct_'+target+'_AUDIT'] for r in rows]))
                x['repairs_'+target]=sum(r[name+'_repair_'+target+'_AUDIT'] for r in rows)
                x['harms_'+target]=sum(r[name+'_break_'+target+'_AUDIT'] for r in rows)
        methods.append(x)
    return methods


def bootstrap21(rows,seed=20261023,n_iter=1600):
    rng=np.random.default_rng(seed);pid=sorted(set(r['query_pid_AUDIT_ONLY'] for r in rows))
    dic={p:np.asarray([i for i,r in enumerate(rows) if r['query_pid_AUDIT_ONLY']==p],int) for p in pid}
    arr={k:np.asarray([r[k] for r in rows],np.float32) for k in (
        'top1_correct_AUDIT','old19_correct_AUDIT','listwise_base_correct_AUDIT','listwise_set_correct_AUDIT')}
    samples={t:[] for t in ('set_minus_old19','set_minus_base','set_minus_top1')}
    for j in range(n_iter):
        ids=rng.integers(0,len(pid),len(pid))
        idx=np.concatenate([dic[pid[z]] for z in ids])
        a=arr['listwise_set_correct_AUDIT'][idx]
        for key,ref in (('set_minus_old19','old19_correct_AUDIT'),('set_minus_base','listwise_base_correct_AUDIT'),('set_minus_top1','top1_correct_AUDIT')):
            samples[key].append(float(np.mean(a-arr[ref])))
    return {key:{'observed_delta_pp':100.*float(np.mean(arr['listwise_set_correct_AUDIT']-arr[ref])),
            'pid_cluster_bootstrap_95ci_pp':[100.*float(x) for x in np.quantile(vals,[.025,.975])]}
       for (key,vals),ref in zip(samples.items(),('old19_correct_AUDIT','listwise_base_correct_AUDIT','top1_correct_AUDIT'))}


def choose_output21(parent,src12,src16,src19,settings,tau,resume):
    parent.mkdir(parents=True,exist_ok=True)
    fingerprint=stable_json({'version':VERSION,'source12':str(src12.resolve()),
      'source16':str(src16.resolve()),'source19':str(src19.resolve()),
      'settings_hash':settings['settings_hash'],'tau':tau,'groups':GROUP_KNN,
      'candidates':CANDIDATE_GROUPS,'features':list(FULL_FEATURE_NAMES),'source_hash':sha256_file(Path(__file__))})
    if resume:
        for p in sorted(parent.glob(EXP21+'*'),reverse=True):
            if (p/'run_info.json').is_file():
                try:run=json.loads((p/'run_info.json').read_text(encoding='utf-8'))
                except (OSError,ValueError):continue
                if run.get('fingerprint')==fingerprint:return p,True
        raise FileNotFoundError('没有与当前实验设置完全一致的实验21输出目录，无法--resume')
    p=parent/(EXP21+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    p.mkdir(parents=True,exist_ok=False)
    atomic_json(p/'run_info.json',{'version':VERSION,'fingerprint':fingerprint,
       'source_exp12':str(src12),'source_exp16':str(src16),'source_exp19':str(src19),
       'best_checkpoint_sha256':BEST_SHA,'no_holdout_label_selection':True,
       'source_texts_train_only':True,'val_test_gallery_captions_never_used':True})
    return p,False


def package21(out):
    files=['run_info.json','irra_exp21_listwise_group_set.py','fit_fold_diagnostics.csv',
           'crossfit_decisions.csv','fit_crossval_cutoffs.json',
           'listwise_base_final_model.json','listwise_set_final_model.json',
           'holdout_all_queries.csv','holdout_method_summary.csv',
           'holdout_graph_audit.json','bootstrap_pid.json','experiment21_report.json',
           'experiment_run.log','实验21_目标与边界.md']
    for n in files:
        if not (out/n).is_file():raise RuntimeError('打包前缺少必须结果文件: '+n)
    p=out/'发送给ChatGPT.zip';tmp=out/'.sending_tmp.zip'
    try:
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
            for name in files:z.write(out/name,arcname=name)
        with zipfile.ZipFile(tmp) as z:
            if z.testzip() is not None or set(z.namelist())!=set(files):raise RuntimeError('ZIP校验失败')
        os.replace(tmp,p)
    finally:
        if tmp.exists():tmp.unlink()
    return p



# ================= EXP22: Nonlinear set evidence vs cross-group attention =================
EXP21_REAL_VERSION='exp21_v1_listwise_group_set_competition'
SEEDS22=(17,29,43)
EPOCHS22=72
HIDDEN22=32
TRAIN_BATCH22=32
LOSS_PAIR22=.30


def find_exp21(parent,explicit=''):
    paths=[Path(explicit).expanduser().resolve()] if explicit else sorted(parent.glob(EXP21+'*'),reverse=True)
    for path in paths:
        if all((path/x).is_file() for x in ('run_info.json','experiment21_report.json','holdout_all_queries.csv','listwise_set_final_model.json')):
            return path
    raise FileNotFoundError('找不到实验21完整目录。请设置 --exp21-dir；结果ZIP本身不能替代实验12特征缓存。')


def audit_exp21(path,src12,src16,src19):
    run=json.loads((path/'run_info.json').read_text(encoding='utf-8'))
    rep=json.loads((path/'experiment21_report.json').read_text(encoding='utf-8'))
    if run.get('version')!=EXP21_REAL_VERSION or rep.get('best_checkpoint_sha256')!=BEST_SHA:
        raise RuntimeError('实验21版本/最佳权重不一致')
    for k,expected in (('source_exp12',src12),('source_exp16',src16),('source_exp19',src19)):
        if Path(run[k]).resolve()!=expected.resolve():raise RuntimeError('实验21来源与实验19/12/16不一致: '+k)
    models={}
    for name in ('listwise_base','listwise_set'):
        models[name]=json.loads((path/(name+'_final_model.json')).read_text(encoding='utf-8'))
    with (path/'holdout_all_queries.csv').open('r',encoding='utf-8-sig',newline='') as f:
        rows={int(z['query_index']):z for z in csv.DictReader(f)}
    if len(rows)!=542 or sorted(rows)!=list(range(542)):
        raise RuntimeError('实验21原始542条查询文件缺失或顺序异常')
    return rep,models,rows


class GroupReasoner22(nn.Module):
    """Same input evidence for both controls; only cross version mixes candidate nodes.

    The cross version applies multi-head attention to features of all candidate identity
    groups for THIS query; no group ID, identity ID, A/C image or labels are inputs.
    """
    def __init__(self,n_features,mode='independent',hidden=HIDDEN22):
        super().__init__()
        if mode not in ('independent','cross'):raise ValueError('unrecognized mode')
        self.mode=mode
        self.enc=nn.Sequential(nn.Linear(n_features,hidden),nn.GELU(),nn.LayerNorm(hidden),
                               nn.Linear(hidden,hidden),nn.GELU())
        if mode=='cross':
            self.attn=nn.MultiheadAttention(hidden,4,dropout=0.,batch_first=True)
            self.head=nn.Sequential(nn.Linear(hidden*3,hidden),nn.GELU(),nn.Linear(hidden,1))
        else:
            self.head=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,1))

    def forward(self,x,mask):
        h=self.enc(x)
        if self.mode=='cross':
            # All K groups compete by attention; padded nodes are forbidden as keys.
            ctx,_=self.attn(h,h,h,key_padding_mask=~mask,need_weights=False)
            out=self.head(torch.cat([h,ctx,h-ctx],dim=-1)).squeeze(-1)
        else:
            out=self.head(h).squeeze(-1)
        return out.masked_fill(~mask,-1e4)


def prep_padded22(records,mu=None,sd=None):
    """Keep unlabelled queries for evaluation, but only fit positive+negative queries.

    Zero-padded training inputs; masks ensure padding never becomes a candidate.
    """
    raw=[]
    for r in records:
        a=np.asarray(r['full'],np.float32)
        y=np.asarray(r['y'],np.float32)
        if a.ndim!=2 or a.shape[1]!=len(FULL_FEATURE_NAMES) or len(a)!=len(y) or len(a)==0:
            raise ValueError('candidate feature schema/length mismatch')
        raw.append((a,y))
    if mu is None:
        pool=np.concatenate([x for x,y in raw if y.sum()>0 and y.sum()<len(y)],0)
        mu=pool.mean(0).astype(np.float32);sd=pool.std(0).astype(np.float32)
        sd=np.maximum(sd,1e-3)
    mu=np.asarray(mu,np.float32);sd=np.asarray(sd,np.float32)
    n=len(raw);k=max(len(x) for x,y in raw);d=len(mu)
    X=np.zeros((n,k,d),np.float32);Y=np.zeros((n,k),np.float32);M=np.zeros((n,k),bool)
    for i,(x,y) in enumerate(raw):
        l=len(x);X[i,:l]=np.clip((x-mu)/sd,-6,6);Y[i,:l]=y;M[i,:l]=True
    return X,Y,M,mu,sd


def fit_nonlinear22(records,mode,seed,epochs=EPOCHS22,device='cpu'):
    """Query-wise listwise + hardest-rival ranking loss, no holdout labels in fit."""
    torch.manual_seed(int(seed));np.random.seed(int(seed))
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('指定CUDA却检测不到GPU')
    raw=[r for r in records if 0<sum(r['y'])<len(r['y'])]
    if len(raw)<5:raise RuntimeError('有效拟合查询过少')
    X,Y,M,mu,sd=prep_padded22(raw)
    net=GroupReasoner22(X.shape[-1],mode=mode).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=.0015,weight_decay=.025)
    rng=np.random.default_rng(seed)
    n=len(X);last=[]
    # Train with equal weight per QUERY, not per number of negative groups.
    for ep in range(epochs):
        order=rng.permutation(n);losses=[]
        net.train()
        for ix in np.array_split(order,max(1,int(np.ceil(n/TRAIN_BATCH22)))):
            xx=torch.from_numpy(X[ix]).to(device)
            yy=torch.from_numpy(Y[ix]).to(device).bool()
            mm=torch.from_numpy(M[ix]).to(device)
            score=net(xx,mm)
            correct=torch.logsumexp(score.masked_fill(~yy,-1e4),1)
            denom=torch.logsumexp(score,1)
            list_loss=(denom-correct).mean()
            # Unlike a uniform softmax, penalise the strongest WRONG competing group.
            best_good=score.masked_fill(~yy,-1e4).max(1).values
            best_bad=score.masked_fill(yy|~mm,-1e4).max(1).values
            pair_loss=F.softplus(.15+best_bad-best_good).mean()
            loss=list_loss+LOSS_PAIR22*pair_loss
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
            opt.step();losses.append(float(loss.detach().cpu()))
        last.append(float(np.mean(losses)))
        if not np.isfinite(last[-1]):raise RuntimeError('训练发生NaN/Inf')
    return {'variant':mode,'seed':int(seed),'epochs':int(epochs),'n_fit_queries':len(raw),
            'feature_names':list(FULL_FEATURE_NAMES),'mean':mu.tolist(),'scale':sd.tolist(),
            'model_state':{k:v.detach().cpu().numpy().tolist() for k,v in net.state_dict().items()},
            'last_loss':last[-1],'training_loss_first_last':[last[0],last[-1]],
            'loss_definition':'listwise + 0.30*softplus(.15+hardest_wrong-best_true)',
            'cross_group_attention':bool(mode=='cross')}


def restore_network22(model,device='cpu'):
    m=GroupReasoner22(len(model['feature_names']),model['variant'])
    state={k:torch.tensor(v,dtype=torch.float32) for k,v in model['model_state'].items()}
    m.load_state_dict(state,strict=True)
    m=m.to(device);m.eval();return m


def predict22(records,model,device='cpu',batch=32):
    X,Y,M,_,_=prep_padded22(records,model['mean'],model['scale'])
    net=restore_network22(model,device)
    res=[]
    with torch.no_grad():
        for i in range(0,len(X),batch):
            xx=torch.from_numpy(X[i:i+batch]).to(device)
            mask=torch.from_numpy(M[i:i+batch]).to(device)
            logits=net(xx,mask)
            p=torch.softmax(logits,dim=1).detach().cpu().numpy()
            for j,row in enumerate(p):
                valid=M[i+j];scores=row[valid]
                rank=np.argsort(-scores,kind='stable')
                choice=int(rank[0]);res.append({'local_index':choice,'confidence':float(scores[choice]),
                  'best_true_group_rank_AUDIT':int(min((np.flatnonzero(rank==ii)[0]+1 for ii,y in enumerate(Y[i+j,valid]) if y>0),default=-1)),
                  'top5_local_indices':[int(z) for z in rank[:5]]})
    return res


def verify_model22(model,seed,mode,epochs):
    if (model['seed']!=int(seed) or model['variant']!=mode or model['epochs']!=int(epochs)
        or model['feature_names']!=list(FULL_FEATURE_NAMES)):
        raise RuntimeError('恢复的模型不符合当前种子/架构/训练协议')


def fit_or_load22(path,records,mode,seed,epochs,device):
    if path.exists():
        model=json.loads(path.read_text(encoding='utf-8'))
        verify_model22(model,seed,mode,epochs);return model
    model=fit_nonlinear22(records,mode,seed,epochs,device)
    atomic_json(path,model)
    return model


def prepare_crossfit22(out,folds,epochs,seeds,device):
    cv=[]
    for seed in seeds:
        for fi,rec in enumerate(folds):
            train=[r for k,part in enumerate(folds) if k!=fi for r in part]
            chosen_models={}
            for mode in ('independent','cross'):
                m=fit_or_load22(out/f'cv_{mode}_seed{seed}_fold{fi}.json',train,mode,seed+fi*1000,epochs,device)
                chosen_models[mode]=m
            preds={mode:predict22(rec,m,device) for mode,m in chosen_models.items()}
            for i,r in enumerate(rec):
                row={'seed':seed,'fold':fi,'query_index':int(r['query_index'])}
                for mode in ('independent','cross'):
                    pr=preds[mode][i];row[mode+'_confidence']=pr['confidence']
                    row[mode+'_correct_AUDIT']=int(bool(r['y'][pr['local_index']]))
                cv.append(row)
        log21(out,f'交叉验证完成 seed={seed}，各折相互隔离')
    cutoff={}
    for seed in seeds:
        per=[x for x in cv if x['seed']==seed]
        cutoff[str(seed)]={}
        for mode in ('independent','cross'):
            v=np.asarray([r[mode+'_confidence'] for r in per],np.float64)
            c=np.asarray([r[mode+'_correct_AUDIT'] for r in per],np.int32)
            cutoff[str(seed)][mode]={}
            # Predefine precision targets on CV; no holdout tuning.
            for target in (.90,.95):
                ok=None
                for th in np.unique(v)[::-1]:
                    mask=v>=th-1e-12
                    if mask.sum()>=1 and float(c[mask].mean())+1e-12>=target:
                        if ok is None or int(mask.sum())>ok['n_cv_accepted']:
                            ok={'threshold':float(th),'n_cv_accepted':int(mask.sum()),
                                'cv_precision':float(c[mask].mean()),'n_cv_queries':len(c)}
                cutoff[str(seed)][mode][str(int(target*100))]=ok or {'threshold':2.,'n_cv_accepted':0,'cv_precision':None,'n_cv_queries':len(c)}
    return cv,cutoff


def evaluate22(d,meta,oldrec,oldmodel,old21models,old21rows,prior19,models,cuts,device,expected_queries=542):
    # First independently reproduce EXP19 and EXP21 decisions for EVERY query.
    features=[];hist=[];max_error19=0.
    for r in oldrec:
        qi=int(r['query_index']);gid,p,idx,_=best_group_choice(r,oldmodel)
        pr=prior19[qi]
        if gid!=int(pr['group_fit_all_source_group_id']) or abs(p-float(pr['group_fit_all_score']))>1e-6:
            raise RuntimeError('EXP19逐条分组与分数复现失败 query='+str(qi))
        max_error19=max(max_error19,abs(p-float(pr['group_fit_all_score'])))
        base,full=extract_query_features(r,d['T1'][qi],d,meta,oldmodel)
        current={'query_index':qi,'gids':r['gids'].tolist(),'base':base.tolist(),
                 'full':full.tolist(),'y':r['true'].tolist()}
        old21=old21rows[qi]
        for name,m in old21models.items():
            oldgid,oldp,oldidx,_=choose_listwise(current,m)
            if oldgid!=int(old21[name+'_gid']) or abs(oldp-float(old21[name+'_confidence']))>2e-6:
                raise RuntimeError(f'EXP21历史逐条复现不一致：{name} 查询{qi}')
        features.append(current)
        hist.append({'query_index':qi,'query_train_text_index':int(d['query_text_idx'][qi]),
          'query_pid_AUDIT_ONLY':int(r['query_pid_AUDIT']),
          'top1_correct_AUDIT':int(r['top1_pid_correct_AUDIT']),
          'old19_correct_AUDIT':int(old21['old19_correct_AUDIT']),
          'old21_set_correct_AUDIT':int(old21['listwise_set_correct_AUDIT']),
          'positive_group_in_pool_AUDIT':int(r['positive_group_in_candidates_AUDIT']),
          'n_candidate_groups':len(r['gids']),
          'true_B_img_beyond16_AUDIT':int(r['true_B_img_beyond16_AUDIT'])})
    if len(hist)!=expected_queries:raise RuntimeError('历史查询数发生改变')
    for seed,pairs in models.items():
        for mode,model in pairs.items():
            ps=predict22(features,model,device)
            assert len(ps)==len(hist)
            for i,(r,pred) in enumerate(zip(features,ps)):
                hi=hist[i];gi=int(pred['local_index']);gid=r['gids'][gi]
                good=int(bool(r['y'][gi]));prefix=f'{mode}_s{seed}'
                ims=meta[gid]['images'];txt=meta[gid]['texts'];th=cuts[str(seed)][mode]
                hi.update({prefix+'_gid':gid,prefix+'_confidence':pred['confidence'],
                           prefix+'_correct_AUDIT':good,
                           prefix+'_true_rank_AUDIT':pred['best_true_group_rank_AUDIT'],
                           prefix+'_n_images':len(ims),prefix+'_n_texts':len(txt),
                           prefix+'_source_B_image_indices':json.dumps([int(d['B_image_idx'][j]) for j in ims]),
                           prefix+'_source_B_text_indices':json.dumps([int(d['B_text_idx'][j]) for j in txt]),
                           prefix+'_top5_gids':json.dumps([r['gids'][j] for j in pred['top5_local_indices']])})
                for target in ('90','95'):
                    accepted=int(pred['confidence']>=float(th[target]['threshold'])-1e-12)
                    hi[prefix+'_accept_'+target]=accepted
                    hi[prefix+'_fallback_correct_'+target+'_AUDIT']=good if accepted else hi['top1_correct_AUDIT']
    return hist,max_error19


def summary22(rows,seeds):
    result=[]
    def add(name,values,n_im=None,n_txt=None,seed=None):
        good=np.asarray(values,np.int32);top=np.asarray([r['top1_correct_AUDIT'] for r in rows],np.int32)
        item={'method':name,'seed':seed,'queries':len(rows),'correct':int(good.sum()),
              'accuracy':float(good.mean()),'repairs':int(np.sum((top==0)&(good==1))),
              'harms':int(np.sum((top==1)&(good==0))),
              'net_repairs':int(good.sum()-top.sum())}
        if n_im is not None:
            item['n_multiview_selected']=int(sum(k>=2 for k in n_im))
            item['n_true_multiview_selected']=int(sum(k>=2 and c for k,c in zip(n_im,good)))
            item['n_multi_text_selected']=int(sum(k>=2 for k in n_txt))
        result.append(item)
    add('original_top1',[r['top1_correct_AUDIT'] for r in rows])
    add('exp19',[r['old19_correct_AUDIT'] for r in rows])
    add('exp21_set',[r['old21_set_correct_AUDIT'] for r in rows])
    for seed in seeds:
        for mode in ('independent','cross'):
            k=f'{mode}_s{seed}';add(mode,[r[k+'_correct_AUDIT'] for r in rows],
                                  [r[k+'_n_images'] for r in rows],
                                  [r[k+'_n_texts'] for r in rows],seed)
            for t in ('90','95'):
                keep=[r for r in rows if r[k+'_accept_'+t]]
                row={'method':mode+'_fallback'+t,'seed':seed,'queries':len(rows),
                   'accepted':len(keep),'coverage':len(keep)/len(rows),
                   'accepted_precision':(sum(r[k+'_correct_AUDIT'] for r in keep)/len(keep)) if keep else None,
                   'correct':sum(r[k+'_fallback_correct_'+t+'_AUDIT'] for r in rows)}
                row['accuracy']=row['correct']/len(rows)
                row['repairs']=sum(r[k+'_accept_'+t] and not r['top1_correct_AUDIT'] and r[k+'_correct_AUDIT'] for r in rows)
                row['harms']=sum(r[k+'_accept_'+t] and r['top1_correct_AUDIT'] and not r[k+'_correct_AUDIT'] for r in rows)
                row['net_repairs']=row['repairs']-row['harms'];result.append(row)
    return result


def bootstrap22(rows,seeds,reps=1400):
    pid=sorted({r['query_pid_AUDIT_ONLY'] for r in rows});by={p:[] for p in pid}
    for i,r in enumerate(rows):by[r['query_pid_AUDIT_ONLY']].append(i)
    rg=np.random.default_rng(2219);out={}
    for seed in seeds:
        arr=np.asarray([r[f'cross_s{seed}_correct_AUDIT'] for r in rows],np.float32)
        for name,ref in (('independent',[r[f'independent_s{seed}_correct_AUDIT'] for r in rows]),
                         ('exp19',[r['old19_correct_AUDIT'] for r in rows]),
                         ('exp21',[r['old21_set_correct_AUDIT'] for r in rows]),
                         ('top1',[r['top1_correct_AUDIT'] for r in rows])):
            d=arr-np.asarray(ref,np.float32);vals=[]
            for _ in range(reps):
                chosen=rg.choice(pid,size=len(pid),replace=True)
                idx=[i for p in chosen for i in by[p]]
                vals.append(100*float(d[idx].mean()))
            out[f'cross_s{seed}_minus_{name}']={'observed_pp':100*float(d.mean()),
                   'pid_cluster_bootstrap_95ci_pp':[float(x) for x in np.quantile(vals,[.025,.975])]}
    return out


def choose_output22(parent,src12,src16,src19,src21,settings,tau,resume,epochs,seeds):
    parent.mkdir(parents=True,exist_ok=True)
    fingerprint=stable_json({'version':VERSION,'source12':str(src12),'source16':str(src16),
      'source19':str(src19),'source21':str(src21),'settings_hash':settings['settings_hash'],
      'tau':tau,'source_sha':sha256_file(Path(__file__)),'epochs':epochs,'seeds':list(seeds),
      'model_hidden':HIDDEN22,'features':list(FULL_FEATURE_NAMES)})
    if resume:
        for out in sorted(parent.glob(EXP22+'*'),reverse=True):
            f=out/'run_info.json'
            if f.is_file() and json.loads(f.read_text(encoding='utf-8')).get('fingerprint')==fingerprint:return out,True
        raise FileNotFoundError('无匹配的实验22原文件夹，不能resume；原始脚本、epoch、seed、缓存来源均须一致')
    out=parent/(EXP22+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    atomic_json(out/'run_info.json',{'version':VERSION,'fingerprint':fingerprint,
       'source_exp12':str(src12),'source_exp16':str(src16),'source_exp19':str(src19),'source_exp21':str(src21),
       'best_checkpoint_sha256':BEST_SHA,'training_caption_only':True,'holdout_untouched':True,
       'training_epochs':epochs,'seeds':list(seeds)})
    return out,False


def package22(out):
    required=['run_info.json','irra_exp22_nonlinear_crossgroup_reasoning.py',
              'fit_fold_diagnostics.csv','fit_crossval_decisions.csv','fit_cv_gate_thresholds.json',
              'holdout_all_queries.csv','holdout_method_summary.csv','bootstrap_pid.json',
              'holdout_graph_audit.json','experiment22_report.json','experiment_run.log','实验22_目标与边界.md']
    for name in required:
        if not (out/name).is_file():raise RuntimeError('缺少结果打包文件: '+name)
    temp=out/'.zip_unfinished';dest=out/'发送给ChatGPT.zip'
    try:
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
            for name in required:z.write(out/name,name)
        with zipfile.ZipFile(temp) as z:
            if z.testzip() is not None or set(z.namelist())!=set(required):raise RuntimeError('ZIP检验失败')
        os.replace(temp,dest)
    finally:
        if temp.exists():temp.unlink()
    return dest


def execute22(args=None):
    parser=argparse.ArgumentParser(description='实验22：独立MLP vs 跨候选组注意力，多文本集合证据与真实错检修复')
    parser.add_argument('--project-root',default=str(ROOT))
    for n in ('exp12','exp16','exp19','exp21'):parser.add_argument(f'--{n}-dir',default='')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--epochs',type=int,default=EPOCHS22)
    parser.add_argument('--seeds',default=','.join(map(str,SEEDS22)))
    a=parser.parse_args(args)
    seeds=tuple(int(x) for x in a.seeds.split(','));epochs=int(a.epochs)
    if len(seeds)!=len(set(seeds)) or any(x<0 for x in seeds) or epochs<1 or not seeds:raise ValueError('seeds / epochs不合法')
    if a.self_test:self_test22();return
    root=Path(a.project_root).expanduser().resolve();parent=root/PARENT
    src12=find_exp12(parent,a.exp12_dir);settings,selected=source_audit(src12,full_hash=True)
    src16=find_exp16(parent,a.exp16_dir);tau,cal=audit_exp16(src16,src12,settings)
    src19=find_exp19_folder(parent,a.exp19_dir);oldmodel,oldrep,prior=audit_exp19(src19,src12,src16)
    src21=find_exp21(parent,a.exp21_dir);rep21,old21models,old21rows=audit_exp21(src21,src12,src16,src19)
    source=read_records(settings['annotation_file'],selected)
    tx,vi,vtx,vim=load_features(src12,settings,source)
    for vv,n in ((tx,'Train T'),(vi,'Train I'),(vtx,'Val T'),(vim,'Val I')):l2check(vv,n)
    vp=np.asarray([q['pid'] for q in source['val_queries']]);ip=np.asarray([q['pid'] for q in source['val_images']])
    if set(map(int,settings['selected_pids']))&set(map(int,vp)):raise RuntimeError('Train/Val PID有交集')
    valr1=100*float(np.mean(ip[rank_topk(vtx,vim,k=1)[:,0]]==vp))
    if abs(valr1-66.433907)>.03:raise RuntimeError('最佳IRRA Val基线无法复现，停止')
    out,resumed=choose_output22(parent,src12,src16,src19,src21,settings,tau,a.resume,epochs,seeds)
    shutil.copy2(Path(__file__),out/'irra_exp22_nonlinear_crossgroup_reasoning.py')
    if resumed and (out/'发送给ChatGPT.zip').is_file() and (out/'experiment22_report.json').is_file():
        with zipfile.ZipFile(out/'发送给ChatGPT.zip') as z:
            if z.testzip() is None:print('实验22已有完整结果：',out,flush=True);return
    log21(out,'实验22开始，冻结IRRA Val基线仅核验: %.6f'%valr1)
    folds,fit_stats=prepare_fit_folds21(out,tx,vi,source['train_texts'],source['train_images'],settings['fit_pids'],tau,oldmodel)
    # Cache fingerprint guards ensure partition, source and feature schema do not change on resume.
    crossfit,cuts=prepare_crossfit22(out,folds,epochs,seeds,a.device)
    atomic_csv(out/'fit_fold_diagnostics.csv',fit_stats,list(fit_stats[0]))
    atomic_csv(out/'fit_crossval_decisions.csv',crossfit,list(crossfit[0]))
    atomic_json(out/'fit_cv_gate_thresholds.json',cuts)
    fullfit=[r for f in folds for r in f];models={}
    for seed in seeds:
        models[seed]={}
        for mode in ('independent','cross'):
            file=out/f'final_{mode}_s{seed}.json'
            models[seed][mode]=fit_or_load22(file,fullfit,mode,seed,epochs,a.device)
        log21(out,'最终拟合两种模型完成 seed='+str(seed))
    d=build_abc_subset(tx,vi,source['train_texts'],source['train_images'],settings['holdout_pids'])
    gs,meta,lut,stats=dataset_fold(d,tau)
    audit=oldrep['holdout']['global_visual_graph']
    for key in ('n_images','n_groups','n_multiview_groups','n_pure_multiview_groups','n_impure_groups','n_proposed_pairs','n_correct_pairs'):
        if int(stats[key])!=int(audit[key]):raise RuntimeError('EXP19图谱审计不一致: '+key)
    if (len(d['T1']),len(d['TB']),len(d['VB']),len(gs))!=(542,1274,637,502):
        raise RuntimeError('历史holdout样本与实验19不一致')
    original,_=make_queries(d,meta,lut,training=False)
    rows,maxerr=evaluate22(d,meta,original,oldmodel,old21models,old21rows,prior,models,cuts,a.device)
    if (sum(r['top1_correct_AUDIT'] for r in rows)!=409
        or sum(r['old19_correct_AUDIT'] for r in rows)!=413
        or sum(r['old21_set_correct_AUDIT'] for r in rows)!=411):
        raise RuntimeError('EXP19/21历史结果未复现')
    methods=summary22(rows,seeds)
    boot=bootstrap22(rows,seeds)
    atomic_csv(out/'holdout_all_queries.csv',rows,list(rows[0]))
    keys=list(dict.fromkeys(x for r in methods for x in r))
    atomic_csv(out/'holdout_method_summary.csv',methods,keys)
    atomic_json(out/'holdout_graph_audit.json',stats)
    atomic_json(out/'bootstrap_pid.json',boot)
    report={'version':VERSION,'objective':'技术难点1：无PID识别查询身份多图多文本来源','best_irra_sha':BEST_SHA,
       'official_val_r1_audit_only':valr1,'internal_fit_pid':1760,'internal_holdout_pid':440,
       'n_eval_queries':len(rows),'n_eval_B_images':len(d['VB']),'n_eval_B_train_captions':len(d['TB']),
       'source_experiments':[str(z) for z in (src12,src16,src19,src21)],
       'historical_exp19_and_21_replayed_querywise':True,'max_exp19_replay_error':maxerr,
       'global_visual_graph':stats,'methods':methods,'bootstrap':boot,'crossfit_cutoffs':cuts,
       'epochs':epochs,'seeds':seeds,'feature_names':list(FULL_FEATURE_NAMES),
       'fit_losses':{str(seed):{mode:models[seed][mode]['training_loss_first_last'] for mode in ('independent','cross')} for seed in seeds},
       'constraints':['Train B captions only; never Val/Test gallery human captions','A/B/C images disjoint',
          'Candidate image groups and selection pool identical to EXP19/21','No PID in prediction features',
          'Train PID used for supervision, holdout PID only for audit','Holdout identities seen by original IRRA during pretraining',
          'This is source identity matching, not information filtering or fusion; official Validation R1 unchanged.',
          'CV cutoffs calibrated per seed on fit identities; not tuned on holdout.']}
    atomic_json(out/'experiment22_report.json',report)
    (out/'实验22_目标与边界.md').write_text(
       '# 实验22：非线性集合 vs 跨组关系竞争\n\n'
       '仅检验用户三难点的第一项：无PID为T1找到可信同身份多图、多描述集合。\n'
       '候选图谱完全复用实验19；对照原IRRA Top1、实验19、实验21 Listwise-Set、'
       '非线性独立集合网络、非线性跨候选组注意力网络。\n'
       '二种神经网络使用完全相同的组级集合特征、训练数据、训练损失和隐层规模；'
       '唯一结构变化是查询内跨组注意力。先在fit身份证据拟合，再评估完全独立的440个训练身份。\n'
       '完整性：严格复现原542条历史查询。所使用的人类描述只来自Train集合B，'
       '真实PID只用于训练标注与结果核验。C没有进入归属推理。\n'
       '研究终点是多个来源图像+多条描述的可信集合，不是选择单独T2；'
       '文本真实性筛选、集合信息融合、后续图像互补和图文联合属于后续独立实验。\n',encoding='utf-8')
    log21(out,'模型与报告导出完毕，开始ZIP完整性校验')
    print('请上传:',package22(out),flush=True)


def self_test22():
    """Actual forward/backprop, mask/permutation, CV, resume, export regression."""
    rng=np.random.default_rng(2211);dim=28
    images=[];txt=[];vv=[];tt=[]
    for pid in range(800,848):
        center=normalized(rng.normal(size=(1,dim)))[0]
        for im in range(4):
            im_idx=len(images);images.append({'pid':pid,'path':f'train/{pid}_{im}.jpg'})
            vv.append(normalized((center+.04*rng.normal(size=dim))[None,:])[0])
            for k in range(2):
                txt.append({'pid':pid,'img_idx':im_idx,'text':f'TRAIN{pid}_{im}_{k}'})
                tt.append(normalized((center+.06*rng.normal(size=dim))[None,:])[0])
    vv=np.asarray(vv,np.float32);tt=np.asarray(tt,np.float32)
    fitids=list(range(800,832));holdids=list(range(832,848))
    with tempfile.TemporaryDirectory(prefix='exp22_selftest_') as td:
        out=Path(td)
        d=build_abc_subset(tt,vv,txt,images,holdids)
        gs,meta,lut,stat=dataset_fold(d,.75)
        oldrec,train=make_queries(d,meta,lut,training=True)
        old=train_binary_linear(train,rounds=30)
        folds,diag=prepare_fit_folds21(out,tt,vv,txt,images,fitids,.75,old,nfold=4)
        seeds=(17,29);epochs=3
        cv,cuts=prepare_crossfit22(out,folds,epochs,seeds,'cpu')
        cv2,cuts2=prepare_crossfit22(out,folds,epochs,seeds,'cpu')
        assert cv==cv2 and cuts==cuts2
        models={z:{mode:fit_or_load22(out/f'final_{mode}_s{z}.json',[r for f in folds for r in f],mode,z,epochs,'cpu')
                   for mode in ('independent','cross')} for z in seeds}
        current=make_fit_records(d,meta,lut,old,training=False)
        # no PID input to inference, and unchanged predictions after restoring a saved model
        pred=predict22(current,models[17]['cross'])
        assert len(pred)==len(d['T1'])==32
        assert all(p['local_index']<len(r['gids']) for p,r in zip(pred,current))
        assert all(any(r['y']) for r in current)
        # Permuting candidates must permute scores but leave the selected real group unchanged.
        rec=current[0];perm=np.random.default_rng(4).permutation(len(rec['gids']))
        copy={**rec,'gids':[rec['gids'][i] for i in perm],
            'full':[rec['full'][i] for i in perm], 'y':[rec['y'][i] for i in perm]}
        pred2=predict22([copy],models[17]['cross'])[0]
        assert rec['gids'][pred[0]['local_index']]==copy['gids'][pred2['local_index']]
        # Padded-group invariance tested via batch difference.
        one=predict22([current[0]],models[17]['cross'])[0]
        assert abs(one['confidence']-pred[0]['confidence'])<1e-5
        assert one['local_index']==pred[0]['local_index']
        # Resumed cache must not change final model weights / metadata.
        reloaded=fit_or_load22(out/'final_cross_s17.json',current,'cross',17,epochs,'cpu')
        assert reloaded==models[17]['cross']
        oldhold,_=make_queries(d,meta,lut,training=False)
        auditrows={}
        for rec0 in oldhold:
            g,p,ix,_=best_group_choice(rec0,old)
            q=int(rec0['query_index']);auditrows[q]={'group_fit_all_source_group_id':g,
                        'group_fit_all_score':p}
        reference={}
        for rec0 in oldhold:
            q=int(rec0['query_index']);b,f=extract_query_features(rec0,d['T1'][q],d,meta,old)
            ref={'gids':rec0['gids'],'base':b,'full':f}
            reference[q]={}
            for nm in ('listwise_base','listwise_set'):
                # old21 on synthetic: lightweight fit 2 epochs, not part of experiment22 main training.
                pass
        old21={name:fit_listwise(current,name,epochs=2) for name in ('listwise_base','listwise_set')}
        for rec0 in oldhold:
            q=int(rec0['query_index']);r=current[q]
            reference[q]={}
            for nm,m in old21.items():
                gid,p,ix,_=choose_listwise(r,m)
                reference[q][nm+'_gid']=gid;reference[q][nm+'_confidence']=p
                reference[q]['old19_correct_AUDIT']=0
                reference[q]['listwise_set_correct_AUDIT']=0
        rows,maxerr=evaluate22(d,meta,oldhold,old,old21,reference,auditrows,models,cuts,'cpu',expected_queries=32)
        assert len(rows)==32 and maxerr<1e-6
        summ=summary22(rows,seeds);b=bootstrap22(rows,seeds,reps=30)
        atomic_json(out/'run_info.json',{'selftest':True})
        atomic_csv(out/'fit_fold_diagnostics.csv',diag,list(diag[0]))
        atomic_csv(out/'fit_crossval_decisions.csv',cv,list(cv[0]))
        atomic_json(out/'fit_cv_gate_thresholds.json',cuts)
        atomic_csv(out/'holdout_all_queries.csv',rows,list(rows[0]))
        atomic_csv(out/'holdout_method_summary.csv',summ,list(dict.fromkeys(k for r in summ for k in r)))
        atomic_json(out/'bootstrap_pid.json',b)
        atomic_json(out/'holdout_graph_audit.json',stat)
        atomic_json(out/'experiment22_report.json',{'selftest':True})
        (out/'experiment_run.log').write_text('self test',encoding='utf-8')
        (out/'实验22_目标与边界.md').write_text('self test',encoding='utf-8')
        shutil.copy2(Path(__file__),out/'irra_exp22_nonlinear_crossgroup_reasoning.py')
        package22(out)
        with zipfile.ZipFile(out/'发送给ChatGPT.zip') as z:
            assert z.testzip() is None
    print('SELF-TEST PASS: nonlinear training, same-group control, cross-attention, permutation, padding, 4fold CV, resume, source, CSV/ZIP',flush=True)


if __name__=='__main__':
    execute22()
