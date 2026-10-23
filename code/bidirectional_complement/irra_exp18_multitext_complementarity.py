# -*- coding: utf-8 -*-
"""实验18：同身份多条文本互补（1/2/4支持）与全图库无PID来源对比。

核心：T1 / naive Top16 / 无PID视觉成组 Top16 / 全图库视觉扩展 / 真实PID教师参考。
训练身份内部进行；T1查询来源A、可取得字幕来源B、评价图库C相互隔离。
B的caption是TRAIN注释，只在本诊断中可访问，不可把VAL/TEST图库描述当作推理时可得。
所有非参考方法不得使用PID、C图像、C标签；教师参考只在训练身份内部。
PID参考只用来估计潜在价值，不是可部署方法，更不是严格数学上界。
最佳IRRA权重与特征缓存均只读，不训练任何IRRA/学生/图像互补模型。
执行：先 --self-test，再无参数正式运行；已有本实验目录中断才使用 --resume。
依赖：仅numpy + Python标准库；绝不import其他实验Python脚本。
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
import argparse, csv, hashlib, json, shutil, tempfile, zipfile, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT=Path(r'/')
PARENT='图文双向互补实验'
EXP12='实验12_最佳IRRA同身份文本互补迁移_'
EXP16='实验16_Top16多图与合法文本来源_'
EXP18='实验18_多文本互补与全图库支持来源_'
VERSION='exp18_v1_multi_text_identity_blind_image_group_budgets124'
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
    fd,tmp=tempfile.mkstemp(prefix='.exp18_',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def atomic_csv(path,rows,fields):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.exp18_',suffix='.tmp',dir=path.parent)
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


def build_disjoint_abc(x,v,texts,images,fit_ids,hold_ids,seed=SEED):
    """Use known TRAIN PID only for reproducible data SPLIT & offline outcome.

    A=query text source; B=support image+its TRAIN captions; C=held-out
    evaluation image. No image enters two sets. All 440 holdout PIDs
    appear in evaluation C (if they have ≥1 image), including query-ineligible.
    """
    hold=set(map(int,hold_ids));fit=set(map(int,fit_ids))
    assert len(hold)==440 and len(fit)==1760 and not fit&hold
    bypid=defaultdict(list)
    for i,im in enumerate(images):
        if int(im['pid']) in hold:bypid[int(im['pid'])].append(i)
    if set(bypid)!=hold:raise RuntimeError('440个holdout身份的图像不完整')
    cand_query_image=[];cand_support_image=[];cand_eval_image=[]
    img_per_pid={}
    for pid in sorted(hold):
        ids=bypid[pid]
        rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(pid)&0xffffffff]))
        ids=np.asarray(ids,dtype=np.int64)[rng.permutation(len(ids))]
        img_per_pid[str(pid)]=int(len(ids))
        cand_eval_image.append(int(ids[0]))
        if len(ids)>=3:
            cand_query_image.append(int(ids[1]))
            cand_support_image.extend(map(int,ids[2:]))
        elif len(ids)==2:
            cand_support_image.append(int(ids[1]))
    A=set(cand_query_image);B=set(cand_support_image);C=set(cand_eval_image)
    if A&B or A&C or B&C:raise AssertionError('A B C sources overlap')
    if len(C)!=440 or not A or not B:raise RuntimeError('三分图像分割样本不足')
    # TEXT source A: every associated caption; TEXT source B: every caption
    query=np.asarray([i for i,t in enumerate(texts) if int(t['img_idx']) in A],dtype=np.int32)
    support_text=np.asarray([i for i,t in enumerate(texts) if int(t['img_idx']) in B],dtype=np.int32)
    bs=np.asarray(sorted(B),np.int32);cc=np.asarray(sorted(C),np.int32)
    if not len(query) or not len(support_text):raise RuntimeError('A/B支持描述数量不足')
    pos=np.asarray([int(t['img_idx']) for t in texts],np.int32)
    remap={int(idx):j for j,idx in enumerate(bs)}
    text_to_image=np.asarray([remap[int(pos[i])] for i in support_text],np.int32)
    if not all(texts[int(i)]['pid']==images[int(pos[i])]['pid'] for i in query):
        raise AssertionError('A描述的身份与图像不一致')
    if not all(texts[int(i)]['pid']==images[int(pos[i])]['pid'] for i in support_text):
        raise AssertionError('B描述身份与其来源图像不一致')
    return {'query_idx':query,'support_text_idx':support_text,'support_img_idx':bs,'eval_img_idx':cc,
            'text_to_B':text_to_image,'text_pos':pos,
            'T1':normalized(x[query]),'TB':normalized(x[support_text]),
            'VB':normalized(v[bs]),'VC':normalized(v[cc]),
            'query_pid':np.asarray([int(texts[int(i)]['pid']) for i in query],np.int64),
            'B_pid':np.asarray([int(images[int(i)]['pid']) for i in bs],np.int64),
            'C_pid':np.asarray([int(images[int(i)]['pid']) for i in cc],np.int64),
            'source_image_counts':img_per_pid,'n_query_pid':len(A),'n_holdout_pid':len(C)}



# -------------- 实验18：多文本互补核心（不读取训练身份真值作决策） --------------
METHODS=('naive_top16','naive_global','visual_top16','visual_anchor_expand','visual_global','pid_reference')


def rank_scores(q, VB, TB, t2b):
    """无PID：T1对支持图像及描述的冻结相似度，只用于候选发现。"""
    si=np.asarray(VB @ q,dtype=np.float32)
    st=np.asarray(TB @ q,dtype=np.float32)
    return si,st,np.asarray(.5*(st+si[t2b]),dtype=np.float32)


def image_anchor(qimage_scores, text_scores, t2b, top):
    """从Top16寻找最匹配查询的锚点图像；只知道图文分数、不知道身份。"""
    capbest=np.full(len(qimage_scores),-np.inf,dtype=np.float32)
    np.maximum.at(capbest,t2b,text_scores)
    score=.5*(qimage_scores+capbest)
    return int(sorted(map(int,top),key=lambda i:(-float(score[i]),i))[0])


def visual_source_group(anchor, universe, neigh, tau):
    """从锚点扩展，要求簇内任意图像两两视觉相似度都达阈值。

    universe 可为 Top16 图像或全B图库（均不读取PID）。
    """
    anchor=int(anchor)
    universe=np.asarray(universe,dtype=np.int32)
    if anchor not in universe:raise AssertionError('anchor not in available image pool')
    eligible=[int(i) for i in universe if int(i)!=anchor and float(neigh[anchor,int(i)])+1e-7>=tau]
    eligible.sort(key=lambda j:(-float(neigh[anchor,j]),j))
    cluster=[anchor]
    for j in eligible:
        if all(float(neigh[j,m])+1e-7>=tau for m in cluster):cluster.append(j)
    return cluster




def partition_all_visual_groups(neigh,tau):
    """基于B所有图像构建全图库不重叠视觉组，任何时候不读PID和C。

    固定图像序号的确定性 complete-link 贪心聚类；不保证每组都对应真实身份。
    """
    n=len(neigh);remaining=set(range(n));groups=[]
    for seed in range(n):
        if seed not in remaining:continue
        remaining.remove(seed);group=[seed]
        cand=[j for j in remaining if float(neigh[seed,j])+1e-7>=tau]
        cand.sort(key=lambda j:(-float(neigh[seed,j]),j))
        for j in cand:
            if j in remaining and all(float(neigh[j,m])+1e-7>=tau for m in group):
                group.append(j);remaining.remove(j)
        groups.append(group)
    if sum(len(g) for g in groups)!=n or len({i for g in groups for i in g})!=n:
        raise RuntimeError('全图库视觉聚类未覆盖所有B图像')
    return groups


def choose_global_group(si,st,t2b,groups):
    """在全图库全部候选组中选择，不要求任何成员位于Top-16。

    组间使用至多两个图像分数和两个caption分数的平均（弱共识），
    不使用真PID、官方Val、评价图C；仅为可检验的固定启发式。
    """
    capbest=np.full(len(si),-np.inf,dtype=np.float32)
    np.maximum.at(capbest,t2b,st)
    best=None;key_best=None
    for gid,g in enumerate(groups):
        img_scores=sorted((float(si[i]) for i in g),reverse=True)[:2]
        txt_scores=sorted((float(capbest[i]) for i in g),reverse=True)[:2]
        val=.5*np.mean(img_scores)+.5*np.mean(txt_scores)
        key=(-float(val),gid)
        if key_best is None or key<key_best:key_best=key;best=gid
    return int(best)

def greedy_caption_set(candidates,k,q,TB,si,t2b,base_relevance,prefer_distinct_images=True):
    """逐条抽取有一定相关性且不重复的文本，优先来自不同图像。

    这里的 diverse 是向量新颖性代理，不宣称语义准确性。没有PID/评价C输入。
    """
    avail=sorted(set(int(i) for i in candidates))
    if not avail:return []
    selected=[];used_images=set()
    while avail and len(selected)<k:
        eligible=[i for i in avail if int(t2b[i]) not in used_images] if prefer_distinct_images else avail
        if not eligible:eligible=avail
        best=None;best_key=None
        for i in eligible:
            rel=float(base_relevance[i])
            cos_t1=float(TB[i]@q)
            if selected:
                novelty=1.-max([cos_t1]+[float(TB[i]@TB[j]) for j in selected])
            else:novelty=1.-cos_t1
            # Diversification is deliberately modest; reliability remains a serious risk.
            utility=rel + .12*novelty
            key=(-utility,int(i))
            if best_key is None or key<best_key:best_key=key;best=i
        selected.append(int(best));avail.remove(best);used_images.add(int(t2b[best]))
    return selected


def metric(scores,true_pid,gallery_pid):
    order=np.argsort(-np.asarray(scores),kind='stable')
    hit=np.asarray(gallery_pid,dtype=np.int64)[order]==int(true_pid)
    ranks=np.flatnonzero(hit)+1
    if len(ranks)==0:raise RuntimeError('Eval C 缺少正确身份')
    ap=float(np.mean(np.arange(1,len(ranks)+1)/ranks))
    return {'rank':int(ranks[0]),'r1':int(ranks[0]<=1),'r5':int(ranks[0]<=5),
            'r10':int(ranks[0]<=10),'ap':ap,'inp':float(len(ranks)/ranks[-1]),
            'n_pos':int(len(ranks))}


def fuse_text_set(t1,T,ids,beta):
    """固定总文本支持权重 beta，不随支持条数增加，控制融合强度混杂。"""
    if not ids:return t1
    tmean=normalized(np.mean(np.asarray(T[np.asarray(ids,dtype=np.int32)]),axis=0)[None,:])[0]
    return normalized(((1.-beta)*t1+beta*tmean)[None,:])[0]


def make_candidates(q,d,neigh,tau,global_groups):
    si,st,rel=rank_scores(q,d['VB'],d['TB'],d['text_to_B'])
    ordered=np.argsort(-si,kind='stable')
    top=ordered[:min(TOPK,len(ordered))]
    anchor=image_anchor(si,st,d['text_to_B'],top)
    loc_group=visual_source_group(anchor,top,neigh,tau)
    anchored_expand=visual_source_group(anchor,np.arange(len(d['VB'])),neigh,tau)
    group_idx=choose_global_group(si,st,d['text_to_B'],global_groups)
    glob_group=global_groups[group_idx]
    to_B=d['text_to_B']
    opts={
        'naive_top16':np.flatnonzero(np.isin(to_B,top)).tolist(),
        'naive_global':list(range(len(d['TB']))),
        'visual_top16':np.flatnonzero(np.isin(to_B,loc_group)).tolist(),
        'visual_anchor_expand':np.flatnonzero(np.isin(to_B,anchored_expand)).tolist(),
        'visual_global':np.flatnonzero(np.isin(to_B,glob_group)).tolist(),
    }
    # True PID reference built separately AFTER noPID candidates; never used for real prediction.
    return {'opts':opts,'ordered':ordered,'top':top,'anchor':anchor,
            'local_group':loc_group,'anchored_expand':anchored_expand,'global_group':glob_group,
            'global_group_index':group_idx,'si':si,'st':st,'rel':rel}


def process_query18(k,d,images,texts,neigh,tau,beta,global_groups):
    q=d['T1'][k];pid=int(d['query_pid'][k]);aidx=int(d['query_idx'][k])
    candidate=make_candidates(q,d,neigh,tau,global_groups)
    opts=candidate['opts']
    # NOTE: offline PID reference isolated; it cannot influence any of four non-oracle modes.
    opts['pid_reference']=np.flatnonzero(d['B_pid'][d['text_to_B']]==pid).tolist()
    if not opts['pid_reference']:raise RuntimeError('该查询身份没有独立支持B')
    q_scores=np.asarray(d['VC'] @ q,dtype=np.float32)
    base=metric(q_scores,pid,d['C_pid'])
    member_ap_cache={};member_r1_cache={}
    row={'query_index':int(k),'pid_OFFLINE_ONLY':pid,
         'T1_caption_index':aidx,'T1_text':texts[aidx]['text'],
         'A_image_path':images[int(d['text_pos'][aidx])]['path'],
         'T1_rank':base['rank'],'T1_r1':base['r1'],'T1_r5':base['r5'],
         'T1_r10':base['r10'],'T1_ap':base['ap'],'T1_inp':base['inp'],
         'anchor_support_image_rank':int(np.flatnonzero(candidate['ordered']==candidate['anchor'])[0]+1),
         'visual_top16_group_images':int(len(candidate['local_group'])),
         'visual_anchor_expand_group_images':int(len(candidate['anchored_expand'])),
         'visual_global_group_images':int(len(candidate['global_group'])),
         'visual_anchor_expand_added_images_after16':int(len(set(candidate['anchored_expand'])-set(candidate['top']))),
         'visual_global_added_images_after16':int(len(set(candidate['global_group'])-set(candidate['top']))),
         'visual_global_group_has_zero_top16_seeds':int(not bool(set(candidate['global_group'])&set(candidate['top']))),
         'has_true_identity_in_B_outside16_OFFLINE_ONLY':int(np.any(d['B_pid'][candidate['ordered'][16:]]==pid)),
         'T1_top16_true_identity_covered_OFFLINE_ONLY':int(np.any(d['B_pid'][candidate['top']]==pid))}
    for method in METHODS:
        allowed=opts[method]
        for budget in BUDGETS:
            label=f'{method}_K{budget}'
            selected=greedy_caption_set(allowed,budget,q,d['TB'],candidate['si'],
                                        d['text_to_B'],candidate['rel'])
            fused=fuse_text_set(q,d['TB'],selected,beta)
            m=metric(d['VC']@fused,pid,d['C_pid'])
            # Posthoc per-member ranking: NOT used to pick supports.
            for member in selected:
                if member not in member_ap_cache:
                    single=fuse_text_set(q,d['TB'],[member],beta)
                    check=metric(d['VC']@single,pid,d['C_pid'])
                    member_ap_cache[member]=check['ap']
                    member_r1_cache[member]=check['r1']
            best_member_ap=max(member_ap_cache[member] for member in selected) if selected else base['ap']
            best_member_r1=max(member_r1_cache[member] for member in selected) if selected else base['r1']
            bs=[int(d['text_to_B'][i]) for i in selected]
            is_correct=[int(d['B_pid'][i]==pid) for i in bs]
            img_ranks=[int(np.flatnonzero(candidate['ordered']==i)[0]+1) for i in bs]
            txids=[int(d['support_text_idx'][i]) for i in selected]
            paths=[images[int(d['support_img_idx'][i])]['path'] for i in bs]
            txs=[texts[i]['text'] for i in txids]
            row.update({
                label+'_rank':m['rank'],label+'_r1':m['r1'],label+'_r5':m['r5'],
                label+'_r10':m['r10'],label+'_ap':m['ap'],label+'_inp':m['inp'],
                label+'_delta_ap':m['ap']-base['ap'],label+'_delta_r1':m['r1']-base['r1'],
                label+'_n_captions':len(selected),label+'_n_source_images':len(set(bs)),
                label+'_best_member_AP_posthoc_ONLY':best_member_ap,
                label+'_multi_better_than_best_member_AP_OFFLINE_ONLY':int(len(selected)>=2 and m['ap']>best_member_ap+1e-9),
                label+'_multi_vs_best_member_AP_delta_OFFLINE_ONLY':float(m['ap']-best_member_ap),
                label+'_multi_better_than_all_member_R1_OFFLINE_ONLY':int(len(selected)>=2 and m['r1']>best_member_r1),
                label+'_n_true_identity_captions_OFFLINE_ONLY':sum(is_correct),
                label+'_n_wrong_identity_captions_OFFLINE_ONLY':len(is_correct)-sum(is_correct),
                label+'_all_correct_PID_OFFLINE_ONLY':int(all(is_correct)),
                label+'_some_wrong_PID_OFFLINE_ONLY':int(not all(is_correct)),
                label+'_n_from_beyond_Top16':int(sum(r>16 for r in img_ranks)),
                label+'_image_ranks':json.dumps(img_ranks),
                label+'_T2_original_train_text_indices':json.dumps(txids),
                label+'_T2_source_paths':json.dumps(paths,ensure_ascii=False),
                label+'_T2_captions':json.dumps(txs,ensure_ascii=False),
            })
    return row


def method_summary(rows):
    ans=[]
    pids=np.asarray([int(r['pid_OFFLINE_ONLY']) for r in rows],np.int64)
    for method in ('T1',)+METHODS:
        budgets=(0,) if method=='T1' else BUDGETS
        for k in budgets:
            label='T1' if method=='T1' else f'{method}_K{k}'
            ranks=np.asarray([int(r[label+'_rank']) for r in rows])
            ap=np.asarray([float(r[label+'_ap']) for r in rows])
            inp=np.asarray([float(r[label+'_inp']) for r in rows])
            t1_r1=np.asarray([int(r['T1_r1']) for r in rows])
            t1_ap=np.asarray([float(r['T1_ap']) for r in rows])
            s={'method':method,'budget':k,'n_queries':len(rows),'n_eval_pids':len(set(pids)),
               'R1_percent':100*float(np.mean(ranks==1)),
               'R5_percent':100*float(np.mean(ranks<=5)),
               'R10_percent':100*float(np.mean(ranks<=10)),
               'mAP_percent':100*float(np.mean(ap)),
               'mINP_percent':100*float(np.mean(inp)),
               'vs_T1_mAP_pp':100*float(np.mean(ap-t1_ap)),
               'vs_T1_R1_pp':100*float(np.mean((ranks==1).astype(int)-t1_r1)),
               'AP_improved_n':int(sum(a>b+1e-9 for a,b in zip(ap,t1_ap))),
               'AP_harmed_n':int(sum(a<b-1e-9 for a,b in zip(ap,t1_ap))),
               'R1_fixed_n':int(sum(r['T1_r1']==0 and r[label+'_r1']==1 for r in rows)),
               'R1_broken_n':int(sum(r['T1_r1']==1 and r[label+'_r1']==0 for r in rows)),
               'warning':'TRAIN PID diagnostic, not official Validation'}
            if k:
                s.update({
                  'mean_n_support_captions':float(np.mean([r[label+'_n_captions'] for r in rows])),
                  'n_exact_budget':int(sum(r[label+'_n_captions']==k for r in rows)),
                  'mean_distinct_support_images':float(np.mean([r[label+'_n_source_images'] for r in rows])),
                  'mean_offtop16_captions':float(np.mean([r[label+'_n_from_beyond_Top16'] for r in rows])),
                  'queries_any_wrongPID_support_OFFLINE_ONLY':int(sum(r[label+'_some_wrong_PID_OFFLINE_ONLY'] for r in rows)),
                  'queries_all_correct_support_OFFLINE_ONLY':int(sum(r[label+'_all_correct_PID_OFFLINE_ONLY'] for r in rows)),
                  'queries_multi_outperforms_best_single_AP_OFFLINE_ONLY':int(sum(r[label+'_multi_better_than_best_member_AP_OFFLINE_ONLY'] for r in rows)),
                  'mean_multi_minus_best_member_AP_pp_OFFLINE_ONLY':float(100*np.mean([r[label+'_multi_vs_best_member_AP_delta_OFFLINE_ONLY'] for r in rows])),
                })
            ans.append(s)
    return ans


def experiment18_diagnostics(rows):
    n=len(rows);res=[]
    for method in METHODS:
        for k in BUDGETS:
            lb=f'{method}_K{k}'
            benefit=[r for r in rows if float(r[lb+'_delta_ap'])>=.01]
            harmful=[r for r in rows if float(r[lb+'_delta_ap'])<=-.01]
            notfull=[r for r in rows if int(r[lb+'_n_captions'])<k]
            mixed=[r for r in rows if int(r[lb+'_some_wrong_PID_OFFLINE_ONLY'])]
            clean=[r for r in rows if not int(r[lb+'_some_wrong_PID_OFFLINE_ONLY'])]
            extra=[r for r in rows if int(r[lb+'_n_from_beyond_Top16'])]
            res.append({'method':method,'budget':k,'n':n,
                'benefit_delta_AP_ge_001':len(benefit),'harm_delta_AP_le_minus001':len(harmful),
                'uncertain_delta_AP_between':n-len(benefit)-len(harmful),
                'n_with_fewer_support_than_budget':len(notfull),
                'n_with_wrongPID_in_support_OFFLINE_ONLY':len(mixed),
                'n_with_selected_T2_from_beyond_Top16':len(extra),
                'n_using_two_or_more_source_images':int(sum(r[lb+'_n_source_images']>=2 for r in rows)),
                'n_multi_AP_better_than_every_single_member_OFFLINE_ONLY':int(sum(r[lb+'_multi_better_than_best_member_AP_OFFLINE_ONLY'] for r in rows)),
                'mean_AP_delta_wrongPID_support':float(np.mean([r[lb+'_delta_ap'] for r in mixed])) if mixed else None,
                'mean_AP_delta_allcorrectPID_support':float(np.mean([r[lb+'_delta_ap'] for r in clean])) if clean else None,
                'note':'Delta AP is retrieval utility proxy, not verified semantic novelty'})
    return res


def pid_cluster_bootstrap(rows,reps=2000):
    ids=np.asarray([r['pid_OFFLINE_ONLY'] for r in rows],dtype=np.int64)
    uniq,inv=np.unique(ids,return_inverse=True)
    cnt=np.bincount(inv,minlength=len(uniq))
    rng=np.random.default_rng(180001);n=len(uniq)
    sampled=rng.integers(0,n,size=(reps,n),dtype=np.int32)
    ans={}
    contrasts=[]
    for method in METHODS:
        contrasts.append((f'{method}_K2','T1'))
        contrasts.append((f'{method}_K4',f'{method}_K1'))
        contrasts.append((f'{method}_K4',f'{method}_K2'))
    for k in BUDGETS:
        contrasts.extend([(f'visual_global_K{k}',f'visual_top16_K{k}'),
                          (f'visual_global_K{k}',f'naive_global_K{k}'),
                          (f'visual_global_K{k}',f'pid_reference_K{k}')])
    for a,b in contrasts:
        for metric in ('ap','r1'):
            delta=np.asarray([float(r[a+'_'+metric])-float(r[b+'_'+metric]) for r in rows],np.float64)
            bypid=np.bincount(inv,weights=delta,minlength=n)
            boots=bypid[sampled].sum(1)/cnt[sampled].sum(1)
            ans[f'{a}__minus__{b}__{metric}_pp']={
                'observed_pp':float(delta.mean()*100),
                'exploratory_CI95_pp':(100*np.quantile(boots,[.025,.975])).tolist(),
                'n_PID':int(n),'n_queries':len(rows)}
    return ans


def output_folder18(parent,src,ex16,settings,resume):
    parent.mkdir(parents=True,exist_ok=True)
    ident={'version':VERSION,'source_exp12':str(src.resolve()),'source_exp16':str(ex16.resolve()),
           'source_settings_hash':settings['settings_hash'],'best_sha256':BEST_SHA,
           'budgets':list(BUDGETS),'seed':SEED,'abc_disjoint':True,
           'group_threshold_from_exp16':True,'beta_source':'exp12'}
    identity_sha=stable_json(ident)
    if resume:
        for p in sorted(parent.glob(EXP18+'*'),reverse=True):
            m=p/'run_info.json'
            if not m.is_file():continue
            try:
                obj=json.loads(m.read_text(encoding='utf-8'))
                if obj.get('identity_sha')==identity_sha:return p,True
            except (OSError,ValueError):pass
        raise RuntimeError('未找到同设置的实验18目录，首次运行请勿使用--resume')
    name=EXP18+datetime.now().strftime('%Y%m%d_%H%M%S')
    out=parent/name;i=2
    while out.exists():out=parent/(name+'_'+str(i));i+=1
    out.mkdir(parents=True)
    atomic_json(out/'run_info.json',{'identity_sha':identity_sha,'ident':ident,
                                     'created_at':datetime.now().isoformat()})
    return out,False


def atomic_jsonl(path,rows):
    path=Path(path);fd,tmp=tempfile.mkstemp(prefix='.exp18_',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def read_chunk(path,start,end):
    rows=[json.loads(line) for line in Path(path).read_text('utf-8').splitlines() if line.strip()]
    if len(rows)!=end-start or [int(r['query_index']) for r in rows]!=list(range(start,end)):
        raise RuntimeError(f'分块续跑文件损坏：{path}')
    for r in rows:
        for m in METHODS:
            for k in BUDGETS:
                lb=f'{m}_K{k}'
                if any(lb+'_'+s not in r for s in ('rank','ap','n_captions','n_source_images')):
                    raise RuntimeError(f'分块字段缺失：{path} {lb}')
    return rows


def pack_zip18(out):
    required=['run_info.json','experiment18_report.json','method_summary.csv',
       'support_utility_diagnostics.csv','multi_text_budget_effects.csv',
       'all_queries.csv','bootstrap_pid.json','实验18_目标与边界.md',Path(__file__).name]
    fd,tmp=tempfile.mkstemp(prefix='.exp18_',suffix='.zip',dir=out);os.close(fd)
    dest=out/'发送给ChatGPT.zip'
    try:
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for nm in required:
                p=out/nm
                if not p.is_file():raise FileNotFoundError('输出文件缺失: '+str(p))
                z.write(p,arcname=nm)
        with zipfile.ZipFile(tmp) as z:
            if z.testzip() is not None or len(z.namelist())!=len(required):
                raise IOError('ZIP测试失败')
        os.replace(tmp,dest)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return dest


def run_core18(out,settings,records,train_x,train_v,threshold,beta):
    fit=set(map(int,settings['fit_pids']));hold=set(map(int,settings['holdout_pids']))
    d=build_disjoint_abc(train_x,train_v,records['train_texts'],records['train_images'],fit,hold)
    for v,name in ((d['T1'],'A text'),(d['TB'],'B captions'),(d['VB'],'B images'),(d['VC'],'C images')):
        l2check(v,name)
    neighbors=np.asarray(d['VB']@d['VB'].T,dtype=np.float32)
    global_groups=partition_all_visual_groups(neighbors,threshold)
    total=len(d['T1']);rows=[];chunk_size=32
    for st in range(0,total,chunk_size):
        ed=min(total,st+chunk_size);piece=out/f'cases_{st:05d}_{ed:05d}.jsonl'
        if piece.is_file():
            chunk=read_chunk(piece,st,ed)
            print(f' RESUME {st}:{ed}',flush=True)
        else:
            chunk=[process_query18(i,d,records['train_images'],records['train_texts'],neighbors,threshold,beta,global_groups)
                   for i in range(st,ed)]
            atomic_jsonl(piece,chunk)
            print(f' processed {ed}/{total}',flush=True)
        rows.extend(chunk)
    if [r['query_index'] for r in rows]!=list(range(total)):
        raise RuntimeError('断点续跑行数/顺序有误')
    summary=method_summary(rows)
    detail=experiment18_diagnostics(rows)
    atomic_csv(out/'all_queries.csv',rows,list(rows[0]))
    atomic_csv(out/'method_summary.csv',summary,sorted(set().union(*(r.keys() for r in summary))))
    atomic_csv(out/'support_utility_diagnostics.csv',detail,sorted(set().union(*(r.keys() for r in detail))))
    effects=[]
    for method in METHODS:
        for ka,kb in ((2,1),(4,1),(4,2)):
            a=f'{method}_K{ka}';b=f'{method}_K{kb}'
            effects.append({'method':method,'more_captions_budget':ka,'fewer_captions_budget':kb,
                'n_queries':total,'n_more_budget_actually_adds_text':int(sum(r[a+'_n_captions']>r[b+'_n_captions'] for r in rows)),
                'n_more_budget_actually_adds_image':int(sum(r[a+'_n_source_images']>r[b+'_n_source_images'] for r in rows)),
                'delta_mAP_pp':float(np.mean([r[a+'_ap']-r[b+'_ap'] for r in rows])*100),
                'delta_R1_pp':float(np.mean([r[a+'_r1']-r[b+'_r1'] for r in rows])*100),
                'n_AP_improved_by_extra':int(sum(r[a+'_ap']>r[b+'_ap']+1e-9 for r in rows)),
                'n_AP_harmed_by_extra':int(sum(r[a+'_ap']<r[b+'_ap']-1e-9 for r in rows))})
    atomic_csv(out/'multi_text_budget_effects.csv',effects,list(effects[0]))
    boot=pid_cluster_bootstrap(rows,1500)
    atomic_json(out/'bootstrap_pid.json',boot)
    stats={'n_queries':total,'n_eval_images':len(d['VC']),'n_eval_PID':len(set(map(int,d['C_pid']))),
      'n_A_query_images':d['n_query_pid'],'n_B_support_images':len(d['VB']),
      'n_B_train_captions':len(d['TB']),
      'n_queries_trueB_outside_top16':sum(r['has_true_identity_in_B_outside16_OFFLINE_ONLY'] for r in rows),
      'n_queries_visual_global_discovered_B_outside_top16':sum(r['visual_global_added_images_after16']>0 for r in rows),
      'n_queries_global_group_has_no_member_in_Top16':sum(r['visual_global_group_has_zero_top16_seeds'] for r in rows),
      'n_global_visual_groups':len(global_groups),
      'n_queries_eligible_K4_real_support_visual_global':sum(r['visual_global_K4_n_captions']>=4 for r in rows)}
    return {'methods':summary,'support_utility_diagnostics':detail,
            'multi_text_budget_effects':effects,'data_stats':stats,'bootstrap':boot}


def main():
    ap=argparse.ArgumentParser(description='实验18：同身份多条文本互补，Top16辅助、全图库视觉扩展、身份已知教师参考')
    ap.add_argument('--project-root',default=str(ROOT));ap.add_argument('--exp12-dir',default='')
    ap.add_argument('--exp16-dir',default='');ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    if args.self_test:self_test();return
    root=Path(args.project_root).expanduser().resolve();parent=root/PARENT
    src=find_exp12(parent,args.exp12_dir)
    settings,selected=source_audit(src,full_hash=True)
    ex16=find_exp16(parent,args.exp16_dir)
    threshold,cal=audit_exp16(ex16,src,settings)
    beta=float(settings['teacher_beta'])
    if not 0<beta<1:raise RuntimeError('融合beta超出范围')
    rec=read_records(settings['annotation_file'],selected)
    tx,vi,vtx,vim=load_features(src,settings,rec)
    for x,n in ((tx,'train text'),(vi,'train image'),(vtx,'Val query'),(vim,'Val image')):l2check(x,n)
    val_image_pid=np.asarray([a['pid'] for a in rec['val_images']],np.int64)
    val_query_pid=np.asarray([a['pid'] for a in rec['val_queries']],np.int64)
    if set(val_image_pid)&set(settings['fit_pids']) or set(val_image_pid)&set(settings['holdout_pids']):
        raise RuntimeError('训练与验证身份标签重叠')
    val_top1=rank_topk(vtx,vim,k=1)
    val_r1=float(100*np.mean(val_image_pid[val_top1[:,0]]==val_query_pid))
    if abs(val_r1-66.43390655517578)>.03:
        raise RuntimeError(f'最佳IRRA复现不通过，R1={val_r1:.5f}')
    out,resumed=output_folder18(parent,src,ex16,settings,args.resume)
    shutil.copy2(Path(__file__).resolve(),out/Path(__file__).name)
    if resumed and (out/'发送给ChatGPT.zip').exists() and (out/'experiment18_report.json').exists():
        with zipfile.ZipFile(out/'发送给ChatGPT.zip') as z:
            if z.testzip() is None and 'experiment18_report.json' in z.namelist():
                print('续跑：输出包完整，无需重复计算：',out,flush=True);return
    print('实验18启动：官方最佳IRRA R1=',val_r1,'视觉阈值=',threshold,
          '多文本beta=',beta,'输出：',out,flush=True)
    res=run_core18(out,settings,rec,tx,vi,threshold,beta)
    report={'version':VERSION,'best_checkpoint_sha256':BEST_SHA,'official_val_baseline_R1':val_r1,
      'source_exp12':str(src),'source_exp16':str(ex16),'exp16_threshold_fit_metadata':cal,
      'teacher_beta_total_fixed':beta,'text_support_budgets':list(BUDGETS),
      'no_irra_training':True,'no_new_student':True,'val_gallery_caption_as_support':False,
      'split':'TRAIN 440-PID holdout: A text-only query, B image/caption support, C independent image gallery',
      'methods':{'naive_top16':'Highest T1-similarity captions within top16 support images, greedy diversity',
                 'naive_global':'All B support captions, greedy diversity (exp17-style baseline)',
                 'visual_top16':'Choose visual group anchored in top16; use only within-top16 member captions',
                 'visual_anchor_expand':'Same Top16 anchor, complete-link visual expansion to all B (source-range isolated comparison)',
                 'visual_global':'Group all B images independently; choose strongest query consensus group even if no image in Top16',
                 'pid_reference':'Actual PID known only for training internal teacher reference, NOT deployable'},
      'fusion':'normalize((1-beta)*T1 + beta*normalize(mean(selected captions))) with constant total beta',
      'warnings':['Not official Validation performance or deployment-ready text source',
        'No PID or evaluation C used by label-free selectors; PID known only in reference and posthoc reports',
        'Frozen IRRA was already trained on TRAIN holdout identities; evaluate selector generalization only',
        'Visual group purity does not imply query identity correctness',
        'Visual group found via B image features is provenance detection, NOT image feature fusion',
        'Training captions are accessible in TRAIN only; Val/Test candidate captions forbidden',
        'Some groups have fewer than K captions; report actual selected K and stratify',
        'C one image per PID: AP/mINP equal reciprocal rank',
        'Embedding novelty/delta AP is NOT automatically factual or semantic complementarity',
        'Best-single-member ranking and set synergy proxy are strictly POSTHOC using C and never used in selection',
        'Greedy diversity weight 0.12 is an unvalidated research heuristic, NOT tuned on Val'],
      **res}
    atomic_json(out/'experiment18_report.json',report)
    (out/'实验18_目标与边界.md').write_text(
      '# 实验18：多条文本互补，而非选出唯一T2\n\n'
      '目的：同身份不同描述之间的可靠互补与有害拒绝；之后单独研究多图互补；最后研究图文互补。\n'
      '训练身份内A查询文本/B文本支持图/C评价图三组图像不重叠。\n'
      'Top16只是局部辅助，visual_anchor_expand从局部向外扩展，visual_global在整个B图库独立成组并选择。\n'
      '固定总beta=.25(来自实验12设置)，比较1/2/4个不同训练caption，尽可能优先不同图像来源。\n'
      'PID Reference只有教师参考意义；任何官方Val/Test人工caption不得成为B支持。\n'
      '当前验证的是多文本数量、支持来源与排序效用代理；尚未验证真正语义可靠性或正式单文本R1提升。\n',
      encoding='utf-8')
    zipped=pack_zip18(out)
    print('实验18完成，结果：',out,'\n上传：',zipped,flush=True)


def self_test():
    """真实端到端模拟：覆盖不完整K预算、视觉组扩展、身份泄漏、断点及ZIP。"""
    rng=np.random.default_rng(1806);D=32
    fit=set(range(1,1761));hold=set(range(5001,5441))
    images=[];texts=[];V=[];T=[]
    for pid in sorted(hold):
        n=5 if pid<5035 else (3 if pid<5070 else 2)
        base=normalized(rng.normal(size=(1,D)))[0]
        for j in range(n):
            idx=len(images);images.append({'pid':pid,'path':f'train/{pid}_{j}.jpg'})
            V.append(normalized((base+.025*rng.normal(size=D))[None,:])[0])
            for c in range(2):
                texts.append({'pid':pid,'img_idx':idx,'text':f'train caption person{pid} image{j} text{c}'})
                T.append(normalized((base+.06*rng.normal(size=D))[None,:])[0])
    V=np.asarray(V,np.float32);T=np.asarray(T,np.float32)
    d=build_disjoint_abc(T,V,texts,images,fit,hold)
    assert d['n_query_pid']==69 and len(d['VC'])==440
    near=d['VB']@d['VB'].T
    # Targeted source expansion test independent of data generator.
    fake=np.eye(3,dtype=np.float32)
    fake[0,1]=fake[1,0]=.91;fake[0,2]=fake[2,0]=.85;fake[1,2]=fake[2,1]=.82
    assert visual_source_group(0,[0,1],fake,.8)==[0,1]
    assert visual_source_group(0,[0,1,2],fake,.8)==[0,1,2]
    assert visual_source_group(0,[0,1,2],fake,.83)==[0,1]
    g=partition_all_visual_groups(fake,.8)
    assert len(g)==1 and sorted(g[0])==[0,1,2]
    # Standalone global group does NOT require membership in a query's Top16.
    qsim=np.asarray([.75,.3,.50],np.float32);tsim=np.asarray([.75,.30,.50],np.float32)
    ix=choose_global_group(qsim,tsim,np.asarray([0,1,2]),[[1],[0],[2]])
    assert ix==1
    q=normalized(rng.normal(size=(1,D)))[0]
    a=normalized(rng.normal(size=(4,D)))
    assert np.allclose(fuse_text_set(q,a,[],.25),q)
    assert len(greedy_caption_set([0,1,2],2,q,a,np.zeros(2),np.array([0,0,1]),
                                 np.array([.9,.8,.7])))==2
    # Distinct-image-first should select image 1 as second support.
    chosen=greedy_caption_set([0,1,2],2,q,a,np.zeros(2),np.array([0,0,1]),np.array([.9,.8,.7]))
    assert len(set([0,0,1][i] for i in chosen))==2
    with tempfile.TemporaryDirectory() as td:
        out=Path(td);settings={'fit_pids':list(fit),'holdout_pids':list(hold)}
        records={'train_texts':texts,'train_images':images}
        a=run_core18(out,settings,records,T,V,.72,.25)
        b=run_core18(out,settings,records,T,V,.72,.25)
        assert a==b and a['data_stats']['n_queries']==138
        assert len(a['methods'])==1+len(METHODS)*len(BUDGETS)
        rows=[r for r in csv.DictReader((out/'all_queries.csv').open('r',encoding='utf-8-sig',newline=''))]
        assert len(rows)==138
        assert all(int(r['visual_global_K1_multi_better_than_best_member_AP_OFFLINE_ONLY'])==0 for r in rows)
        assert all(abs(float(r['visual_global_K1_multi_vs_best_member_AP_delta_OFFLINE_ONLY']))<1e-8 for r in rows)
        assert all(int(r['pid_reference_K4_all_correct_PID_OFFLINE_ONLY'])==1 for r in rows)
        assert all(float(r['T1_ap'])==float(r['T1_inp']) for r in rows)
        assert all(int(r['visual_global_K4_n_captions'])>=int(r['visual_global_K1_n_captions']) for r in rows)
        assert all(int(r['visual_global_K4_n_source_images'])>=int(r['visual_global_K1_n_source_images']) for r in rows)
        assert all(int(r['visual_global_K4_n_from_beyond_Top16'])>=int(r['visual_top16_K4_n_from_beyond_Top16']) for r in rows)
        # Exact same first selected caption under bigger K, preserving comparison.
        for r in rows:
            x=json.loads(r['visual_global_K1_T2_original_train_text_indices'])
            y=json.loads(r['visual_global_K4_T2_original_train_text_indices'])
            assert x==y[:1]
        assert len(list(out.glob('cases_*.jsonl')))==5
        atomic_json(out/'run_info.json',{'self_test':True})
        atomic_json(out/'experiment18_report.json',{'self_test':True})
        (out/'实验18_目标与边界.md').write_text('Synthetic test only',encoding='utf-8')
        shutil.copy2(Path(__file__),out/Path(__file__).name)
        archive=pack_zip18(out)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None and 'all_queries.csv' in z.namelist()
        assert metric([.1,.6,.4],9,[1,9,2])['ap']==1
        assert metric([.9,.6,.4],9,[1,9,2])['ap']==.5
        # Files are closed by context managers, and re-run is cache-safe.
    print('SELF-TEST PASS: multi-caption 1/2/4, 6 methods, disjoint 440 C IDs, full-library independent groups, ranks, resume, ZIP',flush=True)


if __name__=='__main__':
    main()
