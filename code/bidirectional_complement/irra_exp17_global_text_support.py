# -*- coding: utf-8 -*-
"""实验17：最佳IRRA同身份不同描述的全图库文本来源与跨视角效用。

核心：T1-only / Top16 可得T2 / 全图库可得T2 / 真PID T2参考。
训练身份内部进行；T1查询来源A、可取得字幕来源B、评价图库C相互隔离。
B的caption是TRAIN注释，只在本诊断中可访问，不可把VAL/TEST图库描述当作推理时可得。
方法Top16/Global不使用PID、C图像或C标签来选择补充文本。
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
EXP17='实验17_全图库同身份文本来源与效用_'
VERSION='exp17_v1_train_only_disjoint_ABC_budget1'
BEST_SHA='f3ab12028e3406546b836ed6a75c8fc5e02a42146c5da13af423e7f5cf0e7ea6'
ANNOTATION_SHA='fc3d15542f91dd70e23028c2cb40171a8f41734a39afddb9505e352e396cabf0'
CONFIG_SHA='ec2fae46ec4e0cce06372789afe1c877a5cebeddf73654781e11b0484fc30fac'
TOPK=16
SUPPORT_BUDGET=1  # 每种可用方法最多1条额外文本
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
    fd,tmp=tempfile.mkstemp(prefix='.exp16_',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def atomic_csv(path,rows,fields):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.exp16_',suffix='.tmp',dir=path.parent)
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


def choose_support(score,candidate_text_ids):
    """Identical no-label selector for Top16, full gallery and PID reference.

    No truth, no eval C, no oracle score in this function. Stable ties use
    first support-caption index (original training order).
    """
    ids=np.asarray(candidate_text_ids,dtype=np.int32)
    if not len(ids):return -1
    return int(ids[int(np.argmax(np.asarray(score)[ids]))])


def image_and_caption_score(text_q,VB,TB,text_to_B):
    s_image=np.asarray(VB@text_q,dtype=np.float32)
    s_text=np.asarray(TB@text_q,dtype=np.float32)
    # Fixed equally weighted agreement, NO validation hyperparameter search.
    utility=np.asarray(.5*(s_text+s_image[text_to_B]),dtype=np.float32)
    return s_image,utility


def score_metrics(scores,true_pid,gallery_pid):
    """Full independent C gallery, sort on fixed scores, no A/B images."""
    order=np.argsort(-np.asarray(scores),kind='stable')
    hit=np.asarray(gallery_pid,dtype=np.int64)[order]==int(true_pid)
    ranks=np.flatnonzero(hit)+1
    if not len(ranks):raise AssertionError('C gallery missing target identity')
    ap=float(np.mean(np.arange(1,len(ranks)+1,dtype=np.float64)/ranks))
    inp=float(len(ranks)/ranks[-1]);r1=int(ranks[0]==1)
    return {'best_rank':int(ranks[0]),'r1':r1,'r5':int(ranks[0]<=5),
            'r10':int(ranks[0]<=10),'ap':ap,'inp':inp,'n_pos':int(len(ranks))}


def exp15_audit(s,text_a,img_a,text_b,img_b):
    """Teacher-only retrospective audit using A, NOT deployable; no decisions."""
    tt=float(text_a@text_b);ta=float(text_a@img_a);tb=float(text_a@img_b)
    ua=float(text_b@img_a);ub=float(text_b@img_b);ab=float(img_a@img_b)
    feat=np.asarray([tt,ta,tb,ua,ub,ab,ub-ua,tb-ta,ub-tb],np.float64)
    z=np.clip((feat-np.asarray(s['feature_mean']))/np.asarray(s['feature_scale']),-10,10)
    logits=z@np.asarray(s['weights'])+np.asarray(s['bias'])
    logits-=logits.max();p=np.exp(logits);p/=p.sum()
    return int(p.argmax()==2),float(p[0]),float(p[2])


def process_query(k,d,images,texts,image_neighbors,threshold,beta,exp15=None):
    q=d['T1'][k];qp=int(d['query_pid'][k]);aidx=int(d['query_idx'][k])
    image_sim,selection_scores=image_and_caption_score(q,d['VB'],d['TB'],d['text_to_B'])
    ordered=np.argsort(-image_sim,kind='stable')
    first16=ordered[:min(TOPK,len(ordered))]
    # Text candidate indices are determined ONLY by the permitted support B pool.
    in16=np.isin(d['text_to_B'],first16)
    top_choice=choose_support(selection_scores,np.flatnonzero(in16))
    global_choice=choose_support(selection_scores,np.arange(len(d['TB'])))
    # PID reference is a diagnostic only; not an inference choice or true max-AP Oracle.
    true_choice=choose_support(selection_scores,
                               np.flatnonzero(d['B_pid'][d['text_to_B']]==qp))
    if -1 in (top_choice,global_choice,true_choice):raise RuntimeError('候选文本缺失，内部图库划分不合法')
    t1_scores=np.asarray(d['VC']@q)
    origin=score_metrics(t1_scores,qp,d['C_pid'])
    row={'query_index':k,'T1_original_text_index':aidx,'pid_OFFLINE_ONLY':qp,
         'image_A_query_source':images[int(d['text_pos'][aidx])]['path'],
         'T1_train_caption':texts[aidx].get('text',''),
         'T1_r1':origin['r1'],'T1_rank':origin['best_rank'],
         'T1_ap':origin['ap'],'T1_inp':origin['inp'],
         'top16_contains_same_PID_OFFLINE_ONLY':int(np.any(d['B_pid'][first16]==qp)),
         'full_gallery_has_same_PID_outside_top16_OFFLINE_ONLY':int(np.any(d['B_pid'][ordered[16:]]==qp)),
         'nearest_linked_support_imgs_outside16':int(np.any(image_neighbors[first16][:,ordered[16:]]>=threshold-1e-7)) if len(ordered)>16 else 0}
    im_rank=np.empty(len(ordered),dtype=np.int32);im_rank[ordered]=np.arange(1,len(ordered)+1)
    for name,c in (('top16',top_choice),('global',global_choice),('pid_reference',true_choice)):
        b_local=int(d['text_to_B'][c]);b_original=int(d['support_img_idx'][b_local]);t2_idx=int(d['support_text_idx'][c])
        t2=d['TB'][c];mix=normalized(((1-beta)*q+beta*t2)[None,:])[0]
        r=score_metrics(np.asarray(d['VC']@mix),qp,d['C_pid'])
        row.update({f'{name}_support_image_rank':int(im_rank[b_local]),
                    f'{name}_support_img_path':images[b_original]['path'],
                    f'{name}_T2_train_caption':texts[t2_idx].get('text',''),
                    f'{name}_support_PID_correct_OFFLINE_ONLY':int(d['B_pid'][b_local]==qp),
                    f'{name}_image_from_top16':int(b_local in first16),
                    f'{name}_linked_to_top16':int(np.max(image_neighbors[first16,b_local])>=threshold-1e-7),
                    f'{name}_r1':r['r1'],f'{name}_rank':r['best_rank'],
                    f'{name}_r5':r['r5'],f'{name}_r10':r['r10'],
                    f'{name}_ap':r['ap'],f'{name}_inp':r['inp'],
                    f'{name}_delta_ap':r['ap']-origin['ap'],
                    f'{name}_delta_r1':r['r1']-origin['r1']})
        if exp15 is not None:
            A=d['source_train_images'][int(d['text_pos'][aidx])]
            B=d['VB'][b_local]
            accepted,p_harm,p_good=exp15_audit(exp15,q,A,t2,B)
            row[f'{name}_exp15_accept_OFFLINE_ONLY']=accepted
            row[f'{name}_exp15_prob_harmful_OFFLINE_ONLY']=p_harm
            row[f'{name}_exp15_prob_beneficial_OFFLINE_ONLY']=p_good
    # All facts about true PID are written only AFTER prediction to diagnostic columns.
    return row


def summarize_methods(rows):
    modes=['T1','top16','global','pid_reference'];summ=[]
    for mode in modes:
        prefix=mode+'_'
        ranks=np.asarray([int(r[prefix+'rank']) for r in rows],np.int64)
        ap=np.asarray([float(r[prefix+'ap']) for r in rows],np.float64)
        inp=np.asarray([float(r[prefix+'inp']) for r in rows],np.float64)
        vals={'method':mode,'n_queries':len(rows),'r1_percent':float(100*np.mean(ranks<=1)),
              'r5_percent':float(100*np.mean(ranks<=5)),
              'r10_percent':float(100*np.mean(ranks<=10)),
              'mAP_percent':float(100*np.mean(ap)),
              'mINP_percent':float(100*np.mean(inp)),
              'note':'TRAIN-only known-PID support reference; NOT a mathematical upper bound' if mode=='pid_reference' else ''}
        if mode!='T1':
            da=ap-np.asarray([float(r['T1_ap']) for r in rows]);
            vals.update({'AP_improve_n':int(np.sum(da>1e-9)),
                'AP_harm_n':int(np.sum(da<-1e-9)),'AP_unchanged_n':int(np.sum(np.abs(da)<=1e-9)),
                'r1_fixed_n':int(sum(r['T1_r1']==0 and r[prefix+'r1']==1 for r in rows)),
                'r1_broken_n':int(sum(r['T1_r1']==1 and r[prefix+'r1']==0 for r in rows)),
                'selected_support_truePID_pct_OFFLINE_ONLY':100*float(np.mean([r[prefix+'support_PID_correct_OFFLINE_ONLY'] for r in rows])),
                'chosen_outside_top16_pct':100*float(np.mean([r[prefix+'support_image_rank']>16 for r in rows]))})
        summ.append(vals)
    return summ


def utility_diagnostics(rows):
    """Posthoc TRAIN proxy labels; NEVER feed AP outcome into selection."""
    summary=[]
    for mode in ('top16','global','pid_reference'):
        delta=np.asarray([r[f'{mode}_delta_ap'] for r in rows],np.float64)
        benefit=delta>=.01;harm=delta<=-.01;neutral=~(benefit|harm)
        details={'method':mode,'n':len(rows),'beneficial_delta_AP_ge_001_n':int(benefit.sum()),
                 'harmful_delta_AP_le_neg001_n':int(harm.sum()),
                 'uncertain_delta_AP_between_n':int(neutral.sum()),
                 'beneficial_fraction_pct':float(100*benefit.mean()),
                 'harmful_fraction_pct':float(100*harm.mean()),
                 'selected_correct_pid_n_OFFLINE_ONLY':int(sum(r[f'{mode}_support_PID_correct_OFFLINE_ONLY'] for r in rows)),
                 'selected_outside_top16_n':int(sum(r[f'{mode}_support_image_rank']>16 for r in rows))}
        if f'{mode}_exp15_accept_OFFLINE_ONLY' in rows[0]:
            accepted=np.asarray([int(r[f'{mode}_exp15_accept_OFFLINE_ONLY'])==1 for r in rows])
            details.update({'exp15_retrospective_accept_n':int(accepted.sum()),
               'exp15_accepted_beneficial_n':int((accepted&benefit).sum()),
               'exp15_accepted_harmful_n':int((accepted&harm).sum()),
               'exp15_missed_beneficial_n':int((~accepted&benefit).sum()),
               'exp15_harmful_rejected_or_neutral_n':int((~accepted&harm).sum()),
               'exp15_warning':'Uses unavailable query image A. Analysis only, not deployable'})
        summary.append(details)
    return summary


def diagnostics(rows):
    n=len(rows)
    selected=[r for r in rows if r['global_support_image_rank']>16]
    topmiss=[r for r in rows if not r['top16_contains_same_PID_OFFLINE_ONLY']]
    available=[r for r in rows if r['full_gallery_has_same_PID_outside_top16_OFFLINE_ONLY']]
    def frac(seq,pred):return float(sum(1 for r in seq if pred(r))/len(seq)) if seq else None
    return {'n':n,'global_selected_off_top16_n':len(selected),
       'global_selected_off_top16_and_true_pid_n':sum(r['global_support_PID_correct_OFFLINE_ONLY'] for r in selected),
       'global_off16_ap_improvement_rate':frac(selected,lambda r:r['global_delta_ap']>1e-9),
       'queries_true_support_exists_outside_top16_n':len(available),
       'queries_no_true_PID_in_Top16_n':len(topmiss),
       'no_true_PID_Top16_but_global_found_true_PID_n':sum(r['global_support_PID_correct_OFFLINE_ONLY'] for r in topmiss),
       'queries_visual_neighbor_available_outside16_n':sum(r['nearest_linked_support_imgs_outside16'] for r in rows),
       'global_chosen_outside16_with_visual_link_n':sum(r['global_support_image_rank']>16 and r['global_linked_to_top16'] for r in rows),
       'mAP_equals_mINP_expected_single_C_per_PID':True,
       'reason':'Independent C pool uses one image per PID, so AP and INP equal reciprocal rank',
       'limitation':'SUPPORT-B captions are TRAIN annotations only; cannot be retrieved this way in official VAL/TEST'}


def pid_bootstrap(rows,repeats=1000):
    pids=np.asarray([int(r['pid_OFFLINE_ONLY']) for r in rows],np.int64)
    uniq,inv=np.unique(pids,return_inverse=True)
    if len(uniq)<15:return {'reason':'too few identities'}
    rng=np.random.default_rng(170001)
    result={}
    for method in ('top16','global','pid_reference'):
        for metric in ('ap','r1'):
            a=np.asarray([float(r[f'{method}_{metric}'])-float(r[f'T1_{metric}']) for r in rows])
            s=np.bincount(inv,weights=a,minlength=len(uniq));counts=np.bincount(inv,minlength=len(uniq))
            sample=rng.integers(0,len(uniq),size=(repeats,len(uniq)),dtype=np.int32)
            bootstrap=s[sample].sum(1)/counts[sample].sum(1)
            result[f'{method}_minus_T1_{metric}_pp']={
                'observed_pp':float(a.mean()*100),
                'exploratory_CI95_pp':(np.quantile(bootstrap,[.025,.975])*100).tolist(),
                'n_pid':int(len(uniq))}
    # Direct full-vs-top16 paired comparison too.
    a=np.asarray([r['global_ap']-r['top16_ap'] for r in rows],np.float64)
    s=np.bincount(inv,weights=a,minlength=len(uniq));cnt=np.bincount(inv,minlength=len(uniq))
    ix=rng.integers(0,len(uniq),size=(repeats,len(uniq)),dtype=np.int32)
    vals=s[ix].sum(1)/cnt[ix].sum(1)
    result['global_minus_top16_AP_pp']={'observed_pp':float(a.mean()*100),
        'exploratory_CI95_pp':(np.quantile(vals,[.025,.975])*100).tolist(),
        'n_pid':int(len(uniq))}
    return result


def output_folder(parent,src,exp16,settings,resume):
    parent.mkdir(parents=True,exist_ok=True)
    ident={'version':VERSION,'source_exp12':str(src.resolve()),'source_exp16':str(exp16.resolve()),
           'source_settings_hash':settings['settings_hash'],'best_sha256':BEST_SHA,
           'support_budget':SUPPORT_BUDGET,'seed':SEED,'abc_disjoint':True}
    identity_sha=stable_json(ident)
    if resume:
        for p in sorted(parent.glob(EXP17+'*'),reverse=True):
            m=p/'run_info.json'
            if not m.is_file():continue
            try:
                info=json.loads(m.read_text(encoding='utf-8'))
                if info.get('identity_sha')==identity_sha:return p,True
            except (OSError,ValueError):pass
        raise RuntimeError('找不到匹配的实验17目录。首次启动或文件夹被删除后请移除 --resume。')
    name=EXP17+datetime.now().strftime('%Y%m%d_%H%M%S')
    p=parent/name;k=2
    while p.exists():p=parent/(name+'_'+str(k));k+=1
    p.mkdir(parents=True)
    atomic_json(p/'run_info.json',{'identity_sha':identity_sha,'ident':ident,
       'created_at':datetime.now().isoformat()})
    return p,False


def atomic_jsonl(path,rows):
    path=Path(path);fd,tmp=tempfile.mkstemp(prefix='.exp17_',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def load_chunk(path,start,end):
    with open(path,encoding='utf-8') as f:
        rows=[json.loads(s) for s in f if s.strip()]
    if len(rows)!=end-start or [int(r['query_index']) for r in rows]!=list(range(start,end)):
        raise RuntimeError(f'断点分块损坏或编号不符，拒绝继续：{path}')
    return rows


def pack_zip(out):
    names=['run_info.json','experiment17_report.json','method_summary.csv',
           'candidate_accessibility.json','all_queries.csv','bootstrap_pid.json','support_utility_diagnostics.csv',
           '实验17_目标与边界.md',Path(__file__).name]
    p=out/'发送给ChatGPT.zip';fd,tmp=tempfile.mkstemp(prefix='.exp17_',suffix='.zip',dir=out);os.close(fd)
    try:
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for n in names:
                if not (out/n).is_file():raise FileNotFoundError(n)
                z.write(out/n,arcname=n)
        with zipfile.ZipFile(tmp) as z:
            if z.testzip() is not None or len(z.namelist())!=len(names):raise IOError('ZIP校验失败')
        os.replace(tmp,p)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return p


def run_core(out,src_settings,records,data,threshold,beta,selector):
    train_x,train_v,_,_=data
    fit=set(map(int,src_settings['fit_pids']));hold=set(map(int,src_settings['holdout_pids']))
    d=build_disjoint_abc(train_x,train_v,records['train_texts'],records['train_images'],fit,hold)
    l2check(d['T1'],'A query');l2check(d['TB'],'B support text');l2check(d['VB'],'B support image');l2check(d['VC'],'C evaluation image')
    # Image similarity for provenance only, NOT a new image-fusion algorithm.
    neighbors=np.asarray(d['VB']@d['VB'].T,dtype=np.float32)
    # EXP15 model may use inaccessible A, strictly AFTER no-label selection.
    d['source_train_images']=train_v
    total=len(d['query_idx']); rows=[];chunk_size=64
    for start in range(0,total,chunk_size):
        end=min(start+chunk_size,total)
        part=out/f'cases_{start:05d}_{end:05d}.jsonl'
        if part.is_file():
            chunk=load_chunk(part,start,end)
            print(f'  RESUME chunk {start}:{end}',flush=True)
        else:
            chunk=[]
            for k in range(start,end):
                chunk.append(process_query(k,d,records['train_images'],records['train_texts'],
                                           neighbors,threshold,beta,selector))
            atomic_jsonl(part,chunk)
            print(f'  processed {end}/{total}',flush=True)
        rows.extend(chunk)
    if len(rows)!=total or any(row['query_index']!=i for i,row in enumerate(rows)):
        raise AssertionError('Query outcome order/size invalid')
    summary=summarize_methods(rows)
    atomic_csv(out/'method_summary.csv',summary,sorted(set().union(*(s.keys() for s in summary))))
    atomic_csv(out/'all_queries.csv',rows,list(rows[0]))
    detail=utility_diagnostics(rows)
    atomic_csv(out/'support_utility_diagnostics.csv',detail,sorted(set().union(*(r.keys() for r in detail))))
    desc=diagnostics(rows)
    desc.update({'n_support_pool_images':len(d['VB']),'n_support_pool_captions':len(d['TB']),
       'n_eval_gallery_images':len(d['VC']),'n_eval_gallery_PID':len(set(d['C_pid'])),
       'n_A_source_images':d['n_query_pid'],'query_count':total,
       'holdout_image_count_distribution':{
           'one_image':sum(v==1 for v in d['source_image_counts'].values()),
           'two_images':sum(v==2 for v in d['source_image_counts'].values()),
           'three_or_more':sum(v>=3 for v in d['source_image_counts'].values())}})
    atomic_json(out/'candidate_accessibility.json',desc)
    boot=pid_bootstrap(rows)
    atomic_json(out/'bootstrap_pid.json',boot)
    return {'methods':summary,'support_utility_diagnostics':detail,'accessibility':desc,'bootstrap':boot,'n':total}


def main():
    ap=argparse.ArgumentParser(description='实验17：同身份跨图文本T2 Top16 vs 全图库来源，全程冻结最佳IRRA')
    ap.add_argument('--project-root',default=str(ROOT));ap.add_argument('--exp12-dir',default='')
    ap.add_argument('--exp16-dir',default='');ap.add_argument('--exp15-dir',default='')
    ap.add_argument('--resume',action='store_true');ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:self_test();return
    root=Path(args.project_root).expanduser().resolve();parent=root/PARENT
    src=find_exp12(parent,args.exp12_dir)
    src_settings,selected=source_audit(src,full_hash=True)
    ex16=find_exp16(parent,args.exp16_dir)
    threshold,cal=audit_exp16(ex16,src,src_settings)
    # No re-calibration and NO experiment-17 student training.
    b=float(src_settings['teacher_beta'])
    if not 0<b<1:raise RuntimeError('实验12教师融合系数无效')
    ex15=find_exp15(parent,args.exp15_dir)
    # Optional old trained gate requires A and is NOT used for actual method selection.
    selector=optional_exp15_for_audit(ex15,src)
    rec=read_records(src_settings['annotation_file'],selected)
    train_x,train_v,vx,vv=load_features(src,src_settings,rec)
    for a,name in [(train_x,'train text'),(train_v,'train image'),(vx,'val query'),(vv,'val image')]:l2check(a,name)
    # Check official Val R1 using ONLY query embeddings; no Val captions accessed as T2.
    vi_pid=np.asarray([i['pid'] for i in rec['val_images']],dtype=np.int64)
    vt_pid=np.asarray([i['pid'] for i in rec['val_queries']],dtype=np.int64)
    if set(vi_pid)&set(src_settings['fit_pids']) or set(vi_pid)&set(src_settings['holdout_pids']):
        raise RuntimeError('TRAIN/VAL PID overlap')
    top1=rank_topk(vx,vv,k=1)
    val_r1=100*float(np.mean(vi_pid[top1[:,0]]==vt_pid))
    if abs(val_r1-66.43390655517578)>.03:raise RuntimeError(f'最佳IRRA Val指纹指标不符 R1={val_r1}')
    out,resumed=output_folder(parent,src,ex16,src_settings,args.resume)
    shutil.copy2(Path(__file__).resolve(),out/Path(__file__).name)
    print('实验17：最佳IRRA R1=',val_r1,'阈值=',threshold,'beta=',b,'输出:',out,flush=True)
    if resumed and (out/'experiment17_report.json').is_file() and (out/'发送给ChatGPT.zip').is_file():
        with zipfile.ZipFile(out/'发送给ChatGPT.zip') as z:
            if z.testzip() is None and 'experiment17_report.json' in z.namelist():
                print('续跑：完整ZIP已验证，无需重复计算。',flush=True);return
    rs=run_core(out,src_settings,rec,(train_x,train_v,vx,vv),threshold,b,selector)
    report={'version':VERSION,'best_checkpoint_sha256':BEST_SHA,'baseline_val_R1_official':val_r1,
      'source_exp12':str(src),'source_exp16':str(ex16),'source_exp15_optional':str(ex15) if selector else None,
      'beta_exp12_fixed':b,'visual_group_threshold_fit_exp16':threshold,
      'data_split':'Only TRAIN 440 PID holdout, A/query, B/support captions, C/independent gallery images',
      'top16_role':'auxiliary source restriction only; global scans all B without true PID',
      'global_route':'direct full-gallery image and TRAIN caption similarity; report visual links back to Top16 for provenance only',
      'no_model_training':True,'no_original_irra_modification':True,
      'fixed_budget_T2':SUPPORT_BUDGET,
      'selector_criterion':'0.5*(T1 dot support image + T1 dot T2); fixed, label-free, same in all 3 source conditions',
      'fit_threshold_metadata':cal,
      'exp15_gate_only_posthoc':selector is not None,
      'strict_limits':['PID reference knows training PID; only teacher diagnostic, not deployable','No Val/Test candidate captions used',
       'No actual semantic truth labels; delta AP is retrieval utility proxy',
       'One C per PID gives AP==INP==reciprocal rank; use R1, rank and AP with this caveat',
       'Single fixed beta and exactly 1 T2, not a learned beneficial/harmful acceptance gate',
       'Training holdout identities were seen by frozen IRRA training; not fully unseen by backbone',
       'Global T2 metadata would require legal candidate captions; official Val/Test does NOT provide these for inference'],
      **rs}
    atomic_json(out/'experiment17_report.json',report)
    (out/'实验17_目标与边界.md').write_text(
      '# 实验17：同身份多文本互补信息来源与效用（仅训练集机制诊断）\n\n'
      '当前只研究T1与来自不同图像B的T2。Top-16仅是辅助来源，不限定全图库探索。\n'
      '固定最佳IRRA及实验12的融合beta，不训练学生，不修改模型、源码或原始缓存。\n'
      '按身份在TRAIN holdout内划分A查询源、B合法训练文本支持源、C独立评价图像，三集合不交叉。\n'
      'Top16与Global在选择T2时都不读取身份标签，Global可选择Top16之外的B来源；'
      '真PID参考仅用于训练内部潜在价值比较，未事后按AP挑选。每法固定一个T2预算。\n'
      '实验15选择器需要真实来源A图像，因此仅做事后审计，不能作为可部署选择。\n'
      'C每身份仅一张，AP与INP退化成倒数排名，指标只用于严格同协议内部比较。\n'
      '注意：TRAIN人工文字只在训练期允许；不能把Val/Test图库人工caption用作实际推理T2。\n'
      '本实验不宣称Validation R1提升；最终目标仍是文本互补→多视角图像互补→联合互补。\n',encoding='utf-8')
    path=pack_zip(out)
    print('实验17完成，内部指标见 method_summary.csv；结果包:',path,flush=True)


def self_test():
    """Small genuine end-to-end on synthetic identities, plus checks on metadata."""
    rng=np.random.default_rng(17);D=12
    nfit=1760;nhold=440
    # Need only populate holdout IDs, fit PID list remains disjoint for invariants.
    fit=set(range(1,nfit+1));hold=set(range(5001,5001+nhold))
    images=[];texts=[];vf=[];tf=[]
    for pid in sorted(hold):
        n=3 if pid<5033 else (2 if pid<5060 else 1)
        basis=normalized(rng.normal(size=(1,D)))[0]
        for j in range(n):
            vi=normalized((basis+.08*rng.normal(size=D))[None])[0]
            idx=len(images);images.append({'pid':pid,'path':f'train/img_{idx}.jpg'});vf.append(vi)
            for c in range(2):
                ti=normalized((basis+.10*rng.normal(size=D))[None])[0]
                texts.append({'pid':pid,'img_idx':idx,'text':f'caption_{pid}_{j}_{c}'})
                tf.append(ti)
    vf=np.asarray(vf,np.float32);tf=np.asarray(tf,np.float32)
    d=build_disjoint_abc(tf,vf,texts,images,fit,hold)
    assert len(d['query_idx'])==32*2 and len(d['VC'])==440
    assert d['n_query_pid']==32 and not set(d['eval_img_idx'])&set(d['support_img_idx'])
    for name in ['T1','TB','VB','VC']:l2check(d[name],name)
    near=np.asarray(d['VB']@d['VB'].T,np.float32)
    with tempfile.TemporaryDirectory() as td:
        out=Path(td);args={'teacher_beta':.25,'fit_pids':list(fit),'holdout_pids':list(hold)}
        d['source_train_images']=vf
        rows=[process_query(i,d,images,texts,near,.7,.25) for i in range(len(d['query_idx']))]
        result=summarize_methods(rows)
        assert len(result)==4 and result[0]['method']=='T1'
        assert all(r['pid_reference_support_PID_correct_OFFLINE_ONLY']==1 for r in rows)
        assert all(r['top16_support_image_rank']<=16 for r in rows)
        assert all(r['global_support_image_rank']>=1 for r in rows)
        assert all(r['T1_rank']>=1 for r in rows)
        assert all(r['top16_support_img_path'] not in (r['image_A_query_source'],) for r in rows)
        assert all(abs(r['T1_ap']-r['T1_inp'])<1e-9 for r in rows)
        atomic_jsonl(out/'piece.jsonl',rows)
        assert load_chunk(out/'piece.jsonl',0,len(rows))==rows
        atomic_csv(out/'all_queries.csv',rows,list(rows[0]))
        atomic_json(out/'bootstrap_pid.json',pid_bootstrap(rows,100))
        atomic_json(out/'candidate_accessibility.json',diagnostics(rows))
        qual=utility_diagnostics(rows)
        atomic_csv(out/'support_utility_diagnostics.csv',qual,sorted(set().union(*(r.keys() for r in qual))))
        atomic_json(out/'run_info.json',{})
        atomic_json(out/'experiment17_report.json',{'self_test':True})
        atomic_csv(out/'method_summary.csv',result,sorted(set().union(*(r.keys() for r in result))))
        (out/'实验17_目标与边界.md').write_text('自检',encoding='utf-8')
        shutil.copy2(Path(__file__),out/Path(__file__).name)
        z=pack_zip(out)
        with zipfile.ZipFile(z) as f:assert f.testzip() is None
        # Exercise the actual formal stage and CHUNK-RESUME logic twice.
        stage=out/'full_stage'
        stage.mkdir()
        synthetic_settings={'fit_pids':list(fit),'holdout_pids':list(hold)}
        records={'train_texts':texts,'train_images':images}
        one=run_core(stage,synthetic_settings,records,(tf,vf,None,None),.7,.25,None)
        two=run_core(stage,synthetic_settings,records,(tf,vf,None,None),.7,.25,None)
        assert one==two and one['n']==len(rows)
        assert len(list(stage.glob('cases_*.jsonl')))==1
        assert len(list(csv.DictReader((stage/'all_queries.csv').open(encoding='utf-8-sig',newline=''))))==len(rows)
        # adversarial sanity check: source index cannot overlap eval C.
        assert len({int(r['image_A_query_source'].split('_')[-1].split('.')[0]) for r in rows})==32
    # Independent manual order and AP sanity.
    m=score_metrics([0.1,0.7,0.4],true_pid=9,gallery_pid=np.asarray([2,9,3]))
    assert m['best_rank']==1 and m['ap']==1.
    m=score_metrics([0.9,0.7,0.4],true_pid=9,gallery_pid=np.asarray([2,9,3]))
    assert m['best_rank']==2 and m['ap']==.5
    # Label-free chooser must never take a PID argument.
    assert choose_support(np.asarray([.1,.5,.2]),[0,2])==2
    print(f'SELF-TEST PASS: {len(rows)} queries, 440 independent C, 4 comparisons, resume rows, ZIP, AP, no A/B overlap',flush=True)


if __name__=='__main__':
    main()
