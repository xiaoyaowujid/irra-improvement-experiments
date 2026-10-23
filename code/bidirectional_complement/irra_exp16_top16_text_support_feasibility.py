# -*- coding: utf-8 -*-
"""实验16：最佳IRRA Top-16候选多图聚集与合法T2来源可行性核查。

目的严格限定为同身份多文本互补的推理可获得性，而非新的困难负样本、
图像融合或文本生成方法。冻结最佳IRRA、零训练、零参数修改。

1. 训练身份fit部分：仅使用其PID评估图像相似度的候选分组阈值。
2. 训练身份holdout：不使用PID分组，仅作事后真值核查。
3. Val：单条查询T1的特征检索Top-16，图像分组只用图像特征；
   真实Val PID只在计算完全部分组后作离线审计，不参与选择/阈值。
4. Val/Test原始人工caption禁止作为候选图像的补充T2；
   没有外部、真实可用的图像描述来源，本实验明确记录“可用T2=0”，
   不伪造任何检索改进、不运行实验15新筛选器。
5. 同身份跨图图像互补与最终图文融合，均为后续研究，不在本脚本中做。

严格复用实验12的最佳IRRA特征缓存，完全不导入PyTorch，
不重算特征、不写入任何原始缓存、源码或best.pth。

第一次：python irra_exp16_top16_text_support_feasibility.py
自检：  python irra_exp16_top16_text_support_feasibility.py --self-test
续跑：  python irra_exp16_top16_text_support_feasibility.py --resume
指定来源：--exp12-dir "G:\\...\\图文双向互补实验\\实验12_..."
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse, csv, hashlib, json, shutil, tempfile, zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT=Path(r'/')
PARENT='图文双向互补实验'
EXP12='实验12_最佳IRRA同身份文本互补迁移_'
EXP16='实验16_Top16多图与合法文本来源_'
VERSION='exp16_v1_strict_no_val_gallery_captions_no_augmentation'
BEST_SHA='f3ab12028e3406546b836ed6a75c8fc5e02a42146c5da13af423e7f5cf0e7ea6'
ANNOTATION_SHA='fc3d15542f91dd70e23028c2cb40171a8f41734a39afddb9505e352e396cabf0'
CONFIG_SHA='ec2fae46ec4e0cce06372789afe1c877a5cebeddf73654781e11b0484fc30fac'
TOPK=16


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
                    tr_txt.append({'pid':pid,'img_idx':local_i})
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


def image_pairs(top):
    # Return unique candidate-image pairs across a sample of queries.
    pairs=set()
    for row in top:
        for i in range(len(row)-1):
            for j in range(i+1,len(row)):
                a,b=int(row[i]),int(row[j]);pairs.add((a,b) if a<b else (b,a))
    return pairs


def calibrate_threshold(x,v,tpids,ipids,query_limit=1800,seed=20261021,precision_target=.95):
    """Preselect visual-only image cluster threshold using TRAIN FIT true-PID labels.

    Calibration labels are never part of cluster inference and never touch val.
    Select max true-positive pair recall subject to >=95% pair precision, and
    at least 20 observed true positives. If unsupported, abstain (no grouping).
    """
    if len(x)>query_limit:
        rng=np.random.default_rng(seed)
        sample=np.sort(rng.choice(len(x),size=query_limit,replace=False))
    else:sample=np.arange(len(x))
    top=rank_topk(x[sample],v)
    pairs=image_pairs(top)
    if not pairs:raise RuntimeError('No candidate image pairs for calibration')
    ab=np.asarray(sorted(pairs),dtype=np.int32)
    cos=np.einsum('ij,ij->i',v[ab[:,0]],v[ab[:,1]]).astype(np.float64)
    correct=ipids[ab[:,0]]==ipids[ab[:,1]]
    npos=int(correct.sum());nneg=int((~correct).sum())
    if npos<20 or nneg<20:raise RuntimeError('Insufficient fit-domain pos/neg candidate pairs')
    order=np.argsort(-cos,kind='stable')
    s=cos[order]; good=correct[order]
    tp=np.cumsum(good,dtype=np.int64)
    rank=np.arange(1,len(s)+1,dtype=np.int64)
    precision=tp/rank
    # Scores with exact ties must enter together; choose only at tie boundaries.
    boundary=np.r_[s[:-1]!=s[1:],True]
    feasible=np.flatnonzero(boundary & (precision>=precision_target) & (tp>=20))
    if not len(feasible):
        threshold=1.000001; selected=0;positive_found=0;pair_precision=None
        message='No threshold reaches required precision; abstain from pseudo-ID grouping'
    else:
        j=int(feasible[np.argmax(tp[feasible])])
        threshold=float(s[j]);selected=j+1;positive_found=int(tp[j]);pair_precision=float(precision[j]);message='calibrated'
    return threshold,{'threshold':threshold,'training_pid_only':True,'fit_queries':int(len(sample)),
        'unique_top16_image_pairs':int(len(ab)),'positive_pairs':npos,'negative_pairs':nneg,
        'target_pair_precision':precision_target,'selected_pairs':selected,
        'fit_pair_precision':pair_precision,
        'fit_pair_recall':float(positive_found/max(npos,1)), 'status':message}


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


def safe_ratio(a,b):return float(a/b) if b else None


def diagnose(name,top,v,tpid,ipid,image_paths,threshold,save_rows=True):
    """All labels used below *after* predicted grouping, for OFFLINE EVALUATION ONLY."""
    rows=[];total_pair_true=total_pair_pred=total_pair_correct=0
    top1_ok=coverage=positive2=anydup=wrongdup=primary_multi=primary_allpure=0
    falsecluster=0;pred_multigroup=0;gain_opportunity=0
    for q,images in enumerate(top):
        if q and q%1000==0: print(f'{name}: analyzed {q}/{len(top)} queries',flush=True)
        groups=complete_link_clusters(images,v,threshold)
        # Predict with image cosine only; subsequently evaluate with PID labels.
        group0=next(g for g in groups if 0 in g)
        labels=np.asarray(ipid[images]);true=int(tpid[q]);truth_count=int(np.sum(labels==true))
        counts=np.unique(labels,return_counts=True)[1]
        samepair=int(sum(int(c*(c-1)//2) for c in counts))
        total_pair_true+=samepair
        implied_pairs=[]
        for g in groups:
            if len(g)<2:continue
            pred_multigroup+=1
            for a in range(len(g)-1):
                for b in range(a+1,len(g)):
                    implied_pairs.append((g[a],g[b]))
        predicted=len(implied_pairs)
        correct=sum(int(labels[a]==labels[b]) for a,b in implied_pairs)
        total_pair_pred+=predicted;total_pair_correct+=correct
        wrong_merged=predicted>correct
        falsecluster+=int(wrong_merged)
        top1_ok+=int(labels[0]==true)
        coverage+=int(truth_count>=1)
        positive2+=int(truth_count>=2)
        anydup+=int(samepair>0)
        wrongdup+=int(any(int(count)>=2 and int(pid)!=true for pid,count in zip(*np.unique(labels,return_counts=True))))
        primary_multi+=int(len(group0)>=2)
        pure=len(set(map(int,labels[group0])))==1
        primary_allpure+=int(pure) if len(group0)>=2 else 0
        gain_opportunity+=int(len(group0)>=2 and pure and labels[0]==true)
        if save_rows:
            rows.append({'query_index':q,'query_pid_for_audit_only':true,
                'top1_correct_for_audit_only':int(labels[0]==true),
                'true_top16_positive_count':truth_count,
                'true_top16_has_duplicate_pid':int(samepair>0),
                'true_top16_has_multi_positive':int(truth_count>=2),
                'predicted_top1_group_size':len(group0),
                'predicted_top1_group_true_purity_for_audit_only':int(pure),
                'predicted_top1_group_is_correct_for_audit_only':int(pure and labels[0]==true),
                'predicted_top1_additional_image_count':len(group0)-1,
                'top1_gallery_path':image_paths[int(images[0])],
                'top1_group_gallery_paths':' | '.join(image_paths[int(images[i])] for i in group0),
                'candidate_gallery_image_indices':' '.join(map(str,map(int,images))),
                'candidate_group_positions':' | '.join(','.join(map(str,g)) for g in groups),
                'false_cluster_merge_for_audit_only':int(wrong_merged),
                # Val artificially annotated descriptions are NOT read to form T2.
                'deployable_existing_same_pid_T2_count':0,
                'image_to_text_generator_connected':0})
    n=len(top)
    summary={'split':name,'n_queries':n,'gallery_images':len(v),'topk':TOPK,
        'top1_r1_percent':100*top1_ok/n,'top16_positive_coverage_percent':100*coverage/n,
        'top16_multi_positive_fraction':positive2/n,
        'top16_any_duplicate_pid_fraction':anydup/n,
        'top16_wrong_pid_duplicate_fraction':wrongdup/n,
        'predicted_top1_group_multi_fraction':primary_multi/n,
        'predicted_top1_group_pure_multi_fraction':primary_allpure/n,
        'top1_group_true_correct_multi_fraction':gain_opportunity/n,
        'queries_with_false_merge':falsecluster,
        'predicted_cluster_pair_precision':safe_ratio(total_pair_correct,total_pair_pred),
        'predicted_cluster_pair_recall':safe_ratio(total_pair_correct,total_pair_true),
        'true_duplicate_pairs':total_pair_true,'predicted_pairs':total_pair_pred,
        'correct_predicted_pairs':total_pair_correct,
        'legitimate_val_gallery_original_caption_T2':0,
        'external_caption_generator_present':False,
        'real_deployable_T2_generated':0,
        'can_claim_actual_text_complementarity_gain':False}
    return summary,rows


def validate_baseline(r1,expected=66.43390655517578,tol=.03):
    if abs(r1-expected)>tol:
        raise RuntimeError(f'最佳IRRA冻结Val R1不一致：{r1:.6f} vs {expected:.6f}。停止实验。')


def package_output(out):
    zippath=out/'发送给ChatGPT.zip'
    names=[p for p in out.iterdir() if p.is_file() and p.suffix in {'.json','.csv','.md','.py','.log'}]
    fd,tmp=tempfile.mkstemp(prefix='.exp16_',suffix='.zip',dir=out);os.close(fd)
    try:
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
            for p in sorted(names):z.write(p,arcname=p.name)
        with zipfile.ZipFile(tmp) as z:
            bad=z.testzip()
            if bad:raise IOError('ZIP损坏: '+bad)
        os.replace(tmp,zippath)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return zippath


def compute_and_export(out,data,fit_ids,hold_ids,fit_limit=1800,exptop=TOPK,
                       reference_r1=66.43390655517578):
    tx,im,vtx,vim=data['features']
    trainimgs=data['train_images'];traintxt=data['train_texts']
    fit,hold=split_train(tx,im,traintxt,trainimgs,fit_ids,hold_ids)
    fx,fv,ftp,fip,_,_=fit
    hx,hv,htp,hip,_,hii=hold
    # Match the number of identities in the frozen official Val gallery:
    # choose a deterministic 1000-PID subset of 1760 FIT PIDs (never HOLDOUT/Val).
    calibration_pid_count=min(1000,len(fit_ids))
    cal_pids=set(map(int,np.random.default_rng(160001).choice(
        sorted(fit_ids),size=calibration_pid_count,replace=False)))
    cal_t_mask=np.isin(ftp,list(cal_pids))
    cal_i_mask=np.isin(fip,list(cal_pids))
    print(f'Calibrating from FIT-only {calibration_pid_count} PID gallery, '
          f'{int(np.sum(cal_t_mask))} queries / {int(np.sum(cal_i_mask))} images; '
          f'HOLDOUT and VAL are untouched.',flush=True)
    threshold,cal=calibrate_threshold(fx[cal_t_mask],fv[cal_i_mask],
          ftp[cal_t_mask],fip[cal_i_mask],query_limit=fit_limit)
    cal['calibration_gallery_pids']=calibration_pid_count
    cal['calibration_gallery_images']=int(np.sum(cal_i_mask))
    atomic_json(out/'fit_threshold_calibration.json',cal)
    print(f'FIT only visual threshold = {threshold:.7f}; precision = {cal["fit_pair_precision"]}; recall = {cal["fit_pair_recall"]}',flush=True)
    # First prove dataset-agnostic scorer yields correct best-IRRA official val R1.
    vidpid=np.asarray([i['pid'] for i in data['val_images']],dtype=np.int64)
    vqpid=np.asarray([i['pid'] for i in data['val_queries']],dtype=np.int64)
    if set(vidpid)&set(fip) or set(vidpid)&set(hip):
        raise AssertionError('TRAIN and VAL identity overlap')
    valtop=rank_topk(vtx,vim,exptop)
    valr1=float(np.mean(vidpid[valtop[:,0]]==vqpid)*100)
    validate_baseline(valr1,expected=reference_r1)
    atomic_json(out/'best_irra_baseline_audit.json',{'val_r1_percent':valr1,
        'reference_r1_percent':reference_r1,'n_val_queries':len(vtx),
        'n_val_gallery_images':len(vim),'best_model_sha256':BEST_SHA,'passed':True})
    print(f'Best IRRA Val R1 verified: {valr1:.6f}%',flush=True)
    print('Evaluating untouched TRAIN HOLDOUT full Top16...',flush=True)
    htop=rank_topk(hx,hv,exptop)
    holdpaths=[trainimgs[int(k)]['path'] for k in hii]
    hsummary,hrows=diagnose('TRAIN_HOLDOUT',htop,hv,htp,hip,holdpaths,threshold)
    print('Evaluating Val Top16 image groups (only after prediction are labels used)...',flush=True)
    vsummary,vrows=diagnose('VAL_AUDIT_ONLY',valtop,vim,vqpid,vidpid,
                            [e['path'] for e in data['val_images']],threshold)
    fields=list(vrows[0]) if vrows else ['query_index']
    atomic_csv(out/'val_top16_all_queries.csv',vrows,fields)
    atomic_csv(out/'train_holdout_top16_all_queries.csv',hrows,fields)
    atomic_csv(out/'top16_summary.csv',[hsummary,vsummary],list(hsummary))
    note={'purpose':'text_complementarity_supply_feasibility',
        'current_stage':'same-identity different-caption text complementarity',
        'future_stages':['multi-view image complementarity','joint image-text complementarity'],
        'inference_permitted':['query T1','frozen IRRA image features','top16 image paths and pixels','train-fitted visual grouping threshold'],
        'inference_forbidden':['val/test actual PID','val/test artificial/gallery annotated captions','oracle post-hoc selection'],
        'why_no_T2':'No permitted image->caption model or live gallery caption registry is available. Official dataset gallery captions are evaluation annotations and would leak information.',
        'training_reference':'TRAIN captions are available during training but do not make Val candidate captions legitimately available.',
        'no_irra_training_or_modification':True,
        'no_student_training':True,'no_image_complementarity_training':True,
        'no_val_reranking_or_R1_improvement_claim':True}
    atomic_json(out/'caption_availability_and_leakage_audit.json',note)
    report={'version':VERSION,'best_checkpoint_sha256':BEST_SHA,
        'fit_only_threshold':cal,'holdout':hsummary,'validation':vsummary,
        'critical_note':'The pair grouping metrics use PID only AFTER predictions. Val gallery captions must not be used as T2. This experiment has NO final deployable T2 and no retrieval improvement method.',
        'next_dependency':'A deployable, independently trained and provenance-audited image-to-text source must be available before testing selection and T1+T2 on VAL.'}
    atomic_json(out/'experiment16_report.json',report)
    (out/'实验16_目标与边界.md').write_text(
        '# 实验16：同身份不同文本的可获取性\n\n'
        '唯一当前研究目标：有益同身份T2可接受，有害T2可拒绝，最终改善正式单文本检索。\n'
        '本步只检查Top16是否形成真实多图候选以及无标签图像分组能否稳定。\n'
        '该步仅准备补充文本的合法来源，不训练文本学生，不做图像互补，更不做联合图文互补。\n\n'
        '### 推理合法性\n'
        'Top16采用单条Val T1特征；组内相似度只来自候选图像特征。'
        'Val真实PID只作分组完成后的评价，绝不参与分组。\n'
        '禁止把官方Val图像原有的人工caption当作推理可得的T2。'
        '当前无可部署的图像描述生成器或合法图库描述库，故可用真实同身份T2=0。\n'
        'TRAIN caption在训练内部可用于教师实验，但不是Val画廊的可用标注。\n'
        '预计下一步先核查描述来源的合法性和事实准确性，再调用已有实验15选择机制。\n',
        encoding='utf-8')
    return report


def run_folder(parent,source,settings,resume):
    parent.mkdir(parents=True,exist_ok=True)
    target_hash=stable_json(settings)
    if resume:
        for p in sorted(parent.glob(EXP16+'*'),reverse=True):
            path=p/'run_info.json'
            if not path.is_file():continue
            try:
                info=json.loads(path.read_text(encoding='utf-8'))
                if info.get('settings_hash')==target_hash and info.get('source_dir')==str(source.resolve()):
                    return p,True
            except (OSError,ValueError):pass
        raise RuntimeError('找不到可恢复的实验16目录。首次运行或目录删除后请移除 --resume。')
    name=EXP16+datetime.now().strftime('%Y%m%d_%H%M%S')
    path=parent/name
    count=2
    while path.exists():path=parent/(name+'_'+str(count));count+=1
    path.mkdir(parents=True)
    atomic_json(path/'run_info.json',{'version':VERSION,'settings':settings,
        'settings_hash':target_hash,'source_dir':str(source.resolve()),'created_at':datetime.now().isoformat()})
    return path,False


def run_main(args):
    root=Path(args.project_root).expanduser().resolve()
    parent=root/PARENT
    src=find_exp12(parent,args.exp12_dir)
    print('EXP12 source:',src,flush=True)
    settings,selected=source_audit(src,full_hash=True)
    if int(args.calibration_queries)<100:raise ValueError('calibration queries must be >=100')
    if TOPK!=16:raise AssertionError('Top16 fixed')
    rec=read_records(settings['annotation_file'],selected)
    # No actual Val/Test gallery caption contents loaded into candidate lookup.
    fx,fi,vx,vi=load_features(src,settings,rec)
    for array,name in [(fx,'TRAIN text'),(fi,'TRAIN image'),(vx,'VAL text'),(vi,'VAL image')]:l2check(array,name)
    runsettings={'version':VERSION,'topk':TOPK,'fit_query_limit':args.calibration_queries,
        'source_exp12':str(src.resolve()),'exp12_settings_hash':settings['settings_hash'],
        'best_sha':BEST_SHA,'annotation_sha':ANNOTATION_SHA}
    out,resumed=run_folder(parent,src,runsettings,args.resume)
    shutil.copy2(Path(__file__).resolve(),out/Path(__file__).name)
    if resumed and (out/'experiment16_report.json').is_file() and (out/'发送给ChatGPT.zip').is_file():
        with zipfile.ZipFile(out/'发送给ChatGPT.zip') as z:
            if z.testzip() is None:
                print('RESUME: verified completed run:',out,flush=True);return
    data={'features':(fx,fi,vx,vi),'train_images':rec['train_images'],
          'train_texts':rec['train_texts'],'val_images':rec['val_images'],
          'val_queries':rec['val_queries']}
    report=compute_and_export(out,data,set(settings['fit_pids']),set(settings['holdout_pids']),
                              fit_limit=args.calibration_queries)
    package_output(out)
    print('EXP16 COMPLETED; all source data/model are read-only.',flush=True)
    print('Holdout grouping pair precision:',report['holdout']['predicted_cluster_pair_precision'],flush=True)
    print('Val grouping pair precision (offline audit):',report['validation']['predicted_cluster_pair_precision'],flush=True)
    print('Report:',out/'发送给ChatGPT.zip',flush=True)


def self_test():
    rng=np.random.default_rng(1626)
    dim=16;v=[];tx=[];trainim=[];traintxt=[];valim=[];valtxt=[];vval=[];tval=[]
    # 30 disjoint TRAIN identities; 12 disjoint validation identities; >=16 gallery candidates.
    for pid in range(30):
        cen=rng.normal(size=dim);cen/=np.linalg.norm(cen)
        for k in range(3):
            f=cen+.04*rng.normal(size=dim);f/=np.linalg.norm(f)
            trainim.append({'pid':pid,'path':f'train/{pid}_{k}.jpg'});v.append(f)
            for m in range(2):
                t=f+.08*rng.normal(size=dim);t/=np.linalg.norm(t)
                traintxt.append({'pid':pid,'img_idx':len(trainim)-1});tx.append(t)
    for pid in range(100,112):
        cen=rng.normal(size=dim);cen/=np.linalg.norm(cen)
        for k in range(3):
            f=cen+.04*rng.normal(size=dim);f/=np.linalg.norm(f)
            valim.append({'pid':pid,'path':f'val/{pid}_{k}.jpg'});vval.append(f)
            for m in range(2):
                t=f+.08*rng.normal(size=dim);t/=np.linalg.norm(t)
                valtxt.append({'pid':pid,'img_idx':len(valim)-1});tval.append(t)
    v=np.asarray(v,np.float32);tx=np.asarray(tx,np.float32)
    vv=np.asarray(vval,np.float32);tv=np.asarray(tval,np.float32)
    assert len(set(z['pid'] for z in valim)&set(z['pid'] for z in trainim))==0
    t=rank_topk(tx,v);assert t.shape==(len(tx),16)
    fitted=split_train(tx,v,traintxt,trainim,set(range(20)),set(range(20,30)))
    fx,fi,ftp,fip,_,_=fitted[0]
    th,details=calibrate_threshold(fx,fi,ftp,fip,query_limit=120)
    assert th<=1.000001 and details['positive_pairs']>0
    htop=rank_topk(fitted[1][0],fitted[1][1])
    summary,rows=diagnose('TRAIN_HOLDOUT',htop,fitted[1][1],fitted[1][2],fitted[1][3],
            [trainim[int(j)]['path'] for j in fitted[1][5]],th)
    vtop=rank_topk(tv,vv)
    print('Evaluating Val Top16 image groups (only after prediction are labels used)...',flush=True)
    vsummary,vrows=diagnose('VAL_AUDIT_ONLY',vtop,vv,
        np.asarray([r['pid'] for r in valtxt]),np.asarray([r['pid'] for r in valim]),
        [r['path'] for r in valim],th)
    assert summary['n_queries']==60 and vsummary['n_queries']==72
    assert all(row['deployable_existing_same_pid_T2_count']==0 for row in vrows)
    assert summary['predicted_pairs']>=0 and vsummary['predicted_pairs']>=0
    # Complete-link non-transitivity protection: A-B and B-C connected, A-C not -> no triplet.
    u=np.asarray([[1,0],[.8,.6],[.28,.96]],dtype=np.float32)
    groups=complete_link_clusters(np.array([0,1,2]),u,.75)
    assert sorted(map(len,groups))==[1,2],groups
    with tempfile.TemporaryDirectory() as d:
        base=Path(d);s=base/'src';s.mkdir();st={'hello':1}
        out,yes=run_folder(base,s,st,False)
        assert not yes
        atomic_json(out/'experiment16_report.json',{'fit':details,'val':vsummary})
        atomic_csv(out/'val_top16_all_queries.csv',vrows,list(vrows[0]))
        (out/'irra_exp16_top16_text_support_feasibility.py').write_text('test',encoding='utf-8')
        z=package_output(out)
        with zipfile.ZipFile(z) as f:assert f.testzip() is None
        retried,yes=run_folder(base,s,st,True)
        assert yes and retried==out
        # Full no-checkpoint end-to-end smoke test. The synthetic baseline reference
        # is computed from its artificial labels and is NOT used by production.
        input_data={'features':(tx,v,tv,vv),'train_images':trainim,
            'train_texts':traintxt,'val_images':valim,'val_queries':valtxt}
        synth_r1=100*float(np.mean(np.asarray([z['pid'] for z in valim])[vtop[:,0]]==
                                   np.asarray([z['pid'] for z in valtxt])))
        end_to_end=compute_and_export(out,input_data,set(range(20)),set(range(20,30)),
                                     fit_limit=120,reference_r1=synth_r1)
        assert end_to_end['validation']['n_queries']==len(valtxt)
        assert (out/'top16_summary.csv').is_file()
        z=package_output(out)
        with zipfile.ZipFile(z) as f:
            assert f.testzip() is None
            assert 'experiment16_report.json' in f.namelist()
            assert 'caption_availability_and_leakage_audit.json' in f.namelist()
    print('SELF TEST PASSED: synthetic full pipeline, Top16 retrieval, fit-only threshold, PID-disjoint holdout, complete-link grouping, no Val caption T2, atomic CSV/JSON/ZIP, resume.')
    print('Note: actual EXP12 cache and Windows filesystem compatibility require a real run.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',default=str(ROOT))
    parser.add_argument('--exp12-dir',default='')
    parser.add_argument('--calibration-queries',type=int,default=1800)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    else:run_main(args)


if __name__=='__main__':
    main()
