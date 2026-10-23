# -*- coding: utf-8 -*-
"""实验13：实验12筛选器的真实同身份T2接受/拒绝与第三视角C效用诊断。

唯一研究问题：被实验12过滤掉的真实同PID、不同图像T2是否实际上能改善
第三视角C与固定困难负图像D之间的判别边际？接受组与拒绝组谁更有用？

无任何训练、无推理阶段PID选择、无属性词规则、无参数扫描；不写IRRA源码/权重。
严格复现实验12的 holdout 440 PID、支持T2抽样、gate和novelty选择、
第三视角选取、以T1确定D的规则。C只在筛选完成后用于事后诊断。
输入必须是本机完整实验12输出目录及其已有的最佳IRRA特征缓存。
"""
from __future__ import annotations

import os
# 避免NumPy的多份BLAS/OpenMP线程库竞争；本文件不导入torch。
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_var] = '1'

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(r'G:\PyCharmProjects\IRRA\IRRA-main')
PROJECT_FOLDER = '图文双向互补实验'
EXP12_PREFIX = '实验12_最佳IRRA同身份文本互补迁移_'
EXP13_PREFIX = '实验13_被拒绝互补文本第三视角效用_'
EXPECTED_BEST_SHA = 'f3ab12028e3406546b836ed6a75c8fc5e02a42146c5da13af423e7f5cf0e7ea6'
EXPECTED_ANNOTATION_SHA = 'fc3d15542f91dd70e23028c2cb40171a8f41734a39afddb9505e352e396cabf0'
VERSION = 'exp13_original_exp12_holdout_rebuild_v1'
LOG_PATH = None


def say(msg):
    print(msg, flush=True)
    if LOG_PATH is not None:
        with LOG_PATH.open('a', encoding='utf-8') as f:
            f.write(str(msg) + '\n')


def sha256_file(path, chunk=4 * 1024 * 1024):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def stable_hash(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.exp13_', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.exp13_', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def normalize(mat):
    arr = np.asarray(mat, dtype=np.float32)
    return arr / np.maximum(np.linalg.norm(arr, axis=-1, keepdims=True), 1e-12)


def source_exp12(folder, explicit=''):
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f'--exp12-dir not found: {p}')
        choices = [p]
    else:
        choices = sorted((d for d in folder.glob(EXP12_PREFIX + '*') if d.is_dir()), reverse=True)
    for p in choices:
        if all((p / x).is_file() for x in ('run_info.json', 'gate_model.json',
                                            'internal_teacher_diagnostic.json',
                                            'preparation_complete.json',
                                            'selected_train_pids.json', 'baseline_audit.json')):
            return p
    raise FileNotFoundError('No completed EXP12 directory. Specify --exp12-dir with its FULL experiment12 folder path. '
                            'EXP12 ZIP alone does not contain per-sample frozen-feature caches.')


def check_source(exp12, skip_big_sha=False):
    run = json.loads((exp12 / 'run_info.json').read_text(encoding='utf-8'))
    opt = SimpleNamespace(**run['settings'])
    gate = json.loads((exp12 / 'gate_model.json').read_text(encoding='utf-8'))
    prep = json.loads((exp12 / 'preparation_complete.json').read_text(encoding='utf-8'))
    old = json.loads((exp12 / 'internal_teacher_diagnostic.json').read_text(encoding='utf-8'))
    audit = json.loads((exp12 / 'baseline_audit.json').read_text(encoding='utf-8'))
    pids = json.loads((exp12 / 'selected_train_pids.json').read_text(encoding='utf-8'))
    if run['experiment'] != '实验12_最佳IRRA同身份文本互补迁移':
        raise ValueError('Input does not identify as the original EXP12 run')
    if pids != opt.selected_pids or len(pids) != 2200 or len(opt.holdout_pids) != 440:
        raise ValueError('EXP12 PID list is different/incomplete')
    if set(opt.fit_pids) & set(opt.holdout_pids):
        raise ValueError('EXP12 fit and holdout identity leakage')
    if prep.get('settings_hash') != opt.settings_hash or not prep.get('support_selection_ready'):
        raise ValueError('EXP12 preparation metadata inconsistent')
    if not audit.get('passed') or audit.get('checkpoint_sha256') != EXPECTED_BEST_SHA:
        raise ValueError('EXP12 best-IRRA baseline audit invalid')
    if not isinstance(gate.get('weights'), list) or len(gate['weights']) != 9:
        raise ValueError('EXP12 gate features do not match original 9-dimensional gate')
    if gate.get('label') != 'same-pid-vs-similarity-matched-wrong-pid':
        raise ValueError('EXP12 gate provenance differs from original')
    if opt.fp_train['checkpoint_sha256'] != EXPECTED_BEST_SHA or opt.fp_val['checkpoint_sha256'] != EXPECTED_BEST_SHA:
        raise ValueError('EXP12 used the wrong IRRA checkpoint')
    if opt.fp_train['annotations_sha256'] != EXPECTED_ANNOTATION_SHA:
        raise ValueError('EXP12 annotation SHA differs from verified original')
    for key in ('checkpoint', 'annotation_file', 'config_file'):
        if not Path(getattr(opt, key)).is_file():
            raise FileNotFoundError(f'EXP12 source file missing: {key}={getattr(opt,key)}')
    if not skip_big_sha:
        if sha256_file(opt.checkpoint) != EXPECTED_BEST_SHA:
            raise ValueError('Best checkpoint SHA mismatch: STOP')
        if sha256_file(opt.annotation_file) != EXPECTED_ANNOTATION_SHA:
            raise ValueError('Original annotation SHA mismatch: STOP')
        if sha256_file(opt.config_file) != opt.fp_train['config_sha256']:
            raise ValueError('Original config SHA mismatch: STOP')
    return opt, gate, old, audit


def read_train_annotations(ann, chosen_pids):
    """Recreate exact order from EXP12 read_annotations(train) -> get_subset()."""
    allow = set(int(k) for k in chosen_pids)
    rows = json.loads(Path(ann).read_text(encoding='utf-8'))
    if not isinstance(rows, list):
        raise ValueError('IRRA reid_raw.json must be a list')
    images, texts = [], []
    selected_img_ids, selected_text_ids = [], []
    for r in rows:
        if r.get('split') != 'train':
            continue
        pid = int(r['id'])
        if not r.get('captions') or not isinstance(r['captions'], list):
            raise ValueError('No captions in training image')
        ii = len(images)
        images.append({'pid': pid, 'file_path':r['file_path']})
        if pid in allow:
            selected_img_ids.append(ii)
        for cap in r['captions']:
            ti = len(texts)
            texts.append({'pid': pid, 'img_idx': ii, 'text': cap})
            if pid in allow:
                selected_text_ids.append(ti)
    if not images or not texts or not selected_img_ids or not selected_text_ids:
        raise ValueError('Training annotations or selected identities missing')
    selected_img = [images[i] for i in selected_img_ids]
    selected_txt = [texts[i] for i in selected_text_ids]
    mapping = {orig: k for k, orig in enumerate(selected_img_ids)}
    pos = np.asarray([mapping[t['img_idx']] for t in selected_txt], np.int32)
    if not all(selected_txt[i]['pid'] == selected_img[int(pos[i])]['pid'] for i in range(len(pos))):
        raise AssertionError('Training caption-to-image index mismatch')
    return images, texts, selected_img_ids, selected_text_ids, selected_img, selected_txt, pos


def check_cache(directory, prefix, fingerprint, count, dim=512):
    directory = Path(directory)
    f = directory / (prefix + '_features.npy')
    m = directory / (prefix + '_cache_meta.json')
    if not f.is_file() or not m.is_file():
        raise FileNotFoundError(f'EXP12 frozen-feature cache missing: {f} / {m}')
    meta = json.loads(m.read_text(encoding='utf-8'))
    arr = np.load(f, mmap_mode='r', allow_pickle=False)
    if (meta.get('fingerprint') != fingerprint or meta.get('done') != count or
            meta.get('shape') != [count, dim] or arr.shape != (count, dim) or
            arr.dtype != np.float32):
        raise ValueError(f'Incompatible EXP12 cache; no cross-model reuse allowed: {f}')
    return arr


def load_holdout(opt):
    all_images, all_texts, image_ids, text_ids, images, texts, pos = read_train_annotations(
        opt.annotation_file, opt.selected_pids)
    digest = hashlib.sha256(np.asarray(image_ids, dtype=np.int32).tobytes()).hexdigest()
    imgfp = {'train_base':opt.fp_train, 'selected_image_indices_sha256':digest,
             'selected_image_count':len(image_ids), 'format_version':1}
    tdir = opt.train_text_cache_dir or opt.run_dir
    idir = opt.train_image_cache_dir or opt.run_dir
    tx_mem = check_cache(tdir, 'train_text', opt.fp_train, len(all_texts))
    im_mem = check_cache(idir, 'train_image', imgfp, len(image_ids))
    tx = normalize(tx_mem[np.asarray(text_ids, dtype=np.int32)])
    im = normalize(im_mem)
    keep_t = np.asarray([i for i,t in enumerate(texts) if int(t['pid']) in set(opt.holdout_pids)], dtype=np.int32)
    keep_i = np.asarray([i for i,r in enumerate(images) if int(r['pid']) in set(opt.holdout_pids)], dtype=np.int32)
    loc = {int(orig):k for k,orig in enumerate(keep_i)}
    hpos = np.asarray([loc[int(pos[i])] for i in keep_t],dtype=np.int32)
    htxt = [texts[int(i)] for i in keep_t]
    himg = [images[int(i)] for i in keep_i]
    hx = normalize(tx[keep_t]); hv = normalize(im[keep_i])
    if not all(htxt[i]['pid'] == himg[hpos[i]]['pid'] for i in range(len(hpos))):
        raise AssertionError('Holdout caption-image PID mapping corrupted')
    return hx, hv, htxt, himg, hpos


# === Exact copy of the EXP12 selection logic, restricted to holdout ===
def construct_pairs(x, texts, pos, seed, wrong_pool=64):
    rng = np.random.default_rng(seed)
    by_pid = defaultdict(list)
    by_image = defaultdict(list)
    pids = np.asarray([int(t['pid']) for t in texts], dtype=np.int64)
    for i,t in enumerate(texts):
        by_pid[int(t['pid'])].append(i)
        by_image[int(pos[i])].append(i)
    supporters=np.full(len(texts),-1,dtype=np.int32)
    impostors=np.full(len(texts),-1,dtype=np.int32)
    errors=np.full(len(texts),np.nan,dtype=np.float32)
    for i,t in enumerate(texts):
        cross=[j for j in by_pid[int(t['pid'])] if int(pos[j])!=int(pos[i])]
        if not cross:continue
        supporters[i]=int(cross[int(rng.integers(0,len(cross)))])
        target=float(x[i]@x[supporters[i]])
        pool=rng.integers(0,len(x),size=wrong_pool*2)
        pool=pool[pids[pool]!=pids[i]][:wrong_pool]
        if len(pool)==0:raise RuntimeError('EXP12 wrong-PID draw failed')
        sim=x[pool]@x[i]
        z=int(np.argmin(np.abs(sim-target)))
        impostors[i]=int(pool[z])
        errors[i]=abs(float(sim[z])-target)
    ok=supporters>=0
    if np.any(pids[supporters[ok]]!=pids[ok]) or np.any(pos[supporters[ok]]==pos[ok]):
        raise AssertionError('EXP12 same-PID different-image support corrupted')
    if np.any(pids[impostors[ok]]==pids[ok]):
        raise AssertionError('EXP12 wrong-PID control corrupted')
    return supporters,impostors,errors,by_image


def evidence_features(x,v,pos,anchors,supporters):
    t=x[anchors];a=v[pos[anchors]];u=x[supporters];b=v[pos[supporters]]
    prod=lambda p,q:np.einsum('ij,ij->i',p,q,optimize=True)
    tt=prod(t,u);ta=prod(t,a);tb=prod(t,b)
    ua=prod(u,a);ub=prod(u,b);ab=prod(a,b)
    return np.stack([tt,ta,tb,ua,ub,ab,ub-ua,tb-ta,ub-tb],axis=1).astype(np.float32)


def gate_probability(features,gate):
    mu=np.asarray(gate['mean']);s=np.asarray(gate['scale']);w=np.asarray(gate['weights'])
    scores=np.clip(np.clip((np.asarray(features)-mu)/s,-10,10)@w+gate['bias'],-35,35)
    return (1./(1.+np.exp(-scores))).astype(np.float32)


def rebuild_holdout(x,v,texts,pos,gate,support_seed):
    supporter,wrong,errors,by_image=construct_pairs(x,texts,pos,support_seed+1)
    valid=np.flatnonzero(supporter>=0)
    p=evidence_features(x,v,pos,valid,supporter[valid])
    n=evidence_features(x,v,pos,valid,wrong[valid])
    prob=np.full(len(x),np.nan,np.float32)
    wrong_prob=np.full(len(x),np.nan,np.float32)
    prob[valid]=gate_probability(p,gate)
    wrong_prob[valid]=gate_probability(n,gate)
    novelty=np.full(len(x),np.nan,np.float32)
    novelty[valid]=1.-np.einsum('ij,ij->i',x[valid],x[supporter[valid]])
    sel=np.zeros(len(x),np.uint8)
    cut=float(gate['novelty_threshold'])
    sel[valid]=((prob[valid]>=float(gate['threshold']))&(novelty[valid]>cut)).astype(np.uint8)
    return {'support':supporter,'wrong':wrong,'matching_error':errors,
            'novelty':novelty,'positive_probability':prob,'wrong_probability':wrong_prob,
            'selection':sel,'valid_indices':valid}


def select_third_view(images,texts,pos,pack):
    images_by_pid=defaultdict(list)
    for i,r in enumerate(images):images_by_pid[int(r['pid'])].append(i)
    ids,third=[],[]
    for i in pack['valid_indices']:
        src=int(pos[i]);sec=int(pos[pack['support'][i]])
        rest=[j for j in images_by_pid[int(texts[i]['pid'])] if j!=src and j!=sec]
        if rest:
            ids.append(int(i));third.append(rest[0])
    return np.asarray(ids,np.int32),np.asarray(third,np.int32)


def teacher_pair_details(x,v,texts,images,pos,pack,beta):
    """All selection computed before C. Reproduce EXP12's fixed T1 hard-negative D."""
    idx,c=select_third_view(images,texts,pos,pack)
    if len(idx)==0:raise RuntimeError('No three-view cases; cannot do the requested experiment')
    a=x[idx]
    s=pack['support'][idx]
    combined=normalize((1-beta)*a+beta*x[s])
    accepted=pack['selection'][idx].astype(bool)
    gp=np.asarray([r['pid'] for r in images])
    qp=np.asarray([texts[i]['pid'] for i in idx])
    scores=a@v.T
    scores[qp[:,None]==gp[None,:]]=-np.inf
    neg=np.argmax(scores,axis=1)
    vc=v[c];vd=v[neg]
    baseline=np.einsum('ij,ij->i',a,vc)-np.einsum('ij,ij->i',a,vd)
    allmargin=np.einsum('ij,ij->i',combined,vc)-np.einsum('ij,ij->i',combined,vd)
    delta=allmargin-baseline
    # Evaluate Filtered as in EXP12: rejected T2 simply falls back to T1.
    filtered=np.where(accepted,allmargin,baseline)
    return {'idx':idx,'third':c,'hard_negative':neg,'accepted':accepted,
            'base_margin':baseline,'all_margin':allmargin,'filtered_margin':filtered,
            'potential_delta':delta,'gate_probability':pack['positive_probability'][idx],
            'novelty':pack['novelty'][idx], 'support':s,
            'matching_error':pack['matching_error'][idx]}


def group_statistics(details, texts, eps=1e-7):
    d=details['potential_delta'];ac=details['accepted']; base=details['base_margin']
    improved=d>eps; harmed=d < -eps; neutral=~(improved|harmed)
    wins0=base>0;wins1=details['all_margin']>0
    def make_group(mask,name):
        k=int(mask.sum())
        if k==0:raise RuntimeError('Empty '+name+' group; cannot compare acceptance and rejection')
        return {'group':name,'n':k,'fraction_of_eligible':float(k/len(d)),
                'potential_improve_rate':float(improved[mask].mean()),
                'potential_harm_rate':float(harmed[mask].mean()),
                'potential_neutral_rate':float(neutral[mask].mean()),
                'mean_potential_delta':float(d[mask].mean()),
                'median_potential_delta':float(np.median(d[mask])),
                'mean_abs_potential_delta':float(np.abs(d[mask]).mean()),
                't1_pair_win_rate':float(wins0[mask].mean()),
                't1_plus_t2_pair_win_rate':float(wins1[mask].mean()),
                'pair_corrected':int((~wins0[mask]&wins1[mask]).sum()),
                'pair_broken':int((wins0[mask]&~wins1[mask]).sum()),
                'mean_gate_probability':float(details['gate_probability'][mask].mean()),
                'mean_novelty_proxy':float(details['novelty'][mask].mean())}
    groups=[make_group(np.ones(len(d),dtype=bool),'all_eligible'),
            make_group(ac,'accepted'),make_group(~ac,'rejected')]
    harmful_count=int((ac&harmed).sum())
    missed_count=int((~ac&improved).sum())
    potential_useful=int(improved.sum());potential_harmful=int(harmed.sum())
    pid=np.asarray([texts[i]['pid'] for i in details['idx']]);cluster_count=len(np.unique(pid))
    summary={'n_eligible':int(len(d)),'distinct_train_holdout_pids':int(cluster_count),
             'accepted':int(ac.sum()),'rejected':int((~ac).sum()),
             'rejected_but_potentially_helpful':missed_count,
             'accepted_but_potentially_harmful':harmful_count,
             'rejected_share_among_potentially_helpful':float(missed_count/potential_useful) if potential_useful else None,
             'accepted_share_among_potentially_harmful':float(harmful_count/potential_harmful) if potential_harmful else None,
             'potential_helpful_count':potential_useful,'potential_harmful_count':potential_harmful,
             'threshold_for_benefit':eps,'groups':groups}
    return summary,pid


def bootstrap_pid(details,pid,reps=1500,seed=20261021):
    """Cluster bootstrap fixed diagnostic; does NOT tune gate/threshold."""
    ids=np.unique(pid)
    idxs=[np.flatnonzero(pid==k) for k in ids]
    rng=np.random.default_rng(seed)
    d=details['potential_delta'];a=details['accepted']
    eps=1e-7
    stat=[]
    for _ in range(reps):
        picked=rng.integers(0,len(ids),size=len(ids))
        idc=np.concatenate([idxs[j] for j in picked])
        ok=a[idc];dd=d[idc]
        if not ok.any() or ok.all():continue
        stat.append([float((dd[~ok]>eps).mean()),
                     float((dd[ok]>eps).mean()),
                     float(dd[ok].mean()-dd[~ok].mean()),
                     float((dd[~ok]<-eps).mean()),
                     float((dd[ok]<-eps).mean())])
    stat=np.asarray(stat)
    if len(stat)<reps*0.95:
        raise RuntimeError('Too many bootstrap runs with empty groups')
    keys=['rejected_helpful_fraction','accepted_helpful_fraction',
          'accepted_minus_rejected_mean_margin_delta',
          'rejected_harmful_fraction','accepted_harmful_fraction']
    return {'resamples':int(len(stat)),'cluster_count':len(ids),'percentile_95':{
        k:[float(np.quantile(stat[:,i],.025)),float(np.quantile(stat[:,i],.975))]
        for i,k in enumerate(keys)}}


def strict_reproduce(details,pack,prior,beta,atol=0.00003):
    """Check EXP12 published holdout teacher diagnostics, NOT source-only comparison."""
    d=prior.get('holdout_teacher_third_view') or {}
    if int(d.get('n_triple_view',-1))!=len(details['idx']):
        raise AssertionError(f'EXP12 triple-view count mismatch: current {len(details["idx"])} vs prior {d.get("n_triple_view")}')
    ac=pack['selection'][pack['valid_indices']]
    prior_accept=float(prior['holdout']['true_accept_rate_proxy'])
    if abs(float(ac.mean())-prior_accept)>atol:
        raise AssertionError('EXP12 full holdout acceptance does not reproduce')
    for key,vals in [('t1_only',details['base_margin']),
                     ('all_same_pid',details['all_margin']),
                     ('filtered_same_pid',details['filtered_margin'])]:
        pub=d['methods'][key]
        check={'third_view_pair_win_rate':float((vals>0).mean()),
               'mean_margin':float(vals.mean())}
        if key!='t1_only':
            dd=vals-details['base_margin']
            check.update({'margin_improved_fraction':float((dd>1e-7).mean()),
                          'margin_harmed_fraction':float((dd< -1e-7).mean()),
                          'mean_margin_delta':float(dd.mean())})
        for metric,now in check.items():
            expected=float(pub[metric])
            if abs(now-expected)>atol:
                raise AssertionError(f'EXP12 {key} {metric} mismatch: {now} vs {expected}; STOP, no mixed results')
    return {'matched':True,'exp12_triple_view_count':len(details['idx']),
            'exp12_holdout_support_accept_rate':float(ac.mean()),
            'tolerance':atol, 'methods_reproduced':list(d['methods'])}


def detailed_cases(details,texts,images,pos,eps=1e-7):
    rows=[]
    for j,i in enumerate(details['idx']):
        i=int(i); s=int(details['support'][j]); c=int(details['third'][j]); n=int(details['hard_negative'][j])
        delta=float(details['potential_delta'][j]);taken=bool(details['accepted'][j]);
        behavior='improved' if delta>eps else ('harmed' if delta < -eps else 'neutral')
        rows.append({'query_index_holdout':i,'pid':int(texts[i]['pid']),
                     'support_index_holdout':s,'accepted_by_gate':int(taken),
                     'potential_effect':behavior,
                     'gate_probability':float(details['gate_probability'][j]),
                     'novelty_proxy':float(details['novelty'][j]),
                     'wrong_text_match_error':float(details['matching_error'][j]),
                     't1_margin_C_vs_D':float(details['base_margin'][j]),
                     't1_t2_margin_C_vs_D':float(details['all_margin'][j]),
                     'potential_delta':delta,
                     't1_pair_correct':int(details['base_margin'][j]>0),
                     't1_t2_pair_correct':int(details['all_margin'][j]>0),
                     't1_caption':texts[i]['text'],
                     't2_caption':texts[s]['text'],
                     'image_A_path':images[int(pos[i])]['file_path'],
                     'image_B_path':images[int(pos[s])]['file_path'],
                     'image_C_path':images[c]['file_path'],
                     'fixed_negative_D_path':images[n]['file_path']})
    return rows


def review_cases(all_cases,max_cases=200):
    categories=[('rejected','improved'),('rejected','harmed'),('accepted','improved'),('accepted','harmed')]
    limits=[100,40,40,20]
    chosen=[];used=set()
    for (group,effect),limit in zip(categories,limits):
        subset=sorted((r for r in all_cases if
              ('accepted' if r['accepted_by_gate'] else 'rejected')==group and
              r['potential_effect']==effect),key=lambda r:(-abs(r['potential_delta']),r['query_index_holdout']))
        for r in subset[:limit]:
            idx=r['query_index_holdout']
            if idx not in used:chosen.append(r);used.add(idx)
    for r in sorted(all_cases,key=lambda row:-abs(row['potential_delta'])):
        if len(chosen)>=max_cases:break
        if r['query_index_holdout'] not in used:
            chosen.append(r);used.add(r['query_index_holdout'])
    return [{**r,'人工_T2是否真实新增':'','人工_信息可靠/有害/未知':'','人工_说明':''}
            for r in chosen[:max_cases]]


def zip_reports(folder):
    folder=Path(folder)
    include=[p for p in folder.iterdir() if p.is_file() and p.name!='发送给ChatGPT.zip'
             and p.suffix.lower() in {'.py','.csv','.json','.md','.log','.txt'}]
    target=folder/'发送给ChatGPT.zip'
    fd,temp=tempfile.mkstemp(prefix='.exp13_zip_',suffix='.zip',dir=str(folder));os.close(fd)
    try:
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as archive:
            for p in sorted(include):archive.write(p,arcname=p.name)
        with zipfile.ZipFile(temp) as archive:
            bad=archive.testzip()
            if bad:raise IOError('Bad ZIP entry: '+bad)
            if 'experiment13_report.json' not in archive.namelist():
                raise IOError('Missing EXP13 report inside ZIP')
        os.replace(temp,target)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    return target


def new_or_resume(folder,src,source_hash,resume):
    folder.mkdir(parents=True,exist_ok=True)
    if resume:
        for p in sorted(folder.glob(EXP13_PREFIX+'*'),reverse=True):
            m=p/'run_info.json'
            if not m.is_file():continue
            try:
                info=json.loads(m.read_text(encoding='utf-8'))
                if info.get('source_exp12')==str(src) and info.get('source_hash')==source_hash:
                    return p,True
            except (OSError,ValueError):pass
        raise RuntimeError('No matching experiment13 to resume. Clear --resume to create a new EXP13 directory.')
    name=EXP13_PREFIX+datetime.now().strftime('%Y%m%d_%H%M%S')
    p=folder/name;i=2
    while p.exists():p=folder/(name+'_'+str(i));i+=1
    p.mkdir()
    return p,False


def run_analysis(opt,gate,prior,src,out):
    say('Reading immutable best-IRRA train features and EXP12 holdout identities ...')
    x,v,tx,im,pos=load_holdout(opt)
    if len(tx)!=int(json.loads((src/'preparation_complete.json').read_text(encoding='utf-8'))['holdout_samples']):
        raise AssertionError('EXP12 holdout sample count differs from cache reconstruction')
    say(f'Holdout captions={len(tx)}, gallery={len(im)}, PID={len({t["pid"] for t in tx})}')
    pack=rebuild_holdout(x,v,tx,pos,gate,int(opt.support_seed))
    say(f'EXP12 identical T2 selection reconstructed, {len(pack["valid_indices"])} eligible pairs')
    details=teacher_pair_details(x,v,tx,im,pos,pack,float(opt.teacher_beta))
    reproduced=strict_reproduce(details,pack,prior,float(opt.teacher_beta))
    say('EXP12 teacher diagnostics reproduced exactly within tolerance.')
    summary,pid=group_statistics(details,tx)
    uncertainty=bootstrap_pid(details,pid)
    rows=detailed_cases(details,tx,im,pos)
    review=review_cases(rows)
    group_cols=list(summary['groups'][0]);atomic_csv(out/'group_summary.csv',summary['groups'],group_cols)
    atomic_csv(out/'all_third_view_cases.csv',rows,list(rows[0]))
    atomic_csv(out/'人工核查案例.csv',review,list(review[0]))
    report={'experiment':'13','scope':'EXP12 internal holdout teacher selection audit ONLY',
            'exp12_path':str(src),'verified_checkpoint_sha256':EXPECTED_BEST_SHA,
            'exp12_settings_hash':opt.settings_hash,'reproduced_exp12':reproduced,
            'summary':summary,'pid_cluster_bootstrap':uncertainty,
            'limitations':[
                'Third-view C and the true TRAIN PID are used for POST-HOC diagnosis only; not for T2 selection or deployment',
                'Useful/harmful means the sign of the third-view pairwise margin delta against a fixed T1-mined negative D; not manually verified semantic correctness',
                'Only samples with A,B,C distinct images are eligible; results do not represent all training captions or Val',
                'EXP12 same-PID gate is not a genuine same-person factual-harm classifier and text-similar wrong-PID controls were weakly matched',
                'Two captions from the same PID and multiple PID samples are dependent; cluster bootstrap is exploratory',
                'This is teacher-side analysis; no new student training and no new retrieval R1 gain is claimed',
                'Do not tune EXP12 selector using this C diagnostic without an independent PID-separated evaluation' ]}
    atomic_json(out/'experiment13_report.json',report)
    text=(
        '# 实验13：实验12筛选器接受/拒绝与第三视角真实收益\n\n'
        '主问题：被拒绝的同PID真实T2，是否在第三张同身份图像C上其实有帮助？\n\n'
        '完全复用实验12的440个Train holdout PID、同PID不同图像T2与固定种子、筛选器和阈值、'
        'T1选定的固定负样本D、融合系数beta。C只用于选择后的事后评价。\n\n'
        '潜在收益 = [(T1+T2)·C - (T1+T2)·D] - [T1·C - T1·D]，各融合特征均归一化。\n\n'
        '分组报告接受组/拒绝组中的边际改善、恶化、零变化、转正、转错，并按PID簇bootstrap。\n'
        '有益/有害的定义是固定D上的判别边际变化，不等于T2的语义事实正确/错误。\n'
        '没有训练新模型，不改变原IRRA与实验12筛选器，不检索Val/Test，不做阈值调参。\n'
        '本研究诊断“漏掉有益文本”的潜在问题，不直接证明新检索方法R1提升。\n')
    (out/'实验13_目标与边界.md').write_text(text,encoding='utf-8')
    say('EXP13 groups: '+json.dumps(summary['groups'],ensure_ascii=False))
    say(f'Rejected but posthoc beneficial: {summary["rejected_but_potentially_helpful"]}; '
        f'accepted but posthoc harmful: {summary["accepted_but_potentially_harmful"]}')
    return report


def synthetic_self_test():
    """No checkpoint or project needed; exercise exact pair logic and end-to-end file export."""
    rng=np.random.default_rng(20261021)
    # 4 IDs x 3 views x 2 captions: all can have an independent C.
    ims,tx,features,views,pos=[],[],[],[],[]
    for pid in range(4):
        latent=normalize(rng.normal(size=(1,32)))[0]
        for vi in range(3):
            ii=len(ims)
            ims.append({'pid':pid,'file_path':f'P{pid}_V{vi}.jpg'})
            vv=normalize((latent + .35*rng.normal(size=32))[None,:])[0]
            views.append(vv)
            for cap in range(2):
                z=normalize((latent+.5*rng.normal(size=32))[None,:])[0]
                features.append(z)
                tx.append({'pid':pid,'img_idx':ii,'text':f'pid {pid} view {vi} caption {cap}'})
                pos.append(ii)
    x=np.asarray(features,np.float32);v=np.asarray(views,np.float32);pos=np.asarray(pos,np.int32)
    gate={'mean':[0.]*9,'scale':[1.]*9,'weights':[0.]*9,'bias':0.,
          'threshold':0.5,'novelty_threshold':0.1}
    p=rebuild_holdout(x,v,tx,pos,gate,1212)
    # 在模拟环境故意产生接受/拒绝两组；不是正式实验的阈值。
    gate['novelty_threshold']=float(np.median(p['novelty'][p['valid_indices']]))
    p=rebuild_holdout(x,v,tx,pos,gate,1212)
    detail=teacher_pair_details(x,v,tx,ims,pos,p,.25)
    assert len(detail['idx'])==24
    assert np.all(detail['accepted']==(detail['novelty']>gate['novelty_threshold']))
    summary,pids=group_statistics(detail,tx)
    assert summary['accepted']+summary['rejected']==24
    assert np.max(np.abs(detail['filtered_margin']-np.where(detail['accepted'],detail['all_margin'],detail['base_margin'])))<1e-6
    # Changing ONLY C must not alter acceptance (C is not an input to the gate).
    old=p['selection'].copy()
    c2=v.copy();c2[detail['third']]*=-1
    detail2=teacher_pair_details(x,c2,tx,ims,pos,p,.25)
    assert np.array_equal(p['selection'],old) and np.array_equal(detail['accepted'],detail2['accepted'])
    with tempfile.TemporaryDirectory() as td:
        out=Path(td)
        rows=detailed_cases(detail,tx,ims,pos)
        atomic_csv(out/'all_third_view_cases.csv',rows,list(rows[0]))
        atomic_csv(out/'group_summary.csv',summary['groups'],list(summary['groups'][0]))
        atomic_json(out/'experiment13_report.json',{'ok':True})
        (out/'experiment13_source.py').write_text('# synthetic',encoding='utf-8')
        z=zip_reports(out)
        with zipfile.ZipFile(z) as f:
            assert {'experiment13_report.json','all_third_view_cases.csv','group_summary.csv'} <= set(f.namelist())
    say('SELF-TEST PASS: samePID different-view matching, gate first / C evaluation after, '
        'pairwise utility, accepted/rejected statistics, CSV/ZIP, atomic output. '
        'Real EXP12 cache and Windows paths still need local execution.')


def main():
    global LOG_PATH
    ap=argparse.ArgumentParser(description='EXP13: audit accepted vs rejected TRUE same-PID T2 using third-view C')
    ap.add_argument('--project-root',default=str(PROJECT_ROOT),help='IRRA-main root')
    ap.add_argument('--exp12-dir',default='',help='Full original experiment12 output directory; optional auto-discovery')
    ap.add_argument('--resume',action='store_true',help='Resume an existing same-source EXP13 output')
    ap.add_argument('--self-test',action='store_true',help='Synthetic self-test; no checkpoint loaded')
    args=ap.parse_args()
    if args.self_test:
        synthetic_self_test();return
    root=Path(args.project_root).resolve()
    folder=root/PROJECT_FOLDER
    if not folder.is_dir():
        raise FileNotFoundError('Experiment folder missing: '+str(folder))
    src=source_exp12(folder,args.exp12_dir)
    say('Using original EXP12: '+str(src))
    opt,gate,prior,audit=check_source(src)
    # Immutable identity of all experiment12 teacher inputs; not the EXP13 result itself.
    src_hash=stable_hash({'exp12_path':str(src),'settings_hash':opt.settings_hash,
             'gate_model_sha256':sha256_file(src/'gate_model.json'),
             'internal_diagnostic_sha256':sha256_file(src/'internal_teacher_diagnostic.json'),
             'selected_pids_sha256':sha256_file(src/'selected_train_pids.json'),
             'source_train_fp':opt.fp_train})
    out,resumed=new_or_resume(folder,src,src_hash,args.resume)
    LOG_PATH=out/'experiment_run.log'
    atomic_json(out/'run_info.json',{'experiment':'13','version':VERSION,
                 'source_exp12':str(src),'source_hash':src_hash,
                 'best_checkpoint_sha256':EXPECTED_BEST_SHA,
                 'created_or_resumed':datetime.now().isoformat(timespec='seconds')})
    shutil.copy2(Path(__file__).resolve(),out/Path(__file__).name)
    say('Experiment13 output: '+str(out))
    try:
        report_path=out/'experiment13_report.json'
        if resumed and report_path.is_file() and (out/'all_third_view_cases.csv').is_file() and (out/'group_summary.csv').is_file() and (out/'人工核查案例.csv').is_file():
            report=json.loads(report_path.read_text(encoding='utf-8'))
            if report.get('exp12_settings_hash')==opt.settings_hash and report.get('reproduced_exp12',{}).get('matched'):
                say('Verified completed EXP13 report present; skip recomputation.')
            else:run_analysis(opt,gate,prior,src,out)
        else:run_analysis(opt,gate,prior,src,out)
    finally:
        if (out/'experiment13_report.json').is_file():
            p=zip_reports(out)
            say('Analysis ZIP: '+str(p))
    say('EXP13 completed; IRRA and all EXP12 data remain unchanged.')


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
