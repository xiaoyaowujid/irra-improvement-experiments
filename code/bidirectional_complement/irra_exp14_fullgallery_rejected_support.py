# -*- coding: utf-8 -*-
"""实验14：被拒绝的同身份T2，是否真的改善第三视角的全图库排序？

仅复用实验12的最佳冻结IRRA特征、440个内部未见训练PID、原T2抽样和筛选器。
严格复现原2020条第三视角C对固定负图D的实验12教师统计之后，
改为用全套内部未见训练身份图库（排除本次的A和B两张图像）重新计算排名。
所有与A/B不同的同PID图像，包括C，均算作正样本；所有不同PID图像为负样本。
对原筛选器已接受/拒绝的两组做T1→All潜在改进与损害比较。
C及真实PID仅用于训练身份内部的事后诊断，不参与选择或超参数拟合。

不训练，不修改IRRA，绝不把训练身份上的教师排名称为Validation R1。
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

PROJECT_ROOT = Path(r'/')
PROJECT_FOLDER = '图文双向互补实验'
EXP12_PREFIX = '实验12_最佳IRRA同身份文本互补迁移_'
EXP14_PREFIX = '实验14_被拒绝文本完整图库效用_'
EXPECTED_BEST_SHA = 'f3ab12028e3406546b836ed6a75c8fc5e02a42146c5da13af423e7f5cf0e7ea6'
EXPECTED_ANNOTATION_SHA = 'fc3d15542f91dd70e23028c2cb40171a8f41734a39afddb9505e352e396cabf0'
VERSION = 'exp14_full_holdout_gallery_no_AB_leak_v1'
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
    fd, tmp = tempfile.mkstemp(prefix='.exp14_', suffix='.tmp', dir=str(path.parent))
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
    fd, tmp = tempfile.mkstemp(prefix='.exp14_', suffix='.tmp', dir=str(path.parent))
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



# ─────────── EXP12 exact numeric-replay guard (copied verbatim in semantics) ───────────
def strict_reproduce(details, pack, prior, beta, atol=0.00003):
    d = prior.get('holdout_teacher_third_view') or {}
    if int(d.get('n_triple_view', -1)) != len(details['idx']):
        raise AssertionError('EXP12 three-view count mismatch; STOP')
    valid_ac = pack['selection'][pack['valid_indices']]
    if abs(float(valid_ac.mean()) - float(prior['holdout']['true_accept_rate_proxy'])) > atol:
        raise AssertionError('EXP12 full holdout gate decision ratio changed; STOP')
    for name, vals in [('t1_only', details['base_margin']),
                       ('all_same_pid', details['all_margin']),
                       ('filtered_same_pid', details['filtered_margin'])]:
        saved = d['methods'][name]
        cur = {'third_view_pair_win_rate': float(np.mean(vals > 0)),
               'mean_margin': float(vals.mean())}
        if name != 't1_only':
            diff = vals - details['base_margin']
            cur.update({'margin_improved_fraction': float(np.mean(diff > 1e-7)),
                        'margin_harmed_fraction': float(np.mean(diff < -1e-7)),
                        'mean_margin_delta': float(diff.mean())})
        for k, now in cur.items():
            if abs(float(saved[k]) - now) > atol:
                raise AssertionError('EXP12 original {} {} differs: {} vs {}; STOP'.format(name, k, now, saved[k]))
    return {'matched': True, 'three_view_count': len(details['idx']),
            'source_exp12_methods': ['t1_only', 'all_same_pid', 'filtered_same_pid'],
            'tolerance': atol}


# ─────────── Full-gallery ranking, never select T2 from target C or ranking ───────────
METHODS = ('t1_only', 'all_same_pid', 'filtered_same_pid')
EPS = 1e-7
EXPECTED_THIRD_VIEW_COUNT = 2020
EXPECTED_ACCEPTED_COUNT = 714
EXPECTED_ELIGIBLE_PID_COUNT = 271


def evaluate_full_gallery(features, gallery, query_pid, gallery_pid,
                          image_a, image_b, image_c):
    """Each query ranks all holdout images EXCEPT its source A and support source B.

    Every OTHER same-PID image is relevant. C stays in the gallery and its
    absolute rank is reported separately. All wrong-PID images are candidates.
    The exclusions are identical for all three methods and use support provenance,
    NOT search rankings or val/test annotations.
    """
    n, dim = features.shape
    g = gallery.shape[0]
    if gallery.shape[1] != dim:
        raise ValueError('Feature dimension mismatch')
    if not (len(query_pid) == len(image_a) == len(image_b) == len(image_c) == n):
        raise ValueError('Query lengths differ')
    if g < 4 or n == 0:
        raise ValueError('Insufficient samples for full gallery')
    if np.any((image_a == image_b) | (image_a == image_c) | (image_b == image_c)):
        raise AssertionError('A, B, C MUST be distinct images')
    if np.any((image_a < 0) | (image_b < 0) | (image_c < 0)) or np.any(
            (image_a >= g) | (image_b >= g) | (image_c >= g)):
        raise AssertionError('Invalid gallery image index')
    if np.any(gallery_pid[image_a] != query_pid) or np.any(
            gallery_pid[image_b] != query_pid) or np.any(gallery_pid[image_c] != query_pid):
        raise AssertionError('Anchor/support/third PID inconsistent')
    # Scores are computed without PID. PID is used only for removing A/B and
    # independent post-hoc evaluation labels, never for fusing/selecting T2.
    scores = (features @ gallery.T).astype(np.float32)
    scores[np.arange(n), image_a] = -np.inf
    scores[np.arange(n), image_b] = -np.inf
    # Sorting stable ensures deterministic tie handling across three methods.
    order = np.argsort(-scores, axis=1, kind='stable')
    first = np.empty(n, np.int32)
    ap = np.empty(n, np.float64)
    inp = np.empty(n, np.float64)
    c_rank = np.empty(n, np.int32)
    c_vs_wrong_rank = np.empty(n, np.int32)
    pcount = np.empty(n, np.int32)
    top1 = np.empty(n, np.int32)
    for k in range(n):
        ranks = order[k]
        remaining = ranks[(ranks != image_a[k]) & (ranks != image_b[k])]
        if remaining.size != g-2:
            raise AssertionError('Did not remove exactly A and B')
        hit = (gallery_pid[remaining] == query_pid[k])
        positive_pos = np.flatnonzero(hit)
        if positive_pos.size == 0:
            raise AssertionError('No legitimate third-image positive after A/B removal')
        if np.count_nonzero(~hit) == 0:
            raise AssertionError('No wrong-PID competitor')
        c_position = np.flatnonzero(remaining == image_c[k])
        if len(c_position) != 1:
            raise AssertionError('C is missing/duplicated in ranking')
        cpos = int(c_position[0])
        first[k] = positive_pos[0]+1
        pcount[k] = len(positive_pos)
        c_rank[k] = cpos+1
        # Rank C if every other same-PID image (except A/B) were ignored.
        c_vs_wrong_rank[k] = 1+int(np.count_nonzero(~hit[:cpos]))
        r = positive_pos + 1
        ap[k] = float(np.mean(np.arange(1, len(r)+1, dtype=np.float64)/r))
        inp[k] = float(len(r) / r[-1])
        top1[k] = int(remaining[0])
    return {'rank': first, 'ap': ap, 'inp': inp, 'c_rank': c_rank,
            'c_vs_wrong_rank': c_vs_wrong_rank, 'positive_count': pcount,
            'top1_gallery': top1}


def full_gallery_results(x, v, texts, images, pos, pack, details, beta):
    idx = details['idx']
    t = x[idx]
    support = details['support']
    source_a = pos[idx]
    source_b = pos[support]
    c = details['third']
    accepted = details['accepted']
    if np.any(source_a == source_b):
        raise AssertionError('Support T2 must come from different image B')
    qt = np.asarray([int(texts[int(i)]['pid']) for i in idx], np.int64)
    gi = np.asarray([int(r['pid']) for r in images], np.int64)
    combined = normalize((1.-beta)*t+beta*x[support])
    selected = np.array(t, copy=True)
    selected[accepted] = combined[accepted]
    out = {}
    for name, f in [('t1_only',t), ('all_same_pid',combined),
                    ('filtered_same_pid',selected)]:
        out[name] = evaluate_full_gallery(f, v, qt, gi, source_a, source_b, c)
    for key in ('rank','ap','inp','c_rank','c_vs_wrong_rank'):
        if not np.array_equal(out['filtered_same_pid'][key][~accepted],
                              out['t1_only'][key][~accepted]):
            raise AssertionError('Rejected group MUST be untouched by Filtered teacher')
        if not np.array_equal(out['filtered_same_pid'][key][accepted],
                              out['all_same_pid'][key][accepted]):
            raise AssertionError('Accepted group MUST equal All teacher')
    if not np.array_equal(out['t1_only']['positive_count'],out['all_same_pid']['positive_count']):
        raise AssertionError('Methods did not use identical candidate sets')
    return out


def percent_metrics(values, mask):
    if not np.any(mask):
        raise AssertionError('Empty method/group')
    rank=values['rank'][mask]
    return {'R1_percent':float(100*np.mean(rank<=1)),
            'R5_percent':float(100*np.mean(rank<=5)),
            'R10_percent':float(100*np.mean(rank<=10)),
            'mAP_percent':float(100*np.mean(values['ap'][mask])),
            'mINP_percent':float(100*np.mean(values['inp'][mask])),
            'mean_first_positive_rank':float(np.mean(rank)),
            'median_first_positive_rank':float(np.median(rank)),
            'mean_C_absolute_rank':float(np.mean(values['c_rank'][mask])),
            'mean_C_rank_among_wrong_PID':float(np.mean(values['c_vs_wrong_rank'][mask]))}


def grouped_results(out, details, texts):
    accepted = details['accepted']
    all_case = np.ones(len(accepted), bool)
    groups = [('all_eligible', all_case), ('accepted',accepted),('rejected',~accepted)]
    main = out['t1_only'];allm = out['all_same_pid'];fil = out['filtered_same_pid']
    pid=np.asarray([int(texts[int(i)]['pid']) for i in details['idx']], np.int64)
    rows=[]
    for group, mask in groups:
        dr=allm['rank'][mask]-main['rank'][mask]
        da=allm['ap'][mask]-main['ap'][mask]
        dc=allm['c_rank'][mask]-main['c_rank'][mask]
        dh=allm['c_vs_wrong_rank'][mask]-main['c_vs_wrong_rank'][mask]
        for method in METHODS:
            row={'group':group,'method':method,'n_queries':int(mask.sum()),
                 'distinct_pid':int(len(np.unique(pid[mask]))),
                 'all_vs_T1_improved_AP_rate':float(np.mean(da>EPS)),
                 'all_vs_T1_harmed_AP_rate':float(np.mean(da< -EPS)),
                 'all_vs_T1_unchanged_AP_rate':float(np.mean(np.abs(da)<=EPS)),
                 'all_vs_T1_mean_delta_AP_pp':float(np.mean(da)*100),
                 'all_vs_T1_better_best_positive_rank_rate':float(np.mean(dr<0)),
                 'all_vs_T1_worse_best_positive_rank_rate':float(np.mean(dr>0)),
                 'all_vs_T1_better_C_absolute_rank_rate':float(np.mean(dc<0)),
                 'all_vs_T1_worse_C_absolute_rank_rate':float(np.mean(dc>0)),
                 'all_vs_T1_better_C_vs_wrong_rank_rate':float(np.mean(dh<0)),
                 'all_vs_T1_worse_C_vs_wrong_rank_rate':float(np.mean(dh>0))}
            row.update(percent_metrics(out[method],mask))
            rows.append(row)
    pbenefit = details['potential_delta']>EPS
    full_benefit = (allm['ap']-main['ap'])>EPS
    p_harm = details['potential_delta']< -EPS
    full_harm = (allm['ap']-main['ap'])< -EPS
    cross={
        'pair_margin_improved_but_full_gallery_AP_not_improved':int((pbenefit&~full_benefit).sum()),
        'pair_margin_harmed_but_full_gallery_AP_improved':int((p_harm&full_benefit).sum()),
        'pair_margin_improved_and_full_gallery_AP_improved':int((pbenefit&full_benefit).sum()),
        'pair_margin_harmed_and_full_gallery_AP_harmed':int((p_harm&full_harm).sum()),
        'rejected_pair_benefit_and_full_gallery_AP_benefit':int((~accepted&pbenefit&full_benefit).sum()),
        'rejected_pair_benefit_but_full_gallery_AP_not_benefit':int((~accepted&pbenefit&~full_benefit).sum()),
        'rejected_full_gallery_AP_benefit':int((~accepted&full_benefit).sum()),
        'rejected_full_gallery_AP_harm':int((~accepted&full_harm).sum()),
        'accepted_full_gallery_AP_benefit':int((accepted&full_benefit).sum()),
        'accepted_full_gallery_AP_harm':int((accepted&full_harm).sum()),
    }
    return rows, cross, pid


def cluster_bootstrap(out, details, pids, repeats=1200, seed=20261021):
    """Exploratory PID-cluster CI; no fitting, threshold changes or method choice."""
    unique=np.unique(pids)
    chunks=[np.flatnonzero(pids==pid) for pid in unique]
    rng=np.random.default_rng(seed)
    base=out['t1_only'];allv=out['all_same_pid'];fil=out['filtered_same_pid']
    da=100*(allv['ap']-base['ap'])
    diff_r1=(allv['rank']==1).astype(np.float32)-(base['rank']==1).astype(np.float32)
    fda=100*(fil['ap']-base['ap'])
    ac=details['accepted']
    samples=[]
    for _ in range(repeats):
        draw=rng.integers(0,len(unique),size=len(unique))
        indices=np.concatenate([chunks[k] for k in draw])
        acc=ac[indices]
        if acc.all() or not acc.any():
            continue
        samples.append([float(da[indices].mean()),float(da[indices][acc].mean()),
                        float(da[indices][~acc].mean()),
                        float(100*diff_r1[indices].mean()),float(fda[indices].mean()),
                        float(da[indices][acc].mean()-da[indices][~acc].mean())])
    mat=np.asarray(samples,dtype=np.float64)
    if len(mat)<repeats*.95:raise RuntimeError('Too many empty bootstrap groups')
    names=['all_minus_t1_mAP_pp','accepted_all_minus_t1_mAP_pp',
           'rejected_all_minus_t1_mAP_pp','all_minus_t1_R1_pp',
           'filtered_minus_t1_mAP_pp','accepted_minus_rejected_delta_AP_pp']
    return {'cluster_unit':'train PID','resamples':len(mat),'distinct_pids':len(unique),
            'exploratory_95pct_percentile_CI':{
                name:[float(np.quantile(mat[:,i],.025)),float(np.quantile(mat[:,i],.975))]
                for i,name in enumerate(names)}}


def cases_full_gallery(out, details, texts, images, pos):
    rows=[]
    main=out['t1_only'];allv=out['all_same_pid'];fil=out['filtered_same_pid']
    for k,qindex in enumerate(details['idx']):
        qi=int(qindex);s=int(details['support'][k]);c=int(details['third'][k]);
        a=int(pos[qi]);b=int(pos[s]);d=int(details['hard_negative'][k]);
        delta=float(allv['ap'][k]-main['ap'][k]);
        rows.append({'query_index_holdout':qi,'pid':int(texts[qi]['pid']),
            'support_index_holdout':s,'accepted_by_exp12_gate':int(details['accepted'][k]),
            'exp12_gate_probability':float(details['gate_probability'][k]),
            'exp12_novelty_proxy':float(details['novelty'][k]),
            'exp13_pair_delta_C_vs_D':float(details['potential_delta'][k]),
            'exp13_pair_effect':'positive' if details['potential_delta'][k]>EPS else
                    ('negative' if details['potential_delta'][k]<-EPS else 'neutral'),
            'full_gallery_delta_AP':delta,
            'full_gallery_effect':'positive' if delta>EPS else ('negative' if delta< -EPS else 'neutral'),
            'full_gallery_delta_R1_correct':int(allv['rank'][k]==1)-int(main['rank'][k]==1),
            'full_gallery_delta_C_absolute_rank':int(allv['c_rank'][k]-main['c_rank'][k]),
            'full_gallery_delta_C_vs_wrong_rank':int(allv['c_vs_wrong_rank'][k]-main['c_vs_wrong_rank'][k]),
            'positive_gallery_count_after_AB_excluded':int(main['positive_count'][k]),
            'T1_rank':int(main['rank'][k]),'All_rank':int(allv['rank'][k]),
            'Filtered_rank':int(fil['rank'][k]),
            'T1_AP':float(main['ap'][k]),'All_AP':float(allv['ap'][k]),
            'Filtered_AP':float(fil['ap'][k]),
            'T1_C_rank':int(main['c_rank'][k]),'All_C_rank':int(allv['c_rank'][k]),
            'Filtered_C_rank':int(fil['c_rank'][k]),
            'T1_C_vs_wrong_rank':int(main['c_vs_wrong_rank'][k]),
            'All_C_vs_wrong_rank':int(allv['c_vs_wrong_rank'][k]),
            'Filtered_C_vs_wrong_rank':int(fil['c_vs_wrong_rank'][k]),
            'A_image_file':images[a]['file_path'],'B_image_file':images[b]['file_path'],
            'C_image_file':images[c]['file_path'],'D_exp13_negative_image_file':images[d]['file_path'],
            'T1_text':texts[qi]['text'],'T2_text':texts[s]['text']})
    return rows


def choose_review_cases(cases, limit=200):
    # Merely sample representative discrepancies; do not tune model from here.
    accepted=lambda r:bool(r['accepted_by_exp12_gate'])
    definitions=[
        (lambda r:not accepted(r) and r['full_gallery_effect']=='positive',70),
        (lambda r:not accepted(r) and r['full_gallery_effect']=='negative',50),
        (lambda r:accepted(r) and r['full_gallery_effect']=='positive',35),
        (lambda r:accepted(r) and r['full_gallery_effect']=='negative',25),
        (lambda r:r['exp13_pair_effect']!=r['full_gallery_effect'],20)]
    chosen=[];seen=set()
    for criterion,num in definitions:
        matches=sorted((r for r in cases if criterion(r)),
                       key=lambda r:(-abs(r['full_gallery_delta_AP']),r['query_index_holdout']))
        for r in matches[:num]:
            if r['query_index_holdout'] not in seen:
                chosen.append(r);seen.add(r['query_index_holdout'])
    for r in sorted(cases,key=lambda x:-abs(x['full_gallery_delta_AP'])):
        if len(chosen)>=limit:break
        if r['query_index_holdout'] not in seen:
            chosen.append(r);seen.add(r['query_index_holdout'])
    return [{**r,'人工_是否新增语义':'','人工_信息可靠有害未知':'','人工_备注':''}
            for r in chosen[:limit]]


def package_reports(out):
    target=out/'发送给ChatGPT.zip'
    files=[p for p in out.iterdir() if p.is_file() and p.name!=target.name and
           p.suffix.lower() in ('.csv','.json','.md','.log','.py','.txt')]
    fd,tmp=tempfile.mkstemp(prefix='.exp14_zip_',suffix='.zip',dir=str(out));os.close(fd)
    try:
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
            for p in sorted(files):z.write(p,arcname=p.name)
        with zipfile.ZipFile(tmp,'r') as z:
            if z.testzip() is not None:raise IOError('ZIP CRC error')
            required={'experiment14_report.json','all_fullgallery_cases.csv',
                      'group_summary.csv','full_gallery_metrics.csv'}
            if not required<=set(z.namelist()):raise IOError('Missing ZIP report components')
        os.replace(tmp,target)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return target


def output_directory(parent,src,identity,resume):
    parent.mkdir(parents=True,exist_ok=True)
    if resume:
        for folder in sorted(parent.glob(EXP14_PREFIX+'*'),reverse=True):
            p=folder/'run_info.json'
            if not p.is_file():continue
            try:
                info=json.loads(p.read_text(encoding='utf-8'))
                if info.get('source_exp12')==str(src) and info.get('source_identity')==identity:
                    return folder,True
            except (OSError,ValueError,KeyError):continue
        raise RuntimeError('No matching EXP14 to resume. First run without --resume, or specify original --exp12-dir.')
    p=parent/(EXP14_PREFIX+datetime.now().strftime('%Y%m%d_%H%M%S'));i=2
    while p.exists():p=parent/(p.name+'_'+str(i));i+=1
    p.mkdir(parents=False)
    return p,False


def run_exp14(opt,gate,prior,src,out):
    say('Read previously cached best-IRRA TRAIN features; no model inference and no training.')
    x,v,tx,im,pos=load_holdout(opt)
    meta=json.loads((src/'preparation_complete.json').read_text(encoding='utf-8'))
    if len(tx)!=int(meta['holdout_samples']):
        raise AssertionError('Holdout text count differs from original EXP12 preparation')
    if len(set(t['pid'] for t in tx))!=len(opt.holdout_pids):
        raise AssertionError('Holdout PID count mismatch')
    p=rebuild_holdout(x,v,tx,pos,gate,int(opt.support_seed))
    d=teacher_pair_details(x,v,tx,im,pos,p,float(opt.teacher_beta))
    replay=strict_reproduce(d,p,prior,float(opt.teacher_beta))
    if len(d['idx'])!=EXPECTED_THIRD_VIEW_COUNT:
        raise AssertionError('Expected the original 2020 three-view cases; got {}'.format(len(d['idx'])))
    if int(d['accepted'].sum()) != EXPECTED_ACCEPTED_COUNT:
        raise AssertionError('EXP13 original accepted T2 count mismatch; STOP')
    eligible_pids={int(tx[int(i)]['pid']) for i in d['idx']}
    if len(eligible_pids) != EXPECTED_ELIGIBLE_PID_COUNT:
        raise AssertionError('EXP13 original eligible distinct PID count mismatch; STOP')
    say('Original EXP12/EXP13 teacher reproduced: {} cases, {} accepted, {} PIDs.'.format(
        len(d['idx']),int(d['accepted'].sum()),len(eligible_pids)))
    out_metrics=full_gallery_results(x,v,tx,im,pos,p,d,float(opt.teacher_beta))
    group_rows,cross,pid=grouped_results(out_metrics,d,tx)
    bootstrap=cluster_bootstrap(out_metrics,d,pid)
    case_rows=cases_full_gallery(out_metrics,d,tx,im,pos)
    review_rows=choose_review_cases(case_rows)
    overall=[r for r in group_rows if r['group']=='all_eligible']
    atomic_csv(out/'group_summary.csv',group_rows,list(group_rows[0]))
    atomic_csv(out/'full_gallery_metrics.csv',overall,list(overall[0]))
    atomic_csv(out/'all_fullgallery_cases.csv',case_rows,list(case_rows[0]))
    discordant=[r for r in case_rows if r['exp13_pair_effect']!=r['full_gallery_effect']]
    atomic_csv(out/'pair_vs_fullgallery_disagreements.csv',discordant,
               list(case_rows[0]))
    atomic_csv(out/'人工核查案例.csv',review_rows,list(review_rows[0]))
    report={
        'experiment':'14', 'scope':'EXP12 original teacher on TRAIN PID holdout: full candidate gallery, no training',
        'source_exp12':str(src),'best_IRRA_SHA256':EXPECTED_BEST_SHA,
        'exp12_settings_hash':opt.settings_hash,'reproduced_original_exp12':replay,
        'holdout_caption_count':len(tx),'full_holdout_gallery_images_before_exclusion':len(im),
        'total_holdout_pid_count':len(opt.holdout_pids),
        'eligible_queries_with_A_B_C_distinct':len(d['idx']),
        'eligible_unique_pid_count':int(len(np.unique(pid))),
        'original_gate_accept_count':int(d['accepted'].sum()),
        'original_gate_reject_count':int((~d['accepted']).sum()),
        'candidate_policy':'Each query compares ALL 440-holdout-PID images except its OWN A and B. '
             'All other same-PID images, including prechosen C, count relevant; every other PID is competing.',
        'analysis_methods':['T1 teacher','T1+All T2 teacher','T1+Original-Filtered T2 teacher'],
        'full_gallery_group_metrics':group_rows,
        'pair_vs_full_gallery_comparison':cross,
        'exploratory_PID_cluster_bootstrap':bootstrap,
        'limitations':[
           'True TRAIN PID and T2 selected with TRAIN PID are known during teacher construction; cannot be used in ordinary Validation/Test retrieval.',
           'C, true PID and resulting ranks/AP are used only AFTER the original T2 filter decision; never for filter fitting.',
           'All original EXP12 supports and threshold are frozen; no new selector/alpha/attribute rule/student training.',
           'A and B are excluded PER QUERY to prevent source-image shortcut; third image C and any other legal same-PID images remain positive.',
           'Internal holdout means unseen to EXP12 gate fitting, but these identities were not excluded from pretraining IRRA: only diagnostic.',
           'Accepted and rejected groups have DIFFERENT T1 baseline retrieval difficulty; group mean improvements are descriptive and do not establish causal gate quality.',
           'The 2020 queries include multiple texts/images of only 271 PID, not independent samples; bootstrap by PID is exploratory.',
           'Original fixed-D pairwise benefit can disagree with full gallery retrieval and is not visually verified semantic novelty.',
           'No official Validation/Test R1 improvement is claimed; this diagnostic compares oracle-provided T2 only.'
        ]}
    atomic_json(out/'experiment14_report.json',report)
    (out/'实验14_目标与边界.md').write_text(
        '# 实验14：被拒绝同身份T2的全候选图库检验\n\n'
        '本次只做实验12原筛选器的事后诊断，直接复用最佳IRRA的冻结训练集缓存。'
        '强制复现实验12的2020条第三视角教师边际统计，绝不重新训练或选择支持。\n\n'
        '唯一改变：把上一轮C对单个D的成对评价升级为完整440个holdout训练身份的图库排序；'
        '针对每个查询只排除原图A、支持图B，保留C及其他同PID图像，所有不同PID图像参与排名。\n\n'
        '对比T1-only、All、Filtered；对接受/拒绝组报告原T1→All的潜在mAP和名次效用。'
        '按PID聚类Bootstrap仅供不确定性分析。\n\n'
        '不能用结果调门控；不能将训练身份教师的R1称为Validation R1；'
        '不能由排名改善自动推断新增语义真实可靠。\n',encoding='utf-8')
    say('Full gallery diagnosis done; {} images, {} C-query cases, gate accepted {}.'.format(
        len(im),len(d['idx']),int(d['accepted'].sum())))
    for line in overall:
        say('  {}: holdout-teacher-gallery R1={:.4f} mAP={:.4f}'.format(
            line['method'],line['R1_percent'],line['mAP_percent']))
    say('  Rejected T2 full-gallery AP gain count: {}'.format(cross['rejected_full_gallery_AP_benefit']))
    return report


def self_test():
    """End-to-end synthetic full-gallery and I/O checks; no IRRA needed."""
    rng=np.random.default_rng(8312)
    dim=32
    images=[];texts=[];vec=[];feat=[];pos=[]
    # Six identities, five distinct images per identity, 2 texts per image.
    for pid in range(6):
        v0=normalize(rng.standard_normal((1,dim)).astype(np.float32))[0]
        for view in range(5):
            vi=normalize((v0+0.2*rng.standard_normal(dim)).reshape(1,-1))[0]
            img_id=len(images)
            images.append({'pid':pid,'file_path':f'{pid}_{view}.jpg'})
            vec.append(vi)
            for cap in range(2):
                texts.append({'pid':pid,'img_idx':img_id,'text':f'person{pid} view{view} caption{cap}'})
                feat.append(normalize((vi+.17*rng.standard_normal(dim)).reshape(1,-1))[0])
                pos.append(img_id)
    x=np.asarray(feat,np.float32);v=np.asarray(vec,np.float32);pos=np.asarray(pos,np.int32)
    gate={'mean':[0.]*9,'scale':[1.]*9,'weights':[0.]*9,'bias':0.,
          'threshold':.5,'novelty_threshold':.0}
    p0=rebuild_holdout(x,v,texts,pos,gate,123)
    nov=p0['novelty'][p0['valid_indices']]
    gate['novelty_threshold']=float(np.median(nov))
    pack=rebuild_holdout(x,v,texts,pos,gate,123)
    details=teacher_pair_details(x,v,texts,images,pos,pack,.25)
    assert 0<details['accepted'].sum()<len(details['idx'])
    methods=full_gallery_results(x,v,texts,images,pos,pack,details,.25)
    groups,cross,pids=grouped_results(methods,details,texts)
    assert len(groups)==9 and len(methods)==3
    assert len(np.unique(pids))==6
    # A/B must never enter candidate results, while C is present.
    for j in range(min(20,len(details['idx']))):
        a=int(pos[details['idx'][j]]);b=int(pos[details['support'][j]])
        assert methods['t1_only']['top1_gallery'][j] not in (a,b)
        assert methods['t1_only']['positive_count'][j]==3
        assert 1<=methods['t1_only']['c_rank'][j]<=len(images)-2
    # Repeated runs are identical; varying C cannot change original gate decisions.
    m2=full_gallery_results(x,v,texts,images,pos,pack,details,.25)
    assert np.array_equal(methods['all_same_pid']['ap'],m2['all_same_pid']['ap'])
    previous=pack['selection'].copy()
    details2=teacher_pair_details(x,v,texts,images,pos,pack,.25)
    assert np.array_equal(previous,pack['selection'])
    assert np.array_equal(details['third'],details2['third'])
    boot=cluster_bootstrap(methods,details,pids,repeats=40)
    assert boot['distinct_pids']==6
    # Check exact original EXP12 replay function on synthetic published stats.
    d={'holdout':{'true_accept_rate_proxy':float(pack['selection'][pack['valid_indices']].mean())},
       'holdout_teacher_third_view':{'n_triple_view':len(details['idx']),'methods':{}}}
    for name,values in [('t1_only',details['base_margin']),('all_same_pid',details['all_margin']),
                        ('filtered_same_pid',details['filtered_margin'])]:
        item={'third_view_pair_win_rate':float((values>0).mean()),'mean_margin':float(values.mean())}
        if name!='t1_only':
            delta=values-details['base_margin']
            item.update({'margin_improved_fraction':float((delta>1e-7).mean()),
                         'margin_harmed_fraction':float((delta< -1e-7).mean()),
                         'mean_margin_delta':float(delta.mean())})
        d['holdout_teacher_third_view']['methods'][name]=item
    assert strict_reproduce(details,pack,d,.25)['matched']
    with tempfile.TemporaryDirectory(prefix='exp14_test_') as tmp:
        base=Path(tmp);fake=base/'fake_exp12';fake.mkdir()
        ident='synthetic-exact-provenance'
        out,resumed=output_directory(base,fake,ident,False)
        assert not resumed
        atomic_json(out/'run_info.json',{'source_exp12':str(fake),'source_identity':ident})
        rows=cases_full_gallery(methods,details,texts,images,pos)
        atomic_csv(out/'all_fullgallery_cases.csv',rows,list(rows[0]))
        atomic_csv(out/'group_summary.csv',groups,list(groups[0]))
        overall=[r for r in groups if r['group']=='all_eligible']
        atomic_csv(out/'full_gallery_metrics.csv',overall,list(overall[0]))
        atomic_json(out/'experiment14_report.json',{'test':True,'reproduced_original_exp12':True})
        chosen=choose_review_cases(rows)
        atomic_csv(out/'人工核查案例.csv',chosen,list(chosen[0]))
        target=package_reports(out)
        with zipfile.ZipFile(target) as zz:
            assert zz.testzip() is None
        resumed_path,resumed=output_directory(base,fake,ident,True)
        assert resumed and resumed_path==out
        parsed=list(csv.DictReader((out/'all_fullgallery_cases.csv').open(encoding='utf-8-sig')))
        assert len(parsed)==len(rows)
        # Verify malicious/corrupted small files cannot create partially complete report.
        assert len(list(csv.DictReader((out/'full_gallery_metrics.csv').open(encoding='utf-8-sig'))))==3
    say('SELF-TEST PASS: deterministic original T2 gate, A/B exclusion, C inclusion, full-holdout gallery R1/mAP/mINP, accepted/rejected split, replay guard, PID bootstrap, CSV/ZIP and resume.')
    say('Real EXP12 frozen feature cache and Windows paths must still be verified on your computer.')


def main():
    global LOG_PATH
    ap=argparse.ArgumentParser(description='Experiment14: original EXP12 T2 gate full-gallery utility, no training')
    ap.add_argument('--project-root',default=str(PROJECT_ROOT))
    ap.add_argument('--exp12-dir',default='',help='Path to completed original Experiment12 output folder')
    ap.add_argument('--resume',action='store_true',help='Resume previously created Experiment14 only')
    ap.add_argument('--self-test',action='store_true',help='Synthetic end-to-end check, no IRRA loaded')
    args=ap.parse_args()
    if args.self_test:
        self_test();return
    root=Path(args.project_root).expanduser().resolve()
    parent=root/PROJECT_FOLDER
    if not parent.is_dir():raise FileNotFoundError(f'Missing experiment root: {parent}')
    src=source_exp12(parent,args.exp12_dir)
    say('Use original EXP12 directory: '+str(src))
    opt,gate,prior,audit=check_source(src)  # includes SHA256 of best.pth; read-only
    identity=stable_hash({
        'source_exp12':str(src),'settings_hash':opt.settings_hash,
        'gate_sha':sha256_file(src/'gate_model.json'),
        'diagnostic_sha':sha256_file(src/'internal_teacher_diagnostic.json'),
        'selected_PID_sha':sha256_file(src/'selected_train_pids.json'),
        'train_feature_provenance':opt.fp_train,'version':VERSION})
    out,resumed=output_directory(parent,src,identity,args.resume)
    LOG_PATH=out/'experiment_run.log'
    if not resumed:
        atomic_json(out/'run_info.json',{'experiment':14,'version':VERSION,'source_exp12':str(src),
          'source_identity':identity,'best_IRRA_sha256':EXPECTED_BEST_SHA,
          'creation_time':datetime.now().isoformat(timespec='seconds')})
    shutil.copy2(Path(__file__).resolve(),out/Path(__file__).name)
    say('Write only to EXP14 folder: '+str(out))
    try:
        result=out/'experiment14_report.json'
        required=[out/n for n in ('all_fullgallery_cases.csv','group_summary.csv',
                                  'full_gallery_metrics.csv','pair_vs_fullgallery_disagreements.csv',
                                  '人工核查案例.csv')]
        if resumed and result.is_file() and all(p.is_file() for p in required):
            report=json.loads(result.read_text(encoding='utf-8'))
            with (out/'all_fullgallery_cases.csv').open(encoding='utf-8-sig',newline='') as f:
                nrows=sum(1 for _ in csv.DictReader(f))
            if (report.get('exp12_settings_hash')==opt.settings_hash and
                report.get('reproduced_original_exp12',{}).get('matched') and
                report.get('eligible_queries_with_A_B_C_distinct')==nrows==EXPECTED_THIRD_VIEW_COUNT):
                say('Completed EXP14 report and case count validated; skip recomputation.')
            else:
                say('Resume output incomplete or inconsistent; recomputing analysis.')
                run_exp14(opt,gate,prior,src,out)
        else:
            run_exp14(opt,gate,prior,src,out)
    finally:
        if (out/'experiment14_report.json').is_file():
            z=package_reports(out)
            say('Analysis ZIP: '+str(z))
    say('EXPERIMENT14 COMPLETE. No IRRA source, checkpoint, EXP12 model, or feature cache was modified.')


if __name__=='__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
