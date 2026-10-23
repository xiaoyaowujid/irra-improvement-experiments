# -*- coding: utf-8 -*-
"""
IRRA 双向邻近图传播第二轮：G0 原始IRRA基线，G1 C3历史方法，G2-G6新的无监督候选。
默认：independent（独立单查询）；使用固定2048条训练集文本构建图文语义参照。
直接在PyCharm运行；可选 --self-test / --resume 输出目录 / --no-cache-reuse。
严格不使用Validation身份标签进行图构建、门控、传播或调参。
Validation身份标签只用于最终指标和事后纯诊断。

方法：
G0 原始IRRA cosine；G1 历史C3；G2 共同近邻结构图；
G3 双向结构+Train文本语义一致图；G4 正向证据受限传播；
G5 原始IRRA上的Top1多视角竞争；G6 G4与Top1竞争组合。

独立模式不使用其他Validation文本做反向索引。所有分支相对原始G0报告变化。
"""
from __future__ import annotations

import os, sys
if sys.platform == 'win32':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(r'G:\PyCharmProjects\IRRA\IRRA-main')
BEST_DIR = ROOT / 'logs' / 'CUHK-PEDES' / '20261009_095215_irra_baseline_b8_resume20'
OUTPUT_ROOT = ROOT / '多种实验方向探索'
CK_SHA = 'f3ab12028e3406546b836ed6a75c8fc5e02a42146c5da13af423e7f5cf0e7ea6'
CFG_SHA = 'ec2fae46ec4e0cce06372789afe1c877a5cebeddf73654781e11b0484fc30fac'
VERSION = 'bidirectional_graph_round2_original_irra_20261022_v1'
EXPECTED = dict(R1=66.41766807405001, R5=85.30367002273465,
                R10=90.56511854498214, mAP=61.85901641845703,
                mINP=47.24031448364258)
METRICS = ('R1', 'R5', 'R10', 'mAP', 'mINP')

# 预先固定：没有利用 Val 标签选择这些超参数。
GRAPH_K = 8
GRAPH_COS_THRESHOLD = 0.70
GRAPH_ALPHA = 0.30
REFERENCE_COUNT = 2048
REFERENCE_SEED = 20261021
IMAGE_BATCH = 4      # 6 GiB GPU 也可用
TEXT_BATCH = 64
SCORE_BATCH = 256
EVAL_BATCH = 128
ZIP_NAME = '发送给ChatGPT_双向邻近图传播_第二轮_G0G6.zip'


def say(*args):
    print(*args, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=str(path.parent), prefix='.' + path.name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def atomic_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=str(path.parent), prefix='.' + path.name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def norm_np(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('特征必须是有限的二维浮点矩阵')
    return np.ascontiguousarray(x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8))


def checked_yaml(path):
    import yaml
    # 已严格绑定此文件 SHA，原训练 yaml.dump 含 !!python/tuple 标签。
    with Path(path).open(encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(cfg, dict):
        raise ValueError('configs.yaml 顶层不是 dict')
    return cfg


def validate_environment():
    paths = [ROOT / 'model' / 'build.py', ROOT / 'datasets' / 'cuhkpedes.py',
             BEST_DIR / 'best.pth', BEST_DIR / 'configs.yaml']
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('IRRA项目文件缺失：\n' + '\n'.join(missing))
    for path, signature in ((paths[2], CK_SHA), (paths[3], CFG_SHA)):
        digest = sha256(path)
        if digest != signature:
            raise RuntimeError(f'Epoch58文件SHA不匹配，拒绝混用：{path}\n实际={digest}\n预期={signature}')
    cfg = checked_yaml(paths[3])
    for key, want in {'dataset_name': 'CUHK-PEDES', 'num_epoch': 60,
                      'batch_size': 8, 'text_length': 77,
                      'loss_names': 'sdm+mlm+id'}.items():
        if cfg.get(key) != want:
            raise RuntimeError(f'配置异常：{key}={cfg.get(key)!r}，预期{want!r}')
    if tuple(cfg.get('img_size', ())) != (384, 128):
        raise RuntimeError('基线图像尺寸不是(384,128)')
    return cfg


def find_dataset_parent(config_root):
    for base in [Path(config_root), ROOT / 'data', ROOT.parent / 'data', ROOT / 'data' / 'CUHK-PEDES']:
        folder = base if base.name.lower() == 'cuhk-pedes' else base / 'CUHK-PEDES'
        if (folder / 'reid_raw.json').is_file() and (folder / 'imgs').is_dir():
            return folder.parent
    raise FileNotFoundError('未找到 CUHK-PEDES/reid_raw.json + imgs')


def load_validation(cfg):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from datasets.cuhkpedes import CUHKPEDES
    cfg = dict(cfg)
    cfg['root_dir'] = str(find_dataset_parent(cfg['root_dir']))
    ds = CUHKPEDES(root=cfg['root_dir'], verbose=False)
    val = ds.val
    actual = (len(val['captions']), len(val['img_paths']),
              len(ds.val_id_container), len(ds.train_id_container))
    if actual != (6158, 3078, 1000, 11003):
        raise RuntimeError(f'不是官方验证集规模: {actual}')
    if any(not Path(p).is_file() for p in val['img_paths']):
        raise FileNotFoundError('验证集图像不全')
    tp = np.asarray(val['caption_pids'], dtype=np.int64)
    vp = np.asarray(val['image_pids'], dtype=np.int64)
    if not np.all(np.isin(tp, vp)):
        raise RuntimeError('某些Validation文本没有对应图库身份')
    return cfg, val, tp, vp


def load_epoch58(cfg, device):
    import importlib
    checkpoint = torch.load(str(BEST_DIR / 'best.pth'), map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('model'), dict):
        raise RuntimeError('检查点缺少model state_dict')
    if checkpoint.get('epoch') is not None and int(checkpoint['epoch']) != 58:
        raise RuntimeError('检查点不是 Epoch58')
    state = checkpoint['model']
    if state and all(x.startswith('module.') for x in state):
        state = {k[7:]: v for k, v in state.items()}
    vi = {int(m.group(1)) for k in state
          if (m := re.match(r'^base_model\.visual\.transformer\.resblocks\.(\d+)\.', k))}
    te = {int(m.group(1)) for k in state
          if (m := re.match(r'^base_model\.transformer\.resblocks\.(\d+)\.', k))}
    if vi != set(range(12)) or te != set(range(12)):
        raise RuntimeError('视觉/文本编码器不是12层')
    if state['classifier.weight'].shape[0] != 11003:
        raise RuntimeError('身份分类头大小异常')
    def shp(name):
        return state['base_model.' + name].shape
    net = dict(embed_dim=int(shp('text_projection')[1]),
               image_resolution=tuple(cfg['img_size']), vision_layers=12,
               vision_width=int(shp('visual.conv1.weight')[0]),
               vision_patch_size=int(shp('visual.conv1.weight')[-1]),
               stride_size=int(cfg['stride_size']),
               context_length=int(shp('positional_embedding')[0]),
               vocab_size=int(shp('token_embedding.weight')[0]),
               transformer_width=int(shp('ln_final.weight')[0]),
               transformer_heads=int(shp('ln_final.weight')[0]) // 64,
               transformer_layers=12)
    from model.clip_model import CLIP
    build = importlib.import_module('model.build')
    old_fn = build.build_CLIP_from_openai_pretrained
    try:
        # 精确重建已训练模型结构，不尝试从网络下载原CLIP权重。
        build.build_CLIP_from_openai_pretrained = lambda *_: (CLIP(**net), {'embed_dim': net['embed_dim']})
        model = build.build_model(SimpleNamespace(**cfg), num_classes=11003)
    finally:
        build.build_CLIP_from_openai_pretrained = old_fn
    if device.type == 'cpu':
        model.float()
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    del checkpoint, state
    say('[模型] Epoch58 权重严格加载，所有参数冻结')
    return model


def cache_close(mm):
    if not isinstance(mm, np.memmap):
        raise TypeError('缓存必须是memmap')
    mapping = getattr(mm, '_mmap', None)
    try:
        mm.flush()
    finally:
        if mapping is not None and not mapping.closed:
            mapping.close()
    if mapping is not None and not mapping.closed:
        raise RuntimeError('Windows mmap句柄未成功关闭')


def cache_open(folder, key, size, signature):
    folder.mkdir(parents=True, exist_ok=True)
    data_path = folder / (key + '.npy')
    meta_path = folder / (key + '.json')
    if data_path.exists() != meta_path.exists():
        raise RuntimeError(f'缓存及元数据不同步: {key}')
    if data_path.exists():
        state = json.loads(meta_path.read_text(encoding='utf-8'))
        if (state.get('signature') != signature or state.get('size') != size
                or not isinstance(state.get('done'), int) or not 0 <= state['done'] <= size):
            raise RuntimeError(f'缓存不能安全续跑: {key}')
        mm = np.lib.format.open_memmap(data_path, mode='r+')
        if mm.shape != (size, 512) or mm.dtype != np.float32:
            cache_close(mm)
            raise RuntimeError(f'缓存尺寸异常: {key}')
        return mm, state['done'], meta_path
    mm = np.lib.format.open_memmap(data_path, mode='w+', dtype=np.float32, shape=(size, 512))
    try:
        mm.flush()
        atomic_json(meta_path, dict(signature=signature, size=size, done=0))
    except BaseException:
        cache_close(mm)
        raise
    return mm, 0, meta_path


def cache_progress(mm, meta, signature, size, done):
    mm.flush()  # 先写数据，再提交已完成位置，断电不会错误跳过未持久化内容
    atomic_json(meta, dict(signature=signature, size=size, done=int(done)))


def extract_features(model_provider, cfg, records, cache_root, signature, key, device, kind):
    from torch.utils.data import DataLoader, Subset
    from datasets.bases import ImageDataset, TextDataset
    from datasets.build import build_transforms
    if kind == 'image':
        items = records['img_paths']
        dset = ImageDataset(records['image_pids'], items,
                            build_transforms(img_size=tuple(cfg['img_size']), is_train=False))
        batch_size = IMAGE_BATCH
    elif kind == 'text':
        items = records['captions']
        dset = TextDataset(records['caption_pids'], items, text_length=int(cfg['text_length']))
        batch_size = TEXT_BATCH
    else:
        raise ValueError(kind)
    sig = signature + '|' + key
    mm, start, meta = cache_open(cache_root, key, len(items), sig)
    try:
        if start < len(items):
            loader = DataLoader(Subset(dset, range(start, len(dset))),
                                batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=False)
            model = model_provider()
            pos = start
            say(f'[特征提取] {key}: {pos}/{len(items)}')
            with torch.inference_mode():
                for batch in loader:
                    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                        raise RuntimeError('IRRA Dataset返回内容不符合约定')
                    x = batch[1].to(device)
                    value = model.encode_image(x) if kind == 'image' else model.encode_text(x)
                    if value.ndim != 2 or value.shape[1] != 512:
                        raise RuntimeError(f'{key}特征形状异常 {tuple(value.shape)}')
                    result = F.normalize(value.float(), dim=1).cpu().numpy()
                    if not np.isfinite(result).all():
                        raise RuntimeError(f'{key}出现NaN/Inf')
                    mm[pos:pos+len(result)] = result
                    pos += len(result)
                    if pos % (batch_size * 32) < len(result) or pos == len(items):
                        cache_progress(mm, meta, sig, len(items), pos)
                        say(f'[特征提取] {key}: {pos}/{len(items)}')
            cache_progress(mm, meta, sig, len(items), pos)
        out = np.array(mm, dtype=np.float32, copy=True)
        if not np.isfinite(out).all() or np.any(np.linalg.norm(out, axis=1) < 0.5):
            raise RuntimeError('特征缓存损坏/提取未完成')
        return norm_np(out)
    finally:
        cache_close(mm)
        del mm


def score_matrix(text, gallery, device):
    result = np.empty((len(text), len(gallery)), dtype=np.float32)
    with torch.no_grad():
        g = torch.from_numpy(np.ascontiguousarray(gallery)).to(device)
        for lo in range(0, len(text), SCORE_BATCH):
            t = torch.from_numpy(np.ascontiguousarray(text[lo:lo+SCORE_BATCH])).to(device)
            result[lo:lo+len(t)] = (t @ g.T).float().cpu().numpy()
    return result


def evaluate(scores, tp, vp, block=EVAL_BATCH):
    scores = np.asarray(scores, dtype=np.float32)
    tp = np.asarray(tp, dtype=np.int64)
    vp = np.asarray(vp, dtype=np.int64)
    if scores.shape != (len(tp), len(vp)) or not np.isfinite(scores).all():
        raise ValueError('评估得分形状错误或有NaN/Inf')
    n = len(tp)
    top1 = np.empty(n, np.int32)
    rank = np.empty(n, np.int32)
    ap = np.empty(n, np.float32)
    inp = np.empty(n, np.float32)
    for lo in range(0, n, block):
        hi = min(n, lo+block)
        order = np.argsort(-scores[lo:hi], axis=1, kind='stable')
        m = vp[order] == tp[lo:hi, None]
        hits = m.sum(axis=1)
        if not (hits > 0).all():
            raise RuntimeError('评估中有查询不含同身份图像')
        top1[lo:hi] = order[:, 0]
        rank[lo:hi] = np.argmax(m, axis=1) + 1
        last = len(vp) - np.argmax(m[:, ::-1], axis=1)
        precision = np.cumsum(m, axis=1) / np.arange(1, len(vp)+1, dtype=np.float32)[None, :]
        ap[lo:hi] = (precision * m).sum(axis=1) / hits
        inp[lo:hi] = hits / last
    metrics = dict(R1=float(np.mean(rank <= 1)*100),
                   R5=float(np.mean(rank <= 5)*100),
                   R10=float(np.mean(rank <= 10)*100),
                   mAP=float(ap.mean()*100), mINP=float(inp.mean()*100))
    return metrics, dict(top1=top1, rank=rank, ap=ap, inp=inp, correct=(rank == 1))


def verify_baseline(s, tp, vp, metrics, synthetic=False):
    if synthetic:
        return {'synthetic_test': True}
    for key, target in EXPECTED.items():
        if abs(metrics[key]-target) > .07:
            raise RuntimeError(f'Epoch58基线 {key}={metrics[key]:.6f}, 历史={target:.6f}, 差异过大，停止实验')
    from utils.metrics import rank
    cmc, ap, inp, _ = rank(torch.from_numpy(s), torch.as_tensor(tp),
                           torch.as_tensor(vp), max_rank=10, get_mAP=True)
    official = dict(R1=float(cmc[0]), R5=float(cmc[4]), R10=float(cmc[9]),
                    mAP=float(ap), mINP=float(inp))
    for key in METRICS:
        if abs(metrics[key]-official[key]) > .02:
            raise RuntimeError(f'本实验与IRRA官方评估不一致 {key}: {metrics[key]} vs {official[key]}')
    return official


def build_mutual_gallery_graph(gallery, device=None, k=GRAPH_K, threshold=GRAPH_COS_THRESHOLD):
    n = len(gallery)
    if not 1 <= k < n:
        raise ValueError('图邻居数无效')
    if device is None:
        device = torch.device('cpu')
    # 与旧C3原实现严格保持同一 torch.topk 路径，避免Torch/Numpy的选邻差异。
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(gallery)).to(device)
        similarity = x @ x.T
        similarity.fill_diagonal_(-float('inf'))
        values, ids = torch.topk(similarity, k, dim=1, sorted=True)
        indices = ids.cpu().numpy().astype(np.int32)
        strengths = values.cpu().numpy().astype(np.float32)
        del x, similarity, values, ids
    mutual = np.zeros_like(strengths, dtype=bool)
    sets = [set(map(int, row)) for row in indices]
    for i in range(n):
        for p, other in enumerate(indices[i]):
            mutual[i, p] = (i in sets[int(other)])
    eligible = mutual & (strengths >= threshold)
    unscaled = np.maximum(strengths, 0) * eligible
    mass = unscaled.sum(axis=1)
    weights = unscaled / np.maximum(mass[:, None], 1e-8)
    stat = dict(gallery_images=int(n), k=int(k), threshold=float(threshold),
                connected_nodes=int(np.sum(mass > 0)), isolated_nodes=int(np.sum(mass <= 0)),
                directed_edges=int(np.sum(eligible)),
                edges_are_directed_for_averaging=True)
    return indices, strengths, weights.astype(np.float32), stat


def graph_smooth(score, idx, weights, alpha=GRAPH_ALPHA, block=64):
    """C3 完全相同的边归一化+无有效边不变化。"""
    n = score.shape[1]
    if idx.shape != weights.shape or idx.shape[0] != n:
        raise ValueError('图库图邻居尺寸异常')
    has_neighbors = (weights.sum(axis=1) > 0).astype(np.float32)[None, :]
    out = np.empty_like(score)
    for lo in range(0, len(score), block):
        src = score[lo:lo+block]
        peers = np.zeros_like(src)
        for slot in range(idx.shape[1]):
            peers += src[:, idx[:, slot]] * weights[None, :, slot]
        out[lo:lo+len(src)] = src + alpha * has_neighbors * (peers-src)
    return out




def train_reference_texts(dataset_root, validation_pids):
    file = Path(dataset_root) / 'CUHK-PEDES' / 'reid_raw.json'
    data = json.loads(file.read_text(encoding='utf-8'))
    texts, pids = [], []
    for row in data:
        if row.get('split') != 'train':
            continue
        pid = int(row['id'])
        for line in row.get('captions', []):
            texts.append(str(line)); pids.append(pid)
    if len(texts) < REFERENCE_COUNT or len(set(pids)) != 11003:
        raise RuntimeError('Train参照文本数量或身份数不符合官方CUHK-PEDES')
    if set(pids).intersection(map(int, validation_pids)):
        raise RuntimeError('训练与验证身份重合，停止')
    rng = np.random.default_rng(REFERENCE_SEED)
    chosen = rng.choice(len(texts), REFERENCE_COUNT, replace=False)
    ref = dict(captions=[texts[int(i)] for i in chosen],
               caption_pids=[pids[int(i)] for i in chosen])
    sig = hashlib.sha256((sha256(file) + json.dumps(ref, sort_keys=True, ensure_ascii=False))
                          .encode('utf-8')).hexdigest()
    return ref, sig



def record_variant(out, fingerprint, name, score, tp, vp, base_metrics, base_detail, description):
    result_path = out / '结果' / (name+'.json')
    query_path = out / '逐查询' / (name+'.csv')
    if result_path.is_file():
        doc = json.loads(result_path.read_text(encoding='utf-8'))
        if doc.get('fingerprint') != fingerprint or not query_path.is_file() or doc.get('csv_sha256') != sha256(query_path):
            raise RuntimeError(f'已有结果与当前实验不匹配/文件遭篡改: {name}')
        say('[续跑复用]', name)
        return doc
    metrics, details = evaluate(score, tp, vp)
    changed = dict(corrected=int(np.sum(~base_detail['correct'] & details['correct'])),
                   broken=int(np.sum(base_detail['correct'] & ~details['correct'])),
                   net_corrected=int(details['correct'].sum()-base_detail['correct'].sum()),
                   top1_changed=int(np.sum(base_detail['top1'] != details['top1'])))
    rows = []
    for q in range(len(tp)):
        before, after = bool(base_detail['correct'][q]), bool(details['correct'][q])
        outcome = ('CORRECTED' if not before and after else
                   'BROKEN' if before and not after else
                   'STILL_CORRECT' if before and after else 'STILL_WRONG')
        rows.append(dict(query_idx=q, true_pid=int(tp[q]),
                         baseline_rank=int(base_detail['rank'][q]), variant_rank=int(details['rank'][q]),
                         baseline_top1_image=int(base_detail['top1'][q]),
                         variant_top1_image=int(details['top1'][q]),
                         baseline_top1_pid=int(vp[base_detail['top1'][q]]),
                         variant_top1_pid=int(vp[details['top1'][q]]),
                         change_type=outcome,
                         delta_AP=float(details['ap'][q]-base_detail['ap'][q]),
                         delta_INP=float(details['inp'][q]-base_detail['inp'][q])))
    cols = ['query_idx','true_pid','baseline_rank','variant_rank',
            'baseline_top1_image','variant_top1_image','baseline_top1_pid',
            'variant_top1_pid','change_type','delta_AP','delta_INP']
    atomic_csv(query_path, rows, cols)
    doc = dict(name=name, version=VERSION, fingerprint=fingerprint,
               note=description, metrics=metrics,
               delta_pp={k:metrics[k]-base_metrics[k] for k in METRICS},
               changes=changed, csv_sha256=sha256(query_path))
    atomic_json(result_path, doc)
    say('[完成]', name, 'R1={:.4f}(Δ{:+.4f}) mAP={:.4f}(Δ{:+.4f}) 修复={} 破坏={}'.format(
        metrics['R1'], doc['delta_pp']['R1'],metrics['mAP'],doc['delta_pp']['mAP'],
        changed['corrected'],changed['broken']))
    return doc



def results_summary(out, fingerprint, baseline):
    result_files = sorted((out/'结果').glob('G*.json'))
    rows = []
    for file in result_files:
        doc = json.loads(file.read_text(encoding='utf-8'))
        if doc.get('fingerprint') != fingerprint:
            raise RuntimeError('统计阶段发现混入的旧结果')
        path = out/'逐查询'/(doc['name']+'.csv')
        if not path.is_file() or sha256(path) != doc.get('csv_sha256'):
            raise RuntimeError('统计阶段逐查询CSV丢失/改变')
        rows.append(dict(experiment=doc['name'],
                         **{key:doc['metrics'][key] for key in METRICS},
                         **{'delta_'+key:doc['metrics'][key]-baseline[key] for key in METRICS},
                         **doc['changes']))
    columns = ['experiment', *METRICS, *('delta_'+x for x in METRICS),
               'corrected','broken','net_corrected','top1_changed']
    atomic_csv(out/'01_对照指标总表.csv', rows, columns)
    return rows



def build_zip(out):
    target = out/ZIP_NAME
    part = out/('.'+ZIP_NAME+'.tmp')
    try:
        with zipfile.ZipFile(part,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,
                             allowZip64=True) as z:
            for file in sorted(out.rglob('*')):
                if not file.is_file() or file in (part, target):
                    continue
                if '缓存' in file.relative_to(out).parts:
                    continue
                if file.suffix.lower() in ('.npy','.pth','.pt'):
                    continue
                z.write(file,file.relative_to(out).as_posix())
        with zipfile.ZipFile(part,'r') as z:
            problem = z.testzip()
            if problem is not None:
                raise RuntimeError('ZIP CRC校验失败: '+problem)
        os.replace(part,target)
    finally:
        if part.exists():
            part.unlink()
    return target


# ========================== 第二轮核心算法 ==========================
STRUCTURE_K = 16
STRUCTURE_SIM = 0.68
STRUCTURE_COMMON_MIN = 2
STRUCTURE_STRONG_FALLBACK = 0.94
SEMANTIC_KEEP_QUANTILE = 0.35
NEW_GRAPH_ALPHA = 0.35
POSITIVE_ALPHA = 0.80
POSITIVE_TOPK = 35
POSITIVE_GATE_TEMP = 0.025
COMPETE_TOPK_NEW = 15
COMPETE_MAX_GAP = 0.04
COMPETE_PEER_WINDOW = 0.065
COMPETE_MIN_PEERS = 2
COMPETE_MIN_EVIDENCE = 0.66
COMPETE_EVIDENCE_MARGIN = 0.38
PREVIOUS_VERSION = 'bidirectional_graph_g0g5_20261022_v1'


def graph_topk(gallery, k, device):
    n = len(gallery)
    if not 2 <= k < n:
        raise ValueError('图库数量太少，无法建立指定邻近图')
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(gallery)).to(device)
        sim = x @ x.T
        sim.fill_diagonal_(-float('inf'))
        values, indices = torch.topk(sim, k, dim=1, sorted=True)
        out_idx = indices.cpu().numpy().astype(np.int32)
        out_sim = values.cpu().numpy().astype(np.float32)
    return out_idx, out_sim


def normalized_weights(raw):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError('无效传播权重')
    total = raw.sum(axis=1, keepdims=True)
    return np.ascontiguousarray(raw / np.maximum(total, 1e-8), dtype=np.float32)


def structural_graph(gallery, device, k=STRUCTURE_K, min_common=STRUCTURE_COMMON_MIN):
    """只对互为K近邻的边查共同近邻；不访问身份标签和文本验证集。"""
    idx, sim = graph_topk(gallery, k, device)
    n = len(gallery)
    sets = [set(int(j) for j in row) for row in idx]
    common = np.zeros_like(idx, dtype=np.uint8)
    mutual = np.zeros_like(sim, dtype=bool)
    for i in range(n):
        for h, j0 in enumerate(idx[i]):
            j = int(j0)
            if i in sets[j]:
                mutual[i,h] = True
                common[i,h] = min(255, len(sets[i].intersection(sets[j])))
    # 强一致边：共同邻居>=2；极高余弦的孤立双视角对也允许连边。
    eligible = mutual & (sim >= STRUCTURE_SIM) & (
        (common >= min_common) | (sim >= STRUCTURE_STRONG_FALLBACK))
    raw = (np.maximum(sim,0) * (1.0 + .22 * common.astype(np.float32))) * eligible
    weights = normalized_weights(raw)
    edges = weights > 0
    stats = dict(k=k, threshold=STRUCTURE_SIM, min_common=min_common,
                 strong_fallback=STRUCTURE_STRONG_FALLBACK,
                 connected=int(np.sum(weights.sum(axis=1)>0)),
                 directed_edges=int(edges.sum()),
                 mean_common_on_edges=float(common[edges].mean()) if edges.any() else 0.0)
    return idx, sim, common, weights, stats


def semantic_profile(reference_scores):
    """Train描述->每张Val图像的响应剖面；剖面与Val文本查询无关。"""
    if reference_scores.ndim != 2 or reference_scores.shape[0] < 32:
        raise ValueError('训练文本参考分数维度错误')
    x = np.asarray(reference_scores, dtype=np.float32)
    # 逐条参考文本在全部图像上的中心化，消除参考文本总体容易/困难的全局偏移。
    centered = x - x.mean(axis=1, keepdims=True)
    profiles = norm_np(centered.T)
    return profiles


def multimodal_graph(idx, sim, structure_weights, profiles,
                     quantile=SEMANTIC_KEEP_QUANTILE):
    """对已有结构边施加图文语义响应一致性过滤，保持双向对称性。"""
    n,k = idx.shape
    if structure_weights.shape != (n,k) or profiles.shape[0] != n:
        raise ValueError('语义图输入形状不匹配')
    similarity = np.empty((n,k),dtype=np.float32)
    for h in range(k):
        similarity[:,h] = np.einsum('ij,ij->i',profiles,profiles[idx[:,h]],optimize=True)
    # 每个节点的语义阈值按自己的topK图像近邻分布校准，不依赖PID。
    local_threshold = np.quantile(similarity,quantile,axis=1).astype(np.float32)
    passed = ((similarity >= local_threshold[:,None]) &
              (similarity >= local_threshold[idx]))
    # 双边约束有可能完全孤立某些节点：这是刻意的保守设计，不默认回退到错误边。
    relative = similarity - .5*(local_threshold[:,None]+local_threshold[idx])
    strength = np.clip(.7 + 2.0 * relative, .25, 1.25)
    raw = structure_weights * passed * strength
    weights = normalized_weights(raw)
    stat = dict(reference_count=int(profiles.shape[1]),quantile=quantile,
                directed_edges=int(np.sum(weights>0)),
                connected=int(np.sum(weights.sum(axis=1)>0)),
                semantics_mean_on_edges=(float(similarity[weights>0].mean())
                                         if (weights>0).any() else None))
    return weights, similarity, stat


def positive_evidence(score, idx, weights, alpha=POSITIVE_ALPHA, block=64):
    """仅吸收更高的可信邻居得分；原本图像分数不被平均拉低。"""
    if score.shape[1] != idx.shape[0] or idx.shape != weights.shape:
        raise ValueError('正证据输入维度错误')
    out=np.empty_like(score)
    k=min(POSITIVE_TOPK,score.shape[1])
    for lo in range(0,len(score),block):
        src=score[lo:lo+block]
        cut=np.partition(src,src.shape[1]-k,axis=1)[:,src.shape[1]-k,None]
        up=np.zeros_like(src)
        confident=np.zeros_like(src)
        for h in range(idx.shape[1]):
            peer=src[:,idx[:,h]]
            w=weights[None,:,h]
            up+=w*np.maximum(peer-src,0)
            # 接近全图库Top35得分的邻居才提供额外证据。
            prob=1./(1.+np.exp(-np.clip((peer-cut)/POSITIVE_GATE_TEMP,-35,35)))
            confident+=w*prob
        out[lo:lo+len(src)]=src+alpha*up*(.15+.85*confident)
    return out


def cluster_evidence(row, candidate, idx, weights, top_value):
    """对一个候选统计强邻居数和加权支持强度。"""
    peer_ids=idx[candidate]
    support=weights[candidate]
    score=row[peer_ids]
    good=(score >= top_value - COMPETE_PEER_WINDOW) & (support>0)
    count=int(good.sum())
    # 支持度以可用邻居总权重为分母；仅用强邻居不足2时不认定为可靠簇。
    strength=float(support[good].sum())
    return count,strength


def top1_competition(score, idx, weights, max_gap=COMPETE_MAX_GAP,
                     max_candidates=COMPETE_TOPK_NEW):
    """多视角证据足够时挑战Top1；只改变当前查询的一个候选，不使用PID。"""
    if score.shape[1] != idx.shape[0]:
        raise ValueError('竞争方法图库数量错误')
    output=score.copy()
    events=[]
    k=min(max_candidates,score.shape[1])
    for q,row in enumerate(score):
        order=np.argsort(-row,kind='stable')[:k]
        leader=int(order[0]); best=float(row[leader])
        count0,mass0=cluster_evidence(row,leader,idx,weights,best)
        # 对相近的前列候选比较多视角共识。只处理Top1证据较弱的查询。
        if count0>=2 and mass0 >= .55:
            continue
        possible=[]
        for col0 in order[1:]:
            col=int(col0)
            gap=best-float(row[col])
            if gap > max_gap:
                break
            count,mass=cluster_evidence(row,col,idx,weights,best)
            if (count>=COMPETE_MIN_PEERS and mass>=COMPETE_MIN_EVIDENCE and
                    mass-mass0>=COMPETE_EVIDENCE_MARGIN):
                possible.append((mass-mass0, mass, -gap, col, count))
        if not possible:
            continue
        advantage, mass, negative_gap, chosen, count = max(possible)
        # 强多视角支持才执行一次明确换位；记录全部尝试，供事后正确/错误诊断。
        gap=-negative_gap
        output[q,chosen]=np.nextafter(np.float32(best),np.float32(np.inf))
        events.append(dict(query_idx=int(q), original_top1=int(leader),
                           selected_candidate=int(chosen), score_gap=float(gap),
                           leader_peer_count=count0,leader_evidence=mass0,
                           candidate_peer_count=count,candidate_evidence=mass,
                           evidence_margin=float(advantage)))
    return output,events


def graph_edge_audit(idx, weights, gallery_pids, common=None, semantic=None):
    """只用于事后评估，绝对不反馈给筛边、加权与竞争模块。"""
    valid=weights>0
    same=gallery_pids[:,None]==gallery_pids[idx]
    denom=max(1,int(valid.sum()))
    return dict(directed_edges=int(valid.sum()),
                same_pid_edges=int(np.sum(valid&same)),
                different_pid_edges=int(np.sum(valid&~same)),
                same_pid_fraction=float(np.sum(valid&same)/denom),
                connected=int(np.sum(np.any(valid,axis=1))),
                common_mean=float(common[valid].mean()) if common is not None and valid.any() else None,
                semantic_mean=float(semantic[valid].mean()) if semantic is not None and valid.any() else None,
                labels_used_for_building_graph=False)


def settings_dict(mode):
    return dict(version=VERSION, mode=mode, ck_sha=CK_SHA, cfg_sha=CFG_SHA,
                graph_k=GRAPH_K, graph_threshold=GRAPH_COS_THRESHOLD, graph_alpha=GRAPH_ALPHA,
                structure_k=STRUCTURE_K, structure_sim=STRUCTURE_SIM,
                structure_common=STRUCTURE_COMMON_MIN,
                structure_fallback=STRUCTURE_STRONG_FALLBACK,
                semantic_quantile=SEMANTIC_KEEP_QUANTILE,
                new_graph_alpha=NEW_GRAPH_ALPHA,positive_alpha=POSITIVE_ALPHA,
                positive_topk=POSITIVE_TOPK,positive_temp=POSITIVE_GATE_TEMP,
                compete_k=COMPETE_TOPK_NEW,compete_gap=COMPETE_MAX_GAP,
                compete_window=COMPETE_PEER_WINDOW,compete_peers=COMPETE_MIN_PEERS,
                compete_evidence=COMPETE_MIN_EVIDENCE,
                compete_margin=COMPETE_EVIDENCE_MARGIN,
                reference_count=REFERENCE_COUNT, reference_seed=REFERENCE_SEED)


def data_fingerprint(cfg,val,ref_sig):
    body = dict(checkpoint_sha=CK_SHA,config_sha=CFG_SHA,
                images=val['img_paths'],gallery_pids=val['image_pids'],
                texts=val['captions'],query_pids=val['caption_pids'],
                img_size=cfg['img_size'],text_length=cfg['text_length'],
                mode='independent',reference_signature=ref_sig,
                parameters=settings_dict('independent'))
    return hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False).encode('utf8')).hexdigest()


def old_data_fingerprint(cfg,val,ref_sig):
    """只用于校验上一轮缓存可以原样复用，参数完全按上一轮写入记录。"""
    old=dict(version=PREVIOUS_VERSION,mode='independent',
             baseline_checkpoint_sha256=CK_SHA,baseline_config_sha256=CFG_SHA,
             graph_k=8,graph_cos_threshold=.7,graph_alpha=.30,
             rrf_k=60,rrf_blend=.20,query_sim_tau=.08,query_gate_floor=.30,
             reverse_rank_scale=64.,compete_topk=50,compete_margin=.03,
             compete_strength=.15,train_reference_count=2048,seed=20261021)
    body=dict(checkpoint_sha=CK_SHA,config_sha=CFG_SHA,
              images=val['img_paths'],gallery_pids=val['image_pids'],
              texts=val['captions'],query_pids=val['caption_pids'],
              img_size=cfg['img_size'],text_length=cfg['text_length'],
              mode='independent',reference_signature=ref_sig,
              parameters=old)
    return hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False).encode('utf8')).hexdigest()


def prepare_folder(fingerprint,resume='',synthetic=False):
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    settings=settings_dict('independent')
    if resume:
        out=Path(resume).resolve()
        if not out.is_dir() or out.parent.resolve()!=OUTPUT_ROOT.resolve():
            raise RuntimeError('续跑文件夹必须是多种实验方向探索目录下的本轮子目录')
        meta=out/'00_运行元数据.json'
        if not meta.is_file():
            raise RuntimeError('续跑目录无实验元数据')
        prev=json.loads(meta.read_text(encoding='utf8'))
        if prev.get('fingerprint')!=fingerprint or prev.get('settings')!=settings:
            raise RuntimeError('与本轮实验配置或数据指纹不同，禁止混合续跑')
    else:
        name='双向邻近图传播_第二轮_原始IRRA基线_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        out=OUTPUT_ROOT/name
        out.mkdir(parents=True,exist_ok=False)
        atomic_json(out/'00_运行元数据.json',dict(
            fingerprint=fingerprint,settings=settings,synthetic=bool(synthetic),
            baseline='原始IRRA图文归一化余弦，不用历史任何重排序得分',
            protocol='independent: 6158条Val查询互不作为其他查询的参照；2048 Train描述独立建图',
            labels_policy='Validation PID只允许用于评估与图边事后审计',
            model='Epoch58 frozen',date='2026-10-22'))
        try:
            shutil.copy2(Path(__file__),out/'本轮实验单文件脚本.py')
        except OSError as exc:
            say('[警告] 保存脚本副本失败:',exc)
    for folder in ('缓存','结果','逐查询','分析'):
        (out/folder).mkdir(parents=True,exist_ok=True)
    return out


def reuse_previous_cache(cfg,val,ref_sig,new_fp,out,disabled=False):
    """自动查找上一轮相同模型/相同数据/同Train参照的完整缓存，不误复用重排分数。"""
    if disabled:
        return False
    names=('gallery_3078','query_6158','train_reference_2048')
    sizes=(3078,6158,2048)
    target=out/'缓存'
    if any((target/(x+'.npy')).exists() for x in names):
        return False  # 本轮缓存已有内容，按本轮自己的签名继续
    previous_fp=old_data_fingerprint(cfg,val,ref_sig)
    sources=sorted(OUTPUT_ROOT.glob('双向邻近图传播_independent_*'),reverse=True)
    for source in sources:
        try:
            metadata=json.loads((source/'00_运行元数据.json').read_text(encoding='utf8'))
            if metadata.get('fingerprint')!=previous_fp:
                continue
            settings=metadata.get('settings',{})
            if (settings.get('version')!=PREVIOUS_VERSION or
                    settings.get('baseline_checkpoint_sha256')!=CK_SHA or
                    settings.get('baseline_config_sha256')!=CFG_SHA or
                    settings.get('mode')!='independent'):
                continue
            for name,count in zip(names,sizes):
                path=source/'缓存'/(name+'.npy')
                meta=source/'缓存'/(name+'.json')
                info=json.loads(meta.read_text(encoding='utf8'))
                if info.get('signature')!=previous_fp+'|'+name or info.get('done')!=count or info.get('size')!=count:
                    raise RuntimeError('源缓存校验不一致')
                x=np.load(path,mmap_mode='r')
                valid=(x.shape==(count,512) and x.dtype==np.float32 and
                       np.isfinite(x).all() and
                       np.all((np.linalg.norm(x,axis=1)>.95)&(np.linalg.norm(x,axis=1)<1.05)))
                del x
                if not valid:
                    raise RuntimeError('源特征内容不是有限且归一化的512维矩阵')
            # 通过所有校验后，全部复制并提交新缓存元数据。
            for name,count in zip(names,sizes):
                shutil.copyfile(source/'缓存'/(name+'.npy'), target/(name+'.npy'))
                atomic_json(target/(name+'.json'),dict(signature=new_fp+'|'+name,size=count,done=count))
            say('[缓存复用] 从上一轮相同数据的原始IRRA特征复用，不复用任何G1-G5分数:',source)
            return True
        except (OSError,ValueError,RuntimeError,KeyError,TypeError) as exc:
            # 防止复制中途断电/磁盘满留下半套缓存，让下次启动错误认为可续跑。
            for key in names:
                for suffix in ('.npy','.json'):
                    partial=target/(key+suffix)
                    if partial.exists():
                        partial.unlink()
            say('[跳过候选缓存]',source.name,str(exc))
            continue
    say('[缓存] 未发现严格匹配的上一轮缓存，将正常提取原始IRRA特征')
    return False


def export_competition_events(out,name,events,tp,vp,source_detail,original_detail):
    rows=[]
    for event in events:
        row=dict(event)
        q=event['query_idx']
        row['query_true_pid']=int(tp[q])
        row['original_top1_pid']=int(vp[event['original_top1']])
        row['selected_candidate_pid']=int(vp[event['selected_candidate']])
        row['was_input_correct']=bool(vp[event['original_top1']]==tp[q])
        row['was_original_IRRA_correct']=bool(original_detail['correct'][q])
        row['selection_correct']=bool(tp[q]==vp[event['selected_candidate']])
        rows.append(row)
    columns=['query_idx','original_top1','selected_candidate','score_gap',
             'leader_peer_count','leader_evidence','candidate_peer_count',
             'candidate_evidence','evidence_margin', 'query_true_pid',
             'original_top1_pid','selected_candidate_pid','was_input_correct',
             'was_original_IRRA_correct','selection_correct']
    atomic_csv(out/'分析'/(name+'.csv'),rows,columns)
    return dict(activated=len(events),
                input_wrong_selection_correct=sum((not r['was_input_correct']) and r['selection_correct'] for r in rows),
                input_correct_selection_wrong=sum(r['was_input_correct'] and not r['selection_correct'] for r in rows),
                original_IRRA_wrong_selection_correct=sum((not r['was_original_IRRA_correct']) and r['selection_correct'] for r in rows),
                original_IRRA_correct_selection_wrong=sum(r['was_original_IRRA_correct'] and not r['selection_correct'] for r in rows))


def run_pipeline(text,images,text_pids,image_pids,ref,out,fingerprint,
                 device=None,synthetic=False):
    """所有G0-G6均以原始IRRA score计算，严格单查询、无Val标签参与算法。"""
    start=time.perf_counter()
    device=device or torch.device('cpu')
    # 保持与上一轮原始IRRA数值完全一致：提取阶段已归一化，不能再次归一化。
    text=np.ascontiguousarray(text,dtype=np.float32)
    images=np.ascontiguousarray(images,dtype=np.float32)
    ref=np.ascontiguousarray(ref,dtype=np.float32)
    for name,arr in (('query',text),('gallery',images),('reference',ref)):
        if arr.ndim!=2 or arr.shape[1]!=512 or not np.isfinite(arr).all():
            raise RuntimeError(f'{name}特征异常')
        norm=np.linalg.norm(arr,axis=1)
        if not np.all((norm>.98)&(norm<1.02)):
            raise RuntimeError(f'{name}特征并非已归一化的IRRA特征，拒绝产生不可靠基线')
    original=score_matrix(text,images,device)
    metrics0, details0=evaluate(original,text_pids,image_pids)
    official=verify_baseline(original,text_pids,image_pids,metrics0,synthetic=synthetic)
    atomic_json(out/'00_基线核验.json',dict(metrics=metrics0,official=official,
                  expected=(None if synthetic else EXPECTED),
                  baseline='IRRA cosine before any processing',
                  num_queries=len(text),num_gallery=len(images)))
    record_variant(out,fingerprint,'G0_原始IRRA',original,
                   text_pids,image_pids,metrics0,details0,
                   '未经重排序的IRRA余弦分数；所有新方法与其比较')
    say('[原始IRRA基线]',metrics0)

    # G1 只做严格原始C3复现，不作为新方法起点。
    old_idx,old_sim,old_w,old_stat=build_mutual_gallery_graph(images,device=device)
    atomic_json(out/'分析'/'01_C3原视觉图审计.json',
                dict(graph=old_stat,labels_audit=graph_edge_audit(old_idx,old_w,image_pids)))
    old_scores=graph_smooth(original,old_idx,old_w,alpha=GRAPH_ALPHA)
    record_variant(out,fingerprint,'G1_历史C3',old_scores,
                   text_pids,image_pids,metrics0,details0,
                   '历史C3固定Top8 / 0.70 / alpha0.30，直接处理原始IRRA')
    del old_scores,old_idx,old_sim,old_w

    idx,sim,common,struct_w,stat_s=structural_graph(images,device)
    audit_s=graph_edge_audit(idx,struct_w,image_pids,common=common)
    atomic_json(out/'分析'/'02_结构图审计.json',dict(graph=stat_s,labels_audit=audit_s))
    say('[G2结构图]',stat_s,'[仅事后同PID边占比]',round(audit_s['same_pid_fraction'],4))
    struct_score=graph_smooth(original,idx,struct_w,alpha=NEW_GRAPH_ALPHA)
    record_variant(out,fingerprint,'G2_共同邻居结构传播',struct_score,
                   text_pids,image_pids,metrics0,details0,
                   'Top16 mutual + 共同邻居>=2/极高相似度兜底，原始余弦结构传播')
    del struct_score

    ref_scores=score_matrix(ref,images,device)
    profiles=semantic_profile(ref_scores)
    del ref_scores
    mm_w,sem_sim,stat_m=multimodal_graph(idx,sim,struct_w,profiles)
    audit_m=graph_edge_audit(idx,mm_w,image_pids,common=common,semantic=sem_sim)
    atomic_json(out/'分析'/'03_多模态图审计.json',dict(graph=stat_m,labels_audit=audit_m))
    say('[G3多模态图]',stat_m,'[仅事后同PID边占比]',round(audit_m['same_pid_fraction'],4))
    mm_score=graph_smooth(original,idx,mm_w,alpha=NEW_GRAPH_ALPHA)
    record_variant(out,fingerprint,'G3_多模态一致图传播',mm_score,
                   text_pids,image_pids,metrics0,details0,
                   'G2图结构+固定Train文本->图像语义响应一致，静态融合图传播')
    del mm_score

    # G4只有正向邻居支持，消除C3对高得分Top1的非选择性拉低。
    promoted=positive_evidence(original,idx,mm_w)
    record_variant(out,fingerprint,'G4_多视角正证据传播',promoted,
                   text_pids,image_pids,metrics0,details0,
                   '仅当可信邻居对候选得分更高时才正向支持，查询Top35门控')

    # G5是直接在原始IRRA上做一次局部Top1竞争；G6则叠加G4之后竞争。
    solo,events=top1_competition(original,idx,mm_w)
    record_variant(out,fingerprint,'G5_原始IRRA_Top1竞争',solo,
                   text_pids,image_pids,metrics0,details0,
                   '原始IRRA+多邻居共识挑战Top1，分数接近才触发')
    stat5=export_competition_events(out,'04_G5竞争逐例',events,text_pids,image_pids,details0,details0)
    del solo,events
    _, promoted_detail=evaluate(promoted,text_pids,image_pids)
    joined,events6=top1_competition(promoted,idx,mm_w)
    record_variant(out,fingerprint,'G6_正证据加Top1竞争',joined,
                   text_pids,image_pids,metrics0,details0,
                   'G4正向证据传播+局部Top1多视角竞争')
    stat6=export_competition_events(out,'05_G6竞争逐例',events6,text_pids,image_pids,promoted_detail,details0)
    del joined,promoted,events6,profiles,sem_sim
    atomic_json(out/'分析'/'06_竞争触发统计.json',dict(G5=stat5,G6=stat6,
                caveat='竞争后果使用Val PID事后标注；PID没有参与触发与选人'))

    rows=results_summary(out,fingerprint,metrics0)
    atomic_json(out/'02_完整实验摘要.json',dict(
        version=VERSION,mode='independent',duration_sec=time.perf_counter()-start,
        fingerprint=fingerprint,original_IRRA_baseline=metrics0,
        results=rows,graph_structure=stat_s,graph_multimodal=stat_m,
        audit_structure=audit_s,audit_multimodal=audit_m,
        G5_competition=stat5,G6_competition=stat6,
        caveat='本轮G2-G6为机制假设，不保证提升；不得以本Validation调参后的结果直接替代Test泛化验证'))
    path=build_zip(out)
    say('[全流程完成]',out,'\n[上传ZIP]',path)
    return rows,path


def run_real(args):
    torch.set_num_threads(min(6,os.cpu_count() or 6))
    torch.manual_seed(REFERENCE_SEED)
    np.random.seed(REFERENCE_SEED)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('未检测到CUDA：请核查irra_clean环境或加 --device cpu')
    device=torch.device(args.device)
    cfg=validate_environment()
    cfg,val,tp,vp=load_validation(cfg)
    ref_records,ref_sig=train_reference_texts(cfg['root_dir'],tp)
    fingerprint=data_fingerprint(cfg,val,ref_sig)
    out=prepare_folder(fingerprint,args.resume)
    say('[本轮输出]',out)
    reuse_previous_cache(cfg,val,ref_sig,fingerprint,out,disabled=args.no_cache_reuse)
    provider_holder={'model':None}
    def provider():
        if provider_holder['model'] is None:
            provider_holder['model']=load_epoch58(cfg,device)
        return provider_holder['model']
    gallery=extract_features(provider,cfg,val,out/'缓存',fingerprint,
                             'gallery_3078',device,'image')
    text=extract_features(provider,cfg,val,out/'缓存',fingerprint,
                          'query_6158',device,'text')
    ref=extract_features(provider,cfg,ref_records,out/'缓存',fingerprint,
                         'train_reference_2048',device,'text')
    provider_holder['model']=None
    if device.type=='cuda':
        torch.cuda.empty_cache()
    run_pipeline(text,gallery,tp,vp,ref,out,fingerprint,device)


def self_test():
    """CPU独立协议：边构建/图传播/改排触发/评价/断点续跑/压缩/篡改校验。"""
    from unittest.mock import patch
    rng=np.random.default_rng(20261022)
    with tempfile.TemporaryDirectory(prefix='IRRA_graph_round2_') as tmp:
        root=Path(tmp)
        # memmap断点续跑：模拟写入2/5，回读，错误签名拒绝。
        mm,done,meta=cache_open(root/'cache','synthetic',5,'safe')
        assert done==0
        mm[:2]=0.5
        cache_progress(mm,meta,'safe',5,2)
        cache_close(mm);del mm
        mm,done,meta=cache_open(root/'cache','synthetic',5,'safe')
        assert done==2 and np.all(mm[:2]==0.5)
        cache_close(mm);del mm
        try:
            cache_open(root/'cache','synthetic',5,'wrong')
        except RuntimeError:
            pass
        else:
            raise AssertionError('错误缓存签名未被拦截')
        prototypes=norm_np(rng.standard_normal((9,512)))
        vp=np.repeat(np.arange(9),5)
        tp=np.repeat(np.arange(9),4)
        gallery=norm_np(prototypes[vp]+.017*rng.standard_normal((len(vp),512)))
        text=norm_np(prototypes[tp]+.045*rng.standard_normal((len(tp),512)))
        text[0]=prototypes[1]
        ref=norm_np(rng.standard_normal((160,512)))
        idx,sim,common,w,st=structural_graph(gallery,torch.device('cpu'),k=16)
        assert st['directed_edges']>0,'结构图没有可测的边'
        # 构造足够密集的Train语义响应以覆盖语义图。
        ref_scores=score_matrix(ref,gallery,torch.device('cpu'))
        prof=semantic_profile(ref_scores)
        mm_w,sem,_=multimodal_graph(idx,sim,w,prof)
        assert np.sum(mm_w>0)>0,'多模态图完全为空，无法测试'
        a=np.flatnonzero(w.sum(axis=1)>0)[0]
        b=int(idx[a,np.flatnonzero(w[a]>0)[0]])
        altered=prof.copy()
        altered[b]=-altered[a]
        filtered,_,_=multimodal_graph(idx,sim,w,altered)
        assert not np.array_equal(filtered,mm_w), '语义变化未影响可信图权重'
        raw=score_matrix(text,gallery,torch.device('cpu'))
        corrected=positive_evidence(raw,idx,mm_w)
        assert np.max(np.abs(corrected-raw))>1e-7,'G4核心传播分支未运行'
        # G5条件分支定向单测：榜首孤立，第二候选拥有两个高分好友。
        handmade=np.zeros((1,6),dtype=np.float32)
        handmade[0]=[.9,.89,.875,.87,.1,.1]
        idx2=np.array([[4,5],[2,3],[1,3],[1,2],[0,5],[0,4]],dtype=np.int32)
        wt2=np.array([[.5,.5],[.5,.5],[.5,.5],[.5,.5],[.5,.5],[.5,.5]],dtype=np.float32)
        selected,event=top1_competition(handmade,idx2,wt2)
        assert len(event)==1 and selected.argmax(axis=1)[0]==1, '可信竞争应当能实际换Top1'
        # 不允许通过调换其他Val查询影响独立查询本身。
        assert np.allclose(positive_evidence(raw[:2],idx,mm_w),corrected[:2])
        # 严格复用旧的真实512维缓存的协议/签名测试（在临时目录里模拟旧版）。
        with patch.dict(globals(),{'OUTPUT_ROOT':root/'previous_run_fixture'}):
            fake_cfg={'img_size':[384,128],'text_length':77}
            fake_val={'img_paths':['a.jpg','b.jpg'], 'image_pids':[1,2],
                      'captions':['a caption','b caption'],'caption_pids':[1,2]}
            previous_fp=old_data_fingerprint(fake_cfg,fake_val,'fake_train_ref')
            olddir=OUTPUT_ROOT/'双向邻近图传播_independent_fixture'
            (olddir/'缓存').mkdir(parents=True)
            atomic_json(olddir/'00_运行元数据.json',dict(
                fingerprint=previous_fp,settings=dict(
                    version=PREVIOUS_VERSION,mode='independent',
                    baseline_checkpoint_sha256=CK_SHA,baseline_config_sha256=CFG_SHA)))
            for label,count in (('gallery_3078',3078),('query_6158',6158),('train_reference_2048',2048)):
                cached=np.lib.format.open_memmap(olddir/'缓存'/(label+'.npy'),mode='w+',
                                                 dtype='float32',shape=(count,512))
                cached[:]=0;cached[:,0]=1
                cache_close(cached);del cached
                atomic_json(olddir/'缓存'/(label+'.json'),
                            dict(signature=previous_fp+'|'+label,size=count,done=count))
            newdir=OUTPUT_ROOT/'round2_output'
            (newdir/'缓存').mkdir(parents=True)
            assert reuse_previous_cache(fake_cfg,fake_val,'fake_train_ref','new_fp',newdir)
            for label,count in (('gallery_3078',3078),('query_6158',6158),('train_reference_2048',2048)):
                mm,done,meta=cache_open(newdir/'缓存',label,count,'new_fp|'+label)
                assert done==count and np.all(mm[:3,0]==1)
                cache_close(mm);del mm
        with patch.dict(globals(),{'OUTPUT_ROOT':root/'runs'}):
            out=prepare_folder('synthetic_round2','',synthetic=True)
            rows,z=run_pipeline(text,gallery,tp,vp,ref,out,'synthetic_round2',
                                torch.device('cpu'),synthetic=True)
            assert len(rows)==7 and z.is_file()
            with zipfile.ZipFile(z) as zz:
                assert zz.testzip() is None and '01_对照指标总表.csv' in zz.namelist()
                assert not any(item.startswith('缓存/') for item in zz.namelist())
            # 同目录续跑必须原样核验结果，而不是重新写入错误指标。
            second=prepare_folder('synthetic_round2',str(out),synthetic=True)
            assert second==out
            run_pipeline(text,gallery,tp,vp,ref,out,'synthetic_round2',
                         torch.device('cpu'),synthetic=True)
            changed=out/'逐查询'/'G4_多视角正证据传播.csv'
            with changed.open('a',encoding='utf8') as f:
                f.write('tampered\n')
            try:
                run_pipeline(text,gallery,tp,vp,ref,out,'synthetic_round2',
                             torch.device('cpu'),synthetic=True)
            except RuntimeError as exc:
                assert '篡改' in str(exc) or '不匹配' in str(exc)
            else:
                raise AssertionError('结果篡改后未拒绝续跑')
    say('[SELF-TEST PASS] G0-G6，独立查询，结构/语义图，正向证据，真实Top1竞争，续跑，CSV/JSON/ZIP')


def main():
    p=argparse.ArgumentParser(description='双向邻近图传播第二轮：原始IRRA G0 + 6组无训练对照')
    p.add_argument('--self-test',action='store_true',help='纯CPU完整合成测试，不依赖IRRA文件')
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--resume',type=str,default='',help='本轮输出目录，续跑用')
    p.add_argument('--no-cache-reuse',action='store_true',help='不从上一轮匹配的特征缓存自动复制')
    p.add_argument('--mode',choices=('independent',),default='independent',
                   help='固定独立查询模式，防止意外混用transductive协议')
    args=p.parse_args()
    if args.self_test:
        self_test()
    else:
        run_real(args)


if __name__=='__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
