# -*- coding: utf-8 -*-
"""
IRRA 双向邻近图传播 G0-G5，单文件、无训练、完整 CUHK-PEDES Validation。

运行：直接用 PyCharm (irra_clean) 点击 Run。默认 transductive 模式。
独立查询对照（以固定 Train 文本作反向检索参照）：
    python IRRA_双向邻近图传播_G0_G5_单文件.py --mode independent
合成数据自检：
    python IRRA_双向邻近图传播_G0_G5_单文件.py --self-test
续跑：
    python IRRA_双向邻近图传播_G0_G5_单文件.py --resume "...\\双向邻近图传播_时间戳"

六组（分数越大越相似）：
G0 原始 IRRA 余弦。
G1 历史 C3：图库 Mutual Top8 + image cos >= .70，按 image cos 加权，alpha=.30。
G2 图文双向 Reciprocal Rank Fusion（原始余弦/双向RRF按行尺度校准融合）。
G3 对 G2 分数执行原始 C3 图像传播。
G4 对 G2 执行查询感知图传播：视觉可信邻居+图文分数相合+反向排名可信度。
G5 在 G4 基础上对缺乏邻域/反向支持的孤立高分候选做弱竞争抑制。

NOTE: transductive = 利用完整 Val 文本集合构建反向图文排名，是批量查询协议；
independent = 固定 Train 文本描述作为反向参照，每条 Val 查询的修正不依赖其他Val查询。
PID只用于计算检索指标和**事后**同身份边诊断，任何重排序/参数计算都不用Val PID。
参数在运行前固定。Validation用于探索，不用Test选择参数。
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
VERSION = 'bidirectional_graph_g0g5_20261022_v1'
EXPECTED = dict(R1=66.43390655517578, R5=85.28743743896484,
                R10=90.58135986328125, mAP=61.86301040649414,
                mINP=47.24529266357422)
METRICS = ('R1', 'R5', 'R10', 'mAP', 'mINP')

# 预先固定：没有利用 Val 标签选择这些超参数。
GRAPH_K = 8
GRAPH_COS_THRESHOLD = 0.70
GRAPH_ALPHA = 0.30
RRF_K = 60
RRF_BLEND = 0.20
QUERY_SIM_TAU = 0.08
QUERY_GATE_FLOOR = 0.30
REVERSE_RANK_SCALE = 64.0
COMPETE_TOPK = 50
COMPETE_MARGIN = 0.03
COMPETE_STRENGTH = 0.15
REFERENCE_COUNT = 2048
REFERENCE_SEED = 20261021
IMAGE_BATCH = 4      # 6 GiB GPU 也可用
TEXT_BATCH = 64
SCORE_BATCH = 256
EVAL_BATCH = 128
ZIP_NAME = '发送给ChatGPT_双向邻近图传播_G0G5.zip'


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


def backward_ranks(score, mode, ref_score=None):
    """每张图像反向检索文本时当前查询的序位。训练参照法不使用其他Val查询。"""
    n_query, n_gallery = score.shape
    if n_query + 1 > 65535:
        raise ValueError('查询规模大于uint16反向排名容量')
    result = np.empty(score.shape, dtype=np.uint16)
    if mode == 'transductive':
        for j in range(n_gallery):
            order = np.argsort(-score[:, j], kind='stable')
            result[order, j] = np.arange(1, n_query+1, dtype=np.uint16)
    elif mode == 'independent':
        if ref_score is None or ref_score.shape[1] != n_gallery:
            raise ValueError('独立查询模式必须提供Train文本参考分数')
        count = len(ref_score)
        if count + 1 >= 65535:
            raise ValueError('训练参照样本太多')
        for j in range(n_gallery):
            ascending = np.sort(ref_score[:, j])
            # rank = 1 + 参考文本里有多少分数比当前查询高
            result[:, j] = (1 + count - np.searchsorted(ascending, score[:, j], side='right')).astype(np.uint16)
    else:
        raise ValueError('未知模式: ' + mode)
    return result


def bidirectional_rrf(score, reverse_ranks, rrf_k=RRF_K, beta=RRF_BLEND, block=128):
    """完整正向名次+独立反向名次，RRF再行内仿射校准，与余弦线性融合。"""
    if score.shape != reverse_ranks.shape:
        raise ValueError('双向排名矩阵尺寸不一致')
    n, width = score.shape
    out = np.empty_like(score)
    for lo in range(0, n, block):
        hi = min(n, lo+block)
        src = score[lo:hi]
        order = np.argsort(-src, axis=1, kind='stable')
        forward = np.empty(order.shape, dtype=np.int32)
        forward[np.arange(hi-lo)[:, None], order] = np.arange(1, width+1, dtype=np.int32)[None, :]
        rrf = 1.0 / (rrf_k + forward) + 1.0 / (rrf_k + reverse_ranks[lo:hi].astype(np.int32))
        rrf = rrf.astype(np.float32)
        orig_mu, orig_std = src.mean(axis=1, keepdims=True), src.std(axis=1, keepdims=True)
        rrf_mu, rrf_std = rrf.mean(axis=1, keepdims=True), rrf.std(axis=1, keepdims=True)
        scaled = orig_mu + orig_std * ((rrf-rrf_mu) / np.maximum(rrf_std, 1e-8))
        out[lo:hi] = (1-beta)*src + beta*scaled
    return out


def query_aware_graph(score, original, reverse, idx, weights,
                      alpha=GRAPH_ALPHA, tau=QUERY_SIM_TAU, floor=QUERY_GATE_FLOOR,
                      reverse_scale=REVERSE_RANK_SCALE, block=64):
    """G4。无PID。边权按当前查询的跨模态得分相合程度及反向名次动态调整。

    返回 G4 得分，以及原始分数在C3可信邻域里的平均值（用于G5的孤立高分诊断）。
    """
    if score.shape != original.shape or score.shape != reverse.shape:
        raise ValueError('query aware输入得分尺寸不一致')
    n = score.shape[1]
    if idx.shape != weights.shape or len(idx) != n:
        raise ValueError('图尺寸不一致')
    out = np.empty_like(score)
    base_peer = np.empty_like(score)
    connected = (weights.sum(axis=1) > 0).astype(np.float32)[None, :]
    for lo in range(0, len(score), block):
        hi = min(len(score), lo+block)
        src = score[lo:hi]
        original_src = original[lo:hi]
        rank_src = reverse[lo:hi].astype(np.float32)
        row_sum = np.zeros_like(src)
        weighted = np.zeros_like(src)
        peer = np.zeros_like(src)
        for slot in range(idx.shape[1]):
            js = idx[:, slot]
            w = weights[None, :, slot]
            s_j = src[:, js]
            agreement = floor + (1-floor)*np.exp(-np.abs(src - s_j) / tau)
            reverse_reliability = 1.0 / np.sqrt(1.0 + rank_src[:, js]/reverse_scale)
            gate = w * agreement * reverse_reliability
            row_sum += gate
            weighted += gate*s_j
            peer += w*original_src[:, js]
        new_peer = weighted / np.maximum(row_sum, 1e-8)
        # 保护性强度：即使重新归一化，也不让弱支持边获得全部alpha。
        effective_alpha = alpha * np.minimum(1.0, row_sum)
        out[lo:hi] = src + connected*effective_alpha*(new_peer-src)
        base_peer[lo:hi] = peer
    return out, base_peer


def competitive_penalty(g4, original, base_peer, reverse, weights,
                        topk=COMPETE_TOPK, margin=COMPETE_MARGIN,
                        strength=COMPETE_STRENGTH, block=128):
    """G5。仅前topk个当前候选，在具有视觉可信邻居时，抑制孤立高分且反向支持薄弱的候选。"""
    if not (g4.shape == original.shape == base_peer.shape == reverse.shape):
        raise ValueError('竞争抑制输入尺寸错误')
    out = g4.copy()
    connected = weights.sum(axis=1) > 0
    n_col = g4.shape[1]
    k = min(topk, n_col)
    for lo in range(0, len(g4), block):
        hi = min(lo+block, len(g4))
        candidate = np.argpartition(-g4[lo:hi], kth=k-1, axis=1)[:, :k]
        row = np.arange(hi-lo)[:, None]
        ids = candidate
        isolated_surplus = np.maximum(0.0, original[lo:hi][row, ids]
                                     - base_peer[lo:hi][row, ids] - margin)
        # 排名落到固定参照的64名之后，才逐渐增加处罚；没有反向证据则不误做竞争。
        rank_weakness = np.clip((reverse[lo:hi][row, ids].astype(np.float32)
                                 - REVERSE_RANK_SCALE) / (8*REVERSE_RANK_SCALE), 0, 1)
        penalty = strength * isolated_surplus * rank_weakness * connected[ids]
        out[lo:hi][row, ids] -= penalty
    return out


def edge_diagnostics(idx, strength, weight, vp):
    # Val身份仅在图构建完成后用作指标诊断，绝不进入图权重构造。
    mask = weight > 0
    same = vp[:, None] == vp[idx]
    return dict(undirected_edges=int(mask.sum()//2),
                directed_edges=int(mask.sum()),
                same_pid_directed=int((mask & same).sum()),
                different_pid_directed=int((mask & ~same).sum()),
                same_pid_fraction=float((mask & same).sum()/max(1,mask.sum())),
                average_visual_similarity=float(strength[mask].mean()) if mask.any() else None,
                pid_used_for_propagation=False)


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


def settings_dict(mode):
    return dict(version=VERSION, mode=mode, baseline_checkpoint_sha256=CK_SHA,
                baseline_config_sha256=CFG_SHA, graph_k=GRAPH_K,
                graph_cos_threshold=GRAPH_COS_THRESHOLD, graph_alpha=GRAPH_ALPHA,
                rrf_k=RRF_K, rrf_blend=RRF_BLEND,
                query_sim_tau=QUERY_SIM_TAU, query_gate_floor=QUERY_GATE_FLOOR,
                reverse_rank_scale=REVERSE_RANK_SCALE,
                compete_topk=COMPETE_TOPK, compete_margin=COMPETE_MARGIN,
                compete_strength=COMPETE_STRENGTH,
                train_reference_count=(REFERENCE_COUNT if mode=='independent' else 0),
                seed=REFERENCE_SEED)


def prepare_folder(fingerprint, mode, resume, synthetic=False):
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    settings = settings_dict(mode)
    if resume:
        out = Path(resume).resolve()
        if not out.is_dir() or out.parent.resolve() != OUTPUT_ROOT.resolve():
            raise RuntimeError('续跑路径必须位于“多种实验方向探索”目录下一层')
        f = out/'00_运行元数据.json'
        if not f.is_file():
            raise RuntimeError('续跑缺少00_运行元数据.json')
        prev = json.loads(f.read_text(encoding='utf-8'))
        if prev.get('fingerprint') != fingerprint or prev.get('settings') != settings:
            raise RuntimeError('续跑目录的模型/数据/实验参数不一致，禁止混写')
    else:
        name='双向邻近图传播_'+mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        out=OUTPUT_ROOT/name
        out.mkdir(parents=True,exist_ok=False)
        atomic_json(out/'00_运行元数据.json',dict(
            version=VERSION, fingerprint=fingerprint, settings=settings,
            synthetic=bool(synthetic),
            protocol=('全Validation文本用于反向排名：transductive'
                      if mode=='transductive' else '固定Train2048文本参考：每条Val查询独立'),
            labels_policy='Validation PID 仅用于metrics/edge_diagnostics，不进入算法；仅在Val探索',
            warnings=['不根据Val PID学习模型或建图','Val多分支对比不能代替最终独立Test',
                      'G2的双向排名在两个模式下参考集不同，不能直接横向混称为同一协议']))
        try:
            shutil.copy2(Path(__file__),out/'本轮实验单文件脚本.py')
        except OSError as e:
            say('[提示] 脚本副本未写入:',e)
    for folder in ('结果','逐查询','分析','缓存'):
        (out/folder).mkdir(parents=True,exist_ok=True)
    return out


def data_fingerprint(cfg, val, mode, train_ref_signature=''):
    body = dict(checkpoint_sha=CK_SHA, config_sha=CFG_SHA,
                images=val['img_paths'], gallery_pids=val['image_pids'],
                texts=val['captions'], query_pids=val['caption_pids'],
                img_size=cfg['img_size'],text_length=cfg['text_length'],
                mode=mode, reference_signature=train_ref_signature,
                parameters=settings_dict(mode))
    return hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False).encode('utf8')).hexdigest()


def run_pipeline(query, gallery, query_pid, gallery_pid, mode, out, fingerprint,
                 ref=None, device=None, synthetic=False):
    start = time.perf_counter()
    if device is None:
        device = torch.device('cpu')
    original = score_matrix(query,gallery,device)
    baseline, base_detail = evaluate(original,query_pid,gallery_pid)
    official = verify_baseline(original, query_pid, gallery_pid, baseline, synthetic=synthetic)
    atomic_json(out/'00_基线核验.json',dict(metrics=baseline,official=official,
               expected=(None if synthetic else EXPECTED),
               num_queries=len(query),num_gallery=len(gallery),mode=mode))
    result = {}
    result['G0'] = record_variant(out,fingerprint,'G0_原始IRRA',original,
                                 query_pid,gallery_pid,baseline,base_detail,'无传播原始余弦')
    idx,strength,weights,graph_stats = build_mutual_gallery_graph(gallery,device=device)
    graph_stats.update(edge_diagnostics(idx,strength,weights,gallery_pid))
    atomic_json(out/'分析'/'01_视觉邻近图事后诊断.json',graph_stats)
    say('[视觉图]',graph_stats)
    c3 = graph_smooth(original,idx,weights)
    result['G1'] = record_variant(out,fingerprint,'G1_C3历史可信图复核',c3,
                                 query_pid,gallery_pid,baseline,base_detail,
                                 '历史C3：互为Top8、视觉相似度>=.70、视觉相似度归一加权、alpha=.30')
    del c3

    ref_score = None
    if mode == 'independent':
        if ref is None:
            raise RuntimeError('独立模式的Train文本参考特征为空')
        ref_score = score_matrix(ref,gallery,device)
    reverse = backward_ranks(original, mode,ref_score)
    atomic_json(out/'分析'/'02_双向名次统计.json',dict(
        mode=mode,reference_size=(len(query) if mode=='transductive' else len(ref)),
        reverse_rank_min=int(reverse.min()), reverse_rank_max=int(reverse.max()),
        reverse_rank_median=float(np.median(reverse)),
        top1_reverse_rank_median=float(np.median(reverse[np.arange(len(query)),base_detail['top1']])),
        reverse_definition=('rank among all Val captions (transductive)' if mode=='transductive'
                            else '1 + number of fixed Train reference captions scoring higher (independent)')))
    del ref_score

    two_way = bidirectional_rrf(original,reverse)
    result['G2'] = record_variant(out,fingerprint,'G2_双向名次融合',two_way,
                                 query_pid,gallery_pid,baseline,base_detail,
                                 '图文正向名次+反向名次RRF(k=60)，行内尺度配平，与余弦按0.2融合')

    combined = graph_smooth(two_way,idx,weights)
    result['G3'] = record_variant(out,fingerprint,'G3_双向融合再C3传播',combined,
                                 query_pid,gallery_pid,baseline,base_detail,
                                 'G2双向RRF+历史C3静态图传播')
    del combined

    aware, peer = query_aware_graph(two_way,original,reverse,idx,weights)
    result['G4'] = record_variant(out,fingerprint,'G4_查询感知双向可信图',aware,
                                 query_pid,gallery_pid,baseline,base_detail,
                                 'G2+图文得分一致性及反向排名门控，逐查询动态加权可信图')

    suppressed = competitive_penalty(aware,original,peer,reverse,weights)
    result['G5'] = record_variant(out,fingerprint,'G5_双向图与困难负竞争抑制',suppressed,
                                 query_pid,gallery_pid,baseline,base_detail,
                                 'G4+仅前50候选中孤立高分且反向排名较弱者的轻量抑制，无PID')
    del suppressed, aware,peer, two_way

    summary = results_summary(out,fingerprint,baseline)
    elapsed = time.perf_counter()-start
    atomic_json(out/'02_完整实验摘要.json',dict(mode=mode,elapsed_sec=elapsed,
                fingerprint=fingerprint,baseline=baseline,
                results=[dict(name=r['experiment'],**{k:r[k] for k in METRICS},
                              corrected=r['corrected'],broken=r['broken']) for r in summary],
                graph=graph_stats,
                interpretation='G1是C3旧实验复现；G2-G5为待验证实验；独立与transductive协议不可混比'))
    zip_path=build_zip(out)
    say('[全部完成] 保存路径:',out)
    say('[上传ZIP]',zip_path)
    return summary,zip_path


def run_real(args):
    torch.set_num_threads(min(6,os.cpu_count() or 6))
    np.random.seed(REFERENCE_SEED)
    torch.manual_seed(REFERENCE_SEED)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，请在irra_clean环境运行或指定 --device cpu')
    device=torch.device(args.device)
    cfg=validate_environment()
    cfg,val,tp,vp=load_validation(cfg)
    reference_records,ref_sig = (None,'')
    if args.mode=='independent':
        reference_records,ref_sig=train_reference_texts(cfg['root_dir'],tp)
    fingerprint=data_fingerprint(cfg,val,args.mode,ref_sig)
    out=prepare_folder(fingerprint,args.mode,args.resume)
    say('[本轮输出]',out)
    model=None
    def get_model():
        nonlocal model
        if model is None:
            model=load_epoch58(cfg,device)
        return model
    gallery=extract_features(get_model,cfg,val,out/'缓存',fingerprint,
                             'gallery_3078',device,'image')
    query=extract_features(get_model,cfg,val,out/'缓存',fingerprint,
                           'query_6158',device,'text')
    ref=None
    if args.mode=='independent':
        ref=extract_features(get_model,cfg,reference_records,out/'缓存',fingerprint,
                             'train_reference_2048',device,'text')
    del model
    run_pipeline(query,gallery,tp,vp,args.mode,out,fingerprint,ref,device)


def self_test():
    """CPU合成集：算法、缓存断点、CSV/JSON/ZIP、原样续跑、篡改拒绝。"""
    from unittest.mock import patch
    rng=np.random.default_rng(20261022)
    old_root=OUTPUT_ROOT
    with tempfile.TemporaryDirectory(prefix='bidirectional_graph_') as td:
        root=Path(td)
        # 模拟特征缓存3/8处中断、恢复、错误指纹拒绝
        mm,done,meta=cache_open(root/'cache','image',8,'correct')
        assert done==0
        mm[:3]=1.0
        cache_progress(mm,meta,'correct',8,3)
        cache_close(mm);del mm
        mm,done,meta=cache_open(root/'cache','image',8,'correct')
        assert done==3 and np.all(mm[:3]==1.0)
        cache_close(mm);del mm
        try:
            cache_open(root/'cache','image',8,'wrong')
        except RuntimeError:
            pass
        else:
            raise AssertionError('错误缓存指纹未被拒绝')
        # 9个身份、每人多个正图；合成数据里掺噪，强制测试各分支完整输出。
        base=norm_np(rng.standard_normal((9,512)).astype(np.float32))
        vp=np.repeat(np.arange(9),4)
        tp=np.repeat(np.arange(9),5)
        # 使同身份图像强相似，确保Mutual Top8且>=.70的图传播边非空。
        g=norm_np(base[vp]+0.012*rng.standard_normal((len(vp),512)).astype(np.float32))
        t=norm_np(base[tp]+0.09*rng.standard_normal((len(tp),512)).astype(np.float32))
        # 故意植入一条Top1错误，用于检验逐查询CORRECTED/BROKEN统计链路。
        t[0] = base[1]
        ref=norm_np(rng.standard_normal((128,512)).astype(np.float32))
        ix,st,wt,stats=build_mutual_gallery_graph(g,device=torch.device('cpu'))
        assert stats['directed_edges']>0, '合成集必须触发可信图传播'
        rs=score_matrix(t,g,torch.device('cpu'))
        ref_sc=score_matrix(ref,g,torch.device('cpu'))
        rev=backward_ranks(rs,'independent',ref_sc)
        assert np.array_equal(rev[:1],backward_ranks(rs[:1],'independent',ref_sc)), \
               '独立模式不应依赖其他验证查询'
        _, base_det=evaluate(rs,tp,vp)
        assert not base_det['correct'].all(), '合成集必须含错误Top1'
        assert rev.max()>64, '合成集要覆盖竞争抑制的反向弱排名'
        # 对所有候选人为制造一次孤立高分+反向弱信号，单测G5确实惩罚。
        forced_original=rs.copy()
        forced_peer=forced_original.copy()
        forced_original[0,0]=0.9
        forced_peer[0,0]=0.15
        forced_rev=rev.copy()
        forced_rev[0,0]=400
        forced_g5=competitive_penalty(forced_original,forced_original,forced_peer,
                                       forced_rev,wt,topk=50)
        assert forced_g5[0,0] < forced_original[0,0], 'G5竞争抑制分支没有生效'
        qa,peer=query_aware_graph(rs,rs,rev,ix,wt)
        assert np.max(np.abs(qa-rs))>1e-6, 'G4查询感知传播分支没有生效'
        # 运行两个协议，并重新进入已有目录校验幂等续跑
        with patch.dict(globals(),{'OUTPUT_ROOT':root/'all_runs'}):
            for mode in ('transductive','independent'):
                fp='selftest_'+mode
                out=prepare_folder(fp,mode,'',synthetic=True)
                rows,zpath=run_pipeline(t,g,tp,vp,mode,out,fp,ref,torch.device('cpu'),synthetic=True)
                assert len(rows)==6
                assert {r['experiment'].split('_')[0] for r in rows}==set('G'+str(i) for i in range(6))
                assert zpath.is_file()
                with zipfile.ZipFile(zpath) as z:
                    assert z.testzip() is None
                    assert '01_对照指标总表.csv' in z.namelist()
                    assert not any(p.startswith('缓存/') for p in z.namelist())
                again=prepare_folder(fp,mode,str(out),synthetic=True)
                assert out==again
                run_pipeline(t,g,tp,vp,mode,out,fp,ref,torch.device('cpu'),synthetic=True)
                # 模拟结果损坏，必须拒绝在续跑中静默复用。
                path=out/'逐查询'/'G2_双向名次融合.csv'
                with path.open('a',encoding='utf-8') as f:
                    f.write('tamper\n')
                try:
                    run_pipeline(t,g,tp,vp,mode,out,fp,ref,torch.device('cpu'),synthetic=True)
                except RuntimeError as exc:
                    assert '篡改' in str(exc) or '不匹配' in str(exc)
                else:
                    raise AssertionError('未阻止已篡改的逐查询文件')
    say('[SELF-TEST PASS] 6组×2模式；全链路评估；mmap续跑；JSON/CSV/ZIP；篡改拒绝')


def main():
    parser=argparse.ArgumentParser(description='IRRA Epoch58双向邻近图传播G0-G5（单文件）')
    parser.add_argument('--self-test',action='store_true',help='CPU合成数据完整自测，不需要IRRA项目')
    parser.add_argument('--mode',choices=('transductive','independent'),default='transductive',
                        help='默认全Val文本批量查询；independent使用固定Train2048文本作为反向参照')
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--resume',type=str,default='',help='续跑：本轮子目录路径，禁止跨协议复用')
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run_real(args)


if __name__=='__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
