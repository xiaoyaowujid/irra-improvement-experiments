# -*- coding: utf-8 -*-
"""IRRA Epoch58 多种实验方向探索 / 第二轮：单文件、冻结权重、全量验证。
独立方向 B=Train参考Hub校正; C=可信图传播; E=翻转自适应;
O=开放集; D=监控退化; R=风险决策; P=候选成对比较;
M=跨视角正样本学习; H=梯度方向诊断。
模型不联网，不改变原IRRA权重。先用 --self-test，正式运行直接运行。
中断 --resume 之前的目录。输出为 IRRA-main/多种实验方向探索/第二轮_时间戳/。
实验性结果不等同独立Test结论。B1、R批次阈值属于transductive对照。
"""
from __future__ import annotations
import os, sys
if sys.platform == 'win32':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse, csv, gc, hashlib, json, math, re, shutil, tempfile, time, traceback, zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(r'G:\PyCharmProjects\IRRA\IRRA-main')
BEST_DIR = ROOT / 'logs' / 'CUHK-PEDES' / '20261009_095215_irra_baseline_b8_resume20'
OUTPUT_ROOT = ROOT / '多种实验方向探索'
VERSION = 'wide_exploration_round2_v3_20261021'
CHECKPOINT_SHA = 'f3ab12028e3406546b836ed6a75c8fc5e02a42146c5da13af423e7f5cf0e7ea6'
CONFIG_SHA = 'ec2fae46ec4e0cce06372789afe1c877a5cebeddf73654781e11b0484fc30fac'
EXPECTED = {'R1': 66.43390655517578, 'R5': 85.28743743896484,
            'R10': 90.58135986328125, 'mAP': 61.86301040649414,
            'mINP': 47.24529266357422}
BASELINE_TOLERANCE_PP = 0.07
IMAGE_BATCH = 4           # 6 GB CUDA
TEXT_BATCH = 64
SCORE_BATCH = 256
SCORE_EVAL_BATCH = 128
SEED = 20261021
EXPERIMENTS = ('B', 'C', 'E', 'O', 'D', 'R', 'P', 'M', 'H')
ZIP_NAME = '发送给ChatGPT_第二轮多方向探索.zip'


def say(*a):
    print(*a, flush=True)


def atomic_json(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=str(p.parent), prefix='.' + p.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush(); os.fsync(f.fileno())
        os.replace(name, p)
    finally:
        if Path(name).exists():
            Path(name).unlink()


def write_csv(p, records, fields):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=str(p.parent), prefix='.' + p.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader(); w.writerows(records)
        os.replace(name, p)
    finally:
        if Path(name).exists():
            Path(name).unlink()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()



def load_irra_yaml(path):
    """解析经 SHA256 校验的 IRRA yaml.dump 格式，安全支持 python/tuple。"""
    import yaml
    with Path(path).open(encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(cfg, dict):
        raise RuntimeError('IRRA configs.yaml 根节点必须是字典')
    return cfg


def validate_files():
    needed = [ROOT / 'model' / 'build.py', ROOT / 'datasets' / 'cuhkpedes.py',
              BEST_DIR / 'best.pth', BEST_DIR / 'configs.yaml']
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError('原始项目/最佳检查点缺失：\n' + '\n'.join(missing))
    for p, expected in [(BEST_DIR / 'best.pth', CHECKPOINT_SHA),
                        (BEST_DIR / 'configs.yaml', CONFIG_SHA)]:
        got = sha256(p)
        if got != expected:
            raise RuntimeError(f'拒绝使用未知的 Epoch58 文件：{p}\n期望 {expected}\n实际 {got}')
    # IRRA 的 configs.yaml 由 yaml.dump 保存，含 !!python/tuple 标签。
    # 已在上方核验配置文件 SHA256，采用与历史验证脚本一致的 FullLoader。
    # SafeLoader 会在 img_size / stride_size 等 tuple 字段直接抛 ConstructorError。
    cfg = load_irra_yaml(BEST_DIR / 'configs.yaml')
    for k, wanted in {'dataset_name': 'CUHK-PEDES', 'num_epoch': 60,
                       'batch_size': 8, 'text_length': 77,
                       'loss_names': 'sdm+mlm+id'}.items():
        if cfg.get(k) != wanted:
            raise RuntimeError(f'配置不符 {k}: {cfg.get(k)} != {wanted}')
    if tuple(cfg['img_size']) != (384, 128):
        raise RuntimeError('图像尺度不是最佳 Epoch58 配置(384,128)')
    return cfg


def dataset_parent(config_root):
    roots = [Path(config_root), ROOT / 'data', ROOT.parent / 'data',
             ROOT / 'data' / 'CUHK-PEDES']
    for base in roots:
        folder = base if base.name.lower() == 'cuhk-pedes' else base / 'CUHK-PEDES'
        if (folder / 'reid_raw.json').is_file() and (folder / 'imgs').is_dir():
            return folder.parent
    raise FileNotFoundError('找不到 CUHK-PEDES/reid_raw.json 和 imgs；检查 yaml root_dir 和项目 data 目录')


def load_dataset(cfg):
    if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
    from datasets.cuhkpedes import CUHKPEDES
    c = dict(cfg)
    c['root_dir'] = str(dataset_parent(cfg['root_dir']))
    ds = CUHKPEDES(root=c['root_dir'], verbose=False)
    val = ds.val
    actual = (len(val['captions']), len(val['img_paths']),
              len(ds.val_id_container), len(ds.train_id_container))
    if actual != (6158, 3078, 1000, 11003):
        raise RuntimeError(f'Val规模错误 {actual}，应为(6158,3078,1000,11003)')
    if any(not Path(p).is_file() for p in val['img_paths']):
        raise FileNotFoundError('图库图像缺失')
    tp = np.asarray(val['caption_pids'], dtype=np.int64)
    vp = np.asarray(val['image_pids'], dtype=np.int64)
    if not np.all(np.isin(tp, vp)):
        raise RuntimeError('Val存在缺少同PID图库图片的查询')
    return c, val, tp, vp


def load_model(cfg, device):
    """由最佳权重恢复CLIP结构，避免原仓库在build_model时联网或查找CLIP预训练文件。"""
    import importlib
    ck = torch.load(str(BEST_DIR / 'best.pth'), map_location='cpu', weights_only=False)
    if not isinstance(ck, dict) or not isinstance(ck.get('model'), dict):
        raise RuntimeError('checkpoint未包含model')
    if ck.get('epoch') is not None and int(ck['epoch']) != 58:
        raise RuntimeError(f'加载的不是Epoch58: {ck.get("epoch")}')
    state = ck['model']
    if state and all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    rxv = re.compile(r'^base_model\.visual\.transformer\.resblocks\.(\d+)\.')
    rxt = re.compile(r'^base_model\.transformer\.resblocks\.(\d+)\.')
    vision = {int(m.group(1)) for k in state if (m := rxv.match(k))}
    texts = {int(m.group(1)) for k in state if (m := rxt.match(k))}
    if vision != set(range(12)) or texts != set(range(12)):
        raise RuntimeError('模型不是已知的12层视觉与12层文本编码器')
    if int(state['classifier.weight'].shape[0]) != 11003:
        raise RuntimeError('分类头不是11003个Train身份')
    def shp(s): return state['base_model.' + s].shape
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
    build = importlib.import_module('model.build')
    from model.clip_model import CLIP
    original = build.build_CLIP_from_openai_pretrained
    try:
        build.build_CLIP_from_openai_pretrained = lambda *_: (CLIP(**net), {'embed_dim': net['embed_dim']})
        model = build.build_model(SimpleNamespace(**cfg), num_classes=11003)
    finally:
        build.build_CLIP_from_openai_pretrained = original
    if device.type == 'cpu': model.float()
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    del ck, state
    say('[Checkpoint] 严格加载 Epoch58，冻结所有参数')
    return model


def fingerprint(cfg, val):
    source = {'version': VERSION, 'checkpoint_sha': CHECKPOINT_SHA,
              'config_sha': CONFIG_SHA, 'images': val['img_paths'],
              'gallery_pids': val['image_pids'], 'texts': val['captions'],
              'text_pids': val['caption_pids'], 'size': cfg['img_size'],
              'length': cfg['text_length']}
    return hashlib.sha256(json.dumps(source, sort_keys=True,
                 ensure_ascii=False).encode('utf-8')).hexdigest()


def close_memmap(arr):
    """清理Win mmap: 不把其视图返回给调用者。"""
    if not isinstance(arr, np.memmap):
        raise TypeError('期望np.memmap')
    m = getattr(arr, '_mmap', None)
    try:
        arr.flush()
    finally:
        if m is not None and not m.closed:
            m.close()
    if m is not None and not m.closed:
        raise RuntimeError('缓存mmap关闭失败')


def load_cache(folder, key, count, fp):
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / (key + '.npy'); meta = folder / (key + '.json')
    if p.exists() != meta.exists():
        raise RuntimeError(f'缓存文件与元数据不同步: {key}')
    signature = fp + '|' + key
    if p.exists():
        state = json.loads(meta.read_text(encoding='utf-8'))
        if state.get('signature') != signature or state.get('count') != count or \
           not isinstance(state.get('done'), int) or not 0 <= state['done'] <= count:
            raise RuntimeError(f'缓存无法安全续跑: {key}')
        arr = np.lib.format.open_memmap(p, mode='r+')
        if arr.shape != (count, 512) or arr.dtype != np.float32:
            close_memmap(arr)
            raise RuntimeError(f'缓存维度错误: {key}')
        return arr, int(state['done']), meta
    arr = np.lib.format.open_memmap(p, mode='w+', dtype='float32', shape=(count, 512))
    try:
        arr.flush()
        atomic_json(meta, {'signature': signature, 'count': count, 'done': 0})
    except BaseException:
        close_memmap(arr)
        raise
    return arr, 0, meta


def update_cache(arr, meta, signature, count, done):
    arr.flush()
    atomic_json(meta, {'signature': signature, 'count': count, 'done': int(done)})


def normalize_np(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def transform_images(x, mode):
    if mode == 'original': return x
    if mode == 'flip': return x.flip(dims=[-1])
    if mode == 'gray':
        # 原官方CLIP图像归一化参数；反标准化后转灰度再重新标准化。
        means = torch.tensor((0.48145466, 0.4578275, 0.40821073),
                             device=x.device, dtype=x.dtype)[None, :, None, None]
        stds = torch.tensor((0.26862954, 0.26130258, 0.27577711),
                            device=x.device, dtype=x.dtype)[None, :, None, None]
        rgb = x * stds + means
        gray = (rgb[:, 0:1] * .299 + rgb[:, 1:2] * .587 + rgb[:, 2:3] * .114)
        return (gray.expand_as(rgb) - means) / stds
    if mode in ('edge_mask', 'center_mask'):
        # 在标准化图像中填0，等价于填CLIP通道平均色，而不是黑色。
        w = x.size(-1)
        cut = max(1, int(round(w * .18)))
        mask = torch.ones((1, 1, 1, w), device=x.device, dtype=x.dtype)
        if mode == 'edge_mask':
            mask[..., :cut] = 0; mask[..., w-cut:] = 0
        else:
            mask[..., cut:w-cut] = 0
        return x * mask
    raise ValueError('未知图像干预: ' + mode)


def split_caption(text):
    parts = [p.strip() for p in re.split(r'(?<=[,;.])\s+', text.strip()) if p.strip()]
    return parts or [text]


def transform_texts(texts, mode):
    if mode == 'original': return list(texts)
    out = []
    for t in texts:
        parts = split_caption(str(t))
        if mode == 'first_clause':
            selected = parts[0]
        elif mode == 'reverse_clauses':
            selected = ' '.join(reversed(parts))
        else:
            raise ValueError(mode)
        out.append(selected if selected.strip() else str(t))
    return out


def extract(model, cfg, val, cache_root, fp, key, device,
            kind, variant='original'):
    from torch.utils.data import DataLoader, Subset
    from datasets.bases import ImageDataset, TextDataset
    from datasets.build import build_transforms
    if kind == 'image':
        values = val['img_paths']; pids = val['image_pids']
        data = ImageDataset(pids, values,
                build_transforms(img_size=tuple(cfg['img_size']), is_train=False))
        bs = IMAGE_BATCH
    else:
        values = transform_texts(val['captions'], variant)
        pids = val['caption_pids']
        data = TextDataset(pids, values, text_length=int(cfg['text_length']))
        bs = TEXT_BATCH
    arr, start, mp = load_cache(cache_root, key, len(values), fp)
    try:
        if start < len(values):
            loader = DataLoader(Subset(data, range(start, len(data))),
                       batch_size=bs, shuffle=False, num_workers=0, pin_memory=False)
            pos = start
            say(f'[Embedding] {key}: 从 {start}/{len(values)} 开始')
            model = model() if callable(model) else model
            with torch.inference_mode():
                for batch in loader:
                    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                        raise RuntimeError('IRRA Dataset 返回格式不符')
                    z = batch[1].to(device)
                    if kind == 'image':
                        z = transform_images(z, variant)
                        h = model.encode_image(z)
                    else:
                        h = model.encode_text(z)
                    if h.ndim != 2 or h.shape[1] != 512:
                        raise RuntimeError(f'encode_{kind}输出维度错误 {tuple(h.shape)}')
                    h = F.normalize(h.float(), dim=1)
                    values_np = h.cpu().numpy()
                    if not np.all(np.isfinite(values_np)):
                        raise RuntimeError(f'{key}存在NaN/Inf，offset={pos}')
                    nxt = pos + len(values_np)
                    arr[pos:nxt] = values_np
                    pos = nxt
                    if pos % (bs * 32) < len(values_np) or pos == len(values):
                        update_cache(arr, mp, fp + '|' + key, len(values), pos)
                        say(f'[Embedding] {key}: {pos}/{len(values)}')
            update_cache(arr, mp, fp + '|' + key, len(values), pos)
        result = np.array(arr, dtype=np.float32, copy=True)
        if not np.all(np.isfinite(result)):
            raise RuntimeError(f'{key}数据损坏或未写完')
        return normalize_np(result)
    finally:
        close_memmap(arr)
        del arr


def cosine_scores(t, g, device):
    t = np.ascontiguousarray(np.asarray(t, dtype=np.float32))
    g = np.ascontiguousarray(np.asarray(g, dtype=np.float32))
    if t.ndim != 2 or g.ndim != 2 or t.shape[1] != g.shape[1]:
        raise ValueError('相似度输入维度不匹配')
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(g)):
        raise RuntimeError('输入向量NaN/Inf')
    out = np.empty((len(t), len(g)), dtype=np.float32)
    with torch.no_grad():
        gall = torch.from_numpy(g).to(device)
        for i in range(0, len(t), SCORE_BATCH):
            query = torch.from_numpy(t[i:i+SCORE_BATCH]).to(device)
            out[i:i+len(query)] = (query @ gall.T).float().cpu().numpy()
        del gall
    return out


def evaluate(score, tp, vp, batch=SCORE_EVAL_BATCH):
    """官方意义的图像级多正样本指标；分批避免大数组重复峰值。"""
    score = np.asarray(score, dtype=np.float32)
    tp = np.asarray(tp, dtype=np.int64); vp = np.asarray(vp, dtype=np.int64)
    if score.shape != (len(tp), len(vp)):
        raise ValueError('Score矩阵形状不符')
    if not np.all(np.isfinite(score)):
        raise RuntimeError('Score存在NaN/Inf')
    n = len(tp)
    top1 = np.empty(n, dtype=np.int32)
    rank = np.empty(n, dtype=np.int32)
    ap = np.empty(n, dtype=np.float32)
    inp = np.empty(n, dtype=np.float32)
    for i in range(0, n, batch):
        j = min(n, i + batch)
        ids = np.argsort(-score[i:j], axis=1, kind='stable')
        match = vp[ids] == tp[i:j, None]
        hits = match.sum(axis=1)
        if not np.all(hits):
            raise RuntimeError(f'有查询不存在正图库: {i}:{j}')
        best = np.argmax(match, axis=1) + 1
        last = len(vp) - np.argmax(match[:, ::-1], axis=1)
        ranks = np.arange(1, len(vp)+1, dtype=np.float32)[None, :]
        values = np.cumsum(match, axis=1) / ranks
        top1[i:j] = ids[:, 0]
        rank[i:j] = best
        ap[i:j] = (values * match).sum(axis=1) / hits
        inp[i:j] = hits / last
    metrics = {'R1': float(np.mean(rank <= 1) * 100),
               'R5': float(np.mean(rank <= 5) * 100),
               'R10': float(np.mean(rank <= 10) * 100),
               'mAP': float(np.mean(ap) * 100),
               'mINP': float(np.mean(inp) * 100)}
    return metrics, {'rank': rank, 'top1': top1, 'ap': ap, 'inp': inp,
                     'correct': rank == 1}


def verify_baseline(score, tp, vp, m):
    for k, want in EXPECTED.items():
        if abs(m[k] - want) > BASELINE_TOLERANCE_PP:
            raise RuntimeError(f'完整Val基线{k}异常: 得到 {m[k]:.5f}% 预期 {want:.5f}%')
    # 和原始官方 utils.metrics.rank 再做一次独立交叉验证。
    from utils.metrics import rank
    cmc, ap, inp, _ = rank(torch.from_numpy(score),
                torch.as_tensor(tp), torch.as_tensor(vp), max_rank=10, get_mAP=True)
    official = dict(R1=float(cmc[0]), R5=float(cmc[4]), R10=float(cmc[9]),
                    mAP=float(ap), mINP=float(inp))
    for k in EXPECTED:
        if abs(m[k] - official[k]) > .02:
            raise RuntimeError(f'独立评估与官方指标不一致: {k}, 本脚本{m[k]}, 官方{official[k]}')
    return official


def rows_for_comparison(name, base, current, tp, vp):
    fields = ['query_idx', 'pid', 'baseline_rank', 'variant_rank',
              'baseline_top1_idx', 'variant_top1_idx', 'baseline_top1_pid',
              'variant_top1_pid', 'baseline_correct', 'variant_correct',
              'change_type', 'delta_ap', 'delta_inp']
    rows = []
    for i, p in enumerate(tp):
        a, b = bool(base['correct'][i]), bool(current['correct'][i])
        cat = ('CORRECTED' if not a and b else
               'BROKEN' if a and not b else
               'BOTH_CORRECT' if a and b else 'BOTH_WRONG')
        rows.append({'query_idx': i, 'pid': int(p),
            'baseline_rank': int(base['rank'][i]),
            'variant_rank': int(current['rank'][i]),
            'baseline_top1_idx': int(base['top1'][i]),
            'variant_top1_idx': int(current['top1'][i]),
            'baseline_top1_pid': int(vp[base['top1'][i]]),
            'variant_top1_pid': int(vp[current['top1'][i]]),
            'baseline_correct': int(a), 'variant_correct': int(b),
            'change_type': cat,
            'delta_ap': round(float(current['ap'][i]-base['ap'][i]), 7),
            'delta_inp': round(float(current['inp'][i]-base['inp'][i]), 7)})
    return rows, fields


def read_results(out, fp):
    results = {}
    for p in sorted((out / '结果').glob('*.json')):
        if p.name == '00_baseline.json': continue
        d = json.loads(p.read_text(encoding='utf-8'))
        if d.get('fingerprint') != fp or d.get('version') != VERSION:
            raise RuntimeError('续跑时发现不匹配的旧结果: ' + str(p))
        per_csv = out / '逐查询' / (d['name'] + '.csv')
        if not per_csv.is_file() or not d.get('csv_sha256') or sha256(per_csv) != d['csv_sha256']:
            raise RuntimeError('结果的逐查询CSV已更改/损坏，拒绝读取: ' + str(per_csv))
        results[d['name']] = d
    return results


def update_summary(out, fp, baseline):
    results = read_results(out, fp)
    records = []
    for name, result in sorted(results.items()):
        m = result['metrics']
        records.append({'name': name, 'family': result['family'],
                        **{k: round(m[k], 6) for k in EXPECTED},
                        **{'delta_'+k: round(m[k]-baseline[k], 6) for k in EXPECTED},
                        'corrected': result['change']['corrected'],
                        'broken': result['change']['broken'],
                        'net_correct': result['change']['net_correct'],
                        'rank_changed': result['change']['top1_image_changed']})
    cols = ['name', 'family', *EXPECTED.keys(),
            *('delta_'+k for k in EXPECTED),
            'corrected', 'broken', 'net_correct', 'rank_changed']
    write_csv(out / '01_全部实验指标总表.csv', records, cols)
    return records


def record_variant(out, fp, name, family, score, tp, vp, bm, bd, note):
    path = out / '结果' / (name + '.json')
    csvpath = out / '逐查询' / (name + '.csv')
    if path.exists():
        d = json.loads(path.read_text(encoding='utf-8'))
        if d.get('fingerprint') != fp or d.get('version') != VERSION:
            raise RuntimeError('已有结果属于其他缓存，拒绝复用: ' + name)
        if not csvpath.exists() or d.get('csv_sha256') != sha256(csvpath):
            raise RuntimeError('JSON/逐查询CSV不完整或校验失败，拒绝续跑: ' + name)
        say('[已有结果，跳过]', name)
        return d
    m, per = evaluate(score, tp, vp)
    change = {'corrected': int(np.sum(~bd['correct'] & per['correct'])),
              'broken': int(np.sum(bd['correct'] & ~per['correct'])),
              'net_correct': int(np.sum(per['correct']) - np.sum(bd['correct'])),
              'top1_image_changed': int(np.sum(bd['top1'] != per['top1']))}
    rows, cols = rows_for_comparison(name, bd, per, tp, vp)
    write_csv(csvpath, rows, cols)
    d = {'name': name, 'family': family, 'version': VERSION,
         'fingerprint': fp, 'metrics': m, 'delta_vs_baseline':
            {k: m[k] - bm[k] for k in EXPECTED}, 'change': change,
         'note': note, 'full_validation': True, 'num_queries': len(tp),
         'num_gallery': len(vp), 'csv_sha256': sha256(csvpath)}
    atomic_json(path, d)
    update_summary(out, fp, bm)
    say('[指标]', name, 'R1={:.4f} (Δ{:+.4f})'.format(m['R1'], m['R1']-bm['R1']),
        'mAP={:.4f} (Δ{:+.4f})'.format(m['mAP'], m['mAP']-bm['mAP']),
        '纠错', change['corrected'], '破坏', change['broken'])
    return d



# ━━━━━━━━━━━━━━━━━━━━━━━━ 第二轮新实验定义 ━━━━━━━━━━━━━━━━━━━━━━━━
REFERENCE_COUNT = 2048
REFERENCE_SMALL = 512
GRAPH_K = 8
PAIR_IDS = 320                   # 两张图 / 身份，仅 Train 数据
PAIR_BATCH = 96
PAIR_STEPS = 240
MULTIPOS_STEPS = 240
TRAIN_SEEDS = (42, 43)
RISK_FRACTION = .20
OPEN_UNKNOWN_PID_FRACTION = .20


def train_annotation(rootdir, val_pids):
    """原始标注 Train，不调用 val 对应身份监督；源文件 SHA 绑定缓存。"""
    file = Path(rootdir) / 'CUHK-PEDES' / 'reid_raw.json'
    with file.open(encoding='utf-8') as f:
        rows = json.load(f)
    train = []
    for row in rows:
        if row.get('split') != 'train':
            continue
        pid = int(row['id'])
        path = str(row['file_path']).replace('\\', '/')
        folder = file.parent
        candidates = [folder / 'imgs' / path, folder / path]
        image = next((p for p in candidates if p.is_file()), None)
        if image is None:
            raise FileNotFoundError(f'Train 图像缺失: {path}')
        captions = row.get('captions') or []
        if not captions:
            raise RuntimeError('Train 图像没有描述')
        train.append((pid, str(image), tuple(str(x) for x in captions)))
    identities = {x[0] for x in train}
    if len(identities) != 11003 or len(train) < 30000:
        raise RuntimeError(f'Train 协议异常: {len(train)} images, {len(identities)} identities')
    if identities & set(map(int, val_pids)):
        raise RuntimeError('Train 与 Val 身份交叉，禁止训练')
    return train, sha256(file)


def prepare_train_samples(train):
    """预设固定身份/样本选择；完全不看 Val 标签和任何指标。"""
    all_captions = [(pid, caption) for pid, _, caps in train for caption in caps]
    rng = np.random.default_rng(20261021)
    sample = rng.choice(len(all_captions), REFERENCE_COUNT, replace=False)
    ref_texts = [all_captions[int(i)] for i in sample]
    # 训练对必须是同身份不同图，不是同一图复制。
    by_pid = {}
    for pid, path, captions in train:
        by_pid.setdefault(pid, []).append((path, captions[0]))
    valid_pid = [pid for pid, items in sorted(by_pid.items()) if len(items) >= 2]
    if len(valid_pid) < PAIR_IDS:
        raise RuntimeError('Train 双视角身份数量不够')
    chosen = rng.choice(valid_pid, PAIR_IDS, replace=False)
    pairs = []
    for pid in chosen:
        first, second = rng.choice(len(by_pid[int(pid)]), 2, replace=False)
        for ix in (first, second):
            path, caption = by_pid[int(pid)][int(ix)]
            pairs.append((int(pid), path, caption))
    return ref_texts, pairs


def load_train_embeddings(model_factory, cfg, train, annotation_sha, val_fp, cache, device):
    refs, pairs = prepare_train_samples(train)
    pids_r = [p for p, _ in refs]; texts_r = [s for _, s in refs]
    pids_i = [p for p, _, _ in pairs]; paths_i = [path for _, path, _ in pairs]
    txt_p = [text for _, _, text in pairs]
    sub_fp = hashlib.sha256((val_fp + annotation_sha +
        json.dumps({'ref': refs, 'pair': pairs, 'setting': [REFERENCE_COUNT, PAIR_IDS]},
                   ensure_ascii=False)).encode('utf-8')).hexdigest()
    pseudo_ref = {'caption_pids': pids_r, 'captions': texts_r}
    pseudo_pair_img = {'image_pids': pids_i, 'img_paths': paths_i}
    pseudo_pair_txt = {'caption_pids': pids_i, 'captions': txt_p}
    ref = extract(model_factory, cfg, pseudo_ref, cache, sub_fp,
                  'train_reference_2048', device, 'text')
    # image & text pair features are only needed for the supervised P/M/H branches.
    return ref, pairs, sub_fp, pseudo_pair_img, pseudo_pair_txt


def train_pair_features(model_factory, cfg, sub_fp, cache, device, image_records, text_records):
    x = extract(model_factory, cfg, image_records, cache, sub_fp,
                'train_pairs_images', device, 'image')
    y = extract(model_factory, cfg, text_records, cache, sub_fp,
                'train_pairs_texts', device, 'text')
    pids = np.array(image_records['image_pids'], dtype=np.int64)
    if not np.array_equal(pids, np.array(text_records['caption_pids'], dtype=np.int64)):
        raise RuntimeError('训练正样本图文对应关系被破坏')
    assert len(pids) == PAIR_IDS * 2
    return y, x, pids


def hubness_reference(query, gallery, ref, device, neighbors=20):
    """训练文本建立每张图库图像的固定偏置，查询不参与偏置拟合。"""
    if len(ref) < neighbors:
        raise ValueError('reference insufficient')
    r = cosine_scores(ref, gallery, device)
    h = np.partition(r, len(r) - neighbors, axis=0)[-neighbors:].mean(axis=0)
    return np.asarray(2. * query - h[None, :], dtype=np.float32), h


def gallery_graph(gallery, device, k=8):
    with torch.no_grad():
        a = torch.from_numpy(np.ascontiguousarray(gallery)).to(device)
        sim = a @ a.T
        sim.fill_diagonal_(-float('inf'))
        values, ids = torch.topk(sim, k, dim=1, sorted=True)
        idx = ids.cpu().numpy().astype(np.int32)
        strengths = values.cpu().numpy().astype(np.float32)
    return idx, strengths


def graph_scores(scores, indices, strengths, alpha=.30, strategy='plain', block=64):
    """每个图库节点按无标签邻居做传播，没有有效邻居时保持原分数。"""
    n = scores.shape[1]
    if indices.shape != strengths.shape or indices.shape[0] != n:
        raise ValueError('kNN索引/相似度维度不符')
    if np.any(indices < 0) or np.any(indices >= n):
        raise ValueError('图邻居索引越界')
    valid = np.ones_like(strengths, dtype=bool)
    if strategy in ('mutual', 'mutual_strong'):
        neighbor_sets = [set(map(int, r)) for r in indices]
        for i in range(n):
            for j, other in enumerate(indices[i]):
                valid[i, j] = i in neighbor_sets[int(other)]
    if strategy == 'mutual_strong':
        valid &= strengths >= .70  # 事先固定，不根据Val指标调整
    if strategy == 'plain':
        weights = np.ones_like(strengths, dtype=np.float32)
    elif strategy == 'mutual':
        weights = valid.astype(np.float32)
    elif strategy == 'mutual_strong':
        weights = np.maximum(strengths, 0) * valid
    else:
        raise ValueError(strategy)
    total = weights.sum(axis=1)
    weights = weights / np.maximum(total[:, None], 1.e-8)
    out = np.empty_like(scores)
    for start in range(0, len(scores), block):
        src = scores[start:start+block]
        mixed = np.zeros_like(src)
        for j in range(indices.shape[1]):
            mixed += src[:, indices[:, j]] * weights[None, :, j]
        valid_node = (total > 0).astype(np.float32)[None, :]
        out[start:start+len(src)] = src + (alpha * valid_node) * (mixed - src)
    return out, dict(connected_nodes=int((total > 0).sum()), isolated_nodes=int((total == 0).sum()),
                     directed_edges=int((weights > 0).sum()))


def fused_gallery(g, flip, policy):
    cos = (g * flip).sum(axis=1)
    if policy == 'equal':
        w = np.full(len(g), .5, dtype=np.float32)
    elif policy == 'conservative':
        w = np.full(len(g), .25, dtype=np.float32)
    elif policy == 'agreement':
        # 只根据图像本身与翻转图像的一致性；对所有查询的规则相同。
        w = .10 + .40 * np.clip((cos - .75) / .22, 0., 1.)
    else:
        raise ValueError(policy)
    return normalize_np((1. - w[:, None]) * g + w[:, None] * flip), dict(
        flip_weight_mean=float(w.mean()), flip_weight_std=float(w.std()),
        original_flip_cos_mean=float(cos.mean()))


def image_degrade(x, policy):
    """归一化张量上的受控退化，不声称模拟真实JPEG或人体轮廓。"""
    if policy == 'blur5':
        return F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
    if policy == 'lowres_4x':
        small = F.interpolate(x, size=(x.shape[-2] // 4, x.shape[-1] // 4),
                              mode='bilinear', align_corners=False)
        return F.interpolate(small, size=x.shape[-2:], mode='bilinear', align_corners=False)
    if policy == 'dim':
        mean = x.new_tensor((.48145466, .4578275, .40821073))[None, :, None, None]
        std = x.new_tensor((.26862954, .26130258, .27577711))[None, :, None, None]
        rgb = (x * std + mean).clamp(0., 1.)
        return (((rgb * .55 + .03).clamp(0., 1.)) - mean) / std
    raise ValueError(policy)


# extract() is the same proven dataset/tokenization path as the first round, with
# additional deterministic image perturbations. Cache variant name binds operation.
_original_transform_images = transform_images

def transform_images(x, mode):
    if mode in ('blur5', 'lowres_4x', 'dim'):
        return image_degrade(x, mode)
    return _original_transform_images(x, mode)


def auc_roc(confidence, label_positive):
    """AUROC, including average-rank tie handling; no sklearn dependency."""
    confidence = np.asarray(confidence, dtype=np.float64)
    labels = np.asarray(label_positive, dtype=bool)
    if len(confidence) != len(labels) or not np.isfinite(confidence).all():
        raise ValueError('AUROC input incorrect')
    n1 = int(labels.sum()); n0 = len(labels)-n1
    if not n1 or not n0: raise ValueError('AUROC needs two classes')
    order = np.argsort(confidence, kind='stable')
    s = confidence[order]; rank_sum = 0.
    i=0
    while i<len(s):
        j=i+1
        while j<len(s) and s[j]==s[i]: j+=1
        rank_sum += float(labels[order[i:j]].sum()) * (i + j + 1.) / 2.
        i=j
    return (rank_sum - n1*(n1+1.)/2.) / (n1*n0)


def open_set_evaluate(out, base_scores, query_pids, gallery_pids):
    """身份级held-out图库模拟未知行人；不是原6158x3078闭集指标。"""
    known_ids = np.array(sorted(set(map(int, gallery_pids))), dtype=np.int64)
    rng = np.random.default_rng(20261021)
    n_unknown = int(round(len(known_ids)*OPEN_UNKNOWN_PID_FRACTION))
    removed = set(map(int, rng.choice(known_ids, n_unknown, replace=False)))
    retained_gallery = np.array([int(p) not in removed for p in gallery_pids], dtype=bool)
    s = np.ascontiguousarray(base_scores[:, retained_gallery], dtype=np.float32)
    has_person = np.array([int(p) not in removed for p in query_pids], dtype=bool)
    maxval = s.max(axis=1)
    two = np.partition(s, -2, axis=1)[:, -2:]
    margins = np.abs(two[:,1] - two[:,0])
    auc_max = auc_roc(maxval, has_person)
    auc_margin = auc_roc(margins, has_person)
    # 只对确实存在目标的查询计算检索；类别缺席查询不定义R1/mAP。
    close_metrics, details = evaluate(s[has_person], query_pids[has_person],
                                       gallery_pids[retained_gallery])
    rows=[]
    for i in range(len(query_pids)):
        rows.append(dict(query_idx=i,pid=int(query_pids[i]),present=int(has_person[i]),
                    confidence_max=float(maxval[i]),confidence_gap=float(margins[i])))
    write_csv(out/'开放集'/'O_query_scores.csv', rows, list(rows[0]))
    report = dict(name='O1_fixed_unknown_identities', unknown_pid_fraction=OPEN_UNKNOWN_PID_FRACTION,
                  unknown_pid_count=len(removed), known_query_count=int(has_person.sum()),
                  unknown_query_count=int((~has_person).sum()),
                  gallery_after_removal=int(retained_gallery.sum()),
                  auroc_from_max_similarity=float(auc_max),
                  auroc_from_top2_gap=float(auc_margin),
                  known_only_retrieval_metrics=close_metrics,
                  not_directly_comparable_with_full_gallery=True,
                  note='删除身份仅用于模拟未知身份，不用该身份标签训练任何检测器或调阈值；AUROC是Val探索性诊断。')
    atomic_json(out/'开放集'/'O1_open_set.json',report)
    return report


def margin_risk(score):
    top = np.partition(score, -2, axis=1)[:, -2:]
    return np.abs(top[:,0] - top[:,1])


def risk_route(scores, rerank, q=.20):
    gaps = margin_risk(scores)
    k = max(1, int(round(len(gaps)*q)))
    selection = np.zeros(len(gaps), dtype=bool)
    selection[np.argsort(gaps, kind='stable')[:k]] = True
    result = scores.copy()
    result[selection] = rerank[selection]
    return result, selection


def paired_bootstrap(out, name, tp, b_detail, v_detail, seed=123, nrep=600):
    """身份聚类配对区间；探索性未做多重比较校正。"""
    groups, idx = np.unique(tp, return_inverse=True)
    counts = np.bincount(idx, minlength=len(groups))
    rng = np.random.default_rng(seed)
    raw = {
        'R1': np.asarray(v_detail['correct'], np.float32) - np.asarray(b_detail['correct'],np.float32),
        'mAP': v_detail['ap'] - b_detail['ap'],
        'mINP': v_detail['inp'] - b_detail['inp']}
    result=[]
    for key, change in raw.items():
        num = np.bincount(idx, weights=change, minlength=len(groups))
        bs = np.empty(nrep, dtype=np.float64)
        for k in range(nrep):
            draw = rng.integers(len(groups), size=len(groups))
            bs[k] = num[draw].sum() / counts[draw].sum() *100.
        result.append(dict(name=name, metric=key,
             observed_delta_pp=float(np.mean(change)*100.),
             ci_025_pp=float(np.quantile(bs,.025)), ci_975_pp=float(np.quantile(bs,.975)),
             clusters=len(groups), repeats=nrep))
    write_csv(out/'统计'/(name+'_PID_bootstrap.csv'),result,list(result[0]))


def write_variant(out, fp, name, family, s, tp, vp, bm, bd, note):
    data = record_variant(out,fp,name,family,s,tp,vp,bm,bd,note)
    # 即使旧结果可恢复，统计文件也能重建；全部从当前分数计算。
    _, det = evaluate(s,tp,vp)
    paired_bootstrap(out,name,tp,bd,det)
    return data


def atomic_torch(path, state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=str(path.parent),suffix='.pt.tmp');os.close(fd)
    try:
        torch.save(state,name)
        os.replace(name,path)
    finally:
        if Path(name).exists():Path(name).unlink()

class PairComparator(torch.nn.Module):
    """反对称的真正候选对判别器：score(A,B)=-score(B,A)。"""
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(1024,64),
              torch.nn.ReLU(),torch.nn.Linear(64,1))
    def forward(self, query, a, b):
        delta=a-b
        z=torch.cat((query*delta,delta), dim=-1)
        return (self.net(z)-self.net(-z)).squeeze(-1)


class PositiveAdapter(torch.nn.Module):
    """身份内跨视角正例训练：只调整文字特征，完全不更新IRRA。"""
    def __init__(self):
        super().__init__()
        self.fc1=torch.nn.Linear(512,64)
        self.fc2=torch.nn.Linear(64,512)
        torch.nn.init.zeros_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
    def forward(self, x):
        return F.normalize(x+.15*self.fc2(F.relu(self.fc1(x))),dim=-1)


def create_pair_split(pid):
    identity = np.unique(pid)
    rng = np.random.default_rng(12345)
    identity=rng.permutation(identity)
    threshold=int(.8*len(identity))
    train_ids=set(map(int, identity[:threshold]))
    mask=np.array([int(k) in train_ids for k in pid])
    pos=np.empty(len(pid),dtype=np.int32)
    for ix, key in enumerate(pid):
        matches=np.where(pid==key)[0]
        if len(matches)!=2: raise RuntimeError('每个训练身份必须恰好有两张图像')
        pos[ix]=int(matches[1] if matches[0]==ix else matches[0])
    return mask,pos


def make_negative_pool(t,g,pid, mask, top=12):
    """从 Train 中的不同 PID 选困难负例，不接触任何 Val 身份。"""
    eligible=np.where(mask)[0]
    sim=t[eligible] @ g[eligible].T
    same=pid[eligible,None]==pid[None,eligible]
    sim[same]=-np.inf
    if len(eligible)<=top+2:raise RuntimeError('训练身份数不足')
    order=np.argpartition(sim,-top,axis=1)[:,-top:]
    # 固定排序以保证跨系统复现
    scores=np.take_along_axis(sim,order,axis=1)
    order=np.take_along_axis(order,np.argsort(-scores,axis=1,kind='stable'),axis=1)
    return eligible,eligible[order]


def train_pair_comparator(t,g,pid,seed,device,fp,checkpoint):
    mask,positive=create_pair_split(pid)
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    net=PairComparator().to(device)
    if checkpoint.exists():
        ck=torch.load(str(checkpoint),map_location='cpu',weights_only=False)
        if ck.get('fingerprint')!=fp or ck.get('seed')!=seed:
            raise RuntimeError('候选对训练状态与当前实验不匹配')
        net.load_state_dict(ck['model'],strict=True)
        return net.eval(),ck['holdout_accuracy']
    eligible,hard=make_negative_pool(t,g,pid,mask)
    tr=torch.from_numpy(np.ascontiguousarray(t)).to(device)
    gg=torch.from_numpy(np.ascontiguousarray(g)).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=7e-4,weight_decay=1e-4)
    net.train()
    for it in range(PAIR_STEPS):
        loc=rng.integers(len(eligible),size=PAIR_BATCH)
        chosen=eligible[loc]
        bad=hard[loc,rng.integers(hard.shape[1],size=PAIR_BATCH)]
        # 一半随机异身份，保证并非只学习一小簇相似人
        draw=rng.choice(eligible,size=PAIR_BATCH)
        collide=pid[draw]==pid[chosen]
        while np.any(collide):
            draw[collide]=rng.choice(eligible,size=int(collide.sum()))
            collide=pid[draw]==pid[chosen]
        bad[:PAIR_BATCH//2]=draw[:PAIR_BATCH//2]
        truth=(rng.random(PAIR_BATCH)>.5).astype(np.float32)
        good=torch.as_tensor(positive[chosen],device=device)
        wrong=torch.as_tensor(bad,device=device)
        first=torch.where(torch.as_tensor(truth,device=device).bool(),good,wrong)
        second=torch.where(torch.as_tensor(truth,device=device).bool(),wrong,good)
        target=torch.as_tensor(truth*2-1,dtype=torch.float32,device=device)
        logits=net(tr[torch.as_tensor(chosen,device=device,dtype=torch.long)],gg[first],gg[second])
        loss=F.softplus(-target*logits).mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    net.eval()
    # 身份互斥的Train内部holdout，只用于诊断，绝不据此挑Validation规则。
    hold=np.where(~mask)[0]
    rng2=np.random.default_rng(20261021)
    npool=np.array(hold,dtype=int)
    hold_scores=[]
    with torch.no_grad():
        for st in range(0,len(hold),64):
            ix=hold[st:st+64]
            neg=rng2.choice(npool,size=len(ix))
            wrong=pid[neg]==pid[ix]
            while wrong.any():
                neg[wrong]=rng2.choice(npool,size=int(wrong.sum()))
                wrong=pid[neg]==pid[ix]
            ixt=torch.as_tensor(ix,device=device,dtype=torch.long)
            pst=torch.as_tensor(positive[ix],device=device,dtype=torch.long)
            ngt=torch.as_tensor(neg,device=device,dtype=torch.long)
            z=net(tr[ixt],gg[pst],gg[ngt]).detach().cpu().numpy()
            hold_scores.extend(z.tolist())
    acc=float((np.asarray(hold_scores)>0).mean())
    atomic_torch(checkpoint,dict(fingerprint=fp,seed=seed,model={k:v.detach().cpu() for k,v in net.state_dict().items()},
                                 holdout_accuracy=acc,steps=PAIR_STEPS))
    return net,acc


def pair_topk_rerank(score,t,g,net,device,k=10,batch_q=32):
    """仅在原始TopK内部重排。所有其余图像的相对顺序保持不变。"""
    result=score.copy()
    n=len(score)
    chosen=np.argsort(-score,axis=1,kind='stable')[:,:k].astype(np.int32)
    iu,ju=np.triu_indices(k,k=1)
    net.eval()
    with torch.no_grad():
        gallery=torch.from_numpy(np.ascontiguousarray(g)).to(device)
        for st in range(0,n,batch_q):
            stop=min(n,st+batch_q)
            candidates=chosen[st:stop]
            gallery_b=gallery[torch.as_tensor(candidates,device=device)]
            query=torch.from_numpy(np.ascontiguousarray(t[st:stop])).to(device)
            a=gallery_b[:,iu,:];b=gallery_b[:,ju,:]
            logits=net(query[:,None,:].expand(-1,len(iu),-1),a,b)
            delta=logits.detach().cpu().numpy()
            wins=np.zeros((len(candidates),k),dtype=np.float32)
            p=1./(1.+np.exp(-np.clip(delta,-40,40)))
            for j,(u,v) in enumerate(zip(iu,ju)):
                wins[:,u]+=p[:,j]
                wins[:,v]+=1.-p[:,j]
            # 同分时根据原始排序确定性破同分；绝不看任何真实PID。
            wins+= (k-np.arange(k))[None,:]*1e-5
            new_order=np.argsort(-wins,axis=1,kind='stable')
            for row_local, global_row in enumerate(range(st,stop)):
                old_values=score[global_row,candidates[row_local]].copy()
                result[global_row,candidates[row_local,new_order[row_local]]]=old_values
    return result


def train_positive_adapter(t,g,pid,seed,device,fp,checkpoint):
    mask,positive=create_pair_split(pid)
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    net=PositiveAdapter().to(device)
    if checkpoint.exists():
        ck=torch.load(str(checkpoint),map_location='cpu',weights_only=False)
        if ck.get('fingerprint')!=fp or ck.get('seed')!=seed:
            raise RuntimeError('跨视角Adapter训练状态不匹配')
        net.load_state_dict(ck['model'],strict=True)
        return net.eval(),ck['holdout_pos_cos']
    tr=torch.from_numpy(np.ascontiguousarray(t)).to(device)
    gg=torch.from_numpy(np.ascontiguousarray(g)).to(device)
    train_ix=np.where(mask)[0]
    hold_ix=np.where(~mask)[0]
    opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4)
    net.train()
    for it in range(MULTIPOS_STEPS):
        chosen=rng.choice(train_ix,size=PAIR_BATCH,replace=True)
        choose=torch.as_tensor(chosen,device=device,dtype=torch.long)
        target=torch.as_tensor(positive[chosen],device=device,dtype=torch.long)
        emb=net(tr[choose]); true=gg[target]
        loss=(1.-(emb*true).sum(dim=1)).mean() + .10*(1.-(emb*tr[choose]).sum(dim=1)).mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    net.eval()
    with torch.no_grad():
        held=torch.as_tensor(hold_ix,device=device,dtype=torch.long)
        target=torch.as_tensor(positive[hold_ix],device=device,dtype=torch.long)
        hold_cos=float((net(tr[held])*gg[target]).sum(dim=-1).mean().item())
    atomic_torch(checkpoint,dict(fingerprint=fp,seed=seed,model={k:v.detach().cpu() for k,v in net.state_dict().items()},
                                 holdout_pos_cos=hold_cos,steps=MULTIPOS_STEPS))
    return net,hold_cos


def adapt_text(model, t, device, block=512):
    model.eval();out=[]
    with torch.no_grad():
        for i in range(0,len(t),block):
            out.append(model(torch.from_numpy(np.ascontiguousarray(t[i:i+block])).to(device)).cpu().numpy())
    return normalize_np(np.concatenate(out))


def gradient_conflict_report(out,t,g,pid):
    """仅描述梯度方向，不由方向宣称具体参数更新的因果收益。"""
    mask,pos=create_pair_split(pid)
    rng=np.random.default_rng(20261021)
    choices=np.where(mask)[0]
    samples=rng.choice(choices,size=min(500,len(choices)),replace=True)
    rows=[]
    for k,i in enumerate(samples):
        other=choices[pid[choices]!=pid[i]]
        j=int(rng.choice(other))
        # 在固定的对角尺度适配器w=1处求解析梯度：
        # L_pos=-dot(w*t, image_pos), L_neg=+dot(w*t, image_neg)。
        gp=-(t[i]*g[pos[i]])
        gn=t[i]*g[j]
        cos=float(np.dot(gp,gn)/(max(float(np.linalg.norm(gp)*np.linalg.norm(gn)),1e-12)))
        rows.append(dict(probe=k,train_query_index=int(i),train_pid=int(pid[i]),
                    train_negative_index=j,gradient_cos=cos,conflict=int(cos<0)))
    write_csv(out/'梯度'/'H_训练信号梯度方向.csv',rows,list(rows[0]))
    data={'train_only':True,'n':len(rows),'negative_fraction':float(np.mean([r['conflict'] for r in rows])),
          'mean_cos':float(np.mean([r['gradient_cos'] for r in rows])),
          'meaning':'仅对角线尺度参数的一阶梯度代理；不是实际整体网络参数冲突率，也不是训练提升结论'}
    atomic_json(out/'梯度'/'H_解释.json',data)
    return data


def experimental_settings():
    return dict(version=VERSION,checkpoint_sha256=CHECKPOINT_SHA,config_sha256=CONFIG_SHA,
                ref_count=REFERENCE_COUNT,reference_small=REFERENCE_SMALL, graph_k=GRAPH_K,
                pair_ids=PAIR_IDS,pair_batch=PAIR_BATCH,pair_steps=PAIR_STEPS,
                multipos_steps=MULTIPOS_STEPS,train_seeds=list(TRAIN_SEEDS),
                unknown_fraction=OPEN_UNKNOWN_PID_FRACTION,risk_fraction=RISK_FRACTION,
                image_batch=IMAGE_BATCH,text_batch=TEXT_BATCH,score_batch=SCORE_BATCH,
                baseline_tolerance_pp=BASELINE_TOLERANCE_PP)


def run_real(resume='', selection=EXPERIMENTS, device_name='cuda'):
    started=time.perf_counter()
    torch.set_num_threads(min(6,os.cpu_count() or 6))
    torch.manual_seed(SEED);np.random.seed(SEED)
    if device_name=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用；请使用irra_clean CUDA版PyTorch')
    device=torch.device(device_name)
    cfg=validate_files()
    cfg,val,tp,vp=load_dataset(cfg)
    stg=experimental_settings()
    rawfp=fingerprint(cfg,val)
    fp=hashlib.sha256((rawfp+json.dumps(stg,sort_keys=True)).encode('utf-8')).hexdigest()
    if resume:
        out=Path(resume).resolve()
        if not out.is_dir() or out.parent.resolve()!=OUTPUT_ROOT.resolve():
            raise RuntimeError('只能续跑「多种实验方向探索」中的第二轮实验子目录')
        state_path=out/'运行元数据.json'
        if not state_path.exists():raise RuntimeError('续跑目录没有运行元数据')
        old=json.loads(state_path.read_text(encoding='utf-8'))
        if old.get('version')!=VERSION or old.get('fingerprint')!=fp or \
           old.get('selection')!=list(selection) or old.get('settings')!=stg:
            raise RuntimeError('续跑实验指纹/设置/分支不一致，拒绝混写')
    else:
        OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
        out=OUTPUT_ROOT/('第二轮_多方向探索_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        out.mkdir(parents=True,exist_ok=False)
        atomic_json(out/'运行元数据.json',dict(version=VERSION,fingerprint=fp,
            selection=list(selection),settings=stg,epoch=58,
            checkpoint=str(BEST_DIR/'best.pth'),dataset='CUHK-PEDES/val',
            warning='部分方法使用无标签查询批次；超参数在运行前固定，没有Val挑参。'))
        try:shutil.copy2(Path(__file__),out/'本轮脚本_备份.py')
        except OSError:pass
    say('[实验输出]',out)
    for dirname in ('结果','逐查询','缓存','统计','开放集','风险分析','梯度'):
        (out/dirname).mkdir(exist_ok=True)

    model=None
    def get_model():
        nonlocal model
        if model is None:
            model=load_model(cfg,device)
        return model

    try:
        gallery=extract(get_model,cfg,val,out/'缓存',fp,'B0_gallery',device,'image')
        query=extract(get_model,cfg,val,out/'缓存',fp,'B0_queries',device,'text')
        s=cosine_scores(query,gallery,device)
        bm,bd=evaluate(s,tp,vp)
        official=verify_baseline(s,tp,vp,bm)
        atomic_json(out/'00_核验最佳Epoch58_完整Val基线.json',{
            'version':VERSION,'fingerprint':fp,'actual_metrics':bm,
            'official_metrics':official,'historic_expected':EXPECTED,
            'historic_delta_pp':{k:bm[k]-EXPECTED[k] for k in EXPECTED},
            'strict_same_file_sha':True,'validation_queries':len(tp),'validation_gallery':len(vp),
            'important':'固定当前运行B0作全部变体的配对基线；历史0.016pp差异不得隐瞒。'})
        say('[Epoch58 全量Val核验]',bm)
        if not (out/'结果'/'00_baseline.json').exists():
            atomic_json(out/'结果'/'00_baseline.json',{
                'name':'B0_original','version':VERSION,'fingerprint':fp,'metrics':bm})
        ref=None; paired=None; refmeta=None; sub_fp=None
        b_scores=None; e_scores=None
        train_ann=None; train_sh=None

        def get_training_records():
            nonlocal train_ann,train_sh
            if train_ann is None:
                train_ann,train_sh=train_annotation(cfg['root_dir'],tp)
            return train_ann,train_sh

        def get_train_reference():
            nonlocal ref, paired, refmeta, sub_fp
            if ref is None:
                train, sha=get_training_records()
                ref,paired,sub_fp,pi,pt=load_train_embeddings(
                    get_model,cfg,train,sha,fp,out/'缓存',device)
                refmeta=(pi,pt)
            return ref

        def get_training_pairs():
            nonlocal paired
            get_train_reference()
            if isinstance(paired,tuple) and len(paired)==3:
                return paired
            imgs,txts=refmeta
            emb=train_pair_features(get_model,cfg,sub_fp,out/'缓存',device,imgs,txts)
            paired=emb
            return emb

        def get_hub():
            nonlocal b_scores
            if b_scores is None:
                ref=get_train_reference()
                b_scores, bias=hubness_reference(s,gallery,ref,device,neighbors=20)
                atomic_json(out/'缓存'/'B_train_reference_metadata.json',dict(
                    reference_count=len(ref),train_annotation_sha256=train_sh,
                    mean_bias=float(bias.mean()),std_bias=float(bias.std()),
                    reference_source='Train only; no Val labels, no Val queries'))
            return b_scores

        def get_flip():
            nonlocal e_scores
            if e_scores is None:
                vf=extract(get_model,cfg,val,out/'缓存',fp,
                           'gallery_horizontal_flip',device,'image','flip')
                vg, diagnostics=fused_gallery(gallery,vf,'equal')
                e_scores=cosine_scores(query,vg,device)
                atomic_json(out/'缓存'/'E_equal_metadata.json',diagnostics)
            return e_scores

        if 'B' in selection:
            ref = get_train_reference()
            ss=cosine_scores(ref[:REFERENCE_SMALL],gallery,device)
            bias=np.partition(ss,len(ss)-20,axis=0)[-20:].mean(axis=0)
            smaller=(2*s-bias[None,:]).astype(np.float32)
            write_variant(out,fp,'B2_train_ref512','B_train_reference',smaller,tp,vp,bm,bd,
                          'Train独立文本512条作为图库偏置参考; 不使用Val查询分布; 2*cos-top20_reference_mean。')
            full=get_hub()
            write_variant(out,fp,'B3_train_ref2048','B_train_reference',full,tp,vp,bm,bd,
                          '固定2048条Train文本作为图库偏置参考; 评估Val单查询时只需预计算图库偏置。')
            # 保留上一轮B1作为非部署型参照，不声称同样设置可上线。
            b1,bias_old=hubness_reference(s,gallery,query,device,neighbors=20)
            write_variant(out,fp,'B1_val_batch_reference','B_transductive_control',b1,tp,vp,bm,bd,
                          '上一轮全无标签Val查询作为参考，transductive、不可视为单查询部署。')
            del smaller,ss,b1,bias_old;gc.collect()

        if 'C' in selection:
            indices,strengths=gallery_graph(gallery,device,GRAPH_K)
            # 三组图策略均不依赖图库的真实PID。
            for policy,name in [('plain','C1_plain8nn_reproduction'),
                                ('mutual','C2_mutual8nn'),
                                ('mutual_strong','C3_mutual_conf70')]:
                extra,diag=graph_scores(s,indices,strengths,.30,policy)
                write_variant(out,fp,name,'C_gallery_graph',extra,tp,vp,bm,bd,
                    f'{policy}图库视觉图，alpha=0.30; 连边统计={diag}; 不用PID建图。')
                atomic_json(out/'缓存'/(name+'_edges.json'),diag)
                del extra
            gc.collect()

        if 'E' in selection:
            flip=extract(get_model,cfg,val,out/'缓存',fp,'gallery_horizontal_flip',device,'image','flip')
            for policy,name in [('equal','E2_equal_reproduction'),
                                ('conservative','E4_conservative25'),
                                ('agreement','E5_consistency_adaptive')]:
                candidate,diag=fused_gallery(gallery,flip,policy)
                result=cosine_scores(query,candidate,device)
                write_variant(out,fp,name,'E_flip_fusion',result,tp,vp,bm,bd,
                              f'原图与翻转图特征融合，{policy}；没有Val PID权重调参；{diag}')
                if policy=='equal':e_scores=result.copy()
                del candidate,result
            del flip;gc.collect()

        if 'O' in selection:
            report=open_set_evaluate(out,s,tp,vp)
            say('[O 开放集]',report['auroc_from_max_similarity'],
                '未知查询',report['unknown_query_count'])

        if 'D' in selection:
            for policy in ('blur5','lowres_4x','dim'):
                image=extract(get_model,cfg,val,out/'缓存',fp,
                              'D_gallery_'+policy,device,'image',policy)
                extra=cosine_scores(query,image,device)
                write_variant(out,fp,'D_'+policy,'D_surveillance_degradation',extra,tp,vp,bm,bd,
                              '统一干预全部图库图像，不变更文本/身份；图像空间受控退化，不是实际相机压缩实测。')
                del image,extra;gc.collect()

        if 'R' in selection:
            bs=get_hub()   # 仅使用Train参考库，避免把Val批次hub当作在线系统。
            es=get_flip()
            for alt,name in [(bs,'R1_lowgap20_train_hub'),(es,'R2_lowgap20_flip_ensemble')]:
                moved,selected=risk_route(s,alt,RISK_FRACTION)
                write_variant(out,fp,name,'R_risk_routed',moved,tp,vp,bm,bd,
                     '低分数间隔的20%无标签Val查询路由到指定方法；批次分位数为transductive门槛，非单查询部署。')
                err=~bd['correct']
                diag=dict(risk_count=int(selected.sum()),
                          total_queries=len(selected),wrong_captured=int((selected&err).sum()),
                          all_wrong=int(err.sum()),
                          covered_error_rate=float(err[selected].mean()),
                          risk_score='original Top1-Top2 cosine gap; batch bottom 20 percent',
                          rule_learned_from_validation_labels=False,
                          note='即使方案有效，也必须使用Train阈值再验证独立测试集')
                atomic_json(out/'风险分析'/(name+'.json'),diag)
                del moved

        if any(k in selection for k in ('P','M','H')):
            tr,gtr,pids=get_training_pairs()
            trkey=hashlib.sha256((fp+sub_fp+str(pids.tolist())+
                       str(TRAIN_SEEDS)).encode('utf-8')).hexdigest()
            if 'H' in selection:
                report=gradient_conflict_report(out,tr,gtr,pids)
                say('[H 梯度信号诊断]',report)
            if 'P' in selection:
                for seed in TRAIN_SEEDS:
                    ck=out/'缓存'/f'P_candidate_pair_seed{seed}.pt'
                    net,accuracy=train_pair_comparator(tr,gtr,pids,seed,device,trkey,ck)
                    result=pair_topk_rerank(s,query,gallery,net,device,k=10)
                    write_variant(out,fp,f'P_pairwise_seed{seed}','P_pairwise_comparator',
                        result,tp,vp,bm,bd,
                        f'冻结IRRA；仅Train图文学习候选对相对判别，双向反对称；'
                        f'Top10内重排，其余图库排序不动；Train identity holdout acc={accuracy:.4f}。')
                    del net,result;gc.collect()
                    if device.type=='cuda':torch.cuda.empty_cache()
            if 'M' in selection:
                for seed in TRAIN_SEEDS:
                    ck=out/'缓存'/f'M_crossview_positive_seed{seed}.pt'
                    net,holdcos=train_positive_adapter(tr,gtr,pids,seed,device,trkey,ck)
                    tval=adapt_text(net,query,device)
                    result=cosine_scores(tval,gallery,device)
                    write_variant(out,fp,f'M_crossview_positive_seed{seed}',
                         'M_positive_consistency',result,tp,vp,bm,bd,
                         f'只用Train同PID的另一个视角训练文字残差Adapter；'
                         f'Val仍一条文本，冻结原始IRRA；Train identity holdout cos={holdcos:.4f}。')
                    del net,tval,result;gc.collect()
                    if device.type=='cuda':torch.cuda.empty_cache()
        totals=update_summary(out,fp,bm)
        doc=[
            '第二轮多方向探索，原权重冻结。所有闭集指标在同一完整6158x3078官方Val上计算。',
            '00_核验最佳Epoch58_完整Val基线.json 为原始基准。历史66.4339与实际值的差异单独记录。',
            '01_全部实验指标总表.csv 为各分支相对同次B0的变化。',
            '结果/*.json + 逐查询/*.csv + 统计/*_PID_bootstrap.csv 可逐身份审核。',
            'B: Train-reference校正是归纳式图库固定偏置；Val reference只是上一轮transductive对照。',
            'C: 图传播邻居仅取无标签视觉特征。E: 视觉翻转自适应不使用身份真值。',
            'O: 开放集是另一个人为构造的图库设置；其Known R1不得与完整闭集R1直接比较。',
            'D: 图像退化代表输入分布敏感性，不是自然监控场景泛化评估。',
            'R: 20%阈值用当前Val无标签分数分位数，是批量路由诊断而非单查询阈值。',
            'P: PairComparator在Train身份互斥子集训练，Top10重排；M: 正样本跨视角Train训练。',
            'H: 简化的对角缩放参数梯度诊断，只能用于提出假说。',
            '本轮全部结果属于探索性多比较，没有自动挑选最优变体或试后选择超参数。',
            '历史Validation已被反复使用，不可当作独立最终Test。',
            '缓存/保存断点、训练权重但不打包到上传ZIP。',
            '本轮暂不含原始CLIP双模型：无额外预训练权重，不应伪造CLIP对照。',
        ]
        (out/'00_先看这里.txt').write_text('\n'.join(doc)+'\n',encoding='utf-8')
        atomic_json(out/'02_实验完成摘要.json',dict(version=VERSION,fingerprint=fp,
             selection=list(selection),closed_set_variants=len(totals),baseline=bm,
             elapsed_seconds=time.perf_counter()-started,variants=totals))
        say('[完成]',len(totals),'个闭集变体，运行秒数',round(time.perf_counter()-started,1))
    except BaseException as ex:
        (out/'错误_异常.txt').write_text(''.join(traceback.format_exception(type(ex),ex,ex.__traceback__)),
                                        encoding='utf-8')
        say('[异常] 错误日志已保存到',out/'错误_异常.txt')
        raise
    finally:
        if model is not None:
            del model
            if device.type=='cuda':torch.cuda.empty_cache()
        try:
            make_zip(out)
        except BaseException as zip_error:
            say('[警告] ZIP打包失败:',repr(zip_error))
        say('[输出目录]',out)



def make_zip(out):
    """结果归档；仅排除大体积缓存和训练ckpt，验证zip CRC。"""
    out=Path(out)
    target=out/ZIP_NAME
    temp=out/('.'+ZIP_NAME+'.partial')
    try:
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for p in sorted(out.rglob('*')):
                if not p.is_file() or p in (target,temp):continue
                rel=p.relative_to(out)
                if '缓存' in rel.parts:continue
                archive.write(p,rel.as_posix())
        with zipfile.ZipFile(temp,'r') as archive:
            broken=archive.testzip()
            if broken:raise RuntimeError('ZIP文件CRC错误: '+broken)
        os.replace(temp,target)
    finally:
        if temp.exists():temp.unlink()
    say('[ZIP生成]',target)



def synthetic_runner_integration_test():
    """用完整 run_real() 跑一遍伪数据，再整目录续跑；不假装是真实CUDA验证。"""
    from unittest.mock import patch
    rng=np.random.default_rng(777)
    with tempfile.TemporaryDirectory(prefix='irra_r2_runner_') as td:
        base=Path(td)
        v=normalize_np(rng.standard_normal((40,512)).astype(np.float32))
        t=normalize_np(rng.standard_normal((65,512)).astype(np.float32))
        vp=np.arange(40,dtype=np.int64)%10
        tp=np.arange(65,dtype=np.int64)%10
        refs=normalize_np(rng.standard_normal((REFERENCE_COUNT,512)).astype(np.float32))
        trainpid=np.arange(PAIR_IDS*2,dtype=np.int64)//2+10000
        train_text=normalize_np(rng.standard_normal((len(trainpid),512)).astype(np.float32))
        train_img=normalize_np(train_text*.8+rng.standard_normal(train_text.shape).astype(np.float32)*.25)
        cfg={'root_dir':str(base),'img_size':(384,128),'text_length':77}
        val=dict(img_paths=['synthetic_'+str(i) for i in range(len(v))],
                 image_pids=vp.tolist(),caption_pids=tp.tolist(),captions=['synthetic']*len(t))
        def fake_embeddings(_,config,values,cache,fp,key,device,kind,variant='original'):
            if key=='B0_gallery':return v.copy()
            if key=='B0_queries':return t.copy()
            if key=='gallery_horizontal_flip':return normalize_np(v+.1*np.roll(v,1,axis=1))
            if key.startswith('D_gallery_'):return normalize_np(v+.12*np.roll(v,2,axis=1))
            raise AssertionError('Unknown mocked extraction key '+key)
        fake_train=(refs,['fake']*(PAIR_IDS*2),'fake-train-sig',{}, {})
        original=__import__(__name__)
        opts=dict(OUTPUT_ROOT=base/'多种实验方向探索',
                  validate_files=lambda:cfg,
                  load_dataset=lambda _: (cfg,val,tp,vp),
                  fingerprint=lambda _cfg,_val:'mocked-input',
                  verify_baseline=lambda _score,_tp,_vp,bm:bm,
                  extract=fake_embeddings,
                  train_annotation=lambda root,ids: (['train-row'],'mock-ann-sha'),
                  load_train_embeddings=lambda *a: fake_train,
                  train_pair_features=lambda *a:(train_text,train_img,trainpid),
                  PAIR_STEPS=8,MULTIPOS_STEPS=8,TRAIN_SEEDS=(42,))
        with patch.multiple(original,**opts):
            choices=tuple(k for k in EXPERIMENTS)
            run_real('',choices,'cpu')
            runs=list((base/'多种实验方向探索').glob('第二轮_*'))
            assert len(runs)==1
            out=runs[0]
            summary=json.loads((out/'02_实验完成摘要.json').read_text(encoding='utf-8'))
            assert summary['closed_set_variants']>=15
            report=out/ZIP_NAME
            assert report.is_file()
            with zipfile.ZipFile(report) as z:
                assert z.testzip() is None
                assert '开放集/O1_open_set.json' in z.namelist()
            # 模拟中断后的部分结果丢失，重新生成且不覆盖其它完成结果。
            key=out/'结果'/'E4_conservative25.json'
            key_csv=out/'逐查询'/'E4_conservative25.csv'
            key.unlink();key_csv.unlink()
            run_real(str(out),choices,'cpu')
            assert key.is_file() and key_csv.is_file()
            assert sha256(key_csv)==json.loads(key.read_text(encoding='utf-8'))['csv_sha256']
            with zipfile.ZipFile(report) as z:
                assert z.testzip() is None
    say('[MockRunner PASS] 全部实际运行分支与Train训练/完整报告/断点恢复/ZIP CRC')


def self_test():
    """CPU 合成全链路；覆盖 YAML、缓存续跑、训练、重排、指标、ZIP。"""
    say('[SelfTest] 开始：不需要IRRA权重、数据集或CUDA')
    rng=np.random.default_rng(20261021)
    with tempfile.TemporaryDirectory(prefix='irra_explore_r2_') as td:
        folder=Path(td)
        y=folder/'config.yaml'
        y.write_text('img_size: !!python/tuple [384, 128]\nstride_size: !!python/tuple [16, 16]\n',encoding='utf-8')
        assert load_irra_yaml(y)['img_size']==(384,128)
        v=normalize_np(rng.standard_normal((40,512)).astype(np.float32))
        t=normalize_np(rng.standard_normal((65,512)).astype(np.float32))
        vp=np.arange(40,dtype=np.int64)%10
        tp=np.arange(65,dtype=np.int64)%10
        device=torch.device('cpu')
        s=cosine_scores(t,v,device)
        bm,bd=evaluate(s,tp,vp,7)
        dcache=folder/'缓存'
        arr,done,meta=load_cache(dcache,'resume_test',40,'signature')
        assert done==0
        arr[:15]=v[:15];update_cache(arr,meta,'signature|resume_test',40,15)
        close_memmap(arr)
        arr,done,meta=load_cache(dcache,'resume_test',40,'signature')
        assert done==15 and np.allclose(np.array(arr[:15]),v[:15])
        arr[15:]=v[15:];update_cache(arr,meta,'signature|resume_test',40,40)
        close_memmap(arr)
        arr,done,meta=load_cache(dcache,'resume_test',40,'signature')
        assert done==40
        close_memmap(arr)
        try:load_cache(dcache,'resume_test',40,'wrong_signature')
        except RuntimeError:pass
        else:raise AssertionError('未拒绝不匹配的续跑缓存')
        out=folder/'输出'
        for sub in ('结果','逐查询','缓存','统计','开放集','风险分析','梯度'):
            (out/sub).mkdir(parents=True,exist_ok=True)
        fp='synthetic'
        h1,_=hubness_reference(s,v,t,device)
        write_variant(out,fp,'B_hubness_synthetic','B',h1,tp,vp,bm,bd,'test')
        ix,strength=gallery_graph(v,device,k=4)
        for policy in ('plain','mutual','mutual_strong'):
            gs,meta=graph_scores(s,ix,strength,.3,policy,block=7)
            assert gs.shape==s.shape and np.isfinite(gs).all()
            write_variant(out,fp,'C_'+policy,'C',gs,tp,vp,bm,bd,'test')
            assert meta['isolated_nodes']+meta['connected_nodes']==40
        f=normalize_np(v+rng.standard_normal(v.shape).astype(np.float32)*.1)
        for policy in ('equal','conservative','agreement'):
            g,_=fused_gallery(v,f,policy)
            assert np.isfinite(g).all()
            write_variant(out,fp,'E_'+policy,'E',cosine_scores(t,g,device),tp,vp,bm,bd,'test')
        for policy in ('blur5','lowres_4x','dim'):
            x=torch.randn(2,3,64,32)
            assert torch.isfinite(image_degrade(x,policy)).all()
        op=open_set_evaluate(out,s,tp,vp)
        assert 0<=op['auroc_from_max_similarity']<=1
        for queryname, candidate in [('risk_hub',h1),('risk_flip',cosine_scores(t,f,device))]:
            res,selected=risk_route(s,candidate,.2)
            assert int(selected.sum())==13
            assert np.array_equal(res[~selected],s[~selected])
            write_variant(out,fp,queryname,'R',res,tp,vp,bm,bd,'test')
        # 合成320*2 Train身份；与Val独立，双视角训练可复现实测参数流。
        pt=np.arange(PAIR_IDS*2,dtype=np.int64)//2+10000
        ti=normalize_np(rng.standard_normal((len(pt),512)).astype(np.float32))
        ig=normalize_np(ti*.7+rng.standard_normal(ti.shape).astype(np.float32)*.3)
        mask,positive=create_pair_split(pt)
        assert np.all(pt[positive]==pt) and not np.any(positive==np.arange(len(pt)))
        pair_ck=out/'缓存'/'train_p.pt'
        net,accuracy=train_pair_comparator(ti,ig,pt,42,device,'training_sig',pair_ck)
        assert 0<=accuracy<=1 and pair_ck.is_file()
        net2,accuracy2=train_pair_comparator(ti,ig,pt,42,device,'training_sig',pair_ck)
        assert abs(accuracy-accuracy2)<1e-7
        allscore=pair_topk_rerank(s,t,v,net2,device,k=10,batch_q=7)
        assert allscore.shape==s.shape and np.isfinite(allscore).all()
        # 重排只排列Top10原有分数，其余3078-K候选保持不变。
        for i in range(len(s)):
            pos=np.argsort(-s[i],kind='stable')[:10]
            maskrow=np.ones(len(v),dtype=bool);maskrow[pos]=False
            assert np.array_equal(allscore[i,maskrow],s[i,maskrow])
            assert np.array_equal(np.sort(allscore[i,pos]),np.sort(s[i,pos]))
        write_variant(out,fp,'P_pairwise','P',allscore,tp,vp,bm,bd,'test')
        mck=out/'缓存'/'train_m.pt'
        mdl,holdcos=train_positive_adapter(ti,ig,pt,42,device,'train_sig',mck)
        assert -1<=holdcos<=1
        mdl2,holdcos2=train_positive_adapter(ti,ig,pt,42,device,'train_sig',mck)
        assert abs(holdcos-holdcos2)<1e-7
        tt=adapt_text(mdl2,t,device)
        write_variant(out,fp,'M_positive','M',cosine_scores(tt,v,device),tp,vp,bm,bd,'test')
        gradient_conflict_report(out,ti,ig,pt)
        assert len(read_results(out,fp))>=10
        data=update_summary(out,fp,bm)
        assert len(data)>=10
        (out/'00_先看这里.txt').write_text('synthetic execution',encoding='utf-8')
        make_zip(out)
        with zipfile.ZipFile(out/ZIP_NAME) as z:
            assert z.testzip() is None
            assert '01_全部实验指标总表.csv' in z.namelist()
            assert not any(k.startswith('缓存/') for k in z.namelist())
        shutil.rmtree(dcache)
        assert not dcache.exists()
    synthetic_runner_integration_test()
    say('[SelfTest PASS] YAML tuple/缓存续跑与拒绝污染/九方向主函数/Train监督与模型续跑/'
        '开放集与风险报告/完整指标与逐查询/图表统计/ZIP/Windows缓存释放')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true',help='CPU合成数据全链路自检，推荐先运行')
    parser.add_argument('--resume',default='',help='上次本脚本生成的完整输出子文件夹')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--experiments',default='BCEODRPMH',
          help='选择分支，比如BCEODR；P候选对监督、M跨视角正例、H梯度诊断需要额外Train数据')
    args=parser.parse_args()
    selection=args.experiments.upper()
    if not selection or len(set(selection))!=len(selection) or any(k not in EXPERIMENTS for k in selection):
        parser.error('experiments只能包含BCEODRPMH中的不重复字母')
    if args.self_test:self_test()
    else:run_real(args.resume,tuple(k for k in EXPERIMENTS if k in selection),args.device)

if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();sys.exit(1)
