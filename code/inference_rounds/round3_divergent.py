# -*- coding: utf-8 -*-
"""IRRA Epoch58 第三轮大范围发散探索（单文件）。
B=Train参考Hub校正 C=可信图+保护TopK E=翻转融合 R=Train校准风险决策；
T=文本扰动 F=Train跨模态通道 K=低维投影 G=视觉拥挤度 V=局部图像证据；
O=开放集原图/Hub/翻转对照（不计入闭集变体数量）。
全部实验冻结原始IRRA，无需额外模型权重，无网络请求。
首次建议 --self-test；中断后 --resume 原子文件夹，禁止不同分支设置混写。
输出到 IRRA-main/多种实验方向探索/第三轮_发散与优化_时间戳/。
Validation长期用于研究，只能报告探索性收益，最终仍需锁定方法进行独立Test。
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
VERSION = 'wide_exploration_round3_v1_20261021'
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
EXPERIMENTS = ('B', 'C', 'E', 'R', 'T', 'F', 'K', 'G', 'V', 'O')
ZIP_NAME = '发送给ChatGPT_第三轮大范围探索.zip'


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



# ━━━━━━━━━━━━━━━━━━━━━━━ 第三轮：机制改进和全新独立方向 ━━━━━━━━━━━━━━━━━━━━━━━

# 第三轮优化续跑：旧结果json+CSV SHA一致且已有bootstrap则无需重新排序六千条查询。
_base_write_variant=write_variant

def write_variant(out, fp, name, family, s, tp, vp, bm, bd, note):
    record=out/'结果'/(name+'.json')
    bootstrap=out/'统计'/(name+'_PID_bootstrap.csv')
    if record.exists() and bootstrap.exists():
        # record_variant会复验json/CSV签名，错配则立即拒绝。
        return record_variant(out,fp,name,family,s,tp,vp,bm,bd,note)
    return _base_write_variant(out,fp,name,family,s,tp,vp,bm,bd,note)


def protected_graph_rerank(original, improved, k, batch=32):
    """冻结原始 Top-k 的图像集合与原始分数；只能重排更深候选。
    使用同一查询的原始深部 score 多重集按图传播的新顺序重新分配。
    注意：该实验故意不能直接改善被保护的 R@k，用于因果隔离。
    """
    if original.shape != improved.shape or not 1 <= k < original.shape[1]:
        raise ValueError('TopK保护输入不匹配')
    n, m = original.shape
    result = np.empty_like(original)
    for start in range(0, n, batch):
        stop = min(n, start + batch)
        for i in range(start, stop):
            old = np.argsort(-original[i], kind='stable')
            head, tail = old[:k], old[k:]
            tail_order = tail[np.argsort(-improved[i, tail], kind='stable')]
            result[i] = original[i]
            original_scores = original[i, tail]
            ordered_tail_values = np.sort(original_scores)[::-1]
            # 消除正好与保护区边界分数相等时的跨区并列。
            ordered_tail_values = np.minimum(ordered_tail_values,
                                    np.nextafter(original[i, head[-1]], -np.inf))
            result[i, tail_order] = ordered_tail_values
            actual = np.argsort(-result[i], kind='stable')[:k]
            if not np.array_equal(actual, head):
                raise RuntimeError(f'保护Top{k}失败，query={i}')
            if not np.array_equal(result[i, head], original[i, head]):
                raise RuntimeError('保护区分数发生变化')
    return result


def degree_gated_graph(s, idx, strength, alpha=.30, strong_threshold=.70):
    """与C3不同：节点度数低时自动降低传播强度；所有系数固定。"""
    n, k = idx.shape
    nei = [set(int(x) for x in row) for row in idx]
    valid = (strength >= strong_threshold)
    for i in range(n):
        for j in range(k):
            valid[i,j] = bool(valid[i,j] and i in nei[int(idx[i,j])])
    w = np.maximum(strength,0) * valid
    degree = valid.sum(axis=1)
    w = w / np.maximum(w.sum(axis=1,keepdims=True),1e-8)
    # 没有邻居时完全不传播；最高有效度达到k时最大alpha。
    factor = alpha * np.minimum(degree / max(k*.5,1),1.)
    out = np.empty_like(s)
    for z in range(0,len(s),32):
        a = s[z:z+32]
        avg = np.zeros_like(a)
        for j in range(k):avg += a[:,idx[:,j]] * w[None,:,j]
        out[z:z+len(a)] = a + factor[None,:] * (avg-a)
    return out, dict(degree_min=int(degree.min()), degree_mean=float(degree.mean()),
                     degree_max=int(degree.max()), isolated=int((degree==0).sum()))


def rank_fusion(a,b,weight=60.0,block=64):
    """独立决策融合：Reciprocal Rank Fusion，非E2特征平均。"""
    if a.shape!=b.shape:raise ValueError('RRF矩阵维度不符')
    n,m=a.shape; output=np.empty_like(a)
    for start in range(0,n,block):
        stop=min(n,start+block)
        for i in range(start,stop):
            rank_a=np.empty(m,np.int32);rank_b=np.empty(m,np.int32)
            rank_a[np.argsort(-a[i],kind='stable')]=np.arange(m,dtype=np.int32)
            rank_b[np.argsort(-b[i],kind='stable')]=np.arange(m,dtype=np.int32)
            output[i] = (1/(weight+1+rank_a)+1/(weight+1+rank_b)).astype(np.float32)
    return output


def channel_reweight(t, g, train_t, train_g, shuffled=False):
    """由Train成对文本图像的坐标级一致性估算固定权重；不见Val身份标签。"""
    if train_t.shape != train_g.shape or train_t.shape[1]!=t.shape[1]:
        raise ValueError('Train成对特征维度异常')
    evidence = np.mean(train_t*train_g,axis=0)
    scale = max(float(np.std(evidence)),1e-8)
    weights = np.clip(1 + .15*(evidence-evidence.mean())/scale,.70,1.30).astype(np.float32)
    if shuffled:weights=weights[np.random.default_rng(3301).permutation(len(weights))]
    tw=normalize_np(t*weights[None,:]);gw=normalize_np(g*weights[None,:])
    return tw,gw,dict(weight_std=float(weights.std()),weight_min=float(weights.min()),
            weight_max=float(weights.max()),training_pair_count=len(train_t),shuffled_control=shuffled)


def compressed_embeddings(t, g, dims, seed=3301):
    """独立随机正交投影；不学习Val标签，可测表征冗余。"""
    d=t.shape[1]
    if not 1<=dims<=d:raise ValueError('压缩维数非法')
    rng=np.random.default_rng(seed)
    proj=rng.standard_normal((d,dims)).astype(np.float64)
    q,_=np.linalg.qr(proj,mode='reduced')
    q=q.astype(np.float32)
    return normalize_np(t@q),normalize_np(g@q)


def image_neighbor_penalty(s, strength, coefficient=.06):
    """仅由无标签Gallery内部视觉邻居定义拥挤度，不同于Train文本Hubness。"""
    crowd = np.mean(strength, axis=1)
    penalty = (crowd-crowd.mean())/max(float(crowd.std()),1e-8)
    out=np.asarray(s-coefficient*penalty[None,:],dtype=np.float32)
    return out,dict(crowding_mean=float(crowd.mean()),crowding_std=float(crowd.std()),
                    coefficient=coefficient)


_original_third_transform_images = transform_images

def transform_images(x, mode):
    if mode in ('top30_mask','bottom30_mask'):
        y=x.clone()
        n=max(1,int(y.shape[-2]*.30))
        if mode=='top30_mask':y[:,:,:n,:]=0
        else:y[:,:,-n:,:]=0
        return y
    return _original_third_transform_images(x,mode)


def open_set_compare(output, variants, tp, vp):
    """同一预定未知身份拆分。所有图像缺失是先删Gallery列，而不是改变真值以优化。"""
    rng=np.random.default_rng(SEED)
    ids=np.unique(vp)
    gone=set(map(int,rng.choice(ids,int(round(len(ids)*OPEN_UNKNOWN_PID_FRACTION)),replace=False)))
    gallery_mask=np.array([int(pid) not in gone for pid in vp],bool)
    known=np.array([int(pid) not in gone for pid in tp],bool)
    result=[]; allrows=[]
    for name,ss in variants.items():
        subset=ss[:,gallery_mask]
        maxval=subset.max(axis=1)
        gap=np.partition(subset,-2,axis=1)[:,-1]-np.partition(subset,-2,axis=1)[:,-2]
        # AUROC是闭集目标存在性判别的诊断，不是经过Train阈值验证的上线拒识。
        known_metrics,_=evaluate(subset[known],tp[known],vp[gallery_mask])
        results=dict(name=name,known_count=int(known.sum()),unknown_count=int((~known).sum()),
                  known_gallery_count=int(gallery_mask.sum()),
                  auroc_max=auc_roc(maxval,known),auroc_top_gap=auc_roc(gap,known),
                  known_R1=known_metrics['R1'],known_mAP=known_metrics['mAP'],
                  note='同样未知PID拆分；绝对拒识阈值尚未在独立Test确认')
        result.append(results)
        for i in range(len(tp)):
            allrows.append(dict(variant=name,query_index=i,query_pid=int(tp[i]),
                present=int(known[i]),max_score=round(float(maxval[i]),7),
                top_gap=round(float(gap[i]),7)))
    write_csv(output/'开放集'/'O_多机制开放集对照.csv',result,list(result[0]))
    write_csv(output/'开放集'/'O_逐查询置信度.csv',allrows,list(allrows[0]))
    return result


def settings_round3():
    return dict(version=VERSION,checkpoint_sha=CHECKPOINT_SHA,config_sha=CONFIG_SHA,
        experiments=list(EXPERIMENTS),graph_k=8,graph_alpha=.30,
        graph_strong_threshold=.70,hub_ref_size=2048,hub_top=20,
        graph_lock_k=[1,5,10],train_risk_quantile=.20,
        flip_feature_weight=.50,flip_rrf_constant=60,
        text_ensemble_weight=.20,channel_weight_range=[.70,1.30],
        random_projection_dims=[256,128],neighbor_penalty=[.04,.10],
        image_mask_fraction=.30,seed=SEED,
        batch_sizes=dict(image=IMAGE_BATCH,text=TEXT_BATCH,score=SCORE_BATCH))


def run_real(resume='', selection=EXPERIMENTS, device_name='cuda'):
    began=time.perf_counter()
    torch.set_num_threads(min(6,os.cpu_count() or 6))
    torch.manual_seed(SEED);np.random.seed(SEED)
    if device_name=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，请切换到irra_clean环境')
    device=torch.device(device_name)
    cfg=validate_files()
    cfg,val,tp,vp=load_dataset(cfg)
    stg=settings_round3()
    rawfp=fingerprint(cfg,val)
    fp=hashlib.sha256((rawfp+json.dumps(stg,sort_keys=True)).encode('utf-8')).hexdigest()
    if resume:
        out=Path(resume).resolve()
        if not out.is_dir() or out.parent.resolve()!=OUTPUT_ROOT.resolve():
            raise RuntimeError('续跑目录必须是「多种实验方向探索」下的第三轮子文件夹')
        oldpath=out/'运行元数据.json'
        if not oldpath.is_file():raise RuntimeError('续跑目录缺少运行元数据.json')
        old=json.loads(oldpath.read_text(encoding='utf-8'))
        if old.get('version')!=VERSION or old.get('fingerprint')!=fp or \
                old.get('settings')!=stg or old.get('selection')!=list(selection):
            raise RuntimeError('续跑设置/指纹/分支与旧实验不一致；拒绝混写')
    else:
        OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
        out=OUTPUT_ROOT/('第三轮_发散与优化_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        out.mkdir(parents=True,exist_ok=False)
        atomic_json(out/'运行元数据.json',dict(version=VERSION,fingerprint=fp,
                    settings=stg,selection=list(selection),epoch=58,
                    checkpoint=str(BEST_DIR/'best.pth'),validation='CUHK-PEDES/val',
                    warnings=['Val多轮使用，探索性结果不可作独立Test',
                    '保护TopK的收益只作用于保护范围之外的排名']))
        try:shutil.copy2(Path(__file__),out/'第三轮脚本_副本.py')
        except OSError:pass
    for name in ('结果','逐查询','缓存','统计','开放集','分析'):
        (out/name).mkdir(exist_ok=True)
    say('[第三轮结果目录]',out)
    model=None
    def model_fn():
        nonlocal model
        if model is None:model=load_model(cfg,device)
        return model
    try:
        gallery=extract(model_fn,cfg,val,out/'缓存',fp,'B0_gallery',device,'image')
        query=extract(model_fn,cfg,val,out/'缓存',fp,'B0_queries',device,'text')
        original=cosine_scores(query,gallery,device)
        bm,bd=evaluate(original,tp,vp)
        official=verify_baseline(original,tp,vp,bm)
        atomic_json(out/'00_原始Epoch58核验.json',dict(baseline=bm,official=official,
            historical_expected=EXPECTED,delta_historical_pp={k:bm[k]-EXPECTED[k] for k in EXPECTED},
            queries=len(tp),gallery=len(vp),fp=fp))
        base_file=out/'结果'/'00_baseline.json'
        if base_file.exists():
            prev=json.loads(base_file.read_text(encoding='utf-8'))
            if prev.get('fingerprint')!=fp or prev.get('metrics')!=bm:
                raise RuntimeError('旧B0与当前基线不同，不允许续跑')
        else:atomic_json(base_file,dict(name='B0_original',fingerprint=fp,version=VERSION,metrics=bm))
        say('[B0 Epoch58]',bm)
        ref=None;train_ann=None;ann_sha=None;train_pairs=None;subfp=None;pair_meta=None
        hub=None;flip_score=None;flip_gallery=None;graph=None;graph_idx=None;graph_strength=None

        def training_annotation():
            nonlocal train_ann,ann_sha
            if train_ann is None:train_ann,ann_sha=train_annotation(cfg['root_dir'],tp)
            return train_ann,ann_sha
        def train_ref():
            nonlocal ref,train_pairs,subfp,pair_meta
            if ref is None:
                data,sha=training_annotation()
                ref,train_pairs,subfp,pi,pt=load_train_embeddings(model_fn,cfg,data,sha,
                                                fp,out/'缓存',device)
                pair_meta=(pi,pt)
            return ref
        def paired_train_features():
            nonlocal train_pairs
            train_ref()
            if isinstance(train_pairs,tuple) and len(train_pairs)==3:return train_pairs
            imgs,txts=pair_meta
            train_pairs=train_pair_features(model_fn,cfg,subfp,out/'缓存',device,imgs,txts)
            return train_pairs
        def hub_score():
            nonlocal hub
            if hub is None:
                hh=train_ref()
                hub,bias=hubness_reference(original,gallery,hh,device,neighbors=20)
                atomic_json(out/'分析'/'B3_训练参考统计.json',dict(reference_size=len(hh),
                    bias_mean=float(bias.mean()),bias_std=float(bias.std()),
                    source='Train captions only; no Validation captions'))
            return hub
        def flip_scores():
            nonlocal flip_score,flip_gallery
            if flip_score is None:
                vf=extract(model_fn,cfg,val,out/'缓存',fp,
                    'gallery_flip',device,'image','flip')
                flip_gallery,diag=fused_gallery(gallery,vf,'equal')
                flip_score=cosine_scores(query,flip_gallery,device)
                atomic_json(out/'分析'/'E2_翻转特征.json',diag)
            return flip_score
        def graph_data():
            nonlocal graph_idx,graph_strength
            if graph_idx is None:
                graph_idx,graph_strength=gallery_graph(gallery,device,GRAPH_K)
            return graph_idx,graph_strength
        def graph_c3():
            nonlocal graph
            if graph is None:
                idx,strength=graph_data()
                graph,diag=graph_scores(original,idx,strength,.30,'mutual_strong')
                atomic_json(out/'分析'/'C3_图统计.json',diag)
            return graph
        def save(name,family,ss,note):
            if ss.shape!=original.shape or not np.isfinite(ss).all():
                raise RuntimeError(f'{name}得分矩阵异常')
            return write_variant(out,fp,name,family,ss,tp,vp,bm,bd,note)

        if 'B' in selection:
            b=hub_score()
            save('B3_Train2048_复现','B_文本偏置校正',b,
                 'Train 2048参考文本的top20均值校正，固定2*cos-bias，单查询可执行')
            mild=(.5*original+.5*b).astype(np.float32)
            save('B4_半强度Hub','B_文本偏置校正',mild,
                 '预设温和系数，原始cosine与B3校正分数等权相加')
            del mild

        if 'C' in selection:
            c=graph_c3()
            save('C3_可信图传播复现','C_无监督图传播',c,
                '互为近邻且sim≥0.70，8NN α0.30，无PID建图')
            for k in (1,5,10):
                guarded=protected_graph_rerank(original,c,k)
                old_rank=np.argsort(-original,axis=1,kind='stable')[:,:k]
                new_rank=np.argsort(-guarded,axis=1,kind='stable')[:,:k]
                if not np.array_equal(old_rank,new_rank):
                    raise RuntimeError(f'C图传播保护Top{k}失败')
                val_report=save(f'C4_保护Top{k}_深层传播','C_保护TopK图传播',guarded,
                    f'仅在原B0的Top{k}以外重排；原Top{k}图像序列与分数保持不变')
                # R@k保证保持不变（包括R@1、R@5、R@10的相关阈值）
                for metric,threshold in (('R1',1),('R5',5),('R10',10)):
                    if k>=threshold and abs(val_report['metrics'][metric]-bm[metric])>1e-8:
                        raise RuntimeError(f'保护Top{k}但{metric}变化：实现错误')
                del guarded
            idx,strength=graph_data()
            density,diag=degree_gated_graph(original,idx,strength)
            save('C5_度数门控传播','C_保护可信节点',density,
                 f'可信邻居越少传播越弱；阈值.70 互惠近邻；{diag}')
            atomic_json(out/'分析'/'C5_节点度统计.json',diag)
            del density

        if 'E' in selection:
            fused=flip_scores()
            save('E2_等权视觉翻转复现','E_视觉不变性',fused,
                '图像CLS=normalize(.5*原图+.5*水平翻转)；此前两轮均有效')
            idx_path=out/'缓存'/'gallery_flip.json'
            # E6是排名融合而非视觉特征平均；需分别计算翻转图库的原始匹配分数。
            vg=extract(model_fn,cfg,val,out/'缓存',fp,'gallery_flip',device,'image','flip')
            raw_flip=cosine_scores(query,vg,device)
            rrf=rank_fusion(original,raw_flip,weight=60.)
            save('E6_RRF双视图排序','E_排名层融合',rrf,
                 '原图与翻转视角的独立rank，用常数60的Reciprocal Rank Fusion')
            max_scores=np.maximum(original,raw_flip).astype(np.float32)
            save('E7_双视图Max','E_候选证据融合',max_scores,
                 '逐候选选择原图与翻转最高余弦；推理无需标签')
            del vg,raw_flip,rrf,max_scores

        if 'R' in selection:
            ref_train=train_ref()
            # 校准低margin阈值：只用Train captions在当前无标签Gallery上的分数，不用Val文本。
            training_scores=cosine_scores(ref_train,gallery,device)
            gap_train=margin_risk(training_scores)
            fixed_threshold=float(np.quantile(gap_train,.20))
            gap_val=margin_risk(original)
            mask=gap_val<=fixed_threshold
            for alternative,name in ((hub_score(),'R3_Train校准_使用B3'),
                                     (flip_scores(),'R4_Train校准_使用E2')):
                ss=original.copy();ss[mask]=alternative[mask]
                save(name,'R_Train固定阈值',ss,
                    'Train2048文本在相同Val无标签Gallery上校准绝对Top1-Top2间隔20%分位；'
                    '线上单查询只需固定阈值，当前阈值选择不使用Val查询/身份标签')
                del ss
            atomic_json(out/'分析'/'R_Train固定阈值说明.json',dict(
                threshold=fixed_threshold,train_ref_count=len(ref_train),
                train_risk_fraction=float((gap_train<=fixed_threshold).mean()),
                val_selected_count=int(mask.sum()),val_selected_fraction=float(mask.mean()),
                val_wrong_capture=int((mask&~bd['correct']).sum()),
                val_error_ratio=float((mask&~bd['correct']).sum()/max(int(mask.sum()),1)),
                threshold_source='Train captions evaluated on current unlabeled gallery',
                warning='使用目标Gallery但不使用Val文本；应用到新图库必须重新无监督预计算阈值'))

        if 'T' in selection:
            for variant,name in (('first_clause','T1_首分句融合'),
                                 ('reverse_clauses','T2_分句重排序融合')):
                probe=extract(model_fn,cfg,val,out/'缓存',fp,'text_'+variant,
                                      device,'text',variant)
                candidate=normalize_np(.80*query+.20*probe)
                save(name,'T_文本鲁棒性',cosine_scores(candidate,gallery,device),
                     '原文本80%+变换文本20%特征融合，不使用Val身份；变换可能改变语义，非严格同义句')
                del probe,candidate

        if 'F' in selection:
            train_t,train_g,pid=paired_train_features()
            for shuffle,name in ((False,'F1_Train模态通道权重'),
                                 (True,'F2_随机打乱通道对照')):
                tt,gg,diag=channel_reweight(query,gallery,train_t,train_g,shuffle)
                save(name,'F_跨模态维度可靠性',cosine_scores(tt,gg,device),
                    f'320 Train PID的640正例推断512维通道权重，幅度范围.7到1.3；{diag}')
                del tt,gg

        if 'K' in selection:
            for n in (256,128):
                tt,gg=compressed_embeddings(query,gallery,n)
                save(f'K_{n}维随机正交投影','K_表示冗余性',cosine_scores(tt,gg,device),
                     f'独立固定随机正交投影512→{n}维，双方使用同一投影；非训练优化')
                del tt,gg

        if 'G' in selection:
            idx,strength=graph_data()
            for coef in (.04,.10):
                scores,stats=image_neighbor_penalty(original,strength,coef)
                save(f'G_图库视觉拥挤惩罚_{int(coef*100)}','G_纯视觉密度',scores,
                     f'无需任何参考文本，仅按Gallery内部视觉kNN拥挤度惩罚；{stats}')
                del scores

        if 'V' in selection:
            for mode,name in (('top30_mask','V1_遮挡上部30'),
                              ('bottom30_mask','V2_遮挡下部30')):
                vg=extract(model_fn,cfg,val,out/'缓存',fp,'image_'+mode,
                         device,'image',mode)
                # 直接区域遮挡用于诊断；为保留图像信息额外测试保守视觉融合。
                masked=cosine_scores(query,vg,device)
                save(name,'V_局部视觉证据消融',masked,
                     '图像上下固定30%粗区域掩码；不等于人体语义分割')
                mixed=normalize_np(.80*gallery+.20*vg)
                save(name+'_与原图融合20','V_局部视觉证据融合',
                     cosine_scores(query,mixed,device),
                     '原图80%+区域遮挡20%的图像特征融合；该实验与直接遮挡区分')
                del vg,masked,mixed

        if 'O' in selection:
            candidates={'B0_原始余弦':original,'B3_TrainHub':hub_score(),
                        'E2_翻转融合':flip_scores()}
            report=open_set_compare(out,candidates,tp,vp)
            say('[O 开放集比较]',[(r['name'],round(r['auroc_max'],4)) for r in report])

        summary=update_summary(out,fp,bm)
        instructions=[
          'IRRA大范围发散第三轮；始终冻结Epoch58（best.pth），仅在完整官方6158x3078 Val评估闭集。',
          '先严格核验SHA256与原始B0，所有变体配对同轮真实B0；不强求指标提升。',
          'B3 Train文本Hubness重复；B4改变Hub惩罚比例。',
          'C3可信图传播重复；C4保护Top1/5/10再调深部排序；C5度数门控。',
          'E2翻转特征融合重复；E6独立RRF排名融合；E7双视图取最大相似度。',
          'R固定阈值由Train文本在无标签目标Gallery上校准，不使用Validation文本设置阈值。',
          'T为文本结构敏感性与不可靠变换融合，变换后的文本不保证完全语义等价。',
          'F为Train跨模态维度可靠性学习及等权幅度的随机排列对照，不在Val训练。',
          'K为随机低维投影；G为图库视觉邻域拥挤度；V为粗区域输入干预。',
          'O模拟开放集对照，和闭集评价分别统计。特别禁止移除未知图库后重用原图库图边。',
          '新方法的阈值、权重、维数在脚本中预设，报告不自动挑最优或组合最优结果。',
          'Validation已被多轮查看，显著性Bootstrap仅探索性；未来论文必须独立Test。',
          '缓存数据和权重不进ZIP，可在原运行子目录通过--resume安全续跑。',
        ]
        (out/'00_阅读说明.txt').write_text('\n'.join(instructions)+'\n',encoding='utf-8')
        atomic_json(out/'02_第三轮完成摘要.json',dict(version=VERSION,fingerprint=fp,
           selection=list(selection),baseline=bm,closed_set_variants=len(summary),
           elapsed_seconds=time.perf_counter()-began,results=summary))
        say('[第三轮完成]',len(summary),'个独立闭集变体，总秒数',round(time.perf_counter()-began,1))
    except BaseException as e:
        (out/'错误_异常.txt').write_text(''.join(traceback.format_exception(type(e),e,e.__traceback__)),
                encoding='utf-8')
        say('[失败日志]',out/'错误_异常.txt')
        raise
    finally:
        if model is not None:
            del model
            if device.type=='cuda':torch.cuda.empty_cache()
        try:make_zip(out)
        except BaseException as exc:say('[ZIP生成失败]',repr(exc))
        say('[输出文件夹]',out)


def make_zip(out):
    target=out/ZIP_NAME
    part=out/('.'+ZIP_NAME+'.partial')
    try:
        with zipfile.ZipFile(part,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as arc:
            for path in sorted(out.rglob('*')):
                if not path.is_file() or path in (target,part):continue
                rel=path.relative_to(out)
                if '缓存' in rel.parts:continue
                arc.write(path,rel.as_posix())
        with zipfile.ZipFile(part) as z:
            broken=z.testzip()
            if broken is not None:raise RuntimeError('ZIP CRC错误：'+broken)
        os.replace(part,target)
    finally:
        if part.exists():part.unlink()
    say('[ZIP完成]',target)


def self_test():
    """合成数据：全分支主流程、CSV、Bootstrap、ZIP、缓存断点恢复。"""
    from unittest.mock import patch
    say('[SelfTest] CPU完整主流程与续跑测试开始')
    rng=np.random.default_rng(21)
    with tempfile.TemporaryDirectory(prefix='irra_r3_all_') as td:
        base=Path(td)
        conf=base/'configs.yaml'
        conf.write_text('img_size: !!python/tuple [384,128]\nstride_size: !!python/tuple [16,16]\n',encoding='utf-8')
        assert load_irra_yaml(conf)['img_size']==(384,128)
        # 固定 TopK 的纯机制单元测试，包括并列排序边界。
        demo=rng.standard_normal((7,19)).astype(np.float32)
        demo[0,:5]=.5
        change=demo + rng.normal(0,.3,demo.shape).astype(np.float32)
        for k in (1,5,10):
            r=protected_graph_rerank(demo,change,k)
            assert np.array_equal(np.argsort(-r,axis=1,kind='stable')[:,:k],
                                  np.argsort(-demo,axis=1,kind='stable')[:,:k])
        x=torch.randn(2,3,60,24)
        for mod in ('top30_mask','bottom30_mask'):
            assert transform_images(x,mod).shape==x.shape
        v=normalize_np(rng.standard_normal((38,512)).astype(np.float32))
        t=normalize_np(rng.standard_normal((52,512)).astype(np.float32))
        vp=np.arange(38,dtype=np.int64)%10
        tp=np.arange(52,dtype=np.int64)%10
        refs=normalize_np(rng.standard_normal((2048,512)).astype(np.float32))
        pids=np.arange(640,dtype=np.int64)//2+10000
        tr=normalize_np(rng.standard_normal((640,512)).astype(np.float32))
        gr=normalize_np(tr*.75+rng.normal(0,.23,tr.shape).astype(np.float32))
        cfg=dict(root_dir=str(base),img_size=(384,128),text_length=77)
        val=dict(img_paths=['sim'+str(j) for j in range(len(v))],
                 image_pids=vp.tolist(),captions=['text']*len(t),caption_pids=tp.tolist())
        def mock_extract(model, config, records, cache, sig, key, dev, kind, variant='original'):
            if key=='B0_gallery':return v.copy()
            if key=='B0_queries':return t.copy()
            if key=='gallery_flip':return normalize_np(v+.1*np.roll(v,1,axis=1))
            if key=='text_first_clause':return normalize_np(t+.08*np.roll(t,2,axis=1))
            if key=='text_reverse_clauses':return normalize_np(t+.08*np.roll(t,3,axis=1))
            if key.startswith('image_top30_mask'):return normalize_np(v+.25*np.roll(v,3,axis=1))
            if key.startswith('image_bottom30_mask'):return normalize_np(v+.25*np.roll(v,4,axis=1))
            raise RuntimeError('unknown mocked '+key)
        patched=dict(OUTPUT_ROOT=base/'多种实验方向探索',
                 validate_files=lambda:cfg,
                 load_dataset=lambda _: (cfg,val,tp,vp),
                 fingerprint=lambda a,b:'synthetic_r3',
                 verify_baseline=lambda a,b,c,d:d,
                 extract=mock_extract,
                 train_annotation=lambda a,b: (['fake_train'],'sha_fake_train'),
                 load_train_embeddings=lambda *args:(refs,'train_pair_records','subfingerprint',{},{}),
                 train_pair_features=lambda *args:(tr,gr,pids),
                 )
        import sys as _sys
        m=_sys.modules[__name__]
        with patch.multiple(m,**patched):
            selection=tuple(EXPERIMENTS)
            run_real('',selection,'cpu')
            dirs=list((base/'多种实验方向探索').glob('第三轮_*'))
            assert len(dirs)==1
            out=dirs[0]
            results=json.loads((out/'02_第三轮完成摘要.json').read_text(encoding='utf-8'))
            assert results['closed_set_variants']>=20
            with zipfile.ZipFile(out/ZIP_NAME) as arc:
                assert arc.testzip() is None
                assert '01_全部实验指标总表.csv' in arc.namelist()
                assert '开放集/O_多机制开放集对照.csv' in arc.namelist()
                assert not any(x.startswith('缓存/') for x in arc.namelist())
            # 模拟个别结果未写完，通过主流程只补缺并再次ZIP。
            deletion=out/'结果'/'C4_保护Top5_深层传播.json'
            deletion_csv=out/'逐查询'/'C4_保护Top5_深层传播.csv'
            deletion.unlink();deletion_csv.unlink()
            run_real(str(out),selection,'cpu')
            assert deletion.exists() and deletion_csv.exists()
            assert json.loads(deletion.read_text(encoding='utf-8'))['csv_sha256']==sha256(deletion_csv)
            # 模拟结果污染：续跑必须拒绝错误设置，不能悄悄混写。
            info=out/'运行元数据.json'
            obj=json.loads(info.read_text(encoding='utf-8'));obj['selection']=['B']
            atomic_json(info,obj)
            try:run_real(str(out),selection,'cpu')
            except RuntimeError as err:assert '拒绝混写' in str(err)
            else:raise AssertionError('没有拒绝篡改的续跑设置')
    say('[SelfTest PASS] 全分支/保护TopK/开放集/CSV/ZIP/续跑补缺/篡改拒绝/YAML')



def main():
    p=argparse.ArgumentParser(description='第三轮：保留有效实验 + 增加不同机制独立探索')
    p.add_argument('--self-test',action='store_true',help='CPU合成数据端到端自检')
    p.add_argument('--resume',default='',help='第三轮上次运行子目录路径，指纹不符会拒绝混写')
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--experiments',default='BCERTFKGVO',
          help='选择互不重复的分支：B偏置 C图 E翻转 R风险 T文本 F通道 K压缩 G图库密度 V区域 O开放集；如BCER')
    opt=p.parse_args()
    selected=opt.experiments.upper()
    if not selected or len(set(selected))!=len(selected) or any(x not in EXPERIMENTS for x in selected):
        p.error('experiments必须是BCERTFKGVO中不重复的字母')
    if opt.self_test:self_test()
    else:run_real(opt.resume,tuple(x for x in EXPERIMENTS if x in selected),opt.device)

if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();sys.exit(1)
