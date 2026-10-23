# -*- coding: utf-8 -*-
"""IRRA Epoch58 第四轮广域机制探索（独立单文件）。
A=历史成功方法同轮复核；P=无监督查询反馈；L=Train文本近邻；
X=Train联合主方向去除；I=频率/颜色干预；U=跨视图一致性；
D=Top20去冗余；S=多尺度输入；Z=文本反事实压力测试。
冻结Epoch58编码器，统一完整CUHK-PEDES Validation 6158x3078；
无额外外部Python脚本、无网络请求；支持--self-test与--resume。
Z删除/替换颜色属性会改变标签语义，其指标仅作敏感性诊断，禁止当成检索优化。
所有结果保存在 IRRA-main/多种实验方向探索/第四轮_广域机制探索_时间戳/。
Val长期用于探索，最终论文须锁定方法后独立Test。
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
VERSION = 'wide_exploration_round4_v1_20261022'
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
EXPERIMENTS = ('A', 'P', 'L', 'X', 'I', 'U', 'D', 'S', 'Z')
ZIP_NAME = '发送给ChatGPT_第四轮广域探索.zip'


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



# ==============================================================
# 第四轮独立机制：不以Validation标签选择参数，无跨分支自动调优
# A=历史成功方案的同轮对照 P=伪相关反馈 L=Train文本语义邻域
# X=Train估计的主特征子空间 I=图像频率与色彩
# U=多视图一致性与不确定性 D=图库候选多样性
# S=图像尺度变化 Z=文本反事实压力测试
# ==============================================================
_previous_transform_images_r4 = transform_images
_previous_transform_texts_r4 = transform_texts


def transform_images(x, mode):
    """输入为官方验证预处理后、CLIP标准化的 BCHW 张量。"""
    if mode in ('lowpass5', 'unsharp5', 'desaturate50', 'crop90'):
        if mode == 'crop90':
            _, _, height, width = x.shape
            hh, ww = max(2, int(height * .90)), max(2, int(width * .90))
            ys, xs = (height-hh)//2, (width-ww)//2
            return F.interpolate(x[:, :, ys:ys+hh, xs:xs+ww], size=(height,width),
                                 mode='bicubic', align_corners=False)
        if mode == 'desaturate50':
            gray = _previous_transform_images_r4(x, 'gray')
            return .5*x+.5*gray
        # 对标准化后图像做空间卷积，不混合通道，均值色边缘padding
        blurred = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        if mode == 'lowpass5':
            return blurred
        return x + .35*(x-blurred)
    return _previous_transform_images_r4(x,mode)


COLOR_WORDS = ('black','white','red','blue','green','yellow','purple',
               'orange','brown','gray','grey','pink')
COLOR_PATTERN = re.compile(r'\b('+'|'.join(COLOR_WORDS)+r')\b', re.I)
COLOR_SWAPS = dict(black='white', white='black', red='blue', blue='red',
                   green='yellow', yellow='green', purple='orange',orange='purple',
                   brown='gray', gray='brown', grey='brown', pink='green')


def transform_texts(texts, mode):
    if mode in ('remove_color', 'swap_color'):
        result=[]
        for t in texts:
            raw=str(t)
            if mode=='remove_color':
                changed=COLOR_PATTERN.sub('',raw)
                changed=re.sub(r'\s+', ' ', changed).strip()
            else:
                def sub(match):
                    w=match.group(0)
                    new=COLOR_SWAPS[w.lower()]
                    return new.capitalize() if w[0].isupper() else new
                changed=COLOR_PATTERN.sub(sub,raw)
            result.append(changed if changed else raw)
        return result
    return _previous_transform_texts_r4(texts,mode)


def query_feedback(query, gallery, initial_score, k=3, strength=.10,
                   positive_only=True):
    """无监督 Rocchio-like 伪相关反馈；TopK由原始分数确定，不知道PID。"""
    if query.shape[1]!=gallery.shape[1] or initial_score.shape != (len(query),len(gallery)):
        raise ValueError('伪相关反馈维度错误')
    order=np.argsort(-initial_score,axis=1,kind='stable')
    top=order[:,:k]
    evidence=normalize_np(gallery[top].mean(axis=1))
    if positive_only:
        output=normalize_np((1-strength)*query + strength*evidence)
    else:
        neg=normalize_np(gallery[order[:,20:40]].mean(axis=1))
        output=normalize_np(query + strength*(evidence-neg))
    return output


def train_text_neighbors(query, train_ref, k=5, strength=.15, gate=None,
                         batch=128):
    """只使用Train无标签文本特征作语义邻域，不访问Validation PID。"""
    q=normalize_np(query); ref=normalize_np(train_ref)
    if k>len(ref):raise ValueError('Train文本参考库小于k')
    out=np.empty_like(q)
    selected=0
    for i in range(0,len(q),batch):
        part=q[i:i+batch]
        sc=part @ ref.T
        ids=np.argsort(-sc,axis=1,kind='stable')[:,:k]
        values=np.take_along_axis(sc, ids,axis=1)
        weights=np.exp(np.clip((values-values.max(axis=1,keepdims=True))/.07,-50,0))
        weights/=weights.sum(axis=1,keepdims=True)
        centroid=normalize_np((ref[ids]*weights[:,:,None]).sum(axis=1))
        combined=normalize_np((1-strength)*part+strength*centroid)
        if gate is not None:
            valid=values[:,0]>=gate
            combined[~valid]=part[~valid]
            selected+=int(valid.sum())
        else:selected+=len(part)
        out[i:i+len(part)]=combined
    return out,dict(selected=selected,total=len(q),gate=gate,k=k,strength=strength)


def train_pca_remove(query,gallery,train_text,train_image,k):
    """使用独立Train成对特征估计共享显著方向，移除前k个主方向。"""
    if train_text.shape!=train_image.shape or train_text.shape[1]!=512:
        raise ValueError('训练参考特征不是成对512维')
    z=np.concatenate([train_text,train_image],axis=0).astype(np.float64)
    center=z.mean(axis=0)
    centered=z-center
    cov=centered.T @ centered / max(len(z)-1,1)
    eig,vec=np.linalg.eigh(cov)
    top=vec[:,-k:].astype(np.float32)
    cen=center.astype(np.float32)
    tq=query-cen[None,:]; gi=gallery-cen[None,:]
    tq-=((tq @ top) @ top.T); gi-=((gi @ top) @ top.T)
    return normalize_np(tq),normalize_np(gi),dict(top_eigenvalues=eig[-k:][::-1].tolist(),
           total_train_embeddings=len(z),removed=k,centering='pooled_train_only')


def stability_scores(original, flip, variant, policy):
    """三个视图的评分方差/最坏情况；不同于上一轮仅将原图和翻转特征平均。"""
    if original.shape!=flip.shape or original.shape!=variant.shape:
        raise ValueError('多视图得分维度不匹配')
    if policy=='mean':return (.5*original+.25*flip+.25*variant).astype(np.float32)
    if policy=='robust':
        mean=(original+flip+variant)/3.
        std=np.sqrt(((original-mean)**2+(flip-mean)**2+(variant-mean)**2)/3.)
        return (mean-.5*std).astype(np.float32)
    if policy=='min':return np.minimum(np.minimum(original,flip),variant).astype(np.float32)
    raise ValueError(policy)


def diversified_top20(scores, visual, lambda_div=.10, protect=1, max_k=20):
    """贪心MMR：保持原Top20候选集与Top1位置；余下19个重排。
    新分数仅被赋予原Top20现有分数的多重集，防止其他图像跨界。
    """
    if scores.shape[1]!=visual.shape[0] or visual.shape[0]!=visual.shape[1]:
        raise ValueError('MMR尺寸错误')
    if not 0 <= lambda_div <= 1 or not 1<=protect<max_k<=scores.shape[1]:
        raise ValueError('MMR系数/保护范围错误')
    out=np.array(scores,copy=True)
    n,m=scores.shape
    for qi in range(n):
        order=np.argsort(-scores[qi],kind='stable')
        top=order[:max_k]
        # 仅候选集内部的局部视觉相似度矩阵
        neighbors=visual[np.ix_(top,top)]
        first=list(range(protect)); avail=list(range(protect,max_k))
        local=scores[qi,top].astype(np.float64)
        scale=max(float(local[0]-local[-1]),1.e-7)
        relevance=(local-local[-1])/scale
        while avail:
            chosen=max(avail,key=lambda j:(float((1-lambda_div)*relevance[j] -
                 lambda_div*np.maximum(neighbors[j,first],0).max()),-j))
            first.append(chosen);avail.remove(chosen)
        perm=top[np.asarray(first,dtype=np.int32)]
        # 对所有候选的位置使用原始前20分数，不改变集合外的原始分数
        original_values=np.sort(scores[qi,top].copy())[::-1]
        out[qi,perm]=original_values
        # 原图的top1始终不变；如果保护的分数与其他分数并列，精度边界可能交换
        if out[qi,order[0]] < np.max(out[qi]):
            raise RuntimeError('去冗余违反Top1保护')
    return out


def fourth_settings():
    return dict(version=VERSION,checkpoint_sha=CHECKPOINT_SHA,config_sha=CONFIG_SHA,
       experiments=list(EXPERIMENTS),seed=SEED,baseline_tolerance_pp=BASELINE_TOLERANCE_PP,
       anchors=['B3','C3','E2','C4Top10'], feedback=[[3,.10],[10,.10],[3,.05]],
       feedback_contrastive=[3,.10],train_neighbor=[5,.15,.65],
       pca_removed=[1,4,8],image_modes=['lowpass5','unsharp5','desaturate50'],
       multi_view=['mean','robust','min'],diversity_lambdas=[.10,.25],
       scale=['crop90','crop90_mix20'],counterfactual=['remove_color','swap_color'],
       score_batch=SCORE_BATCH,image_batch=IMAGE_BATCH,text_batch=TEXT_BATCH)


def run_real(resume='',selection=EXPERIMENTS,device_name='cuda'):
    started=time.perf_counter()
    torch.set_num_threads(min(6,os.cpu_count() or 6))
    np.random.seed(SEED);torch.manual_seed(SEED)
    if device_name=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，请使用irra_clean环境或显式--device cpu')
    device=torch.device(device_name)
    cfg=validate_files()
    cfg,val,tp,vp=load_dataset(cfg)
    setup=fourth_settings()
    raw_fp=fingerprint(cfg,val)
    fp=hashlib.sha256((raw_fp+json.dumps(setup,sort_keys=True)).encode()).hexdigest()
    if resume:
        out=Path(resume).resolve()
        if not out.is_dir() or out.parent.resolve()!=OUTPUT_ROOT.resolve():
            raise RuntimeError('续跑目录必须为多种实验方向探索下的第四轮子目录')
        meta_path=out/'运行元数据.json'
        if not meta_path.is_file():raise RuntimeError('续跑目录元数据缺失')
        past=json.loads(meta_path.read_text(encoding='utf8'))
        if past.get('fingerprint')!=fp or past.get('version')!=VERSION or \
           past.get('settings')!=setup or past.get('selection')!=list(selection):
            raise RuntimeError('续跑设置或数据指纹不同，拒绝混写')
    else:
        OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
        out=OUTPUT_ROOT/('第四轮_广域机制探索_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        out.mkdir(parents=True,exist_ok=False)
        atomic_json(out/'运行元数据.json',dict(fingerprint=fp,version=VERSION,
                settings=setup,selection=list(selection),epoch=58,
                warnings=['Val仅为探索性评估，禁止将本轮多分支最好值当独立验证',
                          'Z文本反事实改变查询真实性，R1等仅代表文本压力测试，绝不作为检索优化',
                          'L使用Train文本无标签参考；P只用测试图库无标签特征',
                          'D只重排原始Top20，绝不从完整图库寻找正PID']))
        try:shutil.copy2(Path(__file__),out/'第四轮脚本_副本.py')
        except OSError:pass
    for name in ('结果','逐查询','缓存','统计','分析','反事实诊断'):
        (out/name).mkdir(parents=True,exist_ok=True)
    say('[第四轮输出目录]',out)
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
              historical=EXPECTED,historical_delta_pp={k:bm[k]-EXPECTED[k] for k in EXPECTED},
              queries=len(tp),gallery=len(vp),fingerprint=fp))
        prior=out/'结果'/'00_baseline.json'
        if prior.exists():
            old=json.loads(prior.read_text(encoding='utf8'))
            if old.get('fingerprint')!=fp or old.get('metrics')!=bm:
                raise RuntimeError('续跑的B0基线变化，拒绝继续')
        else:atomic_json(prior,dict(version=VERSION,fingerprint=fp,metrics=bm,name='B0_original'))
        say('[固定基线]',bm)

        ref=None; train_pairs=None; pair_records=None; subfp=None; train_annot=None
        baseline_flip=None; raw_flip=None; gallery_flip=None; c3=None
        original_unsharp=None; original_crop=None

        def training_ref():
            nonlocal ref,train_pairs,pair_records,subfp,train_annot
            if ref is None:
                train_annot,ann_sha=train_annotation(cfg['root_dir'],tp)
                ref,train_pairs,subfp,pseudo_img,pseudo_txt=load_train_embeddings(
                    model_fn,cfg,train_annot,ann_sha,fp,out/'缓存',device)
                pair_records=(pseudo_img,pseudo_txt)
            return ref
        def train_pair():
            training_ref()
            return train_pair_features(model_fn,cfg,subfp,out/'缓存',device,*pair_records)
        def flip_score():
            nonlocal baseline_flip,raw_flip,gallery_flip
            if baseline_flip is None:
                gallery_flip=extract(model_fn,cfg,val,out/'缓存',fp,
                           'gallery_flip',device,'image','flip')
                raw_flip=cosine_scores(query,gallery_flip,device)
                blend=normalize_np(.5*gallery+.5*gallery_flip)
                baseline_flip=cosine_scores(query,blend,device)
            return baseline_flip
        def unsharp_score():
            nonlocal original_unsharp
            if original_unsharp is None:
                vg=extract(model_fn,cfg,val,out/'缓存',fp,'image_unsharp5',device,'image','unsharp5')
                original_unsharp=cosine_scores(query,vg,device)
                del vg
            return original_unsharp
        def crop_score():
            nonlocal original_crop
            if original_crop is None:
                vg=extract(model_fn,cfg,val,out/'缓存',fp,'image_crop90',device,'image','crop90')
                original_crop=cosine_scores(query,vg,device)
                # 融合特征在 S 分支另行加载缓存；这里只缓存分数
            return original_crop
        def save(name,family,score,note):
            if score.shape!=original.shape or not np.all(np.isfinite(score)):
                raise RuntimeError(f'{name}出现无效得分/形状异常')
            return write_variant(out,fp,name,family,score,tp,vp,bm,bd,note)

        if 'A' in selection:
            r=training_ref()
            h,_=hubness_reference(original,gallery,r,device,neighbors=20)
            save('A_B3_TrainHub复核','A_旧增益同轮锚点',h,
                 '使用Train2048参考文本Top20校正，不使用Val查询分布')
            idx,strength=gallery_graph(gallery,device,8)
            g,stat=graph_scores(original,idx,strength,.30,'mutual_strong')
            save('A_C3_可信图传播复核','A_旧增益同轮锚点',g,
                 '图像互为Top8，视觉sim≥.70，强度加权传播比例.30')
            save('A_C4_保护Top10复核','A_旧增益同轮锚点',
                 protected_graph_rerank(original,g,10),'仅重排原始Top10后图像')
            save('A_E2_翻转融合复核','A_旧增益同轮锚点',flip_score(),
                 '原图与翻转特征50%+50%并归一化，保留上轮成功结构')
            atomic_json(out/'分析'/'A_可信图统计.json',stat)
            del idx,strength,g,h

        if 'P' in selection:
            for k,alpha in ((3,.10),(10,.10),(3,.05)):
                qf=query_feedback(query,gallery,original,k,alpha)
                save(f'P_伪相关反馈_k{k}_a{int(alpha*100)}', 'P_查询反馈',
                     cosine_scores(qf,gallery,device),
                     f'从原始Top{k}无标签图库图像形成反馈向量, query与反馈混合比例{alpha}，可能强化错误Top1')
                del qf
            qf=query_feedback(query,gallery,original,3,.10,False)
            save('P_正负对照反馈','P_对比式查询反馈',cosine_scores(qf,gallery,device),
                 '从Top3取伪正图库、原始第21-40名取伪负图库，反向抵消，全部无标签')
            del qf

        if 'L' in selection:
            refs=training_ref()
            for gate,name in ((None,'L_Train文本邻域无门控'),(.65,'L_Train文本邻域相似度门控')):
                tt,diag=train_text_neighbors(query,refs,5,.15,gate)
                save(name,'L_Train无标签文本近邻',cosine_scores(tt,gallery,device),
                     f'原查询与Top5 Train近邻文本方向混合；{diag}')
                atomic_json(out/'分析'/(name+'.json'),diag)
                del tt

        if 'X' in selection:
            tt,gg,pids=train_pair()
            # 真值PID仅用于Train采样成对特征，不参与PCA求解或Val决策
            for k in (1,4,8):
                tq,gi,diag=train_pca_remove(query,gallery,tt,gg,k)
                save(f'X_Train主方向去除_{k}', 'X_Train共享表征主子空间',
                     cosine_scores(tq,gi,device),
                     f'只由Train图文嵌入拟合联合中心及前{k}个PCA方向，两个模态统一投影')
                atomic_json(out/'分析'/f'X_Train主方向去除_{k}.json',diag)
                del tq,gi

        if 'I' in selection:
            for mode,name in (('lowpass5','I_低通5'),('unsharp5','I_增强细节5'),
                              ('desaturate50','I_降低彩色50')):
                score=unsharp_score() if mode=='unsharp5' else None
                if score is None:
                    vg=extract(model_fn,cfg,val,out/'缓存',fp,'image_'+mode,device,'image',mode)
                    score=cosine_scores(query,vg,device)
                    del vg
                save(name,'I_图像频率与色彩控制',score,
                     f'在CLIP归一化图像张量上做{mode}，模型不训练，作为输入敏感性实验')
                del score

        if 'U' in selection:
            flip_score()
            a=raw_flip; b=unsharp_score()
            for policy,name in (('mean','U_三视图分数均值'),('robust','U_三视图不确定性惩罚'),
                                ('min','U_三视图最差分数')):
                ss=stability_scores(original,a,b,policy)
                save(name,'U_图像证据一致性',ss,
                     f'原图、水平翻转、细节增强三视图；分数融合规则={policy}，非E2的特征融合')
                del ss
            atomic_json(out/'分析'/'U_跨视图差异统计.json',dict(
                  mean_abs_flip=float(np.mean(np.abs(original-a))),
                  mean_abs_unsharp=float(np.mean(np.abs(original-b)))))

        if 'D' in selection:
            visual=gallery @ gallery.T
            for coef in (.10,.25):
                result=diversified_top20(original,visual,coef,protect=1,max_k=20)
                m,_=evaluate(result,tp,vp)
                if abs(m['R1']-bm['R1'])>1.e-8:
                    raise RuntimeError('D保护Top1却改变R1，排名赋值逻辑需修复')
                save(f'D_Top20视觉去冗余_{int(100*coef)}','D_图库候选多样性',result,
                     '候选集及原Top1固定，Top20内其余候选按MMR减少视觉重复，仅R1与R20必定不变，R5/R10允许变化')
                del result
            del visual

        if 'S' in selection:
            ss=crop_score()
            save('S_中心缩放90','S_多尺度输入鲁棒性',ss,
                 '固定中心90%图像区域重新插值至IRRA原始输入尺寸，不使用GT框')
            vg=extract(model_fn,cfg,val,out/'缓存',fp,'image_crop90',device,'image','crop90')
            combined=normalize_np(.8*gallery+.2*vg)
            save('S_原图80缩放20融合','S_多尺度输入融合',
                 cosine_scores(query,combined,device),
                 '原图CLS特征80%与中心90%缩放图CLS特征20%融合，再归一化')
            del vg,combined

        if 'Z' in selection:
            report=[]
            for variant,name in (('remove_color','Z_去掉颜色词'),('swap_color','Z_互换颜色词')):
                transformed=transform_texts(val['captions'],variant)
                count=int(sum(a!=b for a,b in zip(val['captions'],transformed)))
                textvec=extract(model_fn,cfg,val,out/'缓存',fp,
                   'text_'+variant,device,'text',variant)
                ss=cosine_scores(textvec,gallery,device)
                result=save(name,'Z_反事实文本压力测试_不可作检索收益',ss,
                    '已故意删除/更换颜色语义，原PID仍为标签，仅是对输入证据敏感性的压力诊断，不能当检索优化')
                row=dict(name=name,changed_query_count=count,total_query=len(tp),
                         fraction_edited=count/len(tp),
                         mean_feature_cosine=float((textvec*query).sum(axis=1).mean()),
                         mean_score_abs_change=float(np.abs(ss-original).mean()),
                         top1_image_change=int(np.sum(np.argmax(original,axis=1)!=np.argmax(ss,axis=1))),
                         diagnostic_only=True)
                report.append(row)
                del textvec,ss
            write_csv(out/'反事实诊断'/'颜色词压力测试.csv',report,list(report[0]))

        results=update_summary(out,fp,bm)
        lines=[
         '第四轮广域探索：只使用同一个 Epoch58 冻结编码器；Val 6158×3078 官方评估。',
         'A仅复核以前的B3/C3/C4/E2，并不融合，也不将上轮结果作为本轮最优超参数。',
         'P伪相关查询反馈可能造成错误强化；L为Train文本语义近邻，无Val文本群体信息。',
         'X通过Train图文联合PCA估计方向；I为频率/颜色输入扰动；U为视图分数不确定性。',
         'D仅改变Top20内部相对次序，不从图库移除图像，也不改变Top20集合，因此前十命中不变。',
         'S是粗尺度中心放大，不表示准确人物分割。',
         'Z故意修改颜色属性，不能再把变换查询的原PID视作保证正确；数值仅代表对抗敏感性。',
         '所有方法预先固定参数，无PID参与无监督排序，不从Val结果反向选择模型。',
         'Validation已被反复查看；改进属于探索性发现，最终必须另外锁定方法并评估Test。',
         '每个子目录存放完整缓存和可续跑进度。ZIP不包含缓存文件。',
        ]
        (out/'00_阅读说明.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
        atomic_json(out/'02_第四轮完成摘要.json',dict(version=VERSION,fingerprint=fp,
             selection=list(selection),baseline=bm,closed_set_variants=len(results),
             elapsed_seconds=time.perf_counter()-started,results=results,
             warning='Z改写文本语义，相关数值不得称作检索准确率增益'))
        say('[第四轮完成]',len(results),'项，秒数',round(time.perf_counter()-started,1))
    except BaseException as e:
        (out/'错误_异常.txt').write_text(''.join(traceback.format_exception(type(e),e,e.__traceback__)),encoding='utf8')
        say('[异常日志]',out/'错误_异常.txt')
        raise
    finally:
        if model is not None:
            del model
            if device.type=='cuda':torch.cuda.empty_cache()
        try:make_zip(out)
        except BaseException as exc:say('[ZIP生成错误]',repr(exc))
        say('[实验目录]',out)


def make_zip(out):
    target=out/ZIP_NAME
    tmp=out/('.'+ZIP_NAME+'.partial')
    try:
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as f:
            for p in sorted(out.rglob('*')):
                if not p.is_file() or p in (target,tmp):continue
                rel=p.relative_to(out)
                if '缓存' in rel.parts:continue
                f.write(p,rel.as_posix())
        with zipfile.ZipFile(tmp) as f:
            check=f.testzip()
            if check is not None:raise RuntimeError('ZIP文件CRC不符: '+check)
        os.replace(tmp,target)
    finally:
        if tmp.exists():tmp.unlink()
    say('[ZIP完成]',target)


def self_test():
    """CPU无IRRA依赖合成数据全主流程，检验函数/导出/缓存续跑/篡改拒绝。"""
    from unittest.mock import patch
    say('[SelfTest] 测试所有第四轮分支与续跑')
    rng=np.random.default_rng(20261022)
    with tempfile.TemporaryDirectory(prefix='irra_round4_') as td:
        root=Path(td)
        yam=root/'tuple.yaml'
        yam.write_text('img_size: !!python/tuple [384, 128]\n',encoding='utf8')
        assert load_irra_yaml(yam)['img_size']==(384,128)
        # Memmap 模拟掉电续跑：必须只恢复前3行；指纹错误时拒绝读旧缓存。
        mm, done, meta = load_cache(root/'mmap_regression','trial',8,'fp_x')
        assert done==0
        mm[:3] = 2.5
        update_cache(mm,meta,'fp_x|trial',8,3)
        close_memmap(mm);del mm
        mm, done, meta = load_cache(root/'mmap_regression','trial',8,'fp_x')
        assert done==3 and np.all(mm[:3]==2.5)
        close_memmap(mm);del mm
        try:load_cache(root/'mmap_regression','trial',8,'wrong_fp')
        except RuntimeError as e:assert '续跑' in str(e)
        else:raise AssertionError('失效缓存指纹未被拦截')
        raw=torch.randn(2,3,60,24)
        for mode in ('lowpass5','unsharp5','desaturate50','crop90'):
            got=transform_images(raw,mode)
            assert got.shape==raw.shape and bool(torch.isfinite(got).all())
        sample=['A man with red shoes and black pants.', 'A woman carrying a bag.']
        assert transform_texts(sample,'swap_color')[0]!=sample[0]
        assert transform_texts(sample,'remove_color')[1]==sample[1]
        v=normalize_np(rng.standard_normal((38,512)).astype(np.float32))
        t=normalize_np(rng.standard_normal((52,512)).astype(np.float32))
        vp=np.arange(len(v),dtype=np.int64)%10
        tp=np.arange(len(t),dtype=np.int64)%10
        refs=normalize_np(rng.standard_normal((2048,512)).astype(np.float32))
        train_t=normalize_np(rng.standard_normal((640,512)).astype(np.float32))
        train_g=normalize_np(train_t+.33*rng.standard_normal((640,512)).astype(np.float32))
        train_pid=np.arange(640,dtype=np.int64)//2+2000
        config=dict(root_dir=str(root),img_size=(384,128),text_length=77)
        val=dict(img_paths=['synthetic_image_'+str(i) for i in range(len(v))],
            image_pids=vp.tolist(),captions=['Text red clothes.']*26+['Blue bag.']*26,
            caption_pids=tp.tolist())
        def mock_extract(factory,cfg,records,folder,fingerprint,key,device,kind,variant='original'):
            if key=='B0_gallery':return v.copy()
            if key=='B0_queries':return t.copy()
            if key=='gallery_flip':return normalize_np(v+.15*np.roll(v,1,axis=1))
            if key=='image_unsharp5':return normalize_np(v+.12*np.roll(v,2,axis=1))
            if key=='image_crop90':return normalize_np(v+.16*np.roll(v,3,axis=1))
            if key=='image_lowpass5':return normalize_np(v+.14*np.roll(v,4,axis=1))
            if key=='image_desaturate50':return normalize_np(v+.10*np.roll(v,5,axis=1))
            if key=='text_remove_color':return normalize_np(t+.15*np.roll(t,7,axis=1))
            if key=='text_swap_color':return normalize_np(t+.21*np.roll(t,8,axis=1))
            raise RuntimeError('unknown mock embedding: '+key)
        patched=dict(OUTPUT_ROOT=root/'多种实验方向探索',
               validate_files=lambda:config,
               load_dataset=lambda _: (config,val,tp,vp),
               fingerprint=lambda a,b:'fourth_synthetic',
               verify_baseline=lambda a,b,c,d:d,
               extract=mock_extract,
               train_annotation=lambda a,b:(['train_sample'],'dummy_sha'),
               load_train_embeddings=lambda *args:(refs,'dummytrainpairs','subfingerprint',{},{}),
               train_pair_features=lambda *args:(train_t,train_g,train_pid))
        with patch.multiple(sys.modules[__name__],**patched):
            selection=tuple(EXPERIMENTS)
            run_real('',selection,'cpu')
            paths=list((root/'多种实验方向探索').glob('第四轮_*'))
            assert len(paths)==1
            dest=paths[0]
            summary=json.loads((dest/'02_第四轮完成摘要.json').read_text(encoding='utf8'))
            assert summary['closed_set_variants']>=19,summary['closed_set_variants']
            with zipfile.ZipFile(dest/ZIP_NAME) as f:
                assert f.testzip() is None
                assert '01_全部实验指标总表.csv' in f.namelist()
                assert '反事实诊断/颜色词压力测试.csv' in f.namelist()
                assert not any(x.startswith('缓存/') for x in f.namelist())
            missing=dest/'结果'/'P_伪相关反馈_k3_a10.json'
            missing_csv=dest/'逐查询'/'P_伪相关反馈_k3_a10.csv'
            missing.unlink();missing_csv.unlink()
            run_real(str(dest),selection,'cpu')
            assert missing.exists() and missing_csv.exists()
            assert json.loads(missing.read_text(encoding='utf8'))['csv_sha256']==sha256(missing_csv)
            meta=dest/'运行元数据.json'
            obj=json.loads(meta.read_text(encoding='utf8'));obj['selection']=['P']
            atomic_json(meta,obj)
            try:run_real(str(dest),selection,'cpu')
            except RuntimeError as e:assert '拒绝混写' in str(e)
            else:raise AssertionError('元数据被改写后仍允许续跑')
    say('[SelfTest PASS] 频率/颜色/反馈/近邻/PCA/一致性/MMR/缩放/反事实/ZIP/续跑')


def main():
    parser=argparse.ArgumentParser(description='第四轮：IRRA大范围机制发散（单文件）')
    parser.add_argument('--self-test',action='store_true',help='CPU合成数据完整自检，不载入IRRA')
    parser.add_argument('--resume',default='',help='上次第四轮结果子文件夹绝对路径')
    parser.add_argument('--device',default='cuda',choices=('cuda','cpu'))
    parser.add_argument('--experiments',default='APLXIUDSZ',
           help='独立分支字母 A旧方法复核 P查询反馈 L训练文本 X主方向 I频率 U多视图 D去冗余 S尺度 Z文本压力测试')
    options=parser.parse_args()
    selected=options.experiments.upper()
    if not selected or len(set(selected))!=len(selected) or any(x not in EXPERIMENTS for x in selected):
        parser.error('experiments应为APLXIUDSZ的非重复字母集合')
    if options.self_test:self_test()
    else:run_real(options.resume,tuple(x for x in EXPERIMENTS if x in selected),options.device)


if __name__=='__main__':
    try:main()
    except BaseException:traceback.print_exc();sys.exit(1)
