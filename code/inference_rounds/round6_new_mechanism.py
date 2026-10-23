# -*- coding: utf-8 -*-
"""IRRA 第六轮：7大方向，新机制实验，冻结Epoch58权重。
单文件；支持Windows缓存续跑、完整Val验证、PID配对Bootstrap与ZIP。
O/R/T使用验证集无标签查询集合，仅传导式；其余是单查询或图库预处理。
任何方法都不能使用Val PID指导特征、参数或排名；PID仅用于最终评估。
--self-test 使用CPU合成数据验证全流程，不需要IRRA数据。
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
VERSION = 'wide_exploration_round6_v1_20261022'
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
EXPERIMENTS = ('O', 'R', 'T', 'L', 'A', 'S', 'D')
ZIP_NAME = '发送给ChatGPT_第六轮广域新机制探索.zip'


def say(*a):
    print(*a, flush=True)


def windows_retry_replace(src, dst, attempts=24):
    """Windows Defender / PyCharm 暂占目标文件时，有限重试原子替换。

    不直接覆盖一个正被读取的 JSON，也绝不删除旧目标以求替换成功。
    对权限永久拒绝的情况最终仍抛出有说明的错误。
    """
    src, dst = str(src), str(dst)
    for retry in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            if retry == attempts - 1:
                raise PermissionError(
                    f'Windows 拒绝原子替换 {dst}；已重试 {attempts} 次。'
                    f'如在 Excel、编辑器或同步软件中打开，请先关闭占用者。原始文件未删除。'
                ) from exc
            time.sleep(min(0.05 * (1.3 ** retry), 0.6))


def atomic_json(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=str(p.parent), prefix='.' + p.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush(); os.fsync(f.fileno())
        windows_retry_replace(name, p)
    finally:
        if Path(name).exists():
            try: Path(name).unlink()
            except PermissionError: pass


def write_csv(p, records, fields):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=str(p.parent), prefix='.' + p.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader(); w.writerows(records)
            f.flush(); os.fsync(f.fileno())
        windows_retry_replace(name, p)
    finally:
        if Path(name).exists():
            try: Path(name).unlink()
            except PermissionError: pass


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


def _read_cache_progress(folder, key, count, signature, baseline):
    """从不可变进度快照恢复，兼容旧版仅有 gallery_flip.json 的缓存。

    只有完成 JSON 写入、flush/fsync 且有末尾换行的快照才有效。
    断电产生的半截文件跳过；指纹错误的有效快照则拒绝混写。
    """
    latest = int(baseline)
    for p in folder.glob(key + '.progress.*.json'):
        try:
            raw = p.read_bytes()
            if not raw.endswith(b'\n'):
                say('[缓存快照忽略：不完整]', p.name)
                continue
            state = json.loads(raw.decode('utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError):
            say('[缓存快照忽略：无法解析]', p.name)
            continue
        if state.get('signature') != signature or state.get('count') != count:
            raise RuntimeError('缓存进度快照属于其他数据或模型：' + str(p))
        d = state.get('done')
        if type(d) is not int or not (0 <= d <= count):
            raise RuntimeError('缓存进度快照的范围不合法：' + str(p))
        latest = max(latest, d)
    return latest


def load_cache(folder, key, count, fp):
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / (key + '.npy'); meta = folder / (key + '.json')
    if p.exists() != meta.exists():
        raise RuntimeError(f'缓存文件与元数据不同步: {key}')
    signature = fp + '|' + key
    if p.exists():
        state = json.loads(meta.read_text(encoding='utf-8'))
        if state.get('signature') != signature or state.get('count') != count or \
           type(state.get('done')) is not int or not 0 <= state['done'] <= count:
            raise RuntimeError(f'缓存无法安全续跑: {key}')
        restored = _read_cache_progress(folder, key, count, signature, state['done'])
        arr = np.lib.format.open_memmap(p, mode='r+')
        if arr.shape != (count, 512) or arr.dtype != np.float32:
            close_memmap(arr)
            raise RuntimeError(f'缓存维度错误: {key}')
        if restored > state['done']:
            say(f'[恢复增量进度] {key}: 原元数据{state["done"]}，最近快照{restored}')
        return arr, restored, meta
    arr = np.lib.format.open_memmap(p, mode='w+', dtype='float32', shape=(count, 512))
    try:
        arr.flush()
        atomic_json(meta, {'signature': signature, 'count': count, 'done': 0})
    except BaseException:
        close_memmap(arr)
        raise
    return arr, 0, meta


def update_cache(arr, meta, signature, count, done):
    """不再重复覆盖固定 *.json，避开 WinError 5 的最常见触发点。

    每次创建唯一的 .progress.*.json，只增加新文件，不替换、不截断旧文件。
    npy.flush() 必须先完成：快照永不指向尚未同步的特征行。
    """
    if type(done) is not int or not 0 <= done <= count:
        raise ValueError('缓存进度越界')
    arr.flush()
    meta = Path(meta)
    progress = {'signature': signature, 'count': count, 'done': done}
    fd, name = tempfile.mkstemp(dir=str(meta.parent),
        prefix=meta.stem + '.progress.', suffix='.json')
    # 临时文件名本身就是最终快照名：不需要执行 os.replace()。
    # crash 前最后一个不完整快照会因缺少末尾换行而在续跑时被跳过。
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(progress, ensure_ascii=False, allow_nan=False))
            f.write('\n')
            f.flush(); os.fsync(f.fileno())
    except BaseException:
        try: Path(name).unlink()
        except OSError: pass
        raise


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
        if mode == 'first_clause': selected = parts[0]
        elif mode == 'last_clause': selected = parts[-1]
        else: raise ValueError(mode)
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




# ===================== 第六轮独立研究机制 =====================
ROUND_SEED = 20261022
PREFIX = '第六轮_大范围新机制_'
COMPLETED = '02_第六轮完成摘要.json'


def settings():
    return dict(version=VERSION, checkpoint_sha=CHECKPOINT_SHA, config_sha=CONFIG_SHA,
                groups=list(EXPERIMENTS), seed=ROUND_SEED, image_batch=IMAGE_BATCH,
                text_batch=TEXT_BATCH, baseline_tolerance=BASELINE_TOLERANCE_PP,
                sinkhorn_t=.07, sinkhorn_iterations=12, mini_query_batches=[128,512],
                query_graph_k=5, query_graph_threshold=.80, gallery_knn_k=8,
                graph_visual_threshold=.70, color_weight=[.06,.12], mmr_k=20,
                mmr_weights=[.20,.50],
                note='Validation PID仅用于评估，不用于筛选邻居、颜色、参数或分数')


def balanced_sinkhorn_scores(scores, temp=.07, iters=12):
    """完全复刻第五轮低温Sinkhorn，使用整批无标签查询；从不看PID。"""
    s=np.asarray(scores,dtype=np.float32)
    if s.ndim!=2 or temp<=0 or not np.all(np.isfinite(s)):
        raise ValueError('Sinkhorn输入非法')
    z=(s-s.max(axis=1,keepdims=True))/temp
    k=np.exp(np.maximum(z,-50)).astype(np.float32)
    r_target=1/len(s); c_target=1/s.shape[1]
    u=np.ones(len(s),np.float64);v=np.ones(s.shape[1],np.float64)
    for _ in range(iters):
        u=r_target/np.maximum(k@v,1e-100)
        v=c_target/np.maximum(k.T@u,1e-100)
    ans=s+temp*np.log(np.maximum(v,1e-100)).astype(np.float32)[None,:]
    return ans.astype(np.float32),dict(temperature=temp,iterations=iters,query_count=len(s),
             transductive=True,column_bias_std=float(np.std(temp*np.log(v))))


def batched_sinkhorn(scores, batch_size, seed=ROUND_SEED):
    """固定随机分桶，桶与桶之间不交换查询信息；不使用PID。"""
    n=len(scores); rng=np.random.default_rng(seed)
    perm=rng.permutation(n)
    result=np.empty_like(scores)
    stats=[]
    for start in range(0,n,batch_size):
        ids=perm[start:start+batch_size]
        result[ids],diag=balanced_sinkhorn_scores(scores[ids],temp=.07,iters=12)
        stats.append(diag['column_bias_std'])
    return result,dict(batch_size=batch_size,num_groups=len(stats),
                       avg_column_bias_std=float(np.mean(stats)),transductive=True,
                       partition_seed=seed)


def column_calibration(scores, method='logmeanexp', strength=.5, temp=.07):
    """对跨查询列分布做无标签评分刻度矫正，非普通逐行缩放。"""
    s=np.asarray(scores,dtype=np.float32)
    if method=='logmeanexp':
        x=s/temp
        mx=x.max(axis=0,keepdims=True)
        bias=temp*(mx[0]+np.log(np.maximum(np.mean(np.exp(x-mx),axis=0),1e-30)))
        return (s-strength*bias[None,:]).astype(np.float32)
    if method=='zscore':
        mean=s.mean(axis=0,keepdims=True)
        sigma=s.std(axis=0,keepdims=True)
        z=(s-mean)/np.maximum(sigma,.03)
        return ((1-strength)*s+strength*(z*.15)).astype(np.float32)
    raise ValueError(method)


def query_mutual_neighbors(query,device,k=5,threshold=.80):
    """查询文本互相作为近邻，双向检查；不能使用身份标签。"""
    if len(query)<k+2:raise ValueError('查询数量不足')
    sim=cosine_scores(query,query,device)
    np.fill_diagonal(sim,-np.inf)
    ids=np.argpartition(-sim,kth=k-1,axis=1)[:,:k]
    values=np.take_along_axis(sim,ids,axis=1)
    perm=np.argsort(-values,axis=1,kind='stable')
    ids=np.take_along_axis(ids,perm,axis=1)
    values=np.take_along_axis(values,perm,axis=1)
    # 双向kNN，只有互相近邻且相似度>=阈值才允许共享证据
    back=np.any(ids[ids]==np.arange(len(query))[:,None,None],axis=2)
    # back[i,j] 当且仅当 i 属于第 j 个文本邻居自己的近邻列表
    assert back.shape==ids.shape
    mask=(values>=threshold)&back
    del sim
    return ids,values,mask


def query_cooperation(score,ids,values,mask,alpha=.15):
    """各查询共享相近无标签文本的图库分数；没有可信邻居的查询不变。"""
    out=np.array(score,copy=True,dtype=np.float32)
    n=len(score)
    for a in range(0,n,64):
        b=min(n,a+64)
        w=np.maximum(values[a:b]-.5,0)*mask[a:b]
        denom=w.sum(axis=1,keepdims=True)
        used=(denom[:,0]>1e-9)
        if used.any():
            norms=w[used]/denom[used]
            neighbor=score[ids[a:b][used]] # 64 × 5 × gallery
            agg=np.einsum('nk,nkg->ng',norms,neighbor,optimize=True)
            part=out[a:b]
            part[used]=(1-alpha)*score[a:b][used]+alpha*agg
    return out


def clause_scores(original,first,last,mode):
    """首末分句可能省略属性，必须标注文本变化，不能等同语义保持。"""
    if mode=='mean':return (.5*first+.5*last).astype(np.float32)
    if mode=='min':return np.minimum(first,last).astype(np.float32)
    if mode=='max':return np.maximum(first,last).astype(np.float32)
    raise ValueError(mode)


COLOR_NAMES=('black','white','gray','brown','red','orange','yellow','green','blue','purple','pink')
COLOR_REGEX=re.compile(r'\b(black|white|gray|grey|brown|red|orange|yellow|green|blue|purple|pink)\b',re.I)
UPPER_TERMS=('shirt','t-shirt','top','jacket','coat','sweater','hoodie','blouse','sleeve','vest','uniform')
LOWER_TERMS=('pants','trousers','jeans','shorts','skirt','leggings')
IGNORE_TERMS=('shoe','shoes','sneakers','boots','hat','cap','bag','backpack')


def query_color_targets(captions):
    """句内最近衣物词启发式确定颜色与人体上下区域，无人工真实属性或Val PID。"""
    out=np.zeros((len(captions),2,len(COLOR_NAMES)),dtype=np.float32)
    meaningful=0
    for qi,cap in enumerate(captions):
        for clause in re.split(r'[,.!?;]',str(cap).lower()):
            tokens=re.findall(r"[a-z]+(?:-[a-z]+)?",clause)
            if not tokens:continue
            for j,token in enumerate(tokens):
                m=COLOR_REGEX.fullmatch(token)
                if not m:continue
                color=m.group().replace('grey','gray')
                candidates=[]
                for at,word in enumerate(tokens):
                    if word in UPPER_TERMS: candidates.append((abs(j-at),0))
                    elif word in LOWER_TERMS: candidates.append((abs(j-at),1))
                    elif word in IGNORE_TERMS: candidates.append((abs(j-at),-1))
                if candidates:
                    dist,region=min(candidates,key=lambda x:x[0])
                    if dist>5 or region<0:continue
                else:continue
                out[qi,region,COLOR_NAMES.index(color)]=1.0
        if out[qi].sum()>0:meaningful+=1
    return out,meaningful


def colors_from_hsv(hsv):
    """返回离散色号。HSV为uint8，分类阈值固定、非语义分割。"""
    h=hsv[...,0].astype(np.float32)*360/255
    s=hsv[...,1].astype(np.float32)/255
    v=hsv[...,2].astype(np.float32)/255
    idx=np.full(h.shape,COLOR_NAMES.index('gray'),np.int8)
    idx[(s<.19)&(v>.78)]=COLOR_NAMES.index('white')
    idx[v<.23]=COLOR_NAMES.index('black')
    colorful=(s>=.19)&(v>=.23)
    idx[colorful & ((h<15)|(h>=345))]=COLOR_NAMES.index('red')
    idx[colorful & (h>=15)&(h<45)]=COLOR_NAMES.index('orange')
    idx[colorful & (h>=45)&(h<70)]=COLOR_NAMES.index('yellow')
    idx[colorful & (h>=70)&(h<170)]=COLOR_NAMES.index('green')
    idx[colorful & (h>=170)&(h<265)]=COLOR_NAMES.index('blue')
    idx[colorful & (h>=265)&(h<315)]=COLOR_NAMES.index('purple')
    idx[colorful & (h>=315)&(h<345)]=COLOR_NAMES.index('pink')
    is_brown=colorful&(h>=15)&(h<60)&(v<.55)
    idx[is_brown]=COLOR_NAMES.index('brown')
    return idx


def compute_region_colors(image_paths,output_path,signature):
    """保存独立的颜色元数据；不混用模型特征缓存，禁止无指纹覆盖。"""
    from PIL import Image
    out=Path(output_path)
    meta=out.with_suffix('.json')
    if out.is_file() or meta.is_file():
        if not out.is_file() or not meta.is_file():raise RuntimeError('颜色缓存元数据不完整')
        old=json.loads(meta.read_text(encoding='utf-8'))
        if old.get('signature')!=signature or old.get('n')!=len(image_paths):
            raise RuntimeError('颜色缓存指纹不匹配')
        data=np.load(out,allow_pickle=False)
        if data.shape!=(len(image_paths),2,len(COLOR_NAMES)) or not np.isfinite(data).all():
            raise RuntimeError('颜色缓存数据损坏')
        return data.astype(np.float32,copy=False)
    ratios=np.zeros((len(image_paths),2,len(COLOR_NAMES)),np.float32)
    for i,path in enumerate(image_paths):
        with Image.open(path) as im:
            hs=np.asarray(im.convert('RGB').resize((48,144)).convert('HSV'),dtype=np.uint8)
        # 图像已由行人数据集框选，但上下区域不等同精确的衣物像素
        for part,(lo,hi) in enumerate(((24,83),(80,135))):
            labels=colors_from_hsv(hs[lo:hi,8:40])
            ratios[i,part]=np.bincount(labels.ravel(),minlength=len(COLOR_NAMES))/labels.size
        if i and i%500==0:say('[图像颜色区域]',i,'/',len(image_paths))
    out.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(dir=str(out.parent),suffix='.npy',prefix='.color.')
    os.close(fd)
    try:
        np.save(temp,ratios,allow_pickle=False)
        windows_retry_replace(temp,out)
        atomic_json(meta,dict(signature=signature,n=len(image_paths),regions=['upper','lower'],
             method='HSV coarse two-region without person segmentation'))
    finally:
        if Path(temp).exists():Path(temp).unlink()
    return ratios


def attribute_color_scores(targets,regions,mode='both'):
    """文字颜色在两段粗略图像区域中的匹配比例，绝不引用身份标签。"""
    assert targets.shape[1:]==regions.shape[1:]
    if mode=='upper':t=targets[:,0,:]; r=regions[:,0,:]
    elif mode=='lower':t=targets[:,1,:];r=regions[:,1,:]
    elif mode=='both':
        t=targets.reshape(len(targets),-1);r=regions.reshape(len(regions),-1)
    else:raise ValueError(mode)
    num=t@r.T
    den=np.maximum(t.sum(axis=1,keepdims=True),1)
    return (num/den).astype(np.float32)


def visual_neighbor_graph(gallery,device,k=8,min_cos=.70):
    """仅用视觉向量构建双向近邻；节点自身没有边，孤立节点保持原分数。"""
    n=len(gallery)
    if n<=k:raise ValueError('图库容量不足')
    scores=cosine_scores(gallery,gallery,device)
    np.fill_diagonal(scores,-np.inf)
    ids=np.argpartition(-scores,k-1,axis=1)[:,:k]
    sim=np.take_along_axis(scores,ids,axis=1)
    perm=np.argsort(-sim,axis=1,kind='stable')
    ids=np.take_along_axis(ids,perm,axis=1)
    sim=np.take_along_axis(sim,perm,axis=1)
    mutual=np.any(ids[ids]==np.arange(n)[:,None,None],axis=2)
    mask=mutual&(sim>=min_cos)
    weights=np.maximum(sim,0)*mask
    degree=weights.sum(axis=1,keepdims=True)
    weights=np.divide(weights,degree,out=np.zeros_like(weights),where=degree>0)
    return ids,weights,dict(visual_edges=int(mask.sum()//2),
                             isolated=int((degree[:,0]==0).sum()),
                             min_cos=min_cos,neighbors=k,pid_used=False)


def graph_step(s,idx,w,alpha=.3):
    """图传播是实数加权分数传播，不以身份标签连接节点。"""
    out=np.empty_like(s)
    for a in range(0,len(s),64):
        block=s[a:a+64]
        neighbor=block[:,idx] # batch, image, nearest_k
        agg=np.einsum('bnk,nk->bn',neighbor,w,optimize=True)
        noniso=w.sum(axis=1)>0
        out[a:a+len(block)]=np.where(noniso[None,:],(1-alpha)*block+alpha*agg,block)
    return out


def mmr_rerank(s,gallery,device,lam=.2,k=20):
    """单查询贪心MMR，候选集在原始Top20内，图库其余排名保持在Top20之后。"""
    if not (0<=lam<1) or k>=s.shape[1]:raise ValueError('MMR参数错误')
    # GPU可选，视觉两两相似度 (3078²) 不构造庞大多batch矩阵。
    simgg=cosine_scores(gallery,gallery,device)
    out=np.array(s,copy=True)
    # 必须采用完整稳定排序：Top20/21边界分数并列时也不会暗中换候选
    top=np.argsort(-s,axis=1,kind='stable')[:,:k+1]
    for qi,cands in enumerate(top):
        ids=cands[:k]; matrix=simgg[np.ix_(ids,ids)]
        raw=s[qi,ids]; span=max(float(raw.max()-raw.min()),1e-5)
        relevance=(raw-raw.min())/span
        selected=[]; available=list(range(k))
        for j in range(k):
            if not selected:
                # 首选也需接受冗余惩罚；不人为保护原始Top1
                local=matrix.copy()
                np.fill_diagonal(local,-2.0)
                crowd=np.max(local,axis=1)
                nxt=int(np.argmax((1-lam)*relevance-lam*crowd))
            else:
                penalty=np.max(matrix[np.ix_(available,selected)],axis=1)
                utility=(1-lam)*relevance[available]-lam*penalty
                nxt=available[int(np.argmax(utility))]
            selected.append(nxt);available.remove(nxt)
        # 只改变原始Top20内部顺序，固定Top20成员；分数严格高于Top21
        floor=float(s[qi,cands[k]])
        diff=max(float(raw[0])-floor,1e-5)
        spacing=diff/(k+1)
        for place,which in enumerate(selected):
            out[qi,ids[which]]=floor+(k-place)*spacing
    return out


def bootstrap_pid(out,name,pid,b,p,repeats=350):
    ids,inverse=np.unique(np.asarray(pid),return_inverse=True)
    counts=np.bincount(inverse).astype(np.float64)
    rng=np.random.default_rng(ROUND_SEED+17)
    # 同一组bootstrap索引用于各指标
    draws=rng.integers(0,len(ids),size=(repeats,len(ids)))
    denominator=counts[draws].sum(axis=1)
    rows=[]
    for key,access in [('R1',lambda x:x['correct'].astype(float)),
                       ('mAP',lambda x:x['ap'].astype(float)),
                       ('mINP',lambda x:x['inp'].astype(float))]:
        delta=access(p)-access(b)
        sums=np.bincount(inverse,weights=delta,minlength=len(ids))
        observed=float(100*np.mean(delta))
        intervals=100*sums[draws].sum(axis=1)/denominator
        rows.append(dict(name=name,metric=key,delta_pp=observed,
             ci_low_pp=float(np.quantile(intervals,.025)),
             ci_high_pp=float(np.quantile(intervals,.975)),
             clusters=len(ids),repeats=repeats))
    write_csv(out/'统计'/(name+'_PID_bootstrap.csv'),rows,list(rows[0]))


def save_variant(out,fp,name,family,score,tp,vp,bm,bd,note,diagnostic=False):
    if score.shape!=(len(tp),len(vp)) or not np.isfinite(score).all():
        raise RuntimeError('分数形状/数值非法：'+name)
    r=record_variant(out,fp,name,family,score,tp,vp,bm,bd,note)
    stat=out/'统计'/(name+'_PID_bootstrap.csv')
    if not stat.is_file():
        _,per=evaluate(score,tp,vp)
        bootstrap_pid(out,name,tp,bd,per)
    if diagnostic:
        (out/'诊断'/(name+'.txt')).write_text(
           '该实验改变了文本信息内容或使用粗粒度启发式属性；其结果须视为诊断，'
           '不得将外观语义变化当成同条件识别优化。\n',encoding='utf-8')
    return r


def make_zip(out):
    target=out/ZIP_NAME
    tmp=out/('.'+ZIP_NAME+'.partial')
    try:
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for path in sorted(out.rglob('*')):
                if not path.is_file() or path in (target,tmp):continue
                if '缓存' in path.relative_to(out).parts:continue
                z.write(path,path.relative_to(out).as_posix())
        with zipfile.ZipFile(tmp) as z:
            if (bad:=z.testzip()) is not None:raise RuntimeError('ZIP CRC错误:'+str(bad))
        windows_retry_replace(tmp,target)
    finally:
        if tmp.exists():
            try:tmp.unlink()
            except OSError:pass
    say('[第六轮ZIP]',target)


def find_unfinished(fp,setup,selected):
    if not OUTPUT_ROOT.is_dir():return ''
    candidates=[]
    for folder in OUTPUT_ROOT.glob(PREFIX+'*'):
        if (folder/COMPLETED).is_file():continue
        meta=folder/'运行元数据.json'
        if not meta.is_file():continue
        try:obj=json.loads(meta.read_text(encoding='utf-8'))
        except (OSError,ValueError):continue
        if obj.get('fingerprint')==fp and obj.get('settings')==setup and \
              obj.get('selection')==list(selected) and obj.get('version')==VERSION:
            candidates.append(folder)
    if candidates:
        pick=sorted(candidates,key=lambda p:p.name)[-1]
        say('[自动续跑]',pick)
        return str(pick)
    return ''


def run_real(resume='', selection=EXPERIMENTS, device_name='cuda', fresh=False):
    started=time.perf_counter()
    torch.set_num_threads(min(6,os.cpu_count() or 6))
    np.random.seed(ROUND_SEED);torch.manual_seed(ROUND_SEED)
    if device_name=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，请使用原irra_clean解释器')
    device=torch.device(device_name)
    cfg=validate_files()
    cfg,val,tp,vp=load_dataset(cfg)
    setup=settings()
    key=hashlib.sha256((fingerprint(cfg,val)+'|'+json.dumps(setup,sort_keys=True,
                 ensure_ascii=False)).encode()).hexdigest()
    if not resume and not fresh:resume=find_unfinished(key,setup,selection)
    if resume:
        out=Path(resume).resolve()
        if not out.is_dir() or out.parent.resolve()!=OUTPUT_ROOT.resolve() or not out.name.startswith(PREFIX):
            raise RuntimeError('指定续跑目录不属于多种实验方向探索下第六轮')
        meta=out/'运行元数据.json'
        if not meta.is_file():raise RuntimeError('续跑元数据不存在')
        previous=json.loads(meta.read_text(encoding='utf-8'))
        if previous.get('fingerprint')!=key or previous.get('settings')!=setup or \
                previous.get('selection')!=list(selection) or previous.get('version')!=VERSION:
            raise RuntimeError('续跑目录的模型、选择、参数或数据指纹不符，禁止混写')
    else:
        OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
        out=OUTPUT_ROOT/(PREFIX+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        out.mkdir(exist_ok=False)
        atomic_json(out/'运行元数据.json',dict(version=VERSION,fingerprint=key,
               settings=setup,selection=list(selection),epoch=58,
               limitations=['O/R/T使用全部无标签Validation查询，不是在线单查询',
                            'L的分句可能删除身份特征，不等同同语义输入',
                            'A粗略上下区域颜色不是人体语义分割',
                            'Validation历史多轮被使用，所有收益仅探索性']))
        try:shutil.copy2(Path(__file__),out/'第六轮完整脚本副本.py')
        except OSError:pass
    for folder in ('缓存','结果','逐查询','统计','分析','诊断'):
        (out/folder).mkdir(parents=True,exist_ok=True)
    say('[实验目录]',out)
    model=None
    def get_model():
        nonlocal model
        if model is None:model=load_model(cfg,device)
        return model
    try:
        gallery=extract(get_model,cfg,val,out/'缓存',key,'B0_gallery',device,'image')
        query=extract(get_model,cfg,val,out/'缓存',key,'B0_queries',device,'text')
        s0=cosine_scores(query,gallery,device)
        bm,bd=evaluate(s0,tp,vp)
        official=verify_baseline(s0,tp,vp,bm)
        baseline_path=out/'结果'/'00_baseline.json'
        if baseline_path.exists():
            previous=json.loads(baseline_path.read_text(encoding='utf-8'))
            if previous.get('fingerprint')!=key or previous.get('metrics')!=bm:
                raise RuntimeError('基线与已有缓存不一致，拒绝继续')
        else:
            atomic_json(baseline_path,dict(version=VERSION,fingerprint=key,metrics=bm,name='B0'))
        atomic_json(out/'00_原始Epoch58核验.json',dict(baseline=bm,official=official,
              expected_previous_baseline=EXPECTED,checkpoint_sha=CHECKPOINT_SHA,
              config_sha=CONFIG_SHA,num_queries=len(tp),num_gallery=len(vp),fingerprint=key))
        say('[B0官方Val]',bm)
        completed=read_results(out,key)
        def wanted(names):
            return any(n not in completed or not (out/'统计'/(n+'_PID_bootstrap.csv')).is_file() for n in names)
        def save(name,family,score,note,diagnostic=False):
            result=save_variant(out,key,name,family,score,tp,vp,bm,bd,note,diagnostic)
            completed[name]=result
            return result

        if 'O' in selection:
            name='O_全查询低温Sinkhorn_第五轮复核'
            if wanted([name]):
                s,diag=balanced_sinkhorn_scores(s0,.07,12)
                save(name,'O_双边最优传输',s,
                    '传导式，6158个无标签查询共同进行0.07温度/12迭代列平衡')
                atomic_json(out/'分析'/(name+'.json'),diag)
                del s
            for chunk in (128,512):
                name=f'O_随机查询分桶{chunk}_Sinkhorn'
                if wanted([name]):
                    s,diag=batched_sinkhorn(s0,chunk)
                    save(name,'O_分桶矩阵平衡',s,
                         f'传导式，同一批次{chunk}条查询交换信息；随机分桶种子固定，禁止使用PID')
                    atomic_json(out/'分析'/(name+'.json'),diag)
                    del s

        if 'R' in selection:
            for mode,strength,name in [('logmeanexp',.5,'R_列LogSumExp半强度'),
                                       ('logmeanexp',1.,'R_列LogSumExp全强度'),
                                       ('zscore',.25,'R_列方差标准化')]:
                if wanted([name]):
                    sc=column_calibration(s0,mode,strength)
                    save(name,'R_跨查询统计校准',sc,
                         f'传导式，图像列统计{mode}, 权重={strength}, 没有Val身份信息')
                    del sc

        if 'T' in selection:
            names=['T_互惠文本邻居10传播','T_互惠文本邻居25传播','T_宽松文本邻居10传播']
            if wanted(names):
                ids,values,mutual=query_mutual_neighbors(query,device,k=5,threshold=.80)
                atomic_json(out/'分析'/'T_无标签文本互惠近邻.json',dict(
                    k=5,threshold=.80,pairs_directed=int(mutual.sum()),
                    queries_with_neighbors=int(np.any(mutual,axis=1).sum()),
                    num_queries=len(query),identity_labels_used=False,transductive=True))
                for name,alpha,mask in [(names[0],.10,mutual),(names[1],.25,mutual),
                          (names[2],.10,values>=.75)]:
                    if wanted([name]):
                        sc=query_cooperation(s0,ids,values,mask,alpha=alpha)
                        save(name,'T_文本图查询协作',sc,
                            f'传导式：文本互相提供检索证据；k=5, alpha={alpha}; '
                            '宽松版只要求余弦>=.75，非宽松版双向近邻且>=.80')
                        del sc

        if 'L' in selection:
            names=['L_首末分句均值','L_首末分句取小','L_首末分句取大']
            if wanted(names):
                one=extract(get_model,cfg,val,out/'缓存',key,'query_first_clause',
                             device,'text','first_clause')
                two=extract(get_model,cfg,val,out/'缓存',key,'query_last_clause',
                             device,'text','last_clause')
                first=cosine_scores(one,gallery,device)
                last=cosine_scores(two,gallery,device)
                changed=sum(len(split_caption(t))>1 for t in val['captions'])
                atomic_json(out/'分析'/'L_分句覆盖率.json',dict(
                    multi_clause_queries=changed,total=len(tp),
                    note='仅分句文本可能改变身份语义；此分支为敏感性诊断'))
                for name,mode in zip(names,('mean','min','max')):
                    if wanted([name]):
                        sc=clause_scores(s0,first,last,mode)
                        save(name,'L_分句证据分解',sc,
                            f'改变输入描述：只使用首末分句的图文分数{mode}融合，属于信息缺失诊断',True)
                        del sc
                del first,last,one,two

        if 'A' in selection:
            names=['A_颜色上衣区域06','A_颜色下装区域06','A_颜色上下区域12']
            if wanted(names):
                regions=compute_region_colors(val['img_paths'],out/'缓存'/'A_颜色比例.npy',key)
                targets,covered=query_color_targets(val['captions'])
                atomic_json(out/'分析'/'A_颜色启发式覆盖.json',dict(
                    query_with_usable_color=covered,query_count=len(targets),
                    color_categories=COLOR_NAMES,method='fixed English clothing heuristics',
                    no_val_identity_used=True,not_a_person_segmentation=True))
                for name,region,strength in [(names[0],'upper',.06),
                                             (names[1],'lower',.06),
                                             (names[2],'both',.12)]:
                    if wanted([name]):
                        color=attribute_color_scores(targets,regions,region)
                        # 乘权后每条query减去平均是常数项，不影响排名；有助检查数值量纲
                        sc=s0+strength*(color-color.mean(axis=1,keepdims=True))
                        save(name,'A_启发式颜色局部对齐',sc.astype(np.float32),
                             f'单查询、无需身份监督，粗裁剪图像上下段HSV统计；颜色权重={strength}; '
                             '不是精确衣物分割，颜色提取不代表可靠身份证据',True)
                        del sc,color

        if 'S' in selection:
            names=['S_视觉双跳传播','S_视觉三跳传播','S_视觉两跳反差校正']
            if wanted(names):
                idx,w,info=visual_neighbor_graph(gallery,device,k=8,min_cos=.70)
                atomic_json(out/'分析'/'S_视觉图连接统计.json',info)
                step1=graph_step(s0,idx,w,alpha=.30)
                step2=graph_step(step1,idx,w,alpha=.30)
                if wanted([names[0]]):
                    save(names[0],'S_高阶视觉扩散',step2,
                         '无需PID的可信视觉双向邻居，做两次30%迭代分数传播，不保护任何TopK')
                if wanted([names[2]]):
                    sc=(s0+.4*(step2-step1)).astype(np.float32)
                    save(names[2],'S_传播增量反差',sc,
                         '原始分数增加40%二跳与一跳的差分，检验新增邻域证据而非重复一跳')
                    del sc
                if wanted([names[1]]):
                    step3=graph_step(step2,idx,w,alpha=.30)
                    save(names[1],'S_高阶视觉扩散',step3,
                         '无监督互惠视觉图三跳扩散，不使用任何Validation身份标签')
                    del step3
                del idx,w,step1,step2

        if 'D' in selection:
            for strength,name in [(.20,'D_候选MMR去重20'),(.50,'D_候选MMR去重50')]:
                if wanted([name]):
                    sc=mmr_rerank(s0,gallery,device,lam=strength,k=20)
                    # 强制Top20成员不变，精确外部位置不会由MMR修改
                    base_top=np.argsort(-s0,axis=1,kind='stable')[:,:20]
                    new_top=np.argsort(-sc,axis=1,kind='stable')[:,:20]
                    if any(set(x)!=set(y) for x,y in zip(base_top,new_top)):
                        raise RuntimeError('MMR发生Top20候选集合越界，必须停止')
                    save(name,'D_候选边际相关性多样化',sc,
                         f'原始Top20候选保持集合不变，使用视觉余弦MMR去重 lam={strength}; '
                         '不保证R1/R5/R10不变，仅保证Top20集合')
                    del sc

        summary=update_summary(out,key,bm)
        (out/'00_阅读说明.txt').write_text(
            'IRRA第六轮：O整批低温Sinkhorn及分桶；R跨查询列分布校准；'
            'T文本查询互惠近邻；L描述分句；A原始图像HSV粗衣物区域；'
            'S无监督多跳视觉图；D图库候选MMR去冗余。\n'
            '全部Epoch58主模型冻结，所有方法独立对照完整Validation。'
            'O/R/T明确为Transductive；L/A为启发式或输入信息变化敏感性诊断。\n'
            '模型权重与配置校验采用固定SHA256。自检合成数据不代表CUDA真实验证。'
            '数据缓存仅在本目录，ZIP不包含缓存；所有提升为探索性，需要独立Test。\n',
            encoding='utf-8')
        atomic_json(out/COMPLETED,dict(version=VERSION,fingerprint=key,
            baseline=bm,families=list(selection),variants=len(summary),
            elapsed_seconds=round(time.perf_counter()-started,2),results=summary))
        say('[第六轮完成] 变体数',len(summary),'秒',round(time.perf_counter()-started,2))
    except BaseException as error:
        (out/'错误_异常.txt').write_text(''.join(traceback.format_exception(
            type(error),error,error.__traceback__)),encoding='utf-8')
        say('[异常记录]',out/'错误_异常.txt')
        raise
    finally:
        if model is not None:
            del model
            if device.type=='cuda':torch.cuda.empty_cache()
        try:make_zip(out)
        except BaseException as error:say('[ZIP导出异常]',repr(error))
        say('[实验保存目录]',out)


def self_test():
    """不依赖CUDA/项目，在合成数据上验证算法、缓存、全部变体、续跑、ZIP。"""
    from unittest.mock import patch
    say('[第六轮自检] 数学函数...')
    rng=np.random.default_rng(ROUND_SEED)
    n_q=25;n_g=21
    q=normalize_np(rng.normal(size=(n_q,512)).astype(np.float32))
    g=normalize_np(rng.normal(size=(n_g,512)).astype(np.float32))
    score=q@g.T
    for chunk in (7,16):
        s,info=batched_sinkhorn(score,chunk)
        assert s.shape==score.shape and np.isfinite(s).all() and info['transductive']
    s,info=balanced_sinkhorn_scores(score)
    assert s.shape==score.shape and np.isfinite(s).all() and info['transductive']
    for method in ('logmeanexp','zscore'):
        assert np.isfinite(column_calibration(score,method)).all()
    # 构造一组已知文本近邻，覆盖双向匹配行为
    q[:3]=q[0]
    ids,values,masks=query_mutual_neighbors(q,torch.device('cpu'))
    assert np.any(masks) and query_cooperation(score,ids,values,masks).shape==score.shape
    for i in range(len(q)):
        for j,neighbor in enumerate(ids[i]):
            if masks[i,j]:
                assert i in ids[neighbor] and values[i,j]>=.80
    for mode in ('mean','min','max'):
        assert clause_scores(score,score*.8,score*.5,mode).shape==score.shape
    color,covered=query_color_targets(['He wears a red shirt and blue jeans.',
                                      'She wears black shoes.','white jacket'] )
    assert covered==2 and color[0,0,COLOR_NAMES.index('red')]==1
    assert color[0,1,COLOR_NAMES.index('blue')]==1
    hsv=np.zeros((10,20,3),np.uint8)
    idx=colors_from_hsv(hsv)
    assert idx.shape==(10,20)
    # 保证图邻接存在
    g[:3]=g[0]
    idx,w,graph_info=visual_neighbor_graph(g,torch.device('cpu'),k=5,min_cos=.5)
    assert graph_info['visual_edges']>=1 and np.isfinite(w).all()
    for i in range(len(g)):
        for j,neighbor in enumerate(idx[i]):
            if w[i,j]>0:
                assert i in idx[neighbor] and float(g[i]@g[neighbor])>=.5-1e-5
    assert graph_step(score,idx,w).shape==score.shape
    for lam in (.2,.5):
        reranked=mmr_rerank(score,g,torch.device('cpu'),lam,k=8)
        assert reranked.shape==score.shape and np.isfinite(reranked).all()
        for a,b in zip(score,reranked):
            assert set(np.argsort(-a)[:8])==set(np.argsort(-b)[:8])
    with tempfile.TemporaryDirectory(prefix='irra_round6_') as td:
        root=Path(td)
        cache=root/'c'
        mm,_,meta=load_cache(cache,'gallery_flip',5,'sig')
        mm[:3]=1.0; update_cache(mm,meta,'sig|gallery_flip',5,3);close_memmap(mm)
        mm,done,_=load_cache(cache,'gallery_flip',5,'sig')
        assert done==3 and mm[1,3]==1
        close_memmap(mm)
        from PIL import Image
        pics=[]
        for i in range(n_g):
            p=root/f'test_{i}.png'
            im=np.zeros((60,30,3),dtype=np.uint8)
            im[:30,:,:]=[255,0,0];im[30:,:,:]=[0,0,255]
            Image.fromarray(im).save(p)
            pics.append(str(p))
        colors=compute_region_colors(pics,root/'colors.npy','color-test')
        colors2=compute_region_colors(pics,root/'colors.npy','color-test')
        assert np.allclose(colors,colors2)
        fake_tp=np.asarray([i%7 for i in range(n_q)],dtype=np.int64)
        fake_vp=np.asarray([i%7 for i in range(n_g)],dtype=np.int64)
        val=dict(captions=['He wears a red shirt and blue jeans. Walking.']*n_q,
                 caption_pids=fake_tp.tolist(),image_pids=fake_vp.tolist(),img_paths=pics)
        cfg=dict(dataset_name='CUHK-PEDES',img_size=(384,128),text_length=77,
                 root_dir=str(root))
        def fake_extract(_,config,records,folder,fp,key,device,kind,variant='original'):
            if key=='B0_gallery':return g.copy()
            if key=='B0_queries':return q.copy()
            seeded=np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8],16))
            d=len(records['img_paths'] if kind=='image' else records['captions'])
            return normalize_np(seeded.normal(size=(d,512)).astype(np.float32))
        replacements=dict(validate_files=lambda:cfg,
                load_dataset=lambda _: (cfg,val,fake_tp,fake_vp),
                verify_baseline=lambda *_:{},extract=fake_extract)
        with patch.dict(globals(),replacements),patch.dict(globals(),{
                'OUTPUT_ROOT':root/'out'}):
            run_real(device_name='cpu')
            dirs=list((root/'out').glob(PREFIX+'*'))
            assert len(dirs)==1
            out=dirs[0]
            meta=json.loads((out/'运行元数据.json').read_text(encoding='utf-8'))
            completed=read_results(out,meta['fingerprint'])
            assert len(completed)==20,(len(completed),sorted(completed))
            assert (out/COMPLETED).is_file() and (out/ZIP_NAME).is_file()
            with zipfile.ZipFile(out/ZIP_NAME) as z:
                assert z.testzip() is None
                assert len([x for x in z.namelist() if x.startswith('结果/')])==21
            digest=sha256(out/'01_全部实验指标总表.csv')
            run_real(resume=str(out),device_name='cpu')
            assert digest==sha256(out/'01_全部实验指标总表.csv')
            # 缺结果重跑
            name='R_列方差标准化'
            for path in (out/'结果'/(name+'.json'),out/'逐查询'/(name+'.csv'),
                         out/'统计'/(name+'_PID_bootstrap.csv')):path.unlink()
            run_real(resume=str(out),device_name='cpu')
            assert (out/'结果'/(name+'.json')).is_file()
            # 续跑设置防篡改
            mp=out/'运行元数据.json'
            state=json.loads(mp.read_text(encoding='utf-8'))
            state['selection']=['O']
            atomic_json(mp,state)
            try:
                run_real(resume=str(out),device_name='cpu')
                raise AssertionError('没有拒绝篡改续跑设置')
            except RuntimeError as e:
                assert '续跑' in str(e)
    say('[SelfTest PASS] 第六轮20变体、真实颜色缓存、全流程、ZIP、恢复、防篡改通过')


def main():
    parser=argparse.ArgumentParser(description='IRRA第六轮独立单文件，7大新方向20变体')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--experiments',default=''.join(EXPERIMENTS),
                        help='选择ORTLASD中的分支子集，例如ART（颜色+统计校准+文本协作）')
    parser.add_argument('--resume',default='',help='已有第六轮实验子目录')
    parser.add_argument('--fresh',action='store_true',help='强制新建目录，不自动续跑')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    opt=parser.parse_args()
    if opt.self_test:self_test();return
    raw=opt.experiments.upper()
    if not raw or len(set(raw))!=len(raw) or set(raw)-set(EXPERIMENTS):
        raise ValueError('实验选择只能是不重复的ORTLASD字母')
    if opt.resume and opt.fresh:raise ValueError('--resume与--fresh不能同时使用')
    selection=tuple(k for k in EXPERIMENTS if k in raw)
    run_real(opt.resume,selection,opt.device,opt.fresh)


if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc()
        raise
