# -*- coding: utf-8 -*-
"""第五轮：跨模态映射、查询条件度量、子空间共识、量化、图库原型、
全查询矩阵平衡、文本提示、多层空间次序干预。
使用固定 Epoch58 IRRA 和完整 Validation，所有分支独立，不保护旧增益。
V 的空间打乱是反事实压力诊断，不等于可以部署的检索优化。
O 使用全部无标签 Validation 查询，为传导式检索，对应结果不得称为在线单查询收益。
通过 --self-test 可在无 IRRA 工程时进行合成数据自测。
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
VERSION = 'wide_exploration_round5_v1_20261022'
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
EXPERIMENTS = ('M', 'Q', 'W', 'B', 'C', 'O', 'T', 'V')
ZIP_NAME = '发送给ChatGPT_第五轮跨机制探索.zip'


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



# ───────────────────── 第五轮独立研究方向 ─────────────────────
TRAIN_PAIRS = 1024                # 从 Train 中选择 1024 个不同身份，每个身份1个图文配对
PROTOTYPES = 48
RANDOM_SEED = 20261022
PREFIX = '第五轮_跨机制极限探索_'
COMPLETED = '02_第五轮完成摘要.json'


def settings():
    return {'version': VERSION, 'checkpoint_sha': CHECKPOINT_SHA,
            'config_sha': CONFIG_SHA, 'families': list(EXPERIMENTS),
            'train_pairs': TRAIN_PAIRS, 'prototypes': PROTOTYPES,
            'seed': RANDOM_SEED, 'image_batch': IMAGE_BATCH,
            'text_batch': TEXT_BATCH, 'score_batch': SCORE_BATCH,
            'align_ridge': [1.0, 10.0], 'subspace': [8, 16],
            'balance': [0.07, 0.12], 'quantization': ['binary', 'ternary', 'int8'],
            'risk': 'no_val_pid_in_training_or_scoring'}


def train_paired_sample(dataset_root, val_pids):
    """从原始 Train 身份中随机挑选唯一 PID 各一图一文本。"""
    file = Path(dataset_root) / 'CUHK-PEDES' / 'reid_raw.json'
    with file.open(encoding='utf-8') as f:
        records = json.load(f)
    by_pid = {}
    for row in records:
        if row.get('split') == 'train':
            pid = int(row['id'])
            if pid not in by_pid:
                path = str(row['file_path']).replace('\\', '/')
                img = file.parent / 'imgs' / path
                if not img.is_file():
                    raise FileNotFoundError('Train image missing: ' + str(img))
                caps = row.get('captions') or []
                if not caps:
                    raise ValueError('Train caption missing: ' + str(pid))
                by_pid[pid] = (str(img), str(caps[0]))
    if len(by_pid) != 11003 or set(by_pid) & set(map(int, val_pids)):
        raise RuntimeError('Train PID 数量异常或与 Validation 重合')
    rng = np.random.default_rng(RANDOM_SEED)
    chosen = rng.choice(sorted(by_pid), TRAIN_PAIRS, replace=False)
    pids = [int(pid) for pid in chosen]
    images = [by_pid[pid][0] for pid in pids]
    texts = [by_pid[pid][1] for pid in pids]
    return {'image_pids': pids, 'img_paths': images,
            'caption_pids': pids, 'captions': texts}, sha256(file)


def fit_crossmodal_map(tx, im, method='ridge', lam=1.0):
    """只使用 Train 真实图文配对；绝不使用 Validation 正负 PID。"""
    if tx.shape != im.shape or tx.ndim != 2 or len(tx) < tx.shape[1]:
        raise ValueError('训练配对数少于特征维度或形状不符')
    x, y = tx.astype(np.float64), im.astype(np.float64)
    d = x.shape[1]
    if method == 'ridge':
        # 带 identity 锚点的 ridge: XW ≈ Y，W 不应无约束偏离 I。
        weight = np.linalg.solve(x.T @ x + lam * np.eye(d), x.T @ y + lam*np.eye(d))
    elif method == 'procrustes':
        u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
        weight = u @ vt
    else:
        raise ValueError(method)
    if not np.isfinite(weight).all():
        raise RuntimeError('拟合的映射存在数值异常')
    return weight.astype(np.float32)


def apply_crossmodal_map(q, w, strength):
    return normalize_np((1.0-strength) * q + strength * (q @ w))


def query_conditional_similarity(q, gallery, exponent, block=96):
    """查询依赖的非对称加权余弦：w=abs(q)^p，不是单调分数变换。"""
    q = np.asarray(q, dtype=np.float32)
    g = np.asarray(gallery, dtype=np.float32)
    g2 = g * g
    out = np.empty((len(q), len(g)), np.float32)
    for start in range(0, len(q), block):
        t = q[start:start+block]
        w = np.maximum(np.abs(t), 1e-5) ** exponent
        numer = (t*w) @ g.T
        normq = np.sum((t*t)*w, axis=1, keepdims=True)
        normg = w @ g2.T
        out[start:start+len(t)] = numer / np.sqrt(np.maximum(normq*normg, 1e-12))
    return out


def split_subspaces(q, gallery, slices=8, device=None):
    """随机维度分区后逐块独立归一化，返回均值和标准差两张评分矩阵。"""
    if q.shape[1] != gallery.shape[1] or q.shape[1] % slices:
        raise ValueError('子空间划分不整除')
    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(q.shape[1])
    mean = np.zeros((len(q), len(gallery)), np.float32)
    sq = np.zeros_like(mean)
    for ids in np.array_split(perm, slices):
        tq = normalize_np(q[:, ids]); tg = normalize_np(gallery[:, ids])
        sc = cosine_scores(tq, tg, device)
        mean += sc / slices
        sq += (sc*sc) / slices
        del sc
    std = np.sqrt(np.maximum(sq - mean*mean, 0))
    return mean, std


def quantized_pair(q, g, kind):
    if kind == 'binary':
        return normalize_np(np.where(q >= 0, 1., -1.)), normalize_np(np.where(g >= 0, 1., -1.))
    if kind == 'ternary':
        # 固定阈值 0.035，无 Validation 标签参与选择
        quant = lambda x: np.where(x > .035, 1., np.where(x < -.035, -1., 0.)).astype(np.float32)
        return normalize_np(quant(q)), normalize_np(quant(g))
    if kind == 'int8':
        def quant(x):
            scale = np.maximum(np.max(np.abs(x), axis=1, keepdims=True) / 127., 1e-8)
            return normalize_np(np.round(x/scale).clip(-127,127)*scale)
        return quant(q), quant(g)
    raise ValueError(kind)


def cluster_prototypes(g, k=PROTOTYPES, iters=8):
    """无标签 spherical k-means; 索引完全由 gallery 图像特征产生。"""
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(g)
    if k >= n: raise ValueError('k必须小于图库容量')
    anchor = rng.choice(n, k, replace=False)
    centers = np.array(g[anchor], dtype=np.float32, copy=True)
    counts = np.zeros(k, np.int32)
    for it in range(iters):
        assignment = np.argmax(g @ centers.T, axis=1)
        counts = np.bincount(assignment, minlength=k)
        centers_new = np.zeros_like(centers)
        np.add.at(centers_new, assignment, g)
        empty = np.flatnonzero(counts==0)
        if len(empty):
            centers_new[empty] = g[rng.choice(n,len(empty),replace=False)]
        centers = normalize_np(centers_new)
    assignment = np.argmax(g @ centers.T, axis=1)
    counts = np.bincount(assignment, minlength=k)
    if not np.all(counts):
        # 最后一轮偶发空簇：确定性地从有冗余的簇中迁移一个离中心最远的点。
        score = np.sum(g * centers[assignment], axis=1)
        for empty_cluster in np.flatnonzero(counts == 0):
            eligible = np.flatnonzero(counts[assignment] > 1)
            if len(eligible) == 0:
                raise RuntimeError('无法修复空簇：所有簇都只包含一个样本')
            chosen = eligible[np.argmin(score[eligible])]
            donor = int(assignment[chosen])
            assignment[chosen] = int(empty_cluster)
            counts[donor] -= 1
            counts[empty_cluster] += 1
        # 再按最终确定的分组计算原型，避免centers与assignment错位。
        adjusted = np.zeros_like(centers)
        np.add.at(adjusted, assignment, g)
        centers = normalize_np(adjusted)
    return centers, assignment.astype(np.int32), counts


def cluster_augmented(original, q, g, center, membership, alpha=.15, policy='centroid'):
    """粗粒度原型重排，不做 kNN 邻居传播。"""
    if policy == 'centroid':
        coarse = q @ center.T
        addon = coarse[:, membership]
    elif policy == 'within_centered':
        # 对同一原型内的局部匹配分数做零均值化，挑战绝对分数比较机制
        sizes = np.bincount(membership, minlength=len(center)).astype(np.float32)
        pooled = np.zeros((len(q), len(center)), np.float32)
        for cid in range(len(center)):
            pooled[:,cid] = original[:, membership==cid].mean(axis=1)
        addon = original - pooled[:, membership]
    else:
        raise ValueError(policy)
    return ((1-alpha)*original + alpha*addon).astype(np.float32)


def balanced_sinkhorn_scores(scores, temp=.07, iters=12):
    """双边熵正则矩阵缩放。传导式：使用整批Val无标签查询。"""
    s = np.asarray(scores, dtype=np.float32)
    if s.ndim!=2 or temp <= 0: raise ValueError('scores或温度无效')
    # 避免 exp 溢出，逐行减去最大得分不改变同一查询排名。
    z = ((s-s.max(axis=1,keepdims=True)) / temp).astype(np.float32)
    k = np.exp(np.maximum(z,-50.)).astype(np.float32)
    row_target = 1.0 / len(s)
    col_target = 1.0 / s.shape[1]
    a = np.ones(len(s),np.float64)
    b = np.ones(s.shape[1],np.float64)
    for _ in range(iters):
        a = row_target / np.maximum(k @ b, 1e-100)
        b = col_target / np.maximum(k.T @ a, 1e-100)
    # 概率行归一化不改变该行内排行；log b 才是影响排名的列耦合校正。
    r = s + temp * np.log(np.maximum(b, 1e-100)).astype(np.float32)[None,:]
    return np.asarray(r, dtype=np.float32), {'columns_bias_std':float(np.std(temp*np.log(b))),
               'iterations': iters, 'batch_queries':len(s), 'transductive': True}


# 只添加保留原语义的轻量文本提示；与第四轮删换颜色不同。
def transform_texts(texts, mode):
    if mode == 'original': return list(texts)
    if mode == 'prompt_photo': return ['A photo of '+str(s).strip() for s in texts]
    if mode == 'prompt_person': return ['A person: '+str(s).strip() for s in texts]
    if mode == 'prompt_description': return ['The pedestrian is described as '+str(s).strip() for s in texts]
    raise ValueError('未知提示表达模式：'+mode)


_BASE_IMAGE_TRANSFORM=transform_images

def transform_images(x, mode):
    if mode in ('original','flip','gray','edge_mask','center_mask'):
        return _BASE_IMAGE_TRANSFORM(x,mode)
    # 保持位置尺寸不变，打乱水平带的垂直顺序，不提供 PID 监督
    if mode == 'swap_top_bottom':
        h=x.shape[-2]; k=h//2
        return torch.cat([x[...,k:,:],x[...,:k,:]],dim=-2)
    if mode == 'reverse_thirds':
        slices=torch.tensor_split(x,3,dim=-2)
        return torch.cat(list(reversed(slices)),dim=-2)
    raise ValueError('未知图像结构变换：'+mode)


def bootstrap_pid(out, name, pid, b, p, nrep=350):
    """每个身份作为聚类，配对R1/mAP/mINP置信区间；仅探索性。"""
    groups, inverse = np.unique(np.asarray(pid), return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    rng = np.random.default_rng(RANDOM_SEED+13)
    rows = []
    for key, f in [('R1',lambda x:x['correct'].astype(np.float64)),
                   ('mAP',lambda x:x['ap'].astype(np.float64)),
                   ('mINP',lambda x:x['inp'].astype(np.float64))]:
        diff=f(p)-f(b)
        sums=np.bincount(inverse,weights=diff,minlength=len(groups))
        empirical=100*diff.mean()
        boots=[]
        for _ in range(nrep):
            draw=rng.integers(0,len(groups),len(groups))
            boots.append(100*sums[draw].sum()/count[draw].sum())
        rows.append(dict(method=name,metric=key,delta_pp=float(empirical),
                   ci_low=float(np.percentile(boots,2.5)),
                   ci_high=float(np.percentile(boots,97.5)),pid_clusters=len(groups),
                   n_bootstrap=nrep))
    write_csv(out/'统计'/f'{name}_PID_bootstrap.csv',rows,list(rows[0]))
    return rows


def save_variant(out, fp, name, family, score, tp, vp, baseline_metrics,
                 baseline_detail, note, diagnostic=False):
    if score.shape != (len(tp),len(vp)) or not np.all(np.isfinite(score)):
        raise RuntimeError(f'{name}分数矩阵形状/数值异常')
    info = record_variant(out, fp, name, family, score, tp, vp,
                          baseline_metrics, baseline_detail, note)
    stat_path = out/'统计'/f'{name}_PID_bootstrap.csv'
    if not stat_path.exists():
        _, d = evaluate(score,tp,vp)
        bootstrap_pid(out,name,tp,baseline_detail,d)
    if diagnostic:
        path=out/'诊断'/f'{name}_注意事项.txt'
        path.write_text('该实验人为改变了原始查询或图像的观测条件，'
            '对应指标仅是扰动后的敏感性测量，不可称为检索精度优化。\n',encoding='utf-8')
    return info


def settings_hash(fp, setup):
    return hashlib.sha256((fp+'|'+json.dumps(setup,sort_keys=True,ensure_ascii=False)).encode()).hexdigest()


def find_unfinished(fp, setup, selection):
    if not OUTPUT_ROOT.is_dir():return ''
    matches=[]
    for out in OUTPUT_ROOT.glob(PREFIX+'*'):
        if (out/COMPLETED).exists():continue
        p=out/'运行元数据.json'
        if p.is_file():
            try:d=json.loads(p.read_text(encoding='utf-8'))
            except (ValueError,OSError):continue
            if d.get('fingerprint')==fp and d.get('settings')==setup and \
                    d.get('selection')==list(selection) and d.get('version')==VERSION:
                matches.append(out)
    if matches:
        selected=sorted(matches,key=lambda z:z.name)[-1]
        say('[恢复未完成第五轮]',selected)
        return str(selected)
    return ''


def make_zip(out):
    target=out/ZIP_NAME
    tmp=out/('.'+ZIP_NAME+'.partial')
    try:
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(out.rglob('*')):
                if not p.is_file() or p in (target,tmp):continue
                if '缓存' in p.relative_to(out).parts: continue
                z.write(p,p.relative_to(out).as_posix())
        with zipfile.ZipFile(tmp) as z:
            if (bad:=z.testzip()) is not None:raise RuntimeError('ZIP CRC error:'+str(bad))
        windows_retry_replace(tmp,target)
    finally:
        if tmp.exists():tmp.unlink()
    say('[第五轮ZIP]',target)


def run_real(resume='', selection=EXPERIMENTS, device_name='cuda', fresh=False):
    begin=time.perf_counter()
    torch.set_num_threads(min(6,os.cpu_count() or 6))
    np.random.seed(RANDOM_SEED);torch.manual_seed(RANDOM_SEED)
    if device_name=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('当前 CUDA 不可用；请检查 PyCharm 中 irra_clean 解释器')
    dev=torch.device(device_name)
    cfg=validate_files()
    cfg,val,tp,vp=load_dataset(cfg)
    st=settings()
    fp=settings_hash(fingerprint(cfg,val),st)
    if not resume and not fresh:
        resume=find_unfinished(fp,st,selection)
    if resume:
        out=Path(resume).resolve()
        if not out.is_dir() or out.parent.resolve()!=OUTPUT_ROOT.resolve() or not out.name.startswith(PREFIX):
            raise RuntimeError('指定的续跑目录不是多种实验方向探索下第五轮子目录')
        mp=out/'运行元数据.json'
        if not mp.is_file():raise RuntimeError('续跑元数据缺失')
        prev=json.loads(mp.read_text(encoding='utf-8'))
        if (prev.get('fingerprint')!=fp or prev.get('selection')!=list(selection)
                or prev.get('settings')!=st or prev.get('version')!=VERSION):
            raise RuntimeError('续跑目录的模型、数据、选择或参数不一致，禁止混写')
    else:
        OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
        out=OUTPUT_ROOT/(PREFIX+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        out.mkdir(exist_ok=False)
        atomic_json(out/'运行元数据.json',{
            'version':VERSION,'fingerprint':fp,'settings':st,
            'selection':list(selection),'epoch':58,
            'warning':['第五轮有些方法可能严重退化，无筛选删除结果',
                'O方向使用全部Val无标签查询，属于Transductive',
                'V方向改变人体空间结构，仅为压力诊断，不是优化',
                'M使用Train图文配对，与Validation身份无交叉',
                'Validation已反复使用，只能用于科研探索，不能用它证明最终泛化']})
        try:shutil.copy2(Path(__file__),out/'第五轮脚本_副本.py')
        except OSError:pass
    for part in ['结果','逐查询','缓存','统计','分析','诊断']:
        (out/part).mkdir(parents=True,exist_ok=True)
    say('[第五轮实验目录]',out)
    model=None
    def model_fn():
        nonlocal model
        if model is None:model=load_model(cfg,dev)
        return model
    try:
        gallery=extract(model_fn,cfg,val,out/'缓存',fp,'B0_gallery',dev,'image')
        query=extract(model_fn,cfg,val,out/'缓存',fp,'B0_queries',dev,'text')
        score0=cosine_scores(query,gallery,dev)
        bm, bd=evaluate(score0,tp,vp)
        official=verify_baseline(score0,tp,vp,bm)
        path0=out/'结果'/'00_baseline.json'
        if path0.is_file():
            old=json.loads(path0.read_text(encoding='utf-8'))
            if old.get('fingerprint')!=fp or old.get('metrics')!=bm:
                raise RuntimeError('基线在续跑前后发生变化，拒绝继续')
        else:
            atomic_json(path0,dict(version=VERSION,fingerprint=fp,name='B0_original',metrics=bm))
        atomic_json(out/'00_原始Epoch58核验.json',dict(baseline=bm,
             official=official,historical=EXPECTED,queries=len(tp),gallery=len(vp),
             checkpoint_sha=CHECKPOINT_SHA,config_sha=CONFIG_SHA,fingerprint=fp))
        say('[B0完整Val]',bm)
        completed=read_results(out,fp)
        def wanted(names):
            return any(name not in completed or
                   not (out/'统计'/(name+'_PID_bootstrap.csv')).is_file() for name in names)
        def save(name,family,scores,note,diagnostic=False):
            result=save_variant(out,fp,name,family,scores,tp,vp,bm,bd,note,diagnostic)
            completed[name]=result
            return result

        if 'M' in selection:
            names=['M_Ridge1_25','M_Ridge10_50','M_Procrustes_20','M_均值偏移_30']
            if wanted(names):
                sample,source_sha=train_paired_sample(cfg['root_dir'],tp)
                afp=hashlib.sha256((fp+'|'+source_sha+'|'+json.dumps(
                     list(zip(sample['image_pids'],sample['img_paths'],sample['captions'])),
                     ensure_ascii=False)).encode()).hexdigest()
                ti=extract(model_fn,cfg,sample,out/'缓存',afp,'train_one_img_per_id',dev,'image')
                tt=extract(model_fn,cfg,sample,out/'缓存',afp,'train_one_text_per_id',dev,'text')
                if not np.array_equal(sample['image_pids'],sample['caption_pids']):
                    raise RuntimeError('Train图文PID配对错位')
                info={'train_pairs':len(ti),'train_identities':len(set(sample['image_pids'])),
                      'source_annotation_sha256':source_sha, 'val_pid_used':False,
                      'mean_train_positive_cosine':float(np.mean((ti*tt).sum(axis=1)))}
                atomic_json(out/'分析'/'M_Train配对映射.json',info)
                if wanted(names[:2]):
                    for lam,mix,name in [(1.,.25,names[0]),(10.,.50,names[1])]:
                        if wanted([name]):
                            w=fit_crossmodal_map(tt,ti,'ridge',lam)
                            save(name,'M_Train监督跨模态映射',
                              cosine_scores(apply_crossmodal_map(query,w,mix),gallery,dev),
                              f'1024个Train身份各1个真图文配对，ridge={lam}, 凸融合={mix}')
                if wanted([names[2]]):
                    w=fit_crossmodal_map(tt,ti,'procrustes')
                    save(names[2],'M_Train监督跨模态映射',
                         cosine_scores(apply_crossmodal_map(query,w,.20),gallery,dev),
                         'Train监督正交Procrustes映射，20%映射与80%原表示混合')
                if wanted([names[3]]):
                    delta=ti.mean(axis=0)-tt.mean(axis=0)
                    save(names[3],'M_Train分布均值偏移',
                         cosine_scores(normalize_np(query+.30*delta[None,:]),gallery,dev),
                         '1024 Train身份的图文配对均值差，向图像侧偏移30%')
                del ti,tt,sample

        if 'Q' in selection:
            for power,name in [(0.5,'Q_查询条件半次方'),(1.,'Q_查询条件一次方'),
                               (2.,'Q_查询条件平方')]:
                if wanted([name]):
                    s=query_conditional_similarity(query,gallery,power)
                    save(name,'Q_查询依赖度量',s,
                         f'对不同文本使用不同特征维度权重 abs(query)^{power}; 无监督')
                    del s

        if 'W' in selection:
            for parts in (8,16):
                name0=f'W_{parts}块独立余弦均值'
                name1=f'W_{parts}块一致性惩罚'
                if wanted([name0,name1]):
                    m,std=split_subspaces(query,gallery,parts,dev)
                    atomic_json(out/'分析'/f'W_{parts}块统计.json',dict(
                        blocks=parts,mean_std=float(std.mean()),
                        max_std=float(std.max()),seed=RANDOM_SEED))
                    if wanted([name0]):
                        save(name0,'W_随机子空间证据共识',m,
                            f'512维随机划分为{parts}块，每块独立余弦，再平均')
                    if wanted([name1]):
                        save(name1,'W_随机子空间证据共识',m-.15*std,
                            f'{parts}块分数均值减0.15倍跨子空间标准差')
                    del m,std

        if 'B' in selection:
            for mode,name in [('binary','B_符号二值汉明'),('ternary','B_三值编码'),
                              ('int8','B_Int8逐向量量化')]:
                if wanted([name]):
                    t,g=quantized_pair(query,gallery,mode)
                    save(name,'B_极低精度特征表示',cosine_scores(t,g,dev),
                         f'图文各自量化{mode}; 不修改模型权重，不使用标签')
                    del t,g
            if wanted(['B_原始加二值25融合']):
                t,g=quantized_pair(query,gallery,'binary')
                s=cosine_scores(t,g,dev)
                save('B_原始加二值25融合','B_连续与离散混合',
                     .75*score0+.25*s,'保留75%连续余弦与25%符号二值余弦')
                del t,g,s

        if 'C' in selection:
            names=['C_粗原型增强15','C_粗原型增强40','C_簇内中心化20']
            if wanted(names):
                cent,labels,sizes=cluster_prototypes(gallery,PROTOTYPES)
                atomic_json(out/'分析'/'C_图库原型分布.json',{
                    'num_clusters':len(cent),'sizes':sizes.tolist(),
                    'mean_size':float(sizes.mean()),'min_size':int(sizes.min()),
                    'max_size':int(sizes.max()),'pid_used':False})
                if wanted([names[0]]):
                    save(names[0],'C_全局图库聚类原型',
                         cluster_augmented(score0,query,gallery,cent,labels,.15,'centroid'),
                         '图库图像视觉无监督球面KMeans48簇; 原型分数占15%')
                if wanted([names[1]]):
                    save(names[1],'C_全局图库聚类原型',
                         cluster_augmented(score0,query,gallery,cent,labels,.40,'centroid'),
                         '图库图像视觉无监督球面KMeans48簇; 原型分数占40%')
                if wanted([names[2]]):
                    save(names[2],'C_簇内局部对比',
                         cluster_augmented(score0,query,gallery,cent,labels,.20,'within_centered'),
                         '每张图库图像的匹配分数减去其簇内均值,融合20%')

        if 'O' in selection:
            for temp,name in [(.07,'O_Sinkhorn低温'),(.12,'O_Sinkhorn高温')]:
                if wanted([name]):
                    s,diag=balanced_sinkhorn_scores(score0,temp,12)
                    atomic_json(out/'分析'/(name+'.json'),diag)
                    save(name,'O_整批查询-图库双边平衡',s,
                         'Transductive: 全6158条无标签Validation查询共同参与列边际平衡; '
                         f'温度{temp}; 不能作为独立在线单查询结果')
                    del s

        if 'T' in selection:
            for mode,name in [('prompt_photo','T_照片前缀'),
                              ('prompt_person','T_人物前缀'),
                              ('prompt_description','T_描述前缀')]:
                if wanted([name,'T_原文加照片提示半融合']):
                    ptxt=extract(model_fn,cfg,val,out/'缓存',fp,
                                 'query_'+mode,dev,'text',mode)
                    score=cosine_scores(ptxt,gallery,dev)
                    if wanted([name]):
                        save(name,'T_语义保留提示工程',score,
                             f'仅在文本前加 {mode} 固定模板，原标签身份不变；注意77 Token截断')
                    if mode=='prompt_photo' and wanted(['T_原文加照片提示半融合']):
                        save('T_原文加照片提示半融合','T_原文提示双编码',
                             cosine_scores(normalize_np(.5*query+.5*ptxt),gallery,dev),
                             '同一原始描述和照片提示描述的文本特征各占50%')
                    del ptxt,score

        if 'V' in selection:
            for mode,name in [('swap_top_bottom','V_上下半身互换'),
                              ('reverse_thirds','V_三段垂直倒序')]:
                if wanted([name]):
                    gi=extract(model_fn,cfg,val,out/'缓存',fp,
                               'gallery_'+mode,dev,'image',mode)
                    sc=cosine_scores(query,gi,dev)
                    save(name,'V_空间布局反事实_仅诊断',sc,
                         '保持每张图像像素和总尺寸，但打乱人体空间次序；'
                         '此时图像不再代表原始观察，准确率只能用于敏感性分析',True)
                    atomic_json(out/'分析'/(name+'.json'),{
                        'mean_original_transformed_cosine':float((gallery*gi).sum(axis=1).mean()),
                        'mean_score_abs_delta':float(np.abs(sc-score0).mean()),
                        'diagnostic_only':True})
                    del gi,sc

        full=update_summary(out,fp,bm)
        total=len([d for d in full])
        explanation=(
            '第五轮：放弃保护已有方法；M训练配对映射，Q查询加权余弦，W随机子空间共识，'
            'B量化与二值检索，C图库粗聚类原型，O查询-图库双边平衡，'
            'T保留语义的文本提示，V打乱空间顺序诊断。\n'
            '原始Epoch58图文编码器始终冻结，M只拟合512x512的小矩阵，真实监督仅来自Train。\n'
            'O整个Validation无标签查询集参与优化，是Transductive而不是单查询Online。\n'
            'V图像空间结构发生反事实变化，不是同条件性能改进。\n'
            '结果为固定参数的探索性Validation，最终论文必须预注册并使用独立Test。\n'
            '数据缓存保留本目录，ZIP不包含缓存，但包含实验脚本、副本及逐查询结果。\n'
        )
        (out/'00_阅读说明.txt').write_text(explanation,encoding='utf-8')
        atomic_json(out/COMPLETED,dict(version=VERSION,fingerprint=fp,
                     baseline=bm,experiment_families=list(selection),
                     variants=total,elapsed_seconds=round(time.perf_counter()-begin,3),
                     results=full))
        say('[第五轮完成]',total,'个结果，耗时',round(time.perf_counter()-begin,1),'秒')
    except BaseException as error:
        (out/'错误_异常.txt').write_text(''.join(traceback.format_exception(
            type(error),error,error.__traceback__)),encoding='utf-8')
        say('[异常记录]',out/'错误_异常.txt')
        raise
    finally:
        if model is not None:
            del model
            if dev.type=='cuda':torch.cuda.empty_cache()
        try: make_zip(out)
        except BaseException as error:say('[ZIP导出失败]',repr(error))
        say('[第五轮目录]',out)


def self_test():
    """CPU 合成数据端到端：全分支、指标、缓存、续跑、ZIP、参数防篡改。"""
    say('[SelfTest] 第五轮开始')
    from unittest.mock import patch
    rng=np.random.default_rng(RANDOM_SEED)
    q=normalize_np(rng.normal(size=(28,512)).astype('float32'))
    g=normalize_np(rng.normal(size=(21,512)).astype('float32'))
    # 独立数学函数回归检查
    w=np.abs(rng.normal(size=(42,24))).astype('float32')
    y=normalize_np(rng.normal(size=(42,24)).astype('float32'))
    x=normalize_np(w)
    for name in ('ridge','procrustes'):
        transform=fit_crossmodal_map(x,y,name,2.0)
        assert transform.shape==(24,24) and np.isfinite(transform).all()
    for p in (.5,1.,2.):
        t=query_conditional_similarity(q[:4],g,p)
        assert t.shape==(4,21) and np.isfinite(t).all()
    mean,std=split_subspaces(q,g,8,torch.device('cpu'))
    assert mean.shape==(28,21) and np.isfinite(std).all() and (std>=0).all()
    for kind in ('binary','ternary','int8'):
        a,b=quantized_pair(q,g,kind)
        assert a.shape==q.shape and b.shape==g.shape and np.isfinite(a).all()
    cent,membership,count=cluster_prototypes(g,6)
    for strategy in ('centroid','within_centered'):
        t=cluster_augmented(q @ g.T,q,g,cent,membership,.2,strategy)
        assert t.shape==(28,21) and np.isfinite(t).all()
    a,d=balanced_sinkhorn_scores(q @ g.T,.07,8)
    assert a.shape==(28,21) and np.isfinite(a).all() and d['transductive']
    img=torch.rand(2,3,384,128)
    for mode in ('original','swap_top_bottom','reverse_thirds'):
        assert transform_images(img,mode).shape==img.shape
    assert transform_texts(['red jacket'],'prompt_photo')[0]=='A photo of red jacket'
    with tempfile.TemporaryDirectory(prefix='irra_round5_check_') as td:
        tmp=Path(td)
        # 真正缓存文件模拟：确保增量快照恢复且指纹错误时拒绝。
        cache=tmp/'cache'
        mm,done,mp=load_cache(cache,'gallery_flip',4,'x')
        mm[:2]=1.0
        update_cache(mm,mp,'x|gallery_flip',4,2)
        close_memmap(mm)
        mm2,done,mp2=load_cache(cache,'gallery_flip',4,'x')
        assert done==2 and float(mm2[0,0])==1.0
        close_memmap(mm2)
        try:
            mm_bad,_,_=load_cache(cache,'gallery_flip',4,'wrong')
            close_memmap(mm_bad)
            raise AssertionError('缓存指纹不符必须拒绝')
        except RuntimeError:pass
        # 模拟原始 IRRA 特征提取函数，运行 8 类算法的完整主流程。
        fake_tp=np.asarray([i%7 for i in range(len(q))],dtype=np.int64)
        fake_vp=np.asarray([i%7 for i in range(len(g))],dtype=np.int64)
        val=dict(captions=['The person wears a blue coat.']*len(q),
                 caption_pids=fake_tp.tolist(),image_pids=fake_vp.tolist(),
                 img_paths=[f'/fake/{i}.jpg' for i in range(len(g))])
        dummy_config=dict(root_dir=str(tmp),dataset_name='CUHK-PEDES',
                          img_size=(384,128),text_length=77)
        extract_map={'B0_gallery':g,'B0_queries':q}
        def fake_extract(_,cfg,data,folder,fp,key,device,kind,variant='original'):
            if key in extract_map: return extract_map[key].copy()
            seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
            gen=np.random.default_rng(seed)
            n=len(data['img_paths'] if kind=='image' else data['captions'])
            vector=gen.normal(size=(n,512)).astype(np.float32)
            if kind=='image' and n==len(g):vector=.9*g+.1*vector
            if kind=='text' and n==len(q):vector=.9*q+.1*vector
            return normalize_np(vector)
        def fake_train(*_):
            pids=list(range(10000,10000+TRAIN_PAIRS))
            return {'image_pids':pids,'caption_pids':pids,
                    'img_paths':[f'/fake/train/{i}.jpg' for i in pids],
                    'captions':['pedestrian']*len(pids)},'fake_train_sha'
        # 合成 512x512 的真实SVD会增加自测耗时，映射函数另用24维真算过。
        def fake_map(texts,imgs,method='ridge',lam=1):
            return np.eye(texts.shape[1],dtype=np.float32)
        funcs={'validate_files':lambda:dummy_config,
               'load_dataset':lambda cfg:(cfg,val,fake_tp,fake_vp),
               'extract':fake_extract,'verify_baseline':lambda *_:{},
               'train_paired_sample':fake_train,
               'fit_crossmodal_map':fake_map}
        with patch.dict(globals(),funcs), patch.dict(globals(),{'OUTPUT_ROOT':tmp/'out','PROTOTYPES':8}):
            run_real(device_name='cpu')
            dirs=list((tmp/'out').glob(PREFIX+'*'))
            assert len(dirs)==1
            out=dirs[0]
            variants=read_results(out,json.loads((out/'运行元数据.json').read_text())['fingerprint'])
            assert len(variants)==26,(len(variants),list(variants))
            assert (out/COMPLETED).is_file() and (out/ZIP_NAME).is_file()
            with zipfile.ZipFile(out/ZIP_NAME) as z:
                assert z.testzip() is None
                assert len([n for n in z.namelist() if n.startswith('结果/')])==27
            # 完整续跑无重算、不改变指标文件
            old_sha=sha256(out/'01_全部实验指标总表.csv')
            run_real(resume=str(out),device_name='cpu')
            assert sha256(out/'01_全部实验指标总表.csv')==old_sha
            # 删除一个成品结果与统计CSV，再运行，仅补算缺失项。
            missing='Q_查询条件平方'
            (out/'结果'/(missing+'.json')).unlink()
            (out/'逐查询'/(missing+'.csv')).unlink()
            (out/'统计'/(missing+'_PID_bootstrap.csv')).unlink()
            run_real(resume=str(out),device_name='cpu')
            assert (out/'结果'/(missing+'.json')).is_file()
            # 用被篡改的选择元数据续跑必须失败
            meta=out/'运行元数据.json'
            obj=json.loads(meta.read_text(encoding='utf-8'))
            obj['selection']=['M']
            atomic_json(meta,obj)
            try:
                run_real(resume=str(out),device_name='cpu')
                raise AssertionError('续跑的设置变更未被阻止')
            except RuntimeError as exc:
                assert '续跑' in str(exc)
    say('[SelfTest PASS] 数学/缓存/全26变体/导出/恢复/防篡改测试通过')


def main():
    parser=argparse.ArgumentParser(description='IRRA 第五轮 8 大方向 26 变体探索')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--experiments',default=''.join(EXPERIMENTS),
                        help='从 MQWBCOTV 中选择子集，如 QWBC')
    parser.add_argument('--resume',default='',help='已有第五轮输出子目录')
    parser.add_argument('--fresh',action='store_true',help='新建目录，不自动续跑未完成任务')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    options=parser.parse_args()
    if options.self_test:
        self_test();return
    raw=options.experiments.upper()
    if not raw or len(set(raw))!=len(raw) or set(raw)-set(EXPERIMENTS):
        raise ValueError('experiments必须为MQWBCOTV中不重复的字母')
    if options.resume and options.fresh:
        raise ValueError('--resume和--fresh不可同时出现')
    selection=tuple(c for c in EXPERIMENTS if c in raw)
    run_real(options.resume,selection,options.device,options.fresh)


if __name__=='__main__':
    main()
