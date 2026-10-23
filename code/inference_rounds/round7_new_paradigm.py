# -*- coding: utf-8 -*-
"""IRRA 第七轮：7条新的大跨度实验路线，冻结Epoch58权重。
单文件；支持Windows缓存续跑、完整Val验证、PID配对Bootstrap与ZIP。
Y/O使用整批无标签Validation查询，属于传导式；M只使用Train文本，其他为独立查询、视觉变换或诊断。
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
VERSION = 'wide_exploration_round7_v1_20261022'
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
EXPERIMENTS = ('Y', 'C', 'K', 'U', 'V', 'M', 'O')
ZIP_NAME = '发送给ChatGPT_第七轮跨域发散探索.zip'


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
    if mode in ('center_zoom', 'upper_zoom', 'lower_zoom'):
        height, width = x.shape[-2:]
        if mode == 'center_zoom':
            top, bottom = int(height * .08), int(height * .92)
            left, right = int(width * .08), int(width * .92)
        elif mode == 'upper_zoom':
            top, bottom = 0, int(height * .82)
            left, right = 0, width
        else:
            top, bottom = int(height * .18), height
            left, right = 0, width
        return F.interpolate(x[..., top:bottom, left:right], size=(height,width),
                             mode='bilinear', align_corners=False)
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





# =========================== 第七轮：全新实验机制 ===========================
ROUND_SEED = 20261022
PREFIX = '第七轮_新范式探索_'
COMPLETED = '02_第七轮完成摘要.json'
VARIANTS = {
    'Y': ['Y_互惠排名05','Y_互惠排名15'],
    'C': ['C_Top12公共模式25','C_Top40公共模式40'],
    'K': ['K_符号幂次05','K_符号幂次15'],
    'U': ['U_随机扰动微弱','U_随机扰动较强'],
    'V': ['V_中心放大单视图','V_上下缩放多视图'],
    'M': ['M_Train语义桥16','M_Train语义桥48'],
    'O': ['O_全查询Sinkhorn复核','O_非均匀图库容量','O_部分列平衡'],
}


def settings():
    return dict(version=VERSION, checkpoint_sha=CHECKPOINT_SHA, config_sha=CONFIG_SHA,
                families=list(EXPERIMENTS), seed=ROUND_SEED,
                image_batch=IMAGE_BATCH, text_batch=TEXT_BATCH,
                score_eval_batch=SCORE_EVAL_BATCH,
                baseline_tolerance=BASELINE_TOLERANCE_PP,
                query_reciprocal_strengths=[.05,.15],
                candidate_mode=[[12,.25],[40,.4]],
                signed_powers=[.5,1.5],
                uncertainty=[[.035,.5],[.10,.5]], uncertainty_draws=8,
                train_text_refs=2048, semantic_neighbors=[16,48],
                sinkhorn_temperature=.07, sinkhorn_iterations=12,
                no_validation_pid_for_model_selection=True)


def query_reciprocal_score(s, alpha):
    """Use image-to-query reciprocal ranking. Needs entire unlabeled query set.

    Each gallery image ranks *queries*. This is transductive (not online).
    Within-column rank percentile is mapped to centered, robust rank evidence.
    """
    s=np.asarray(s,np.float32)
    n,m=s.shape
    if n<3 or m<2:raise ValueError('invalid query/gallery size')
    if not 0<alpha<1:raise ValueError('invalid mixing factor')
    out=s.copy()
    for start in range(0,m,96):
        end=min(m,start+96)
        block=s[:,start:end]
        order=np.argsort(-block,axis=0,kind='stable')
        ranks=np.empty(order.shape,dtype=np.int32)
        for column in range(order.shape[1]):
            ranks[order[:,column],column]=np.arange(n,dtype=np.int32)
        # score scale ~[.0,.2], percentile 0..1 compressed into [-.10,+.10]
        reciprocal=(.5-ranks.astype(np.float32)/float(n-1)) * .2
        out[:,start:end]+=alpha*reciprocal
    return out


def candidate_common_mode(query,gallery,score,device,k,beta):
    """Remove query-aligned direction shared by the initially highest-ranked images.

    No PID, candidate labels or trained parameters are used.
    Scores are recomputed against the *entire* gallery (no Top-K protection).
    """
    q=normalize_np(query);g=normalize_np(gallery)
    k=min(int(k),len(g))
    if k<2:raise ValueError('at least two candidates required')
    out=np.empty_like(q)
    for start in range(0,len(q),128):
        end=min(len(q),start+128)
        block=score[start:end]
        ids=np.argpartition(-block,kth=k-1,axis=1)[:,:k]
        centers=g[ids].mean(axis=1)
        centers=normalize_np(centers)
        c=q[start:end]
        coef=(c*centers).sum(axis=1,keepdims=True)
        out[start:end]=c-beta*coef*centers
    return cosine_scores(normalize_np(out),g,device)


def signed_power_features(x,p):
    if not .01<p<3:raise ValueError('power out of range')
    y=np.sign(x)*np.power(np.abs(np.asarray(x,dtype=np.float32)),p)
    return normalize_np(y)


def stochastic_consensus(query,gallery,device,noise,penalty,draws=8,seed=ROUND_SEED):
    """Risk-adjusted MC query score; returns diagnostics from feature perturbation.

    Unlike simply averaging noisy embeddings, penalizes image-dependent score std.
    No Validation labels are used to choose random trials or thresholds.
    """
    if draws<4 or noise<=0:raise ValueError('insufficient MC perturbations')
    gen=np.random.default_rng(seed)
    g=torch.from_numpy(np.ascontiguousarray(normalize_np(gallery))).to(device)
    q=normalize_np(query)
    out=np.empty((len(q),len(gallery)),np.float32)
    instability=np.empty(len(q),np.float32)
    with torch.no_grad():
        for start in range(0,len(q),64):
            end=min(len(q),start+64)
            q0=torch.from_numpy(q[start:end]).to(device)
            samples=[]
            for _ in range(draws):
                eps=gen.standard_normal((end-start,q.shape[1])).astype('float32')
                eps=torch.from_numpy(eps).to(device)
                pert=F.normalize(q0+noise*eps,dim=1)
                samples.append((pert@g.T).float())
            sims=torch.stack(samples,0)
            mean=sims.mean(dim=0)
            std=sims.std(dim=0,unbiased=False)
            out[start:end]=(mean-penalty*std).cpu().numpy()
            instability[start:end]=std.mean(dim=1).cpu().numpy()
            del sims
    return out,dict(noise=noise,draws=draws,penalty=penalty,
          mean_score_std=float(instability.mean()),max_score_std=float(instability.max()),
          not_independent_training_seeds=True)


def train_semantic_pivot(query,gallery,reference,device,k,blend,temp=.07):
    """Bridge query to gallery through training captions, not Validation queries.

    Query -> top-k Train captions -> gallery; Train identity labels are not used.
    Softmax weights on selected caption similarity; gallery match computed on text ref.
    """
    ref=normalize_np(reference)
    if len(ref)<k:raise ValueError('Too few training captions')
    qa=cosine_scores(query,ref,device)
    ra=cosine_scores(ref,gallery,device)
    out=cosine_scores(query,gallery,device)
    for start in range(0,len(query),96):
        end=min(start+96,len(query))
        ids=np.argpartition(-qa[start:end],kth=k-1,axis=1)[:,:k]
        sim=np.take_along_axis(qa[start:end],ids,axis=1)
        w=np.exp(np.clip((sim-sim.max(axis=1,keepdims=True))/temp,-50,0))
        w/=np.maximum(w.sum(axis=1,keepdims=True),1e-10)
        pivot=np.einsum('nk,nkg->ng',w,ra[ids],optimize=True)
        out[start:end]=(1-blend)*out[start:end]+blend*pivot
    return out.astype(np.float32)


def gallery_knn_capacity(gallery,device,k=12):
    """Use completely unsupervised image density to define column capacities."""
    g=normalize_np(gallery)
    gram=cosine_scores(g,g,device)
    np.fill_diagonal(gram,-np.inf)
    k=min(k,len(g)-1)
    if k<1:raise ValueError('Gallery too small')
    local=np.partition(gram,-k,axis=1)[:,-k:].mean(axis=1)
    # density higher -> *smaller* available column capacity
    centers=np.median(local)
    weight=np.exp(-2*np.clip(local-centers,-.5,.5))
    weight=weight/weight.sum()
    return weight,dict(density_min=float(local.min()),density_max=float(local.max()),
                       effective_capacity=float(1/np.square(weight).sum()),
                       no_gallery_pid_used=True)


def sinkhorn_with_capacity(scores,capacity=None,temp=.07,iters=12):
    """Column-capacity Sinkhorn with stable row-shift; still transductive."""
    s=np.asarray(scores,np.float32)
    n,m=s.shape
    if capacity is None: capacity=np.full(m,1/m,np.float64)
    capacity=np.asarray(capacity,np.float64)
    if capacity.shape!=(m,) or (capacity<=0).any():raise ValueError('Invalid capacity')
    capacity=capacity/capacity.sum()
    z=np.exp(np.maximum((s-s.max(axis=1,keepdims=True))/temp,-50)).astype(np.float64)
    r=1/n
    u=np.ones(n,np.float64); v=np.ones(m,np.float64)
    for _ in range(iters):
        u=r/np.maximum(z@v,1e-100)
        v=capacity/np.maximum(z.T@u,1e-100)
    return (s+temp*np.log(np.maximum(v,1e-100))[None,:]).astype(np.float32),dict(
             temperature=temp,iterations=iters,transductive=True,
             column_adjustment_std=float(np.std(temp*np.log(np.maximum(v,1e-100)))))


def read_train_caption_reference(cfg,limit=2048,seed=ROUND_SEED):
    """Take deterministic unlabeled subset of *Train* captions, never Val labels."""
    from datasets.cuhkpedes import CUHKPEDES
    ds=CUHKPEDES(root=str(cfg['root_dir']),verbose=False)
    train=ds.train
    caps=train.get('captions')
    if not isinstance(caps,(list,tuple)) or len(caps)<limit:
        raise RuntimeError('Train captions unavailable or fewer than requested references')
    # pid field is used only because TextDataset requires it; all fake zero labels.
    idx=np.random.default_rng(seed).choice(len(caps),size=limit,replace=False)
    values=[str(caps[int(i)]) for i in idx]
    signature=hashlib.sha256(json.dumps(values,ensure_ascii=False).encode()).hexdigest()
    return dict(captions=values,caption_pids=[0]*len(values)),signature


def bootstrap_pid(out,name,pid,b,p,repeats=350):
    ids,inverse=np.unique(np.asarray(pid),return_inverse=True)
    counts=np.bincount(inverse).astype(np.float64)
    rng=np.random.default_rng(ROUND_SEED+17)
    draws=rng.integers(0,len(ids),size=(repeats,len(ids)))
    denominator=counts[draws].sum(axis=1)
    rows=[]
    for key,access in [('R1',lambda x:x['correct'].astype(float)),
                       ('mAP',lambda x:x['ap'].astype(float)),
                       ('mINP',lambda x:x['inp'].astype(float))]:
        diff=access(p)-access(b)
        sums=np.bincount(inverse,weights=diff,minlength=len(ids))
        ci=100*sums[draws].sum(axis=1)/denominator
        rows.append(dict(name=name,metric=key,delta_pp=float(100*np.mean(diff)),
             ci_low_pp=float(np.quantile(ci,.025)),
             ci_high_pp=float(np.quantile(ci,.975)),
             clusters=len(ids),repeats=repeats))
    write_csv(out/'统计'/(name+'_PID_bootstrap.csv'),rows,list(rows[0]))


def save_variant(out,fp,name,family,score,tp,vp,bm,bd,note):
    if score.shape!=(len(tp),len(vp)) or not np.isfinite(score).all():
        raise RuntimeError('分数矩阵非法：'+name)
    r=record_variant(out,fp,name,family,score,tp,vp,bm,bd,note)
    stat=out/'统计'/(name+'_PID_bootstrap.csv')
    if not stat.is_file():
        _,per=evaluate(score,tp,vp)
        bootstrap_pid(out,name,tp,bd,per)
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
    say('[第七轮ZIP]',target)


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


def run_real(resume='',selection=EXPERIMENTS,device_name='cuda',fresh=False):
    started=time.perf_counter()
    torch.set_num_threads(min(6,os.cpu_count() or 6))
    np.random.seed(ROUND_SEED);torch.manual_seed(ROUND_SEED)
    if device_name=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，请使用irra_clean解释器')
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
            raise RuntimeError('续跑目录不属于第七轮')
        state=json.loads((out/'运行元数据.json').read_text(encoding='utf-8'))
        if state.get('fingerprint')!=key or state.get('settings')!=setup or \
          state.get('selection')!=list(selection) or state.get('version')!=VERSION:
            raise RuntimeError('续跑设置/权重/数据指纹不匹配，拒绝混写')
    else:
        OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
        out=OUTPUT_ROOT/(PREFIX+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        out.mkdir(exist_ok=False)
        atomic_json(out/'运行元数据.json',dict(version=VERSION,fingerprint=key,
            settings=setup,selection=list(selection),epoch=58,
            limitations=['Y/O依赖全批无标签Val查询，属于Transductive',
              'U随机扰动是表征稳定性诊断，不保证语义等价',
              'V空间变换可能损失人体部位语义',
              'M参考信息仅来自Train captions；不按Val PID选取',
              '已反复查看Val数据，不得作为论文独立最终测试']))
        try:shutil.copy2(Path(__file__),out/'第七轮完整脚本副本.py')
        except OSError:pass
    for folder in ('缓存','结果','逐查询','统计','分析'):
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
            old=json.loads(baseline_path.read_text(encoding='utf-8'))
            if old.get('fingerprint')!=key or old.get('metrics')!=bm:
                raise RuntimeError('基线和原有缓存不一致')
        else:atomic_json(baseline_path,dict(version=VERSION,fingerprint=key,metrics=bm,name='B0'))
        atomic_json(out/'00_原始Epoch58核验.json',dict(baseline=bm,official=official,
            expected_previous_baseline=EXPECTED,checkpoint_sha=CHECKPOINT_SHA,
            config_sha=CONFIG_SHA,num_queries=len(tp),num_gallery=len(vp),fingerprint=key))
        say('[B0]',bm)
        completed=read_results(out,key)
        def wanted(names):
            return any(n not in completed or not (out/'统计'/(n+'_PID_bootstrap.csv')).is_file()
                       for n in names)
        def save(name,family,s,note,extra=None):
            result=save_variant(out,key,name,family,s,tp,vp,bm,bd,note)
            completed[name]=result
            if extra is not None:atomic_json(out/'分析'/(name+'.json'),extra)
            return result

        if 'Y' in selection:
            for alpha,name in [(0.05,VARIANTS['Y'][0]),(.15,VARIANTS['Y'][1])]:
                if wanted([name]):
                    sc=query_reciprocal_score(s0,alpha)
                    save(name,'Y_双向跨模态互惠排名',sc,
                         f'全查询传导式，图像反向排名查询得到分位数再融合；α={alpha}; 未使用PID')
                    del sc
        if 'C' in selection:
            for (k,beta),name in zip([(12,.25),(40,.4)],VARIANTS['C']):
                if wanted([name]):
                    sc=candidate_common_mode(query,gallery,s0,device,k,beta)
                    save(name,'C_候选公共特征方向移除',sc,
                         f'每条查询从Top{k}视觉候选估计共有方向，移除β={beta}; 全图库重新评分')
                    del sc
        if 'K' in selection:
            for power,name in [(0.5,VARIANTS['K'][0]),(1.5,VARIANTS['K'][1])]:
                if wanted([name]):
                    sc=cosine_scores(signed_power_features(query,power),
                                     signed_power_features(gallery,power),device)
                    save(name,'K_非线性跨模态坐标几何',sc,
                         f'图文两端共享符号幂次变换 p={power}; 无训练，无PID，无阈值调节')
                    del sc
        if 'U' in selection:
            for noise,name in [(0.035,VARIANTS['U'][0]),(.10,VARIANTS['U'][1])]:
                if wanted([name]):
                    sc,diag=stochastic_consensus(query,gallery,device,noise,.5,8)
                    save(name,'U_随机表示扰动与置信惩罚',sc,
                         f'固定种子、8次512维随机扰动 std={noise},均值-0.5*标准差；仅诊断',diag)
                    del sc
        if 'V' in selection and wanted(VARIANTS['V']):
            center=extract(get_model,cfg,val,out/'缓存',key,'V_center_zoom',device,
                           'image','center_zoom')
            if wanted([VARIANTS['V'][0]]):
                sc=cosine_scores(query,center,device)
                save(VARIANTS['V'][0],'V_空间视角敏感性',sc,
                     '将原图中心区域放大为384x128，改变人体边界与背景，属于图像干预诊断')
                del sc
            if wanted([VARIANTS['V'][1]]):
                upper=extract(get_model,cfg,val,out/'缓存',key,'V_upper_zoom',device,
                              'image','upper_zoom')
                lower=extract(get_model,cfg,val,out/'缓存',key,'V_lower_zoom',device,
                              'image','lower_zoom')
                g=normalize_np(.4*gallery+.3*upper+.3*lower)
                sc=cosine_scores(query,g,device)
                save(VARIANTS['V'][1],'V_空间视角敏感性',sc,
                     '原图40%+上段缩放30%+下段缩放30%特征融合；非人体部位分割')
                del sc,g,upper,lower
            del center
        if 'M' in selection and wanted(VARIANTS['M']):
            train_val,sig=read_train_caption_reference(cfg,2048)
            refs=extract(get_model,cfg,train_val,out/'缓存',key+'|TrainCaptionsSHA:'+sig,
                         'M_train_text_reference2048',device,'text')
            atomic_json(out/'分析'/'M_Train语义桥参考集.json',dict(
                train_count=len(refs),train_captions_sha=sig,pids_not_used=True,
                from_validation=False,train_sampling_seed=ROUND_SEED))
            for k,alpha,name in [(16,.15,VARIANTS['M'][0]),(48,.30,VARIANTS['M'][1])]:
                if wanted([name]):
                    sc=train_semantic_pivot(query,gallery,refs,device,k,alpha)
                    save(name,'M_Train语义中介检索',sc,
                         f'查询→Train最近{k}条描述→图像的双跳证据，权重{alpha}，不使用Train身份标签')
                    del sc
            del refs
        if 'O' in selection:
            if wanted([VARIANTS['O'][0]]):
                sc,diag=sinkhorn_with_capacity(s0,None)
                save(VARIANTS['O'][0],'O_全查询OT对照',sc,
                     '全Validation查询一起参与统一容量的熵正则迭代；传导式，不用于线上单查询',diag)
                del sc
            if wanted([VARIANTS['O'][1]]):
                cap,info=gallery_knn_capacity(gallery,device)
                sc,diag=sinkhorn_with_capacity(s0,cap)
                save(VARIANTS['O'][1],'O_非均匀图库容量',sc,
                     '仅根据图库视觉近邻密度设置Sinkhorn容量，没有任何PID或查询身份监督',dict(**diag,**info))
                del sc,cap
            if wanted([VARIANTS['O'][2]]):
                sc,diag=sinkhorn_with_capacity(s0,None)
                sc=(.5*s0+.5*sc).astype(np.float32)
                save(VARIANTS['O'][2],'O_半强度全局平衡',sc,
                     '完整Validation无标签查询列校正减半；仍属传导式',diag)
                del sc

        summary=update_summary(out,key,bm)
        expected_number=sum(len(VARIANTS[c]) for c in selection)
        if len(summary)!=expected_number:
            raise RuntimeError(f'结果数{len(summary)} != 预定{expected_number}')
        (out/'00_阅读说明.txt').write_text(
            'IRRA第七轮：Y双向检索互惠；C候选公共模式；K非余弦坐标幂次；'
            'U随机扰动稳定性；V上下视图缩放；M训练文字语义桥；O非均匀列容量OT。\n'
            '冻结原始Epoch58，严格完整6158x3078 Validation五项检索指标。\n'
            'Y/O传导式，M只用Train文本，U/V是输入稳定性或诊断；'
            '禁止用Validation身份标签建邻接、拟合参数或选择变体。\n'
            '报告所有性能上升或下降。Validation已被多次使用，独立Test才有泛化意义。\n',
            encoding='utf-8')
        atomic_json(out/COMPLETED,dict(version=VERSION,fingerprint=key,baseline=bm,
            branches=list(selection),variants=len(summary),
            elapsed_seconds=round(time.perf_counter()-started,2),results=summary))
        say('[第七轮完成]',len(summary),'变体，耗时秒',round(time.perf_counter()-started,2))
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
    """Synthetic CPU end-to-end (no external weights), including resume & export."""
    from unittest.mock import patch
    rng=np.random.default_rng(ROUND_SEED)
    n_q,n_g=23,21
    q=normalize_np(rng.normal(size=(n_q,512)).astype(np.float32))
    g=normalize_np(rng.normal(size=(n_g,512)).astype(np.float32))
    s=q@g.T
    for a in (.05,.15):assert query_reciprocal_score(s,a).shape==s.shape
    for k,b in [(5,.25),(12,.4)]:
        assert candidate_common_mode(q,g,s,torch.device('cpu'),k,b).shape==s.shape
    for power in (.5,1.5):
        assert signed_power_features(q,power).shape==q.shape
    for noise in (.035,.1):
        score,info=stochastic_consensus(q,g,torch.device('cpu'),noise,.5,8)
        assert np.isfinite(score).all() and info['mean_score_std']>=0
    ref=normalize_np(rng.normal(size=(48,512)).astype(np.float32))
    for k in (8,16):assert train_semantic_pivot(q,g,ref,torch.device('cpu'),k,.15).shape==s.shape
    capacity,info=gallery_knn_capacity(g,torch.device('cpu'),k=5)
    assert np.isclose(capacity.sum(),1) and info['no_gallery_pid_used']
    a,_=sinkhorn_with_capacity(s,None)
    b,_=sinkhorn_with_capacity(s,capacity)
    assert a.shape==s.shape and b.shape==s.shape
    with tempfile.TemporaryDirectory(prefix='irra_r7_') as td:
        root=Path(td)
        arr,_,mp=load_cache(root/'cache','gallery_flip',5,'sig')
        arr[:3]=1
        update_cache(arr,mp,'sig|gallery_flip',5,3)
        close_memmap(arr)
        arr,done,_=load_cache(root/'cache','gallery_flip',5,'sig')
        assert done==3 and arr[2,100]==1
        close_memmap(arr)
        fake_tp=np.array([i%7 for i in range(n_q)],np.int64)
        fake_vp=np.array([i%7 for i in range(n_g)],np.int64)
        val=dict(captions=['He wears a black jacket with blue jeans.']*n_q,
                 caption_pids=fake_tp.tolist(),image_pids=fake_vp.tolist(),
                 img_paths=[str(root/f'{i}.png') for i in range(n_g)])
        cfg=dict(root_dir=str(root),img_size=(384,128),text_length=77)
        train=dict(captions=['Train text '+str(i) for i in range(48)],
                   caption_pids=[0]*48)
        def fake_extract(_,config,records,folder,fp,key,device,kind,variant='original'):
            if key=='B0_gallery':return g.copy()
            if key=='B0_queries':return q.copy()
            if key=='M_train_text_reference2048':return ref.copy()
            d=len(records['img_paths'] if kind=='image' else records['captions'])
            seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
            rg=np.random.default_rng(seed)
            return normalize_np(rg.normal(size=(d,512)).astype(np.float32))
        patches=dict(validate_files=lambda:cfg,
             load_dataset=lambda _:(cfg,val,fake_tp,fake_vp),
             verify_baseline=lambda *_:{},extract=fake_extract,
             read_train_caption_reference=lambda cfg,limit=2048,seed=ROUND_SEED:
                (train,'train-hash'))
        with patch.dict(globals(),patches),patch.dict(globals(),{'OUTPUT_ROOT':root/'out'}):
            run_real(device_name='cpu')
            folders=list((root/'out').glob(PREFIX+'*'))
            assert len(folders)==1,folders
            out=folders[0]
            meta=json.loads((out/'运行元数据.json').read_text('utf-8'))
            result=read_results(out,meta['fingerprint'])
            assert len(result)==15,(len(result),sorted(result))
            assert (out/COMPLETED).exists() and (out/ZIP_NAME).exists()
            with zipfile.ZipFile(out/ZIP_NAME) as z:
                assert z.testzip() is None
                assert len([f for f in z.namelist() if f.startswith('结果/')])==16
            prior=sha256(out/'01_全部实验指标总表.csv')
            run_real(str(out),device_name='cpu')
            assert prior==sha256(out/'01_全部实验指标总表.csv')
            # remove exactly one result to test resume/recalculation
            name=VARIANTS['K'][1]
            for path in (out/'结果'/(name+'.json'),out/'逐查询'/(name+'.csv'),
                         out/'统计'/(name+'_PID_bootstrap.csv')):path.unlink()
            run_real(str(out),device_name='cpu')
            assert (out/'结果'/(name+'.json')).is_file()
            state=json.loads((out/'运行元数据.json').read_text('utf-8'))
            state['selection']=['K']
            atomic_json(out/'运行元数据.json',state)
            try:
                run_real(str(out),device_name='cpu')
                raise AssertionError('Bad metadata was not rejected')
            except RuntimeError as e:
                assert '续跑' in str(e)
    say('[SelfTest PASS] 全15变体、真实文件、缓存续跑、缺项补跑、防篡改及ZIP均通过')


def main():
    parser=argparse.ArgumentParser(description='IRRA第七轮，7大方向15变体，单独冻结Epoch58')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--experiments',default=''.join(EXPERIMENTS),
         help='YCKUVMO子集，例如KVC，选择顺序不影响实验顺序')
    parser.add_argument('--resume',default='',help='已有第七轮实验子目录绝对路径')
    parser.add_argument('--fresh',action='store_true',help='强制新建实验目录')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    options=parser.parse_args()
    if options.self_test:
        self_test();return
    raw=options.experiments.upper()
    if not raw or len(set(raw))!=len(raw) or set(raw)-set(EXPERIMENTS):
        raise ValueError('实验选择只能包含不重复的YCKUVMO字母')
    if options.resume and options.fresh:
        raise ValueError('--resume和--fresh不能同时使用')
    selection=tuple(k for k in EXPERIMENTS if k in raw)
    run_real(options.resume,selection,options.device,options.fresh)


if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc()
        raise
