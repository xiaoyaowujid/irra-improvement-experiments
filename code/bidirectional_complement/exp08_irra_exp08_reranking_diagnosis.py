# -*- coding: utf-8 -*-
"""实验08：IRRA 候选重排序基础网络故障归因（不训练多文本）

独立科研诊断脚本。读取实验07已完成的冻结特征、固定TopK、Warmup模型、
AdamW状态以及困难负样本；不读取或训练任何同身份多文本教师。
先在NumPy子进程做旧分数插值回放，再于Torch子进程进行五种单因素
Warmup续训对照，最后在NumPy子进程计算完整图库指标并自动打包。
无Val PID辅助推理，Val标签只在生成所有分数后用于计算指标。
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import traceback
import tempfile
import shutil
import time
import io
import subprocess
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np

DEFAULT_ROOT = r"G:\PyCharmProjects\IRRA\IRRA-main"
DEFAULT_CONFIG = (r"G:\PyCharmProjects\IRRA\IRRA-main\logs\CUHK-PEDES"
                  r"\20261008_164451_irra_baseline_b8\configs.yaml")
DEFAULT_CKPT = (r"G:\PyCharmProjects\IRRA\IRRA-main\logs\CUHK-PEDES"
                r"\20261008_164451_irra_baseline_b8\best.pth")
PROJECT_FOLDER_NAME = '图文双向互补实验'
EXPERIMENT_NAME = '实验08_候选重排序基础网络诊断'
METHODS = ('exact_replay','lower_lr','strong_protect','fixed_gate','hard_cap')
LOG_FILE = None

def msg(s):
    print(s, flush=True)
    if LOG_FILE is not None:
        with open(LOG_FILE,'a',encoding='utf-8') as f:
            f.write(str(s) + '\n')

def resolve_dataset(root, cfgpath, dataset_root='', annotation_file='', images_root=''):
    """Resolve the real CUHK-PEDES image/annotation pair, never fabricate annotations.

    Accepts reid_raw.json (official IRRA) or caption_all.json (only if it has
    the same required split/id/path/captions schema). Supports both a dataset
    parent and the CUHK-PEDES directory itself. Checks YAML root_dir first,
    then common nearby locations; does a bounded nearby scan only as fallback.
    """
    import yaml
    with open(cfgpath, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config_root = config.get('root_dir', '') if isinstance(config, dict) else ''
    seeds = []
    def add(path):
        if path is None or not str(path).strip():
            return
        path = Path(os.path.expandvars(os.path.expanduser(str(path))))
        choices = [path] if path.is_absolute() else [root / path, cfgpath.parent / path, Path.cwd() / path]
        for c in choices:
            c = c.resolve()
            if c not in seeds:
                seeds.append(c)
    add(dataset_root)
    add(config_root)
    add(root / 'data')
    add(root.parent / 'data')
    add(root.parent.parent / 'data')
    add(root)
    add(root.parent)
    add(Path(__file__).resolve().parent / 'data')
    add(Path.cwd() / 'data')

    explicit_annotation = Path(annotation_file).expanduser().resolve() if annotation_file else None
    explicit_images = Path(images_root).expanduser().resolve() if images_root else None
    candidates = []
    def offer(folder, reason):
        if not folder:
            return
        folder = Path(folder)
        if folder.is_file() and folder.suffix.lower() == '.json':
            folder = folder.parent
        options = ([folder] if folder.name.lower() == 'cuhk-pedes' else
                   [folder / 'CUHK-PEDES', folder / 'data' / 'CUHK-PEDES', folder])
        for d in options:
            if d not in (x[0] for x in candidates):
                candidates.append((d, reason))

    if explicit_annotation:
        if not explicit_annotation.is_file():
            raise FileNotFoundError(f'--annotation-file does not exist: {explicit_annotation}')
        offer(explicit_annotation.parent, 'explicit annotation')
    for seed in seeds:
        offer(seed, 'project/config path')

    def choose():
        matches = []
        for folder, source in candidates:
            img_dir = explicit_images if explicit_images else folder / 'imgs'
            if not img_dir.is_dir():
                continue
            names = ([explicit_annotation.name] if explicit_annotation else
                     ['reid_raw.json', 'caption_all.json'])
            for fname in names:
                ann = explicit_annotation if explicit_annotation else folder / fname
                if ann.is_file():
                    # Official annotation preferred across candidates; explicit and config
                    # roots take precedence when the same annotation type is available.
                    priority = 0 if fname == 'reid_raw.json' else 1
                    if explicit_annotation:
                        priority = -1
                    matches.append((priority, len(matches), folder, ann, img_dir, source))
        if matches:
            matches.sort(key=lambda x:(x[0], x[1]))
            _, _, folder, ann, img_dir, source = matches[0]
            return folder, ann, img_dir, source
        return None

    found = choose()
    if found:
        return found

    # Search only nearby project trees, not entire drives or system folders.
    scan_roots = [root, root.parent, Path(__file__).resolve().parent]
    seen, visited = set(), 0
    blocked = {'.git', '.idea', '.venv', 'venv', 'envs', '__pycache__',
               'node_modules', 'logs', PROJECT_FOLDER_NAME, 'anaconda3', 'anaconda'}
    for top in scan_roots:
        if not top.is_dir() or top in seen:
            continue
        for current, dirs, files in os.walk(top):
            now = Path(current)
            depth = len(now.relative_to(top).parts)
            visited += 1
            if depth >= 5 or visited >= 3000:
                dirs[:] = []
            else:
                dirs[:] = [d for d in dirs if d.lower() not in blocked]
            if 'reid_raw.json' in files or 'caption_all.json' in files:
                offer(now, 'nearby folder scan')
            if visited >= 3000:
                break
        seen.add(top)
        if visited >= 3000:
            break
    found = choose()
    if found:
        return found

    checked = '\n'.join('  - ' + str(d) for d, _ in candidates[:18])
    raise FileNotFoundError(
        'Could not find CUHK-PEDES annotations AND imgs together. '        'The old fixed path is no longer assumed.\n'
        'Expected CUHK-PEDES/{reid_raw.json or caption_all.json, imgs/}.\n'
        f'Searched {len(candidates)} candidate folders (and {visited} nearby directories).\n'
        'Typical locations checked:\n' + checked + '\n'
        'Specify the true dataset folder with --dataset-root "G:\\...\\CUHK-PEDES"; '
        'for separate locations, also use --annotation-file and --images-root. '
        'If both annotation JSON files are genuinely missing, you need to obtain the original file.'
    )

def verify_image_paths(images_dir, ims):
    # Validate full gallery *before* loading a 1.2GB checkpoint.
    missing = []
    for item in ims:
        if not (Path(images_dir) / item['file_path']).is_file():
            missing.append(item['file_path'])
            if len(missing) >= 6:
                break
    if missing:
        raise FileNotFoundError(
            f'Image paths in annotation do not match {images_dir}. '
            f'First missing paths: {missing}. '
            'Check --dataset-root / --images-root and completeness of imgs/. '
            'Images will not be silently skipped.'
        )

def _replace_with_retries(source, destination, retries=8):
    """Windows-safe replace: handles short-lived antivirus/IDE/Excel file locks.

    Both files must be on the same filesystem.  A persistent lock still raises,
    except for optional derived reports handled by write_csv below.
    """
    for attempt in range(retries):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            transient = isinstance(exc, PermissionError) or getattr(exc, 'winerror', None) in (5, 32, 33)
            if not transient or attempt == retries - 1:
                raise
            time.sleep(min(0.10 * (attempt + 1), 0.6))


def _new_temp_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique temporary names avoid reusing a .tmp file still scanned/locked by Windows.
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=str(path.parent))
    return descriptor, Path(name)


def atomic_json(path, obj):
    path = Path(path)
    fd, temp = _new_temp_path(path)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        _replace_with_retries(temp, path)
    finally:
        try: temp.unlink(missing_ok=True)
        except PermissionError: pass


def write_csv(path, rows, cols, allow_locked_fallback=False):
    """Save CSV atomically. A locked *derived* training history may use a fallback.

    Primary evaluation outputs are critical and still raise on write failure.
    The training history is recoverable from *_training_meta.json, and will be
    reconstructed under its canonical name inside the analysis ZIP.
    """
    path = Path(path)
    fd, temp = _new_temp_path(path)
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        try:
            _replace_with_retries(temp, path)
            return path
        except PermissionError as exc:
            if not allow_locked_fallback:
                raise PermissionError(
                    f'Cannot replace {path}; close any open CSV in Excel/PyCharm '                    f'and retry with --resume. Original error: {exc}'
                ) from exc
            # No need to fail the entire experiment just because a *summary* CSV
            # is held open. Models and history JSON have already been committed.
            fallback = path.with_name(
                path.stem + '_recovered_' + datetime.now().strftime('%Y%m%d_%H%M%S')
                + '_' + str(os.getpid()) + path.suffix
            )
            _replace_with_retries(temp, fallback)
            msg(f'WARNING: {path.name} is locked, saved full history to {fallback.name}. '
                'The analysis ZIP will contain a fresh training_history.csv.')
            return fallback
    finally:
        try: temp.unlink(missing_ok=True)
        except PermissionError: pass

def sha256_small_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def fingerprint(project_root, config, checkpoint, annotations, split):
    p = Path(checkpoint).stat()
    return {
        'project_root': str(Path(project_root).resolve()),
        'config_sha256': sha256_small_file(config),
        'annotations_sha256': sha256_small_file(annotations),
        'checkpoint_path': str(Path(checkpoint).resolve()),
        'checkpoint_size': p.st_size,
        'checkpoint_sha256': sha256_small_file(checkpoint),
        'checkpoint_mtime_ns': p.st_mtime_ns,
        'split': split,
        'format_version': 4,
    }

def read_annotations(path, split):
    with open(path, encoding='utf-8') as f:
        ann = json.load(f)
    if not isinstance(ann, list):
        raise ValueError('Expected an annotation JSON list (with split, id, file_path, captions)')
    ims, texts, train_ids = [], [], set()
    for row in ann:
        if row.get('split') == 'train':
            train_ids.add(int(row['id']) - 1)
        if row.get('split') != split:
            continue
        path_value = row['file_path']
        if not isinstance(path_value, str):
            raise ValueError('Invalid file_path in annotations')
        img_idx = len(ims)
        ims.append({'pid': int(row['id']), 'file_path': path_value})
        if not isinstance(row.get('captions'), list) or not row['captions']:
            raise ValueError(f'Invalid/empty captions at image {path_value}')
        for c in row['captions']:
            if not isinstance(c, str):
                raise ValueError('Non-string caption in annotations')
            texts.append({'pid': int(row['id']), 'img_idx': img_idx, 'text': c})
    if not ims or not texts:
        raise ValueError(f'No images/text in split={split}')
    if not train_ids:
        raise ValueError('No training IDs found; annotation may not be the official CUHK-PEDES split file')
    return ims, texts, len(train_ids)

def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

def load_irra_model(project_root, cfgpath, ckptpath, num_classes, device):
    import torch
    import importlib
    import yaml
    with open(cfgpath, encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)  # IRRA configs may use !!python/tuple
    args = SimpleNamespace(**config)
    if not hasattr(args, 'img_size') or not hasattr(args, 'stride_size'):
        raise ValueError('The IRRA configs.yaml must include img_size and stride_size')
    args.img_size = tuple(args.img_size)
    root_str = str(Path(project_root).resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    # Avoid accidentally importing a third-party module called `model`.
    if not (Path(project_root) / 'model' / 'build.py').exists():
        raise FileNotFoundError('Could not find IRRA model/build.py in project root')

    msg('Loading trained IRRA checkpoint to CPU (no OpenAI CLIP download) ...')
    try:
        ckpt = torch.load(ckptpath, map_location='cpu', weights_only=False)
    except TypeError:  # compatibility with older torch
        ckpt = torch.load(ckptpath, map_location='cpu')
    state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    if not isinstance(state, dict):
        raise ValueError('Checkpoint does not contain a state_dict or model key')
    if all(k.startswith('module.') for k in state):
        state = {k[len('module.'):]: v for k, v in state.items()}
    if 'base_model.text_projection' not in state:
        raise KeyError('Missing base_model.text_projection in checkpoint; wrong checkpoint?')

    def shape(key):
        return state['base_model.' + key].shape

    is_vit = 'base_model.visual.proj' in state
    if not is_vit:
        raise NotImplementedError('This experiment currently supports IRRA ViT checkpoints only')
    vit_layers = {int(m.group(1)) for key in state
                  if (m := re.match(r'base_model\.visual\.transformer\.resblocks\.(\d+)\.', key))}
    txt_layers = {int(m.group(1)) for key in state
                  if (m := re.match(r'base_model\.transformer\.resblocks\.(\d+)\.', key))}
    if not vit_layers or not txt_layers:
        raise ValueError('Cannot infer IRRA transformer layer numbers from checkpoint')
    width = shape('visual.conv1.weight')[0]
    cfg = dict(
        embed_dim=shape('text_projection')[1],
        image_resolution=args.img_size,
        vision_layers=len(vit_layers),
        vision_width=width,
        vision_patch_size=shape('visual.conv1.weight')[-1],
        stride_size=args.stride_size,
        context_length=shape('positional_embedding')[0],
        vocab_size=shape('token_embedding.weight')[0],
        transformer_width=shape('ln_final.weight')[0],
        transformer_heads=shape('ln_final.weight')[0] // 64,
        transformer_layers=len(txt_layers),
    )
    # Construct exactly the repo architecture without attempting a second
    # download of the unused initial CLIP weights.
    mb = importlib.import_module('model.build')
    from model.clip_model import CLIP
    original_builder = mb.build_CLIP_from_openai_pretrained
    mb.build_CLIP_from_openai_pretrained = (lambda *_: (CLIP(**cfg), {'embed_dim': cfg['embed_dim']}))
    try:
        actual_classes = int(state['classifier.weight'].shape[0]) if 'classifier.weight' in state else num_classes
        if 'classifier.weight' in state and actual_classes != num_classes:
            raise ValueError(f'Classifier classes {actual_classes} != dataset train IDs {num_classes}')
        model = mb.build_model(args, num_classes=actual_classes)
    finally:
        mb.build_CLIP_from_openai_pretrained = original_builder
    if device.type == 'cpu':
        model.float()  # CLIP fp16 ops are not universally supported on CPU
    model.load_state_dict(state, strict=True)
    del state, ckpt
    model = model.to(device).eval()
    msg(f'Strict checkpoint load OK; feature_dim={cfg["embed_dim"]}, device={device}')
    return model, args, cfg['embed_dim']

def open_cache(path, meta_path, fingerprint_value, n, dim):
    path, meta_path = Path(path), Path(meta_path)
    if path.exists() != meta_path.exists():
        raise RuntimeError(f'Incomplete cache pair {path} / {meta_path}. Start a new run without --resume')
    if meta_path.exists():
        with open(meta_path, encoding='utf-8') as f:
            info = json.load(f)
        if (info['fingerprint'] != fingerprint_value or info['shape'] != [n, dim]
                or not 0 <= info['done'] <= n):
            raise RuntimeError(f'Cache metadata conflict at {meta_path}; use new --output-dir')
        x = np.lib.format.open_memmap(path, mode='r+')
        if x.shape != (n, dim) or x.dtype != np.float32:
            raise RuntimeError('Cache array shape/dtype mismatch')
        return x, int(info['done'])
    x = np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=(n, dim))
    x.flush()
    atomic_json(meta_path, {'fingerprint': fingerprint_value, 'shape': [n, dim], 'done': 0})
    return x, 0

def record_done(path, fp, n, dim, done, arr):
    arr.flush()  # flush data first, then commit the completed prefix
    atomic_json(path, {'fingerprint': fp, 'shape': [n, dim], 'done': done})

def make_image_dataset(ims, imgsroot, img_size):
    # Only define a torch Dataset when encoding images, never in the NumPy-only
    # analysis worker. Class inheritance would otherwise import PyTorch eagerly.
    import torch
    import torchvision.transforms as T

    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.items = ims
            self.root = Path(imgsroot)
            self.transform = T.Compose([
                T.Resize(tuple(img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
            ])
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            from PIL import Image
            imgpath = self.root / self.items[idx]['file_path']
            if not imgpath.is_file():
                raise FileNotFoundError(f'Image missing: {imgpath}')
            with Image.open(imgpath) as im:
                return self.transform(im.convert('RGB'))

    return ImageDataset()

def encode_and_cache(project_root, cfgpath, ckptpath, annotation_path,
                     imgsroot, ims, texts, num_classes, fp, out, opt, device):
    import torch
    # We must get embedding dim before opening caches; can avoid loading the
    # network when both full cache files are present by checking metadata.
    txt_metadata = out / 'text_cache_meta.json'
    img_metadata = out / 'image_cache_meta.json'
    cached_dim = None
    if txt_metadata.exists() and img_metadata.exists():
        with open(txt_metadata, encoding='utf-8') as f:
            tm = json.load(f)
        with open(img_metadata, encoding='utf-8') as f:
            im = json.load(f)
        if tm.get('fingerprint') != fp or im.get('fingerprint') != fp:
            raise RuntimeError('Cache fingerprint conflict. Start a new run without --resume')
        if tm.get('done') == len(texts) and im.get('done') == len(ims):
            cached_dim = tm['shape'][1]
    if cached_dim is not None:
        t, td = open_cache(out / 'text_features.npy', txt_metadata, fp, len(texts), cached_dim)
        v, vd = open_cache(out / 'image_features.npy', img_metadata, fp, len(ims), cached_dim)
        if td == len(texts) and vd == len(ims):
            msg('Complete matching feature caches found: skip model initialization.')
            return np.array(t), np.array(v)

    model, args, dim = load_irra_model(project_root, cfgpath, ckptpath, num_classes, device)
    t, text_done = open_cache(out / 'text_features.npy', txt_metadata, fp, len(texts), dim)
    v, image_done = open_cache(out / 'image_features.npy', img_metadata, fp, len(ims), dim)
    from utils.simple_tokenizer import SimpleTokenizer
    tokenizer = SimpleTokenizer()
    sot, eot = tokenizer.encoder['<|startoftext|>'], tokenizer.encoder['<|endoftext|>']
    text_length = int(getattr(args, 'text_length', 77))
    if text_length <= 0:
        raise ValueError('Invalid text_length')

    def tokenize(sent):
        seq = [sot] + tokenizer.encode(sent) + [eot]
        if len(seq) > text_length:
            seq = seq[:text_length]
            seq[-1] = eot
        return seq + [0] * (text_length - len(seq))

    if text_done < len(texts):
        msg(f'Encoding text: resume {text_done}/{len(texts)}')
        for start in range(text_done, len(texts), opt.text_batch):
            end = min(start + opt.text_batch, len(texts))
            toks = torch.tensor([tokenize(texts[k]['text']) for k in range(start, end)],
                                device=device, dtype=torch.long)
            with torch.inference_mode():
                feats = model.encode_text(toks)
            features = normalize(feats.float().cpu().numpy())
            t[start:end] = features
            if end % (opt.text_batch * 20) < opt.text_batch or end == len(texts):
                record_done(txt_metadata, fp, len(texts), dim, end, t)
                msg(f'Text feature cached: {end}/{len(texts)}')
        record_done(txt_metadata, fp, len(texts), dim, len(texts), t)
    else:
        msg('Text cache complete.')

    if image_done < len(ims):
        msg(f'Encoding images: resume {image_done}/{len(ims)}')
        dataset = make_image_dataset(ims, imgsroot, args.img_size)
        # batch sampler starts at the exact unfinished image index
        from torch.utils.data import DataLoader, Subset
        batch = opt.image_batch
        dataset = Subset(dataset, range(image_done, len(dataset)))
        loader = DataLoader(dataset, batch_size=batch, num_workers=opt.workers, shuffle=False)
        position = image_done
        with torch.inference_mode():
            for batch_images in loader:
                batch_images = batch_images.to(device)
                features = normalize(model.encode_image(batch_images).float().cpu().numpy())
                end = position + len(features)
                v[position:end] = features
                position = end
                if (position - image_done) % (batch * 20) < len(features) or position == len(ims):
                    record_done(img_metadata, fp, len(ims), dim, position, v)
                    msg(f'Image feature cached: {position}/{len(ims)}')
        record_done(img_metadata, fp, len(ims), dim, len(ims), v)
    else:
        msg('Image cache complete.')
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return np.array(t), np.array(v)

def metrics(sim, query_pids, gallery_pids):
    """Exact IRRA-style R1/R5/R10, AP and INP from full gallery ranking."""
    sim = np.asarray(sim, dtype=np.float32)
    query_pids = np.asarray(query_pids)
    gallery_pids = np.asarray(gallery_pids)
    if sim.shape != (len(query_pids), len(gallery_pids)):
        raise ValueError('Similarity shape does not match PID arrays')
    # stable tie order for reproducibility, same across every scenario
    idx = np.argsort(-sim, axis=1, kind='stable')
    ok = gallery_pids[idx] == query_pids[:, None]
    positives = ok.sum(axis=1)
    if np.any(positives == 0):
        raise ValueError('Some query IDs have no positive gallery image')
    ranks = np.arange(1, sim.shape[1] + 1, dtype=np.float32)[None, :]
    cumsum = np.cumsum(ok, axis=1)
    ap = ((cumsum / ranks) * ok).sum(axis=1) / positives
    # IRRA mINP: number of relevant / last relevant rank
    last_relevant = sim.shape[1] - np.argmax(ok[:, ::-1], axis=1)
    inp = positives / last_relevant
    first_relevant = np.argmax(ok, axis=1) + 1
    out = {'R1': 100 * np.mean(first_relevant <= 1),
           'R5': 100 * np.mean(first_relevant <= 5),
           'R10': 100 * np.mean(first_relevant <= 10),
           'mAP': 100 * np.mean(ap), 'mINP': 100 * np.mean(inp)}
    # true cosine of best relevant and best irrelevant gallery image
    positive_score = np.where(gallery_pids[None, :] == query_pids[:, None], sim, -np.inf).max(axis=1)
    negative_score = np.where(gallery_pids[None, :] != query_pids[:, None], sim, -np.inf).max(axis=1)
    return out, {'rank': first_relevant, 'ap': ap, 'inp': inp,
                 'margin': positive_score - negative_score,
                 'positive_score': positive_score, 'negative_score': negative_score,
                 'top1_image': idx[:, 0]}


def cache_is_ready(folder, prefix, fp, count, dim=None):
    folder=Path(folder)
    arr=folder/(prefix+'_features.npy')
    meta=folder/(prefix+'_cache_meta.json')
    if not arr.is_file() or not meta.is_file():
        return False
    try:
        with meta.open(encoding='utf-8') as f:
            m=json.load(f)
        if m['fingerprint'] != fp or m['done'] != count or m['shape'][0] != count:
            return False
        if dim and m['shape'][1] != dim:
            return False
        ar=np.load(arr,mmap_mode='r',allow_pickle=False)
        return list(ar.shape)==m['shape'] and ar.dtype==np.float32
    except (OSError,KeyError,ValueError):
        return False


def new_or_resume(root,args,settings):
    root.mkdir(parents=True,exist_ok=True)
    if args.resume:
        opts=sorted((p for p in root.iterdir() if p.is_dir() and p.name.startswith(EXPERIMENT_NAME+'_')),
                    reverse=True,key=lambda p:p.name)
        for d in opts:
            f=d/'run_info.json'
            if f.is_file():
                try:
                    x=json.loads(f.read_text(encoding='utf-8'))
                    if x.get('settings')==settings:
                        return d,True
                except (OSError,ValueError):
                    pass
        raise FileNotFoundError('No matching experiment07 run to resume. Remove --resume for a new run.')
    name=EXPERIMENT_NAME+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')
    dest=root/name
    idx=2
    while dest.exists():
        dest=root/(name+f'_{idx:02d}')
        idx+=1
    dest.mkdir(parents=True)
    return dest,False


def run_subprocess(stage,out,root):
    command=[sys.executable, str(Path(__file__).resolve()),'--_worker',stage,'--_run-dir',str(out)]
    env=os.environ.copy()
    env.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                'NUMEXPR_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'})
    env.pop('KMP_DUPLICATE_LIB_OK',None)
    msg('\n=== Stage: '+stage+' ===')
    with subprocess.Popen(command,cwd=str(root),env=env,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,bufsize=1,text=True,
                          encoding='utf-8',errors='replace') as p:
        for line in p.stdout:
            msg(line.rstrip('\r\n'))
        code=p.wait()
    if code:
        raise RuntimeError(f'Stage {stage} failed (exit code {code}); see experiment_run.log')


def package_for_chat(out):
    """Always generate a small analysis bundle, even after interrupted training.

    Training history is rendered from per-model completed JSON metadata, NOT from
    an on-disk CSV that might have been locked by Excel or PyCharm.
    """
    out=Path(out)
    zip_path=out/'发送给ChatGPT.zip'
    allow_ext={'.csv','.json','.txt','.log','.md','.py','.png'}
    files=[p for p in out.iterdir() if p.is_file() and p.suffix.lower() in allow_ext]
    history=all_training_records(out)
    fd, temp=_new_temp_path(zip_path)
    os.close(fd)
    try:
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(files):
                if p.name=='training_history.csv' and history:
                    continue  # replace stale or locked physical CSV with fresh reconstruction
                z.write(p,p.name)
            if history:
                sio=io.StringIO(newline='')
                writer=csv.DictWriter(sio,fieldnames=list(history[0]),extrasaction='ignore')
                writer.writeheader();writer.writerows(history)
                z.writestr('training_history.csv',b'\xef\xbb\xbf'+sio.getvalue().encode('utf-8'))
        try:
            _replace_with_retries(temp,zip_path)
        except PermissionError:
            # Don't overwrite a ZIP currently being opened by an archiver.
            backup=out/('发送给ChatGPT_恢复版_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.zip')
            _replace_with_retries(temp,backup)
            zip_path=backup
            msg('WARNING: previous ZIP is locked; wrote '+backup.name+' instead.')
    finally:
        try: temp.unlink(missing_ok=True)
        except PermissionError: pass
    msg(f'Analysis ZIP saved: {zip_path} ({zip_path.stat().st_size/1048576:.2f} MiB)')
    return zip_path


def manifest_for_stage(run_dir):
    out=Path(run_dir)
    manifest=json.loads((out/'run_info.json').read_text(encoding='utf-8'))
    return manifest,SimpleNamespace(**manifest['settings']),out


def encode_text_only(model,device,texts,out,fp,batch_size,d,cache_name='train_text'):
    import torch
    arrpath=out/(cache_name+'_features.npy')
    metapath=out/(cache_name+'_cache_meta.json')
    arr,done=open_cache(arrpath,metapath,fp,len(texts),d)
    if done==len(texts):
        msg(f'Complete training text features: {done}/{len(texts)}, skip encoding')
        return
    from utils.simple_tokenizer import SimpleTokenizer
    tok=SimpleTokenizer()
    sot,eot=tok.encoder['<|startoftext|>'],tok.encoder['<|endoftext|>']
    max_len=77
    def tokenize(sentence):
        ids=[sot]+tok.encode(sentence)+[eot]
        if len(ids)>max_len:
            ids=ids[:max_len]
            ids[-1]=eot
        return ids+[0]*(max_len-len(ids))
    msg(f'Encoding training captions {done}/{len(texts)} ...')
    for start in range(done,len(texts),batch_size):
        end=min(start+batch_size,len(texts))
        tokens=torch.tensor([tokenize(t['text']) for t in texts[start:end]],
                            dtype=torch.long,device=device)
        with torch.inference_mode():
            result=model.encode_text(tokens)
        arr[start:end]=normalize(result.float().cpu().numpy())
        if end%max(batch_size, batch_size*10)<batch_size or end==len(texts):
            record_done(metapath,fp,len(texts),d,end,arr)
            msg(f'Training text feature cached: {end}/{len(texts)}')
    record_done(metapath,fp,len(texts),d,len(texts),arr)


# ============== Reused IRRA encoding and cache infrastructure ==============

def capped_peer_indices(candidates, x, feats, cap):
    """Choose a deterministic similarity-spaced subset when too many supporters exist."""
    if len(candidates) <= cap:
        return candidates
    scores = feats[candidates] @ x
    ordered = np.argsort(scores, kind='stable')
    positions = np.unique(np.linspace(0, len(candidates)-1, cap, dtype=int))
    return [candidates[int(ordered[j])] for j in positions]



def get_subset(ann, selected_pids):
    imgs, texts, _ = read_annotations(ann, 'train')
    allowed = set(map(int, selected_pids))
    img_ids = [i for i, r in enumerate(imgs) if r['pid'] in allowed]
    text_ids = [i for i, r in enumerate(texts) if r['pid'] in allowed]
    if not img_ids or not text_ids:
        raise ValueError('Training PID subset has no matching images/texts')
    return imgs, texts, img_ids, text_ids


def source_val_cache(root, project_folder, explicit, fp, nt, ni):
    if explicit:
        dirs = [Path(explicit).resolve()]
    else:
        dirs = (sorted([p for p in project_folder.iterdir() if p.is_dir() and
                        (p.name.startswith('实验00_') or p.name.startswith('实验01_'))],
                       reverse=True, key=lambda x:x.name) if project_folder.is_dir() else [])
    for folder in dirs:
        if cache_is_ready(folder, 'text', fp, nt) and cache_is_ready(folder, 'image', fp, ni):
            msg('Reuse validation feature cache: ' + str(folder))
            return folder
    if explicit:
        raise ValueError('--val-cache-dir not complete or not checkpoint/split fingerprint matching')
    return None


def source_train_text_cache(root, project_folder, explicit, fp, nt):
    if explicit:
        dirs = [Path(explicit).resolve()]
    else:
        dirs = (sorted([p for p in project_folder.iterdir() if p.is_dir() and
                        p.name.startswith('实验01_')],reverse=True,key=lambda x:x.name)
                if project_folder.is_dir() else [])
    for folder in dirs:
        if cache_is_ready(folder, 'train_text', fp, nt):
            msg('Reuse training caption cache: ' + str(folder))
            return folder
    if explicit:
        raise ValueError('--train-text-cache-dir not complete/fingerprint matching')
    return None


def train_arrays(out, opt):
    ims, texts, img_idx, text_idx = get_subset(Path(opt.annotation_file),
                            json.loads((out/'selected_train_pids.json').read_text(encoding='utf-8')))
    txtbase = Path(opt.train_text_cache_dir) if opt.train_text_cache_dir else out
    full = np.load(txtbase/'train_text_features.npy', mmap_mode='r')
    if full.shape[0] != len(texts):
        raise ValueError('Full train text feature cache is incompatible with annotation text order')
    xf = normalize(np.asarray(full[text_idx],dtype=np.float32))
    vf = normalize(np.load(out/'train_image_features.npy',mmap_mode='r'))
    if vf.shape[0] != len(img_idx) or vf.shape[1] != xf.shape[1]:
        raise ValueError('Training image features inconsistent with subset')
    mini = [texts[i] for i in text_idx]
    miniims = [ims[i] for i in img_idx]
    lookup = {orig: i for i,orig in enumerate(img_idx)}
    positives = np.asarray([lookup[t['img_idx']] for t in mini], dtype=np.int64)
    return xf, vf, mini, miniims, positives


def train_img_fingerprint(opt, img_idx):
    digest = hashlib.sha256(np.asarray(img_idx,dtype=np.int32).tobytes()).hexdigest()
    return {'train_base': opt.fp_train, 'selected_image_indices_sha256':digest,
            'selected_image_count':len(img_idx),'format_version':1}


def hard_negative_mine(x, v, tx_pid, img_pid, k, chunk=96):
    """Frozen IRRA training gallery; exclude ALL images of anchor PID."""
    n=len(x); k=min(k, len(v)-1)
    if k <= 0: raise ValueError('Need >=2 distinct gallery identities for negative mining')
    out=np.empty((n,k),dtype=np.int32)
    for s in range(0,n,chunk):
        e=min(n,s+chunk)
        logits=np.asarray(x[s:e]@v.T,dtype=np.float32)
        logits[tx_pid[s:e,None]==img_pid[None,:]]=-np.inf
        if (np.isneginf(logits).sum(axis=1)>len(v)-k).any():
            raise ValueError('Not enough different PID images for negatives')
        ind=np.argpartition(-logits,kth=k-1,axis=1)[:,:k]
        val=np.take_along_axis(logits,ind,axis=1)
        order=np.argsort(-val,axis=1,kind='stable')
        out[s:e]=np.take_along_axis(ind,order,axis=1)
        if e % (chunk*30)<chunk or e==n:
            msg(f'Hard negatives mined: {e}/{n}')
    return out


def get_adapter_exp02():
    import torch
    from torch import nn
    import torch.nn.functional as F
    class TextResidualAdapter(nn.Module):
        def __init__(self,d,hidden,gain):
            super().__init__()
            self.fc1=nn.Linear(d,hidden)
            self.fc2=nn.Linear(hidden,d)
            nn.init.zeros_(self.fc2.weight);nn.init.zeros_(self.fc2.bias)
            self.gain=float(gain)
        def forward(self,x):
            return F.normalize(x+self.gain*self.fc2(F.gelu(self.fc1(x))),dim=-1)
    return TextResidualAdapter


# ============== Reused train data and image cache infrastructure ===========

def source_train_image_cache(project_folder, explicit, opt, img_idx):
    fp=train_img_fingerprint(opt,img_idx)
    if explicit:
        directories=[Path(explicit).resolve()]
    elif project_folder.is_dir():
        directories=sorted((p for p in project_folder.iterdir() if p.is_dir() and
                            p.name.startswith('实验02_')),key=lambda p:p.name,reverse=True)
    else:
        directories=[]
    for folder in directories:
        if cache_is_ready(folder,'train_image',fp,len(img_idx)):
            # Fingerprint alone checks selection/order and encoder weights; explicit PID equality too.
            f=folder/'selected_train_pids.json'
            expected=opt.selected_pids
            if f.is_file() and json.loads(f.read_text(encoding='utf-8'))==expected:
                msg('Reuse experiment02 training image cache: '+str(folder))
                return folder
    if explicit:
        raise RuntimeError('Explicit --train-image-cache-dir not fully compatible with this PID subset/checkpoint.')
    return None


def train_arrays03(out,opt):
    ims,texts,img_idx,text_idx=get_subset(Path(opt.annotation_file),
                        json.loads((out/'selected_train_pids.json').read_text(encoding='utf-8')))
    txt_dir=Path(opt.train_text_cache_dir) if opt.train_text_cache_dir else out
    img_dir=Path(opt.train_image_cache_dir) if opt.train_image_cache_dir else out
    if not cache_is_ready(txt_dir,'train_text',opt.fp_train,len(texts)):
        raise RuntimeError('Invalid/incomplete full train-text cache')
    if not cache_is_ready(img_dir,'train_image',train_img_fingerprint(opt,img_idx),len(img_idx)):
        raise RuntimeError('Invalid/incomplete selected train-image cache')
    full=np.load(txt_dir/'train_text_features.npy',mmap_mode='r')
    xf=normalize(np.asarray(full[text_idx],dtype=np.float32))
    vf=normalize(np.load(img_dir/'train_image_features.npy',mmap_mode='r'))
    if len(vf)!=len(img_idx) or xf.shape[1]!=vf.shape[1]:
        raise RuntimeError('Train text/image cache dim mismatch')
    mini=[texts[i] for i in text_idx]
    mini_imgs=[ims[i] for i in img_idx]
    lookup={orig:i for i,orig in enumerate(img_idx)}
    pos=np.asarray([lookup[t['img_idx']] for t in mini],dtype=np.int64)
    if not all(mini[k]['pid']==mini_imgs[pos[k]]['pid'] for k in range(len(mini))):
        raise RuntimeError('Training positive image mapping corrupted')
    return xf,vf,mini,mini_imgs,pos


def stage_encode03(out):
    _,opt,out=manifest_for_stage(out)
    all_img,all_txt,img_idx,_=get_subset(Path(opt.annotation_file),opt.selected_pids)
    val_img,val_txt,nclasses=read_annotations(Path(opt.annotation_file),'val')
    tdir=Path(opt.train_text_cache_dir) if opt.train_text_cache_dir else out
    idir=Path(opt.train_image_cache_dir) if opt.train_image_cache_dir else out
    vdir=Path(opt.val_cache_dir) if opt.val_cache_dir else out
    ready_t=cache_is_ready(tdir,'train_text',opt.fp_train,len(all_txt))
    ready_i=cache_is_ready(idir,'train_image',train_img_fingerprint(opt,img_idx),len(img_idx))
    ready_v=(cache_is_ready(vdir,'text',opt.fp_val,len(val_txt)) and
             cache_is_ready(vdir,'image',opt.fp_val,len(val_img)))
    if ready_t and ready_i and ready_v:
        msg('All verified frozen-feature caches ready. No model initialization or image extraction.')
        return
    import torch
    device=torch.device('cuda' if opt.device=='cuda' or
                        (opt.device=='auto' and torch.cuda.is_available()) else 'cpu')
    if opt.device=='cuda' and device.type!='cuda':
        raise RuntimeError('CUDA requested but unavailable')
    model=None
    if not (ready_t and ready_i):
        model,args,d=load_irra_model(Path(opt.project_root),Path(opt.config_file),
                                     Path(opt.checkpoint),nclasses,device)
        if not ready_t:
            encode_text_only(model,device,all_txt,out,opt.fp_train,opt.text_batch,d,
                             cache_name='train_text')
        if not ready_i:
            cache,done=open_cache(out/'train_image_features.npy',out/'train_image_cache_meta.json',
                     train_img_fingerprint(opt,img_idx),len(img_idx),d)
            from torch.utils.data import DataLoader
            dataset=make_image_dataset([all_img[i] for i in img_idx],opt.images_dir,args.img_size)
            # no child DataLoader processes: safer for Windows and PyCharm
            loader=DataLoader(torch.utils.data.Subset(dataset,range(done,len(img_idx))),
                              batch_size=opt.image_batch,shuffle=False,num_workers=0)
            current=done
            with torch.inference_mode():
                for batch in loader:
                    x=normalize(model.encode_image(batch.to(device)).float().cpu().numpy())
                    cache[current:current+len(x)]=x
                    current+=len(x)
                    if current%(opt.image_batch*25)<len(x) or current==len(img_idx):
                        record_done(out/'train_image_cache_meta.json',
                                    train_img_fingerprint(opt,img_idx),len(img_idx),d,current,cache)
                        msg(f'Training images cached: {current}/{len(img_idx)}')
            record_done(out/'train_image_cache_meta.json',
                        train_img_fingerprint(opt,img_idx),len(img_idx),d,current,cache)
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    if not ready_v:
        argsval=SimpleNamespace(text_batch=opt.text_batch,image_batch=opt.image_batch,workers=0)
        encode_and_cache(Path(opt.project_root),Path(opt.config_file),Path(opt.checkpoint),
                         Path(opt.annotation_file),Path(opt.images_dir),val_img,val_txt,
                         nclasses,opt.fp_val,out,argsval,device)
    msg('Frozen encoding stage done.')



def all_training_records(out):
    records=[]
    for file in sorted(Path(out).glob('*_training_meta.json')):
        entry=json.loads(file.read_text(encoding='utf-8'))
        if entry.get('complete'):
            records.extend(entry.get('history', []))
    return records



# ================= EXP07: reliable support triage ====================

def pid_split(all_pids, fraction, seed):
    ids = np.asarray(sorted(map(int, all_pids)), dtype=np.int64)
    if len(ids) < 8: raise ValueError('Need >=8 identities')
    sh = np.random.default_rng(seed).permutation(ids)
    k = min(len(ids)-2, max(4, int(round(len(ids)*fraction))))
    a,b = sorted(map(int, sh[:k])),sorted(map(int,sh[k:]))
    if set(a)&set(b):raise AssertionError('Identity leakage')
    return a,b


def partition(x,v,texts,images,pos,pid_list):
    allow=set(map(int,pid_list))
    t_idx=np.asarray([j for j,r in enumerate(texts) if int(r['pid']) in allow],np.int32)
    i_idx=np.asarray([j for j,r in enumerate(images) if int(r['pid']) in allow],np.int32)
    mapper={int(i):j for j,i in enumerate(i_idx)}
    local_pos=np.asarray([mapper[int(pos[k])] for k in t_idx],np.int32)
    tx=[texts[int(i)] for i in t_idx]; im=[images[int(i)] for i in i_idx]
    if not all(tx[j]['pid']==im[int(local_pos[j])]['pid'] for j in range(len(tx))):
        raise AssertionError('Positive instance mapping mismatch')
    return normalize(x[t_idx]),normalize(v[i_idx]),tx,im,local_pos,t_idx,i_idx


def atomic_npy(path, array):
    path=Path(path)
    fd,temp=_new_temp_path(path)
    try:
        with os.fdopen(fd,'wb') as f:
            np.save(f,np.asarray(array),allow_pickle=False)
            f.flush();os.fsync(f.fileno())
        _replace_with_retries(temp,path)
    finally:
        try:temp.unlink(missing_ok=True)
        except OSError:pass


def mined_cache(out,name,x,v,texts,ims,k,batch,settings_hash):
    path=out/(name+'_negatives.npy');meta=out/(name+'_negatives_meta.json')
    tp=np.asarray([r['pid'] for r in texts]);gp=np.asarray([r['pid'] for r in ims])
    stamp={'settings_hash':settings_hash,'shape':[len(x),k],'pids_sha':hashlib.sha256(tp.tobytes()+gp.tobytes()).hexdigest()}
    if path.is_file() and meta.is_file():
        try:
            old=json.loads(meta.read_text(encoding='utf-8'))
            arr=np.load(path,mmap_mode='r',allow_pickle=False)
            if (old==stamp and arr.shape==(len(x),k) and arr.dtype.kind in 'iu' and
                np.all(arr>=0) and np.all(arr<len(v)) and np.all(gp[arr]!=tp[:,None])):
                msg('Recovered valid negative cache: '+name)
                return np.asarray(arr,dtype=np.int32)
        except (OSError,ValueError,KeyError):pass
    neg=hard_negative_mine(x,v,tp,gp,k,batch)
    if not np.all(gp[neg]!=tp[:,None]):raise AssertionError('Negative PID contamination')
    atomic_npy(path,neg);atomic_json(meta,stamp)
    return neg



# ==========================================================================
# EXP08: BASE RERANKING ONLY.  NO OTHER CAPTIONS, NO SEMANTIC TEACHER.
# The EXP07 warmup and mined training negatives are read-only experiment input.
# ==========================================================================
VERSION08 = 'exp08_base_rerank_20261021_v1'
SOURCE_PREFIX08 = '实验07_条件式文本互补与有害信息拒绝_'
SCALES08 = (0., .1, .25, .5, .75, 1.)


def source08(opt):
    p=Path(opt.source_exp07)
    if not (p/'run_info.json').is_file():
        raise FileNotFoundError('Missing source EXP07 run_info.json at '+str(p))
    d=json.loads((p/'run_info.json').read_text(encoding='utf-8'))
    if 'settings' not in d:raise RuntimeError('Malformed EXP07 source manifest')
    return p,SimpleNamespace(**d['settings'])


def read_source08(out,validation=False):
    _,opt,_=manifest_for_stage(out)
    src,so=source08(opt)
    x,v,txt,ims,pos=train_arrays03(src,so)
    xf,vf,tf,gf,pf,_,_=partition(x,v,txt,ims,pos,so.fit_pids)
    neg=np.load(src/'fit_negative_indices.npy',allow_pickle=False)
    if neg.shape!=(len(xf),int(so.hard_k)) or neg.dtype.kind not in 'iu':
        raise RuntimeError('EXP07 fit negative indices missing or corrupt')
    pid=np.asarray([r['pid'] for r in tf]); ipid=np.asarray([r['pid'] for r in gf]);
    if np.any(neg<0) or np.any(neg>=len(gf)) or np.any(ipid[neg]==pid[:,None]):
        raise AssertionError('Mined negatives illegal: positive PID included or index corrupted')
    if any(tf[j]['pid']!=gf[int(pf[j])]['pid'] for j in range(len(tf))):
        raise AssertionError('Train pos mapping corrupted')
    if not validation:return src,so,(xf,vf,tf,gf,pf,neg)
    vd=Path(so.val_cache_dir) if so.val_cache_dir else src
    vi,vt,_=read_annotations(Path(so.annotation_file),'val')
    if not (cache_is_ready(vd,'text',so.fp_val,len(vt)) and
            cache_is_ready(vd,'image',so.fp_val,len(vi))):
        raise RuntimeError('Original frozen validation feature cache missing or mismatched')
    xx=normalize(np.load(vd/'text_features.npy',mmap_mode='r'))
    vv=normalize(np.load(vd/'image_features.npy',mmap_mode='r'))
    topk=np.load(src/'val_topk_indices.npy',allow_pickle=False)
    if topk.ndim!=2 or topk.shape[0]!=len(xx) or np.any(topk>=len(vv)) or np.any(topk<0):
        raise RuntimeError('EXP07 original Top-K index mismatch')
    if topk.shape[1]>1 and np.any(np.diff(np.sort(topk,axis=1),axis=1)==0):
        raise RuntimeError('EXP07 topk index duplicates')
    if set(t['pid'] for t in vt)&set(so.fit_pids):raise AssertionError('Train vs Val PID leak')
    if set(t['pid'] for t in vt)&set(so.holdout_pids):raise AssertionError('Holdout vs Val PID leak')
    return src,so,(xf,vf,tf,gf,pf,neg),(xx,vv,vi,vt,np.asarray(topk,dtype=np.int32))


def get_pair_class08():
    import torch
    from torch import nn
    import torch.nn.functional as F
    class PairReranker(nn.Module):
        '''Matches original EXP07 parameter keys exactly for checkpoint loading.'''
        def __init__(self,dim,hidden,max_delta,mode='standard',fixed_gate=0.5,cap=0.02):
            super().__init__()
            self.fc1=nn.Linear(dim*2+1,hidden)
            self.fc2=nn.Linear(hidden,hidden)
            self.gate=nn.Linear(hidden,1)
            self.delta=nn.Linear(hidden,1)
            self.max_delta=float(max_delta)
            self.mode=mode
            self.fixed_gate=float(fixed_gate)
            self.cap=float(cap)
            nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
            nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,-.5)
        def forward(self,t,image):
            t=t[:,None,:].expand_as(image)
            cosine=(t*image).sum(-1)
            h=F.gelu(self.fc1(torch.cat([t*image,torch.abs(t-image),cosine[...,None]],-1)))
            h=F.gelu(self.fc2(h))
            gate_logit=self.gate(h).squeeze(-1)
            raw=self.max_delta*torch.tanh(self.delta(h).squeeze(-1))
            gate=torch.sigmoid(gate_logit)
            if self.mode=='fixed_gate':gate=torch.full_like(gate,self.fixed_gate)
            correction=gate*raw
            if self.mode=='hard_cap':correction=torch.clamp(correction,-self.cap,self.cap)
            return cosine+correction,correction,gate_logit,gate
    return PairReranker


def validate_source_checkpoints08(src,so,seed):
    for meth in ('warmup','rank_continue'):
        stem=f'{meth}_s{seed}';ck=src/(stem+'_student.pt');meta=src/(stem+'_training_meta.json')
        score=src/(stem+'_topk_score.npy');gate=src/(stem+'_topk_gate.npy')
        if not all(p.is_file() for p in (ck,meta,score,gate)):
            raise FileNotFoundError('EXP07 incomplete original method '+stem)
        stamp=json.loads(meta.read_text(encoding='utf-8'))
        if stamp.get('settings_hash')!=so.settings_hash or stamp.get('checkpoint_sha256')!=sha256_small_file(ck):
            raise RuntimeError('EXP07 checkpoint fingerprint mismatch: '+stem)


def write_stage08(out,name,rows):
    if rows:write_csv(out/name,rows,list(dict.fromkeys(k for row in rows for k in row)),allow_locked_fallback=True)


def np_diagnostics08(out):
    '''Pure NumPy; no model training or PID access before generating candidate scores.'''
    _,opt,_=manifest_for_stage(out)
    src,so,_,(x,v,ims,txt,tk)=read_source08(out,True)
    sims=np.asarray(x@v.T,dtype=np.float32)
    rows=np.arange(len(x),dtype=np.int32)[:,None]
    b=sims[rows,tk]
    pid=np.asarray([a['pid'] for a in txt]);gp=np.asarray([a['pid'] for a in ims]);
    baseline,br=metrics(sims,pid,gp)
    diag=[];case=[]
    for seed in opt.seeds:
        validate_source_checkpoints08(src,so,seed)
        for method in ('warmup','rank_continue'):
            stem=f'{method}_s{seed}'
            old=np.load(src/(stem+'_topk_score.npy'),mmap_mode='r')
            gates=np.load(src/(stem+'_topk_gate.npy'),mmap_mode='r')
            if old.shape!=b.shape or gates.shape!=b.shape:raise ValueError('Legacy score/gate shape mismatch')
            delta=np.asarray(old,dtype=np.float32)-b
            for alpha in SCALES08:
                # Inference is PID-blind. This is a fixed, posthoc experiment (NOT trained models).
                q=sims.copy();q[rows,tk]=b+alpha*delta
                result,details=metrics(q,pid,gp)
                ok=details['rank']==1;bok=br['rank']==1
                diag.append({'seed':seed,'model':method,'score_scale':alpha,'R1':result['R1'],
                  'R5':result['R5'],'R10':result['R10'],'mAP':result['mAP'],'mINP':result['mINP'],
                  'delta_R1_vs_original_IRRA':result['R1']-baseline['R1'],
                  'repaired_from_irra':int(np.sum(~bok&ok)),
                  'harmed_from_irra':int(np.sum(bok&~ok)),
                  'absolute_delta_mean':float(np.mean(np.abs(delta)))*alpha,
                  'absolute_delta_p95':float(np.percentile(np.abs(delta),95))*alpha,
                  'absolute_delta_p99':float(np.percentile(np.abs(delta),99))*alpha,
                  'positive_correction_fraction':float(np.mean(delta>0)),
                  'mean_gate':float(np.mean(gates)),
                  'gate_gt_08_fraction':float(np.mean(gates>.8)),
                  'source_ranking_protocol':'EXP07 frozen Top-K rerank; posthoc residual interpolation'})
                if alpha in (0.,.25,1.):
                    # record disagreements only, without text content/pid leakage into score
                    changed=np.where(details['rank']!=br['rank'])[0]
                    for i in changed[:100]:
                        case.append({'seed':seed,'method':method,'scale':alpha,
                                     'query_idx':int(i),'pid_evaluation_only':int(pid[i]),
                                     'original_rank':int(br['rank'][i]),'rerank_rank':int(details['rank'][i]),
                                     'original_top1_gallery':int(br['top1_image'][i]),
                                     'rerank_top1_gallery':int(details['top1_image'][i])})
            msg(f'audit legacy {stem}: abs delta mean {np.mean(np.abs(delta)):.6g}, '
                f'gate {np.mean(gates):.4f}')
    write_stage08(out,'legacy_scale_diagnostics.csv',diag)
    write_stage08(out,'legacy_ranking_cases.csv',case)
    atomic_json(out/'legacy_baseline.json',{'original_IRRA':baseline,'n_queries':len(x),
       'gallery_size':len(v),'fixed_shortlist_k':tk.shape[1],
       'note':'All adjustments are PID-blind; PIDs are used only after scores for metrics.'})
    if opt.expected_baseline_r1 is not None and abs(baseline['R1']-opt.expected_baseline_r1)>.05:
        raise RuntimeError('Wrong baseline IRRA fingerprint, do not compare with EXP07')
    msg('EXP08 legacy score-scale diagnostic completed, no model trained.')


def check_model_epoch08(out,stem,shape,opt):
    ck=out/(stem+'_student.pt');meta=out/(stem+'_training_meta.json')
    score=out/(stem+'_topk_score.npy');gate=out/(stem+'_topk_gate.npy')
    if not all(p.is_file() for p in (ck,meta,score,gate)):return False
    try:
        m=json.loads(meta.read_text(encoding='utf-8'))
        a=np.load(score,mmap_mode='r',allow_pickle=False)
        b=np.load(gate,mmap_mode='r',allow_pickle=False)
        return (m['settings_hash']==opt.settings_hash and
                m['checkpoint_sha256']==sha256_small_file(ck) and
                a.shape==shape and b.shape==shape and np.isfinite(a).all() and np.isfinite(b).all())
    except (OSError,KeyError,ValueError):return False


def epoch_train08(model,optimizer,gen,tx,images,pos,neg,method,so,opt,device,fixed):
    import torch
    import torch.nn.functional as F
    n=len(tx);order=torch.randperm(n,generator=gen)
    model.train();sums=defaultdict(float);count=0
    for start in range(0,n,so.train_batch):
        sub=order[start:start+so.train_batch];bs=len(sub)
        t=tx[sub].to(device)
        candidate=torch.cat([pos[sub,None],neg[sub]],dim=1).to(device)
        gallery=images[candidate]
        score,corr,logit,gate=model(t,gallery)
        rank_loss=F.softplus((score[:,1:]-score[:,:1]+so.margin)/so.temperature).mean()
        protect=corr.square().mean()
        multiplier=opt.protect_factor if method=='strong_protect' else 1.
        loss=so.rank_weight*rank_loss+so.protect_weight*multiplier*protect
        # Exact EXP07 original gradient recipe for same-architecture Rank baseline.
        params=list(model.parameters())
        grads=torch.autograd.grad(loss,params,allow_unused=True)
        optimizer.zero_grad(set_to_none=True)
        for p,g in zip(params,grads):p.grad=(torch.zeros_like(p) if g is None else g.detach())
        clipped=torch.nn.utils.clip_grad_norm_(params,5.)
        before=[p.detach().clone() for p in params]
        optimizer.step()
        with torch.no_grad():
            step=math.sqrt(sum(float((p.detach()-q).double().square().sum()) for p,q in zip(params,before)))
        sums['rank_loss']+=float(rank_loss.detach())*bs
        sums['protect_loss']+=float(protect.detach())*bs
        sums['correction_abs_mean']+=float(corr.detach().abs().mean())*bs
        sums['correction_abs_max']+=float(corr.detach().abs().max())*bs
        sums['gate_mean']+=float(gate.detach().mean())*bs
        sums['adamw_step_norm']+=step*bs
        sums['clipped_count']+=int(clipped>5.)
        count+=1
    return {k:float(v)/n for k,v in sums.items() if k!='clipped_count'}|{
        'clipped_batches':int(sums['clipped_count']),'n_batches':count}


def heldout_report08(model,source,so,opt,device):
    '''Internal unseen TRAIN PIDs, never used to train the reranker or choose via Val.'''
    import torch
    x,v,tx,ims,pos=train_arrays03(source,so)
    xh,vh,th,gh,ph,_,_=partition(x,v,tx,ims,pos,so.holdout_pids)
    pids=np.asarray([r['pid'] for r in th]);gpids=np.asarray([r['pid'] for r in gh])
    nn=np.load(source/'holdout_negatives.npy',allow_pickle=False)
    if len(nn)!=len(xh) or np.any(nn<0) or np.any(nn>=len(gh)) or np.any(gpids[nn]==pids[:,None]):
        raise RuntimeError('Holdout negatives are missing or invalid')
    nn=np.asarray(nn[:,:so.hard_k],dtype=np.int64)
    xh=torch.from_numpy(np.array(xh,copy=True));vh=torch.from_numpy(np.array(vh,copy=True)).to(device)
    ph=torch.from_numpy(np.asarray(ph,dtype=np.int64));nn=torch.from_numpy(nn)
    model.eval();better=0;rawloss=0.;gap=0.;corrm=0.;gatem=0.;n=len(xh)
    with torch.inference_mode():
        for s in range(0,n,128):
            e=min(n,s+128);t=xh[s:e].to(device)
            choice=torch.cat([ph[s:e,None],nn[s:e]],dim=1).to(device)
            image=vh[choice]
            score,cor,_,gate=model(t,image)
            margin=score[:,:1]-score[:,1:]
            rawloss+=float(torch.nn.functional.softplus((-margin+so.margin)/so.temperature).sum())
            better+=int((margin>0).sum());gap+=float(margin.sum())
            corrm+=float(cor.abs().sum());gatem+=float(gate.sum())
    pairs=n*(1+nn.shape[1]);rankpairs=n*nn.shape[1]
    return {'holdout_rank_pair_accuracy':better/rankpairs,'holdout_rank_margin_mean':gap/rankpairs,
            'holdout_softplus_mean':rawloss/rankpairs,'holdout_abs_correction_mean':corrm/pairs,
            'holdout_gate_mean':gatem/pairs,'holdout_pids':len(so.holdout_pids),
            'warning':'PID labels only used to build internal TRAIN holdout; NOT standard Val inference.'}


def make_predictions08(model,vx,vg,tk,dev,out,stem,batch=64):
    """Save reranker predictions without keeping any Windows file mapping open.

    At the default 6158 queries x 100 candidates, each float32 prediction array
    occupies only ~2.35 MiB.  In EXP08 v1, the final `arr` loop variable still
    referenced the gate's np.memmap during os.replace, causing WinError 32 even
    with all external programs closed.  Keep arrays in RAM and commit them via
    atomic_npy, which closes its file descriptor *before* os.replace.
    """
    import torch
    n,k=tk.shape
    scores=np.empty((n,k),dtype=np.float32)
    gates=np.empty((n,k),dtype=np.float32)
    model.eval();qb=max(1,min(batch,8192//max(k,1)))
    with torch.inference_mode():
        for start in range(0,n,qb):
            end=min(n,start+qb)
            t=vx[start:end].to(dev)
            ids=torch.from_numpy(np.asarray(tk[start:end],dtype=np.int64)).to(dev)
            prediction,_,_,gate=model(t,vg[ids])
            scores[start:end]=prediction.float().cpu().numpy()
            gates[start:end]=gate.float().cpu().numpy()
    if not (np.isfinite(scores).all() and np.isfinite(gates).all()):
        raise RuntimeError(f'Nonfinite reranking predictions for {stem}; refuse to save checkpoint')
    # No open_memmap and no persistent .partial.npy handles on this write path.
    # Atomic files are written with unique temp names then closed and renamed.
    atomic_npy(out/(stem+'_topk_score.npy'),scores)
    atomic_npy(out/(stem+'_topk_gate.npy'),gates)


def save_epoch08(model,optimizer,gen,out,stem,stats,method,seed,epoch,opt,dev,vx,vg,tk):
    import torch
    make_predictions08(model,vx,vg,tk,dev,out,stem,opt.eval_query_batch)
    saved=out/(stem+'_student.pt');temp=out/(stem+'_student.partial.pt')
    torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                'generator_state':gen.get_state(),'epoch':epoch,'settings_hash':opt.settings_hash,
                'method':method,'seed':seed,'mode':model.mode,'fixed_gate':model.fixed_gate,
                'cap':model.cap,'history':stats},temp)
    _replace_with_retries(temp,saved)
    atomic_json(out/(stem+'_training_meta.json'),{
      'settings_hash':opt.settings_hash,'checkpoint_sha256':sha256_small_file(saved),
      'method':method,'seed':seed,'epoch':epoch,'train_metrics':stats})


def train_base08(out):
    import torch
    torch.set_num_threads(1)
    _,opt,_=manifest_for_stage(out)
    src,so,fit,validation=read_source08(out,True)
    xf,vf,tf,gf,pf,nn=fit;vx,vg,vi,vt,tk=validation
    dev=torch.device('cuda' if opt.device=='cuda' or (opt.device=='auto' and torch.cuda.is_available()) else 'cpu')
    if opt.device=='cuda' and dev.type!='cuda':raise RuntimeError('CUDA requested but not available')
    n,k=tk.shape;shape=(n,k)
    trainx=torch.from_numpy(np.array(xf,copy=True))
    trainv=torch.from_numpy(np.array(vf,copy=True)).to(dev)
    pos=torch.from_numpy(np.asarray(pf,dtype=np.int64));neg=torch.from_numpy(np.asarray(nn,dtype=np.int64))
    valx=torch.from_numpy(np.array(vx,copy=True));valv=torch.from_numpy(np.array(vg,copy=True)).to(dev)
    class08=get_pair_class08()
    rows=[]
    for seed in opt.seeds:
        validate_source_checkpoints08(src,so,seed)
        torch.manual_seed(seed)
        if dev.type=='cuda':torch.cuda.manual_seed_all(seed)
        s=torch.load(src/f'warmup_s{seed}_student.pt',map_location='cpu',weights_only=False)
        if s.get('settings_hash')!=so.settings_hash:raise ValueError('Source warmup settings mismatch')
        # A preset scalar, not fitted using Val scores, Val PIDs, or Val distribution.
        fixedgate=float(opt.fixed_gate)
        for meth in METHODS:
            mode={'fixed_gate':'fixed_gate','hard_cap':'hard_cap'}.get(meth,'standard')
            model=class08(xf.shape[1],so.hidden,so.max_delta,mode,fixedgate,opt.hard_cap).to(dev)
            opti=torch.optim.AdamW(model.parameters(),lr=so.lr,weight_decay=1e-4)
            gen=torch.Generator().manual_seed(seed+10000)
            start_epoch=1
            existing=[e for e in range(1,opt.continue_epochs+1) if check_model_epoch08(
                out,f'{meth}_s{seed}_e{e}',shape,opt)]
            consecutive=0
            for e in range(1,opt.continue_epochs+1):
                if e in existing:consecutive=e
                else:break
            if consecutive:
                prev=out/f'{meth}_s{seed}_e{consecutive}_student.pt'
                save=torch.load(prev,map_location='cpu',weights_only=False)
                model.load_state_dict(save['model_state_dict']);opti.load_state_dict(save['optimizer_state_dict'])
                gen.set_state(save['generator_state']);start_epoch=consecutive+1
                msg(f'EXP08 resumed {meth} seed={seed} after epoch={consecutive}')
            else:
                model.load_state_dict(s['model_state_dict']);opti.load_state_dict(s['optimizer_state_dict'])
            if meth=='lower_lr':
                for group in opti.param_groups:group['lr']=float(so.lr)*opt.lr_fraction
            if start_epoch>opt.continue_epochs:
                msg(f'{meth} s{seed}: complete, skip');continue
            for epoch in range(start_epoch,opt.continue_epochs+1):
                stats=epoch_train08(model,opti,gen,trainx,trainv,pos,neg,meth,so,opt,dev,fixedgate)
                hold=heldout_report08(model,src,so,opt,dev)
                stats.update(hold)
                stem=f'{meth}_s{seed}_e{epoch}'
                save_epoch08(model,opti,gen,out,stem,stats,meth,seed,epoch,opt,dev,valx,valv,tk)
                # Gate all conclusions on exact source EXP07 Rank-only replay.
                # Perform BEFORE training the other branches, rather than only at final eval.
                if meth=='exact_replay' and epoch==opt.continue_epochs:
                    replay=np.load(out/(stem+'_topk_score.npy'),mmap_mode='r')
                    original=np.load(src/f'rank_continue_s{seed}_topk_score.npy',mmap_mode='r')
                    gap=float(np.max(np.abs(np.asarray(replay)-np.asarray(original))))
                    msg(f'Exact EXP07 replay seed {seed}: maximum score error = {gap:.9g}')
                    if gap>opt.replay_tolerance:
                        raise RuntimeError(f'Exact source replay mismatch seed {seed}: {gap:.8g} > '
                                           f'{opt.replay_tolerance}. Stop before comparing changed models.')
                rows.append({'seed':seed,'method':meth,'epoch':epoch,**stats})
                msg(f'{stem}: train loss {stats["rank_loss"]:.5f}; train gate {stats["gate_mean"]:.3f}; '
                    f'holdout pair acc {hold["holdout_rank_pair_accuracy"]:.4f}')
            del model,opti
        if dev.type=='cuda':torch.cuda.empty_cache()
    # Reconstruct all epoch histories, including resumed previous work.
    allrows=[]
    for sd in opt.seeds:
        for meth in METHODS:
            for epoch in range(1,opt.continue_epochs+1):
                stem=f'{meth}_s{sd}_e{epoch}'
                if not check_model_epoch08(out,stem,shape,opt):
                    raise RuntimeError('Missing EXP08 checkpoint/predictions: '+stem)
                z=json.loads((out/(stem+'_training_meta.json')).read_text(encoding='utf-8'))
                allrows.append({'seed':sd,'method':meth,'epoch':epoch,**z['train_metrics']})
    write_stage08(out,'training_history.csv',allrows)
    msg('EXP08 all five base-only training controls complete')


def eval_base08(out):
    _,opt,_=manifest_for_stage(out)
    src,so,fit,(x,v,imgs,txt,tk)=read_source08(out,True)
    scores=np.asarray(x@v.T,dtype=np.float32)
    rr=np.arange(len(x))[:,None];bb=scores[rr,tk]
    pid=np.asarray([t['pid'] for t in txt]);gp=np.asarray([a['pid'] for a in imgs])
    baseline,br=metrics(scores,pid,gp)
    models=[];pairs=[];perquery=[]
    old={}  # old EXP07 score delta for numerical replay consistency
    for seed in opt.seeds:
        for meth in ('warmup','rank_continue'):
            stem=f'{meth}_s{seed}';arr=np.load(src/(stem+'_topk_score.npy'),mmap_mode='r')
            g=np.load(src/(stem+'_topk_gate.npy'),mmap_mode='r')
            if arr.shape!=bb.shape:raise RuntimeError('Original source TopK prediction shape wrong')
            full=scores.copy();full[rr,tk]=arr
            met,det=metrics(full,pid,gp)
            old[(seed,meth)]=(np.array(arr,copy=True),met,det)
            models.append({'method':'legacy_'+meth,'seed':seed,'epoch':-1,**met,
                  'delta_R1_vs_IRRA':met['R1']-baseline['R1'],'gate_mean':float(g.mean()),
                  'delta_abs_mean':float(np.abs(np.asarray(arr)-bb).mean()),
                  'note':'Unchanged EXP07 stored prediction; no new training'})
        for meth in METHODS:
            for ep in range(1,opt.continue_epochs+1):
                stem=f'{meth}_s{seed}_e{ep}'
                if not check_model_epoch08(out,stem,bb.shape,opt):
                    raise RuntimeError('Missing/incomplete EXP08 prediction: '+stem)
                a=np.load(out/(stem+'_topk_score.npy'),mmap_mode='r')
                g=np.load(out/(stem+'_topk_gate.npy'),mmap_mode='r')
                test=scores.copy();test[rr,tk]=a
                m,d=metrics(test,pid,gp)
                ref=old[(seed,'rank_continue')][1]
                legacy_delta=float(np.max(np.abs(np.asarray(a)-old[(seed,'rank_continue')][0]))) if meth=='exact_replay' else None
                models.append({'method':meth,'seed':seed,'epoch':ep,**m,
                  'delta_R1_vs_IRRA':m['R1']-baseline['R1'],
                  'delta_R1_vs_legacy_continue':m['R1']-ref['R1'],
                  'gate_mean':float(g.mean()),'delta_abs_mean':float(np.abs(np.asarray(a)-bb).mean()),
                  'exact_replay_score_max_abs_error':legacy_delta if ep==opt.continue_epochs else None})
                if meth=='exact_replay' and ep==opt.continue_epochs:
                    if legacy_delta>opt.replay_tolerance:
                        raise RuntimeError('EXP08 source replay DOES NOT MATCH original EXP07: '+
                            f'{legacy_delta:.8g} > {opt.replay_tolerance}. Cannot attribute controls until reproducibility is fixed.')
                was=br['rank']==1;now=d['rank']==1
                for idx in np.where(was!=now)[0]:
                    perquery.append({'seed':seed,'method':meth,'epoch':ep,'query_idx':int(idx),
                        'pid_evaluation_only':int(pid[idx]),'original_rank':int(br['rank'][idx]),
                        'rank_new':int(d['rank'][idx]),'repaired':int(not was[idx] and now[idx]),
                        'harmed':int(was[idx] and not now[idx])})
                if ep==opt.continue_epochs:
                    pairs.append({'seed':seed,'method':meth,'repairs_vs_IRRA':int((~was&now).sum()),
                          'harms_vs_IRRA':int((was&~now).sum()),
                          'net_new_correct_vs_IRRA':int(now.sum()-was.sum()),
                          'R1_vs_legacy_continue':m['R1']-ref['R1']})
    write_stage08(out,'method_summary.csv',models)
    write_stage08(out,'case_changes.csv',perquery)
    write_stage08(out,'paired_rank_comparisons.csv',pairs)
    agg=[]
    for meth in METHODS:
        for epoch in range(1,opt.continue_epochs+1):
            subset=[r for r in models if r['method']==meth and r['epoch']==epoch]
            obj={'method':meth,'epoch':epoch,'n_seeds':len(subset)}
            for field in ('R1','R5','R10','mAP','mINP','gate_mean','delta_abs_mean'):
                vals=np.asarray([row[field] for row in subset],dtype=np.float64)
                obj[field+'_mean']=float(vals.mean());obj[field+'_sd']=float(vals.std(ddof=1)) if len(vals)>1 else 0.
            agg.append(obj)
    write_stage08(out,'aggregate_summary.csv',agg)
    legacy=list(csv.DictReader(open(out/'legacy_scale_diagnostics.csv',encoding='utf-8-sig',newline='')))
    atomic_json(out/'experiment08_report.json',{
       'title':'EXP08: Base reranking architecture failure attribution — NO text complementarity training',
       'irra_baseline':baseline,'models':agg,
       'legacy_scale_probe_rows':len(legacy),'training_identities':len(so.fit_pids),
       'internal_unseen_training_identities':len(so.holdout_pids),
       'no_val_pid_at_inference':True,'model_selection_policy':'DIAGNOSTIC ONLY; no adaptive selection via repeatedly used Val',
       'conclusion_policy':['Inspect exact_replay equivalence first','Inspect legacy alpha shrinkage causal score distortion',
             'Compare same warmup + optimizer reset-free controlled branches','Use holdout train IDs to distinguish train-vs-unseen trend',
             'Do not claim overfitting without same-protocol fit and unseen evidence',
             'Report full-gallery fixed TopK re-ranking, not IRRA independent text/image encoding',
             'Do not attribute effects to same-PID multiple descriptions; this experiment trains NONE'],
       'research_objectives':['1. Validated text complementarity beyond regularization','2. Validated multiview image complementarity','3. Strict joint fusion']})
    msg('EXP08 full-gallery evaluation and audit complete.')


def package08(out):
    out=Path(out);zpath=out/'发送给ChatGPT.zip'
    fd,temp=_new_temp_path(zpath);os.close(fd)
    try:
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(out.iterdir()):
                if p.is_file() and p.suffix.lower() in ('.csv','.json','.md','.py','.log','.txt'):
                    z.write(p,p.name)
        _replace_with_retries(temp,zpath)
    finally:
        try:temp.unlink(missing_ok=True)
        except OSError:pass
    msg(f'EXP08 analysis ZIP saved: {zpath} ({zpath.stat().st_size/1024/1024:.2f} MiB)')


def run_worker08(stage,out):
    if stage=='audit':np_diagnostics08(out)
    elif stage=='train':train_base08(out)
    elif stage=='eval':eval_base08(out)
    else:raise ValueError('Bad exp08 stage '+stage)


def self_test08():
    import torch
    rng=np.random.default_rng(808)
    with tempfile.TemporaryDirectory(prefix='exp08_full_test_') as td:
        home=Path(td);source=home/'source_exp07';source.mkdir();out=home/'exp08';out.mkdir()
        d=20;data=[];xx=[];vv=[];vx=[];vg=[]
        for pid in range(1,13):
            center=normalize(rng.standard_normal(d).astype(np.float32))
            for view in range(2):
                data.append({'split':'train','id':pid,'file_path':f'{pid}_{view}.png',
                     'captions':[f'a person number {pid} one',f'a person number {pid} two']})
                vv.append(normalize(center+.18*rng.normal(size=d)))
                for cap in range(2):xx.append(normalize(center+.22*rng.normal(size=d)))
        for pid in range(50,55):
            center=normalize(rng.standard_normal(d).astype(np.float32))
            for view in range(2):
                data.append({'split':'val','id':pid,'file_path':f'{pid}_{view}.png',
                     'captions':[f'val {pid} person one',f'val {pid} person two']})
                vg.append(normalize(center+.18*rng.normal(size=d)))
                for cap in range(2):vx.append(normalize(center+.22*rng.normal(size=d)))
        ann=source/'synthetic_annotations.json';atomic_json(ann,data)
        selected=list(range(1,13));fit,hold=pid_split(selected,.75,776)
        so={'settings_hash':'source_mock_08','project_root':str(home),'annotation_file':str(ann),
            'selected_pids':selected,'fit_pids':fit,'holdout_pids':hold,
            'val_cache_dir':'','train_text_cache_dir':'','train_image_cache_dir':'',
            'fp_train':{'mock08':'t'},'fp_val':{'mock08':'v'},'hidden':16,'max_delta':.08,
            'hard_k':2,'train_batch':12,'rank_weight':.2,'protect_weight':.5,
            'margin':.05,'temperature':.07,'lr':.0003,'device':'cpu'}
        atomic_json(source/'run_info.json',{'settings':so});atomic_json(source/'selected_train_pids.json',selected)
        for name,arr,fp in [('train_text',xx,so['fp_train']),('train_image',vv,train_img_fingerprint(SimpleNamespace(fp_train=so['fp_train']),list(range(len(vv))))),('text',vx,so['fp_val']),('image',vg,so['fp_val'])]:
            a=np.asarray(arr,dtype=np.float32);np.save(source/(name+'_features.npy'),a)
            atomic_json(source/(name+'_cache_meta.json'),{'fingerprint':fp,'done':len(a),
                        'shape':list(a.shape)})
        x=np.asarray(xx,dtype=np.float32);v=np.asarray(vv,dtype=np.float32)
        ims,txt,_,_=get_subset(ann,selected)
        x1,v1,t1,i1,p1,_,_=partition(x,v,read_annotations(ann,'train')[1],read_annotations(ann,'train')[0],
                                   np.repeat(np.arange(len(v)),2),fit)
        hx,hv,ht,hi,hp,_,_=partition(x,v,read_annotations(ann,'train')[1],read_annotations(ann,'train')[0],
                                     np.repeat(np.arange(len(v)),2),hold)
        for nm,arr in [('fit_negative_indices',hard_negative_mine(x1,v1,np.asarray([r['pid'] for r in t1]),np.asarray([r['pid'] for r in i1]),2,16)),
                       ('holdout_negatives',hard_negative_mine(hx,hv,np.asarray([r['pid'] for r in ht]),np.asarray([r['pid'] for r in hi]),2,16))]:
            np.save(source/(nm+'.npy'),arr)
        base=normalize(np.asarray(vx,dtype=np.float32))@normalize(np.asarray(vg,dtype=np.float32)).T
        np.save(source/'val_topk_indices.npy',np.argsort(-base,axis=1,kind='stable')[:,:5].astype(np.int32))
        torch.set_num_threads(1)
        klass=get_pair_class08();
        for seed in (42,43):
            torch.manual_seed(seed);model=klass(d,16,.08)
            optimizer=torch.optim.AdamW(model.parameters(),lr=so['lr'],weight_decay=1e-4)
            stx=torch.from_numpy(np.array(x1,copy=True));sv=torch.from_numpy(np.array(v1,copy=True));
            sp=torch.from_numpy(np.asarray(p1,dtype=np.int64));sn=torch.from_numpy(np.load(source/'fit_negative_indices.npy').astype(np.int64))
            gen=torch.Generator().manual_seed(seed)
            for epoch in (1,2):
                epoch_train08(model,optimizer,gen,stx,sv,sp,sn,'exact_replay',SimpleNamespace(**so),
                              SimpleNamespace(protect_factor=10),torch.device('cpu'),0.5)
            stem=f'warmup_s{seed}'
            torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                        'settings_hash':so['settings_hash']},source/(stem+'_student.pt'))
            for meth in ('warmup','rank_continue'):
                meta=f'{meth}_s{seed}';
                if meth=='rank_continue':
                    gen=torch.Generator().manual_seed(seed+10000)
                    for epoch in (1,2):
                        epoch_train08(model,optimizer,gen,stx,sv,sp,sn,'exact_replay',SimpleNamespace(**so),
                                      SimpleNamespace(protect_factor=10),torch.device('cpu'),0.5)
                    torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                                 'settings_hash':so['settings_hash']},source/(meta+'_student.pt'))
                xval=torch.from_numpy(np.array(vx,dtype=np.float32));vval=torch.from_numpy(np.array(vg,dtype=np.float32))
                make_predictions08(model,xval,vval,np.load(source/'val_topk_indices.npy'),torch.device('cpu'),source,meta,16)
                atomic_json(source/(meta+'_training_meta.json'),{'settings_hash':so['settings_hash'],
                   'checkpoint_sha256':sha256_small_file(source/(meta+'_student.pt'))})
        settings={'source_exp07':str(source),'device':'cpu','seeds':[42,43],
                  'continue_epochs':2,'protect_factor':10.,'lr_fraction':.25,
                  'hard_cap':.02,'fixed_gate':.5,'eval_query_batch':16,'expected_baseline_r1':None,
                  'replay_tolerance':1e-5,'settings_hash':'exp08_synth'}
        atomic_json(out/'run_info.json',{'settings':settings})
        # Real subprocess isolation for audit (NumPy) and training (PyTorch).
        sc=Path(__file__).resolve();env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
        for stage in ('audit','train','eval'):
            done=subprocess.run([sys.executable,str(sc),'--_worker',stage,'--_run-dir',str(out)],
                   env=env,capture_output=True,text=True,timeout=220)
            if done.returncode:raise AssertionError(f'EXP08 self-test {stage} failed\n{done.stdout[-2500:]}\n{done.stderr[-4000:]}')
        files=list(out.glob('*_training_meta.json'))
        assert len(files)==2*5*2,len(files)
        # Simulate a Windows interruption AFTER a completed epoch-1 snapshot.
        # Rebuild only the unfinished epoch-2, retaining sampler+AdamW state.
        snapshot='lower_lr_s42_e2'
        expected_sha=sha256_small_file(out/(snapshot+'_topk_score.npy'))
        for suffix in ('_student.pt','_training_meta.json','_topk_score.npy','_topk_gate.npy'):
            (out/(snapshot+suffix)).unlink()
        broken=subprocess.run([sys.executable,str(sc),'--_worker','train','--_run-dir',str(out)],
               env=env,capture_output=True,text=True,timeout=220)
        if broken.returncode:raise AssertionError('EXP08 interrupted-epoch resume failed:\n'+broken.stdout[-2000:]+broken.stderr[-2000:])
        assert sha256_small_file(out/(snapshot+'_topk_score.npy'))==expected_sha, 'Resume changed predictions'
        for stage in ('train','eval'):
            done=subprocess.run([sys.executable,str(sc),'--_worker',stage,'--_run-dir',str(out)],
                   env=env,capture_output=True,text=True,timeout=220)
            if done.returncode:raise AssertionError(f'EXP08 resume {stage} failed\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}')
        shutil.copy2(sc,out/sc.name);package08(out)
        with zipfile.ZipFile(out/'发送给ChatGPT.zip') as zip:
            assert 'method_summary.csv' in zip.namelist()
            assert 'legacy_scale_diagnostics.csv' in zip.namelist()
            assert 'training_history.csv' in zip.namelist()
            assert not any(p.endswith(('.npy','.pt')) for p in zip.namelist())
    msg('EXP08 SELF-TEST PASSED: reproducibility, 5 controls x 2 seeds x 2 epochs, PID blind eval, resume, ZIP.')


def main08():
    global LOG_FILE
    p=argparse.ArgumentParser(description='EXP08 frozen-IRRA base reranker diagnostic; source EXP07 required.')
    p.add_argument('--project-root',default=DEFAULT_ROOT)
    p.add_argument('--source-exp07',default='')
    p.add_argument('--seeds',default='42,43,44')
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    p.add_argument('--continue-epochs',type=int,default=2)
    p.add_argument('--protect-factor',type=float,default=10.)
    p.add_argument('--lr-fraction',type=float,default=.25)
    p.add_argument('--hard-cap',type=float,default=.02)
    p.add_argument('--fixed-gate',type=float,default=.5)
    p.add_argument('--eval-query-batch',type=int,default=64)
    p.add_argument('--replay-tolerance',type=float,default=.002)
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--_worker',default='',help=argparse.SUPPRESS)
    p.add_argument('--_run-dir',default='',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a._worker:
        try:run_worker08(a._worker,Path(a._run_dir))
        except Exception:traceback.print_exc();sys.exit(1)
        return
    if a.self_test:self_test08();return
    root=Path(a.project_root).resolve()
    if not root.is_dir():p.error('Project root does not exist: '+str(root))
    try:seeds=[int(s.strip()) for s in a.seeds.split(',') if s.strip()]
    except ValueError:p.error('Invalid seeds')
    if len(set(seeds))!=len(seeds) or not seeds or min(seeds)<0:p.error('Provide distinct nonnegative seeds')
    if not (a.continue_epochs>=1 and a.protect_factor>1 and 0<a.lr_fraction<1 and
            0<a.hard_cap<.08 and 0<=a.fixed_gate<=1 and a.eval_query_batch>=1 and a.replay_tolerance>0):
        p.error('Invalid experiment08 hyperparameters')
    parent=root/PROJECT_FOLDER_NAME
    src=Path(a.source_exp07).resolve() if a.source_exp07 else None
    if src is None:
        folders=sorted((q for q in parent.glob(SOURCE_PREFIX08+'*') if q.is_dir() and
                        (q/'run_info.json').is_file()),key=lambda q:q.name,reverse=True)
        for q in folders:
            try:
                opts=json.loads((q/'run_info.json').read_text(encoding='utf-8'))['settings']
                if all((q/f'warmup_s{seed}_student.pt').is_file() and
                       (q/f'rank_continue_s{seed}_student.pt').is_file() for seed in seeds):
                    src=q;break
            except (OSError,ValueError,KeyError):continue
        if src is None:p.error('No complete EXP07 source found; provide --source-exp07 "..."')
    source=src
    old=json.loads((source/'run_info.json').read_text(encoding='utf-8'))['settings']
    if not set(seeds).issubset(set(old.get('seeds',[]))):p.error('Selected seeds missing in source EXP07')
    if not all((source/f'warmup_s{s}_student.pt').is_file() for s in seeds):p.error('Source warmup models missing')
    settings={'source_exp07':str(source),'source_settings_hash':old['settings_hash'],
        'device':a.device,'seeds':seeds,'continue_epochs':a.continue_epochs,
        'protect_factor':a.protect_factor,'lr_fraction':a.lr_fraction,'hard_cap':a.hard_cap,
        'fixed_gate':a.fixed_gate,
        'eval_query_batch':a.eval_query_batch,'expected_baseline_r1':56.83663527119195 if (
                len(read_annotations(Path(old['annotation_file']),'val')[1])==6158) else None,
        'replay_tolerance':a.replay_tolerance,'version':VERSION08}
    settings['settings_hash']=hashlib.sha256(json.dumps(settings,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    out,resumed=new_or_resume(parent,a,settings)
    LOG_FILE=out/'experiment_run.log'
    if not resumed:
        atomic_json(out/'run_info.json',{'experiment':EXPERIMENT_NAME,'settings':settings})
    shutil.copy2(Path(__file__).resolve(),out/Path(__file__).name)
    (out/'实验08_目的与判定边界.md').write_text(
      '# 实验08：基础重排序故障归因\n\n'
      '- 本实验无多文本教师、无可靠性文本选择训练，属于基础网络专项诊断。\n'
      '- 直接使用实验07原始Warmup权重和同一AdamW优化器状态；训练Fit PID、困难负样本、训练顺序一致。\n'
      '- 先重放原始预测，测量把分数修正从1缩小到0时检索怎样变化，避免把评分失真说成过拟合。\n'
      '- 五个单因素训练对照：完全重放、后续低学习率、更强修正保护、固定Gate、硬限制修正幅度。\n'
      '- 真实是否过拟合只能对比训练与内部未见Train身份的表现，不能凭一次Val下降断定。\n'
      '- 指标为原始IRRA固定TopK候选重排后的完整图库检索；检索分数不使用Val PID。\n'
      '- 不得用已多次开发使用的Val最优模型当作独立Test性能。\n',encoding='utf-8')
    try:
        for st in (('audit',) if a.audit_only else ('audit','train','eval')):
            run_subprocess(st,out,root)
        msg('EXP08 finished (audit only)' if a.audit_only else 'EXP08 finished training/analysis.')
    except BaseException as e:
        msg('EXP08 FAILED: '+repr(e));traceback.print_exc();raise
    finally:
        package08(out)


if __name__=='__main__':main08()
