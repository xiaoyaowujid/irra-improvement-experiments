# -*- coding: utf-8 -*-
"""实验04：梯度匹配与身份真实性负对照（冻结 IRRA，三个训练随机种子）

Train-only PID multi-caption teachers + frozen train images + hard negatives;
warm up one Rank student per seed, then fork identical weights AND optimizer state.
Validation: EXACTLY one caption/query, original fixed IRRA image gallery;
validation PIDs are used exclusively by evaluator AFTER prediction.

PyCharm: put this .py under IRRA-main; default run with no parameters.
--self-test: synthetic end-to-end verification, no real dataset/model needed.
--resume: resume a matching timestamped run, skip completed encodings/methods.
--max-train-pids 0: use ALL training identities (slower; default 2200).
--val-cache-dir / --train-text-cache-dir: optional explicit EXP00/EXP01 caches.
Every run emits 图文双向互补实验/实验04_..._YYYYMMDD_HHMMSS/发送给ChatGPT.zip.
Never set KMP_DUPLICATE_LIB_OK; workers isolate IRRA/Torch from NumPy eval.
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
EXPERIMENT_NAME = '实验04_梯度匹配与身份负对照'
METHODS = ('rank_warmup','rank_continue','relation_self','protect_extra','relation_cross','wrongpid_matched','wrongpid_raw')
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
        raise FileNotFoundError('No matching experiment04 run to resume. Remove --resume for a new run.')
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


# ====================== EXP04 controlled diagnostics ======================

def make_teachers04(out, opt, xf, vf, texts, pos, neg):
    """Cross/PID-wrong controls share support counts, and MATCHED has identical
    per-anchor angular distance to its true cross-image teacher (not merely same alpha).
    """
    pids=np.array([r['pid'] for r in texts],dtype=np.int64)
    imgkey=np.array([r['img_idx'] for r in texts],dtype=np.int64)
    bypid=defaultdict(list)
    for i,p in enumerate(pids):bypid[int(p)].append(i)
    rng=np.random.default_rng(opt.random_teacher_seed)
    available={int(p):np.flatnonzero(pids!=p) for p in np.unique(pids)}
    n,d=xf.shape
    names=('relation_cross','wrongpid_matched','wrongpid_raw')
    matrices={name:np.lib.format.open_memmap(out/(name+'_teacher.npy'),mode='w+',dtype='float32',shape=(n,d))
              for name in names}
    rows=[]
    aggregates={name:{'sum_angle':0.,'sum_train_gain':0.,'sum_hold_gain':0.,
                      'count_improved_train':0,'count_improved_hold':0} for name in names}
    angular_errors=[]
    count_support=0;count_mismatch=0;total_support=0
    for i in range(n):
        anchor=xf[i]; pid=int(pids[i]); sample_image=int(imgkey[i]);
        peers=[j for j in bypid[pid] if int(imgkey[j])!=sample_image]
        peers=capped_peer_indices(peers,anchor,xf,opt.max_support)
        count_support+=bool(peers);total_support+=len(peers)
        if peers:
            other=list(map(int,rng.choice(available[pid],len(peers),replace=False)))
            if not all(int(pids[j])!=pid for j in other):
                raise AssertionError('Wrong-PID teacher has same-PID source!')
            count_mismatch+=1
            cross=normalize((1-opt.teacher_strength)*anchor+
                           opt.teacher_strength*xf[peers].mean(axis=0))
            raw=normalize((1-opt.teacher_strength)*anchor+
                         opt.teacher_strength*xf[other].mean(axis=0))
            c=float(np.clip(anchor@cross,-1.,1.))
            tangent=np.asarray(raw-(raw@anchor)*anchor,dtype=np.float64)
            tn=float(np.linalg.norm(tangent))
            if tn<1e-10:
                tangent=rng.normal(size=d).astype(np.float64)
                tangent-=float(tangent@anchor)*anchor
                tn=float(np.linalg.norm(tangent))
            assert tn>1e-10
            matched=normalize(np.asarray(c*anchor+math.sqrt(max(0.,1-c*c))*(tangent/tn),dtype=np.float32))
            err=float(abs(np.linalg.norm(matched-anchor)-np.linalg.norm(cross-anchor)))
            angular_errors.append(err)
            if err>2e-4:raise AssertionError(f'Teacher angular mismatch: {err}')
        else:cross=raw=matched=anchor
        variants={'relation_cross':cross,'wrongpid_matched':matched,'wrongpid_raw':raw}
        vpos=vf[pos[i]]
        negtrain=vf[neg[i,:opt.hard_k]]
        neghold=vf[neg[i,opt.hard_k:]]
        baseline_train=float(anchor@vpos-np.max(negtrain@anchor))
        baseline_hold=float(anchor@vpos-np.max(neghold@anchor))
        example={'query_idx':i,'train_pid':pid,'anchor_text':texts[i]['text'],
                 'num_cross_support':len(peers),
                 'cross_texts':' || '.join(texts[j]['text'] for j in peers),
                 'wrong_pid_texts':' || '.join(texts[j]['text'] for j in other) if peers else '',
                 'angular_match_abs_error':angular_errors[-1] if peers else 0.}
        for name,teacher in variants.items():
            matrices[name][i]=teacher
            agg=aggregates[name]
            angle=1-float(anchor@teacher)
            train=float(teacher@vpos-np.max(negtrain@teacher)-baseline_train)
            hold=float(teacher@vpos-np.max(neghold@teacher)-baseline_hold)
            agg['sum_angle']+=angle;agg['sum_train_gain']+=train;agg['sum_hold_gain']+=hold
            agg['count_improved_train']+=int(train>0);agg['count_improved_hold']+=int(hold>0)
            example[name+'_cos_to_anchor']=float(anchor@teacher)
            example[name+'_train_margin_gain']=train
            example[name+'_held_negative_margin_gain']=hold
        if i%max(1,n//130)==0 or len(rows)<30:
            if len(rows)<150:rows.append(example)
        if (i+1)%5000==0:msg(f'EXP04 teachers {i+1}/{n}')
    for m in matrices.values():m.flush()
    if rows:write_csv(out/'teacher_examples.csv',rows,list(rows[0]))
    for a in aggregates.values():
        a['mean_one_minus_cos']=a.pop('sum_angle')/n
        a['mean_train_margin_gain']=a.pop('sum_train_gain')/n
        a['mean_held_neg_margin_gain']=a.pop('sum_hold_gain')/n
        a['train_margin_improved_rate']=a.pop('count_improved_train')/n
        a['held_neg_margin_improved_rate']=a.pop('count_improved_hold')/n
    stats={'num_train_texts':n,'num_train_images':len(vf),'train_pid_count':len(bypid),
           'cross_support_coverage':count_support/n,'mean_cross_support_count':total_support/n,
           'wrong_pid_count_verified':count_mismatch,
           'angular_distance_match_mean_abs_error':float(np.mean(angular_errors)) if angular_errors else 0.,
           'angular_distance_match_max_abs_error':float(np.max(angular_errors)) if angular_errors else 0.,
           'sources':aggregates,'uses_train_pid_for_support':True,
           'note':'Same frozen train negatives for comparisons. Holdout negatives are different images from the same train gallery, not independent identities.'}
    atomic_json(out/'teacher_diagnostics.json',stats)
    msg('Teacher angular-match max error: '+str(stats['angular_distance_match_max_abs_error']))
    msg('Teacher margin gains: '+str({k:round(v['mean_held_neg_margin_gain'],5) for k,v in aggregates.items()}))


def stage_evidence04(out):
    _,opt,out=manifest_for_stage(out)
    completion=out/'evidence_complete.json'
    needed=['hard_negative_indices.npy','positive_gallery_idx.npy',
            'relation_cross_teacher.npy','wrongpid_matched_teacher.npy','wrongpid_raw_teacher.npy',
            'teacher_diagnostics.json']
    if completion.is_file() and all((out/n).is_file() for n in needed):
        m=json.loads(completion.read_text(encoding='utf-8'))
        if m.get('settings_hash')==opt.settings_hash:
            msg('Verified complete teacher evidence cache; skip.');return
    xf,vf,texts,images,pos=train_arrays03(out,opt)
    pids=np.array([t['pid'] for t in texts]);ipids=np.array([v['pid'] for v in images])
    neg=hard_negative_mine(xf,vf,pids,ipids,opt.hard_k*2,opt.mine_batch)
    assert not np.any(pids[:,None]==ipids[neg])
    np.save(out/'hard_negative_indices.npy',neg)
    np.save(out/'positive_gallery_idx.npy',pos)
    make_teachers04(out,opt,xf,vf,texts,pos,neg)
    atomic_json(completion,{'settings_hash':opt.settings_hash,'time':datetime.now().isoformat()})


def _grad_norm(grads):
    import torch
    return math.sqrt(sum(float(g.detach().square().sum()) for g in grads if g is not None))


def _grad_cos(g1,g2,n1,n2):
    if n1<1e-12 or n2<1e-12:return float('nan')
    return float(sum((a.detach()*b.detach()).sum().item() for a,b in zip(g1,g2)
                    if a is not None and b is not None)/(n1*n2))


def _train_epoch04(adapter,optimizer,order,trainx,gallery,pos,neg,method,teacher,opt,device,measure):
    import torch
    import torch.nn.functional as F
    adapter.train()
    params=list(adapter.parameters())
    totals=defaultdict(float)
    counts=defaultdict(int)
    n=len(order)
    for start in range(0,n,opt.train_batch):
        inds=order[start:start+opt.train_batch]
        x=trainx[inds].to(device)
        z=adapter(x)
        options=torch.cat([pos[inds,None],neg[inds]],dim=1).to(device)
        im=gallery[options]
        pred=torch.einsum('bkd,bd->bk',im,z)
        l_rank=F.softplus((pred[:,1:]-pred[:,:1]+opt.margin)/opt.temperature).mean()
        l_protect=(1-(z*x).sum(-1)).mean()
        base=opt.rank_weight*l_rank+opt.protect_weight*l_protect
        extra=torch.zeros((),device=device)
        aux_coef=0.;ratio=0.;aux_cos=float('nan');raw_grad=0.;base_grad=0.;clipped=0
        if method not in ('rank_warmup','rank_continue'):
            if method=='protect_extra':
                extra=l_protect
            else:
                t=teacher[inds].to(device) if teacher is not None else x
                tgt=torch.einsum('bkd,bd->bk',im,t)
                dz=pred[:,:1]-pred[:,1:]
                dt=(tgt[:,:1]-tgt[:,1:]).detach()
                extra=F.smooth_l1_loss(dz,dt)
            if measure:
                gb=torch.autograd.grad(base,params,retain_graph=True,allow_unused=True)
                ga=torch.autograd.grad(extra,params,retain_graph=True,allow_unused=True)
                base_grad=_grad_norm(gb);raw_grad=_grad_norm(ga)
                aux_cos=_grad_cos(gb,ga,base_grad,raw_grad)
                if base_grad>=opt.min_grad_norm and raw_grad>=opt.min_grad_norm:
                    aux_coef=min(opt.aux_grad_ratio*base_grad/raw_grad,opt.max_aux_scale)
                else:counts['zero_grad_batches']+=1
                ratio=aux_coef*raw_grad/max(base_grad,1e-12)
            else:aux_coef=opt.relation_weight
        loss=base+float(aux_coef)*extra
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_norm=torch.nn.utils.clip_grad_norm_(params,5.0)
        clipped=float(clip_norm>5.)
        optimizer.step()
        w=len(inds)
        values={'rank_loss':l_rank.item(),'protect_loss':l_protect.item(),
                'aux_loss':extra.item(),'total_loss':loss.item(),
                'effective_aux_coef':aux_coef,'realized_aux_ratio':ratio,
                'base_grad_norm':base_grad,'raw_aux_grad_norm':raw_grad,
                'gradient_clipped_rate':clipped}
        for name,val in values.items():totals[name]+=float(val)*w
        if math.isfinite(aux_cos):
            totals['grad_dot_cos']+=aux_cos*w;counts['grad_dot_cos_count']+=w
        counts['batches']+=1
        if aux_coef>=opt.max_aux_scale-1e-9 and method not in ('rank_warmup','rank_continue'):
            counts['aux_coefficient_cap_batches']+=1
    row={k:float(v)/n for k,v in totals.items()}
    row['grad_dot_cos']=float(totals['grad_dot_cos']/counts['grad_dot_cos_count']) if counts['grad_dot_cos_count'] else None
    row['zero_grad_batches']=counts['zero_grad_batches']
    row['aux_coefficient_cap_batches']=counts['aux_coefficient_cap_batches']
    row['num_batches']=counts['batches']
    return row


def _commit_model04(out,adapter,optimizer,stem,history,seed,method,valx,opt,device,kind='continue'):
    import torch
    # Persist features, checkpoint (including warmup optimizer state), THEN metadata.
    feature=out/(stem+'_val_features.npy')
    temporary=out/(stem+'_val_features.partial.npy')
    batch=opt.train_batch
    mm=np.lib.format.open_memmap(temporary,mode='w+',dtype='float32',shape=tuple(valx.shape))
    adapter.eval()
    with torch.inference_mode():
        for start in range(0,len(valx),batch):
            xb=valx[start:start+batch].to(device)
            mm[start:start+len(xb)]=adapter(xb).cpu().numpy()
    mm.flush();del mm
    _replace_with_retries(temporary,feature)
    ck=out/(stem+'_student.pt')
    temp=out/(stem+'_student.partial.pt')
    # Optimizer state needed to fork all continuation models identically.
    torch.save({'state_dict':adapter.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                'seed':seed,'method':method,'dim':valx.shape[1],
                'hidden':opt.hidden,'gain':opt.gain,'settings_hash':opt.settings_hash},temp)
    _replace_with_retries(temp,ck)
    atomic_json(out/(stem+'_training_meta.json'),{
        'complete':True,'settings_hash':opt.settings_hash,'method':method,'seed':int(seed),
        'phase':kind,'history':history,
        'checkpoint_sha256':sha256_small_file(ck),
        'features_shape':list(valx.shape)})


def _model_done04(out,opt,stem,val_shape):
    files=[out/(stem+'_student.pt'),out/(stem+'_training_meta.json'),out/(stem+'_val_features.npy')]
    if not all(f.is_file() for f in files):return False
    try:
        x=json.loads(files[1].read_text(encoding='utf-8'))
        return (x.get('complete') and x.get('settings_hash')==opt.settings_hash and
                x.get('features_shape')==list(val_shape) and
                tuple(np.load(files[2],mmap_mode='r',allow_pickle=False).shape)==tuple(val_shape) and
                sha256_small_file(files[0])==x.get('checkpoint_sha256'))
    except (OSError,ValueError,KeyError):return False


def stage_train04(out):
    import torch
    torch.set_num_threads(1)
    _,opt,out=manifest_for_stage(out)
    device=torch.device('cuda' if opt.device=='cuda' or (opt.device=='auto' and torch.cuda.is_available()) else 'cpu')
    if opt.device=='cuda' and device.type!='cuda':raise RuntimeError('CUDA requested but unavailable')
    xf,vf,texts,images,pos_idx=train_arrays03(out,opt)
    n,d=xf.shape
    valdir=Path(opt.val_cache_dir) if opt.val_cache_dir else out
    xval=normalize(np.asarray(np.load(valdir/'text_features.npy',mmap_mode='r'),dtype=np.float32))
    if xval.shape[1]!=d:raise ValueError('Embedding mismatch between train and val')
    negidx=np.load(out/'hard_negative_indices.npy',allow_pickle=False)
    if negidx.shape!=(n,2*opt.hard_k):raise ValueError('Incompatible hard-negative cache')
    trainx=torch.from_numpy(xf.copy());valx=torch.from_numpy(xval.copy())
    gallery=torch.from_numpy(vf.copy()).to(device)
    pos=torch.from_numpy(pos_idx.copy())
    neg=torch.from_numpy(np.asarray(negidx[:,:opt.hard_k],dtype=np.int64).copy())
    teachers={key:torch.from_numpy(np.asarray(np.load(out/(key+'_teacher.npy'),mmap_mode='r'),dtype=np.float32).copy())
              for key in ('relation_cross','wrongpid_matched','wrongpid_raw')}
    cls=get_adapter_exp02()
    for seed in opt.seeds:
        warmstem=f'rank_warmup_s{seed}'
        if not _model_done04(out,opt,warmstem,tuple(valx.shape)):
            torch.manual_seed(seed)
            if device.type=='cuda':torch.cuda.manual_seed_all(seed)
            adapter=cls(d,opt.hidden,opt.gain).to(device)
            optim=torch.optim.AdamW(adapter.parameters(),lr=opt.lr,weight_decay=1e-4)
            rng=torch.Generator().manual_seed(seed)
            history=[]
            for epoch in range(1,opt.warmup_epochs+1):
                perm=torch.randperm(n,generator=rng)
                stats=_train_epoch04(adapter,optim,perm,trainx,gallery,pos,neg,'rank_warmup',None,opt,device,True)
                history.append({'method':'rank_warmup','seed':seed,'phase':'warmup','epoch':epoch,**stats})
                msg(f'rank_warmup_s{seed} epoch {epoch}/{opt.warmup_epochs}: rank={stats["rank_loss"]:.5f} protect={stats["protect_loss"]:.6f}')
            _commit_model04(out,adapter,optim,warmstem,history,seed,'rank_warmup',valx,opt,device,'warmup')
            del adapter,optim
        else:msg(f'{warmstem}: validated completed checkpoint, skip.')
        checkpoint=torch.load(out/(warmstem+'_student.pt'),map_location='cpu',weights_only=False)
        if checkpoint['settings_hash']!=opt.settings_hash:
            raise ValueError('Warmup checkpoint fingerprint mismatch')
        for method in METHODS[1:]:
            stem=f'{method}_s{seed}'
            if _model_done04(out,opt,stem,tuple(valx.shape)):
                msg(f'{stem}: validated completed checkpoint, skip.');continue
            torch.manual_seed(seed)
            if device.type=='cuda':torch.cuda.manual_seed_all(seed)
            adapter=cls(d,opt.hidden,opt.gain).to(device)
            adapter.load_state_dict(checkpoint['state_dict'])
            optim=torch.optim.AdamW(adapter.parameters(),lr=opt.lr,weight_decay=1e-4)
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            permgen=torch.Generator().manual_seed(seed+10000)
            teacher=teachers.get(method)
            history=[]
            for epoch in range(1,opt.continue_epochs+1):
                perm=torch.randperm(n,generator=permgen)
                stats=_train_epoch04(adapter,optim,perm,trainx,gallery,pos,neg,method,teacher,opt,device,True)
                history.append({'method':method,'seed':seed,'phase':'continue','epoch':epoch,**stats})
                msg(f'{stem} epoch {epoch}/{opt.continue_epochs}: rank={stats["rank_loss"]:.5f} aux={stats["aux_loss"]:.6f} ratio={stats["realized_aux_ratio"]:.3f} cos={stats["grad_dot_cos"]}')
            _commit_model04(out,adapter,optim,stem,history,seed,method,valx,opt,device)
            del adapter,optim
            if device.type=='cuda':torch.cuda.empty_cache()
    records=all_training_records(out)
    if records:
        cols=list(dict.fromkeys(k for r in records for k in r))
        write_csv(out/'training_history.csv',records,cols,allow_locked_fallback=True)
    msg('EXP04 students trained; optimizer-state matched branches completed.')


def evaluate04(out):
    _,opt,out=manifest_for_stage(out)
    val_ims,val_txt,_=read_annotations(Path(opt.annotation_file),'val')
    valdir=Path(opt.val_cache_dir) if opt.val_cache_dir else out
    original=normalize(np.asarray(np.load(valdir/'text_features.npy',mmap_mode='r'),dtype=np.float32))
    gallery=normalize(np.asarray(np.load(valdir/'image_features.npy',mmap_mode='r'),dtype=np.float32))
    qpid=np.asarray([t['pid'] for t in val_txt],dtype=np.int64)
    gpid=np.asarray([t['pid'] for t in val_ims],dtype=np.int64)
    train_pids=set(map(int,opt.selected_pids))
    if train_pids & set(map(int,qpid)):
        raise RuntimeError('TRAIN/VAL PID contamination')
    baseline,bres=metrics(original @ gallery.T,qpid,gpid)
    if opt.expected_baseline_r1 is not None and abs(baseline['R1']-opt.expected_baseline_r1)>0.05:
        raise RuntimeError('Original IRRA R1 different from expected fixed checkpoint; abort comparison')
    base_right=bres['rank']==1
    records=[];all_query=[];cases=[];paired=[]
    for seed in opt.seeds:
        outcomes={}
        for method in METHODS:
            stem=f'{method}_s{seed}'
            if not _model_done04(out,opt,stem,tuple(original.shape)):
                raise RuntimeError(f'Missing or invalid completed student {stem}')
            features=normalize(np.asarray(np.load(out/(stem+'_val_features.npy'),mmap_mode='r'),dtype=np.float32))
            scores,res=metrics(features @ gallery.T,qpid,gpid)
            right=res['rank']==1
            outcomes[method]=(scores,right,res)
            row={'method':method,'seed':seed,'evaluation_type':'STANDARD_SINGLE_TEXT',**scores,
                 **{'delta_'+k+'_vs_irra':scores[k]-baseline[k] for k in baseline},
                 'recovered_vs_irra':int(np.count_nonzero(~base_right&right)),
                 'harmed_vs_irra':int(np.count_nonzero(base_right&~right)),
                 'mean_margin_delta_vs_irra':float(np.mean(res['margin']-bres['margin'])),
                 'mean_cos_to_irra':float(np.mean((features*original).sum(axis=1)))}
            records.append(row)
            msg(f'seed {seed}: {method} R1={scores["R1"]:.4f} (ΔIRRA {scores["R1"]-baseline["R1"]:+.4f}) mAP={scores["mAP"]:.4f}')
            for j,t in enumerate(val_txt):
                r={'method':method,'seed':seed,'query_idx':j,'pid':int(qpid[j]),'text':t['text'],
                   'base_rank':int(bres['rank'][j]),'new_rank':int(res['rank'][j]),
                   'base_correct':int(base_right[j]),'new_correct':int(right[j]),
                   'base_margin':float(bres['margin'][j]),'new_margin':float(res['margin'][j]),
                   'delta_margin':float(res['margin'][j]-bres['margin'][j]),
                   'cosine_to_orig':float(features[j]@original[j])}
                all_query.append(r)
                if bool(base_right[j])!=bool(right[j]):cases.append({**r,'type':'fixed' if right[j] else 'harmed'})
        ref=outcomes['rank_continue']
        for method in METHODS:
            score,right,res=outcomes[method]
            newwins=int(np.count_nonzero(~ref[1]&right))
            newloss=int(np.count_nonzero(ref[1]&~right))
            paired.append({'seed':seed,'method':method,'reference':'rank_continue',
                           'R1_diff_pp':score['R1']-ref[0]['R1'],
                           'mAP_diff_pp':score['mAP']-ref[0]['mAP'],
                           'fixed_vs_rank_continue':newwins,'harmed_vs_rank_continue':newloss,
                           'net_correct':newwins-newloss})
            for r in records:
                if r['method']==method and r['seed']==seed:
                    r['delta_R1_vs_rank_continue']=score['R1']-ref[0]['R1']
                    r['delta_mAP_vs_rank_continue']=score['mAP']-ref[0]['mAP']
                    break
    baseline_row={'method':'original_IRRA','seed':-1,'evaluation_type':'STANDARD_SINGLE_TEXT',**baseline,
                  **{'delta_'+k+'_vs_irra':0. for k in baseline},
                  'recovered_vs_irra':0,'harmed_vs_irra':0,'mean_margin_delta_vs_irra':0.,'mean_cos_to_irra':1.,
                  'delta_R1_vs_rank_continue':0.,'delta_mAP_vs_rank_continue':0.}
    records.insert(0,baseline_row)
    write_csv(out/'method_summary.csv',records,list(records[1]))
    write_csv(out/'paired_rank_comparisons.csv',paired,list(paired[0]))
    write_csv(out/'query_deltas.csv',all_query,list(all_query[0]))
    write_csv(out/'case_changes.csv',cases,list(cases[0]) if cases else list(all_query[0])+['type'])
    aggregate=[]
    for method in ('original_IRRA',)+METHODS:
        sub=[x for x in records if x['method']==method]
        item={'method':method,'n_seeds':len(sub)}
        for m in ('R1','R5','R10','mAP','mINP'):
            vals=np.asarray([r[m] for r in sub])
            item[m+'_mean']=float(vals.mean());item[m+'_sd']=float(vals.std(ddof=1)) if len(vals)>1 else 0.
            item['delta_'+m+'_vs_irra_mean']=item[m+'_mean']-baseline[m]
        differences=[r for r in paired if r['method']==method]
        item['delta_R1_vs_rank_continue_mean']=float(np.mean([r['R1_diff_pp'] for r in differences])) if differences else None
        item['delta_mAP_vs_rank_continue_mean']=float(np.mean([r['mAP_diff_pp'] for r in differences])) if differences else None
        aggregate.append(item)
    write_csv(out/'aggregate_summary.csv',aggregate,list(aggregate[1]))
    history=all_training_records(out)
    diagnostics={}
    for m in METHODS[1:]:
        rows=[r for r in history if r['method']==m]
        diagnostics[m]={
            'realized_aux_grad_ratio_mean':float(np.mean([r['realized_aux_ratio'] for r in rows])),
            'aux_gradient_cosine_mean':float(np.nanmean([r['grad_dot_cos'] for r in rows if r['grad_dot_cos'] is not None])) if any(r['grad_dot_cos'] is not None for r in rows) else None,
            'coef_cap_batches':sum(r['aux_coefficient_cap_batches'] for r in rows),
            'zero_grad_batches':sum(r['zero_grad_batches'] for r in rows),
            'clipped_gradient_rate_mean':float(np.mean([r['gradient_clipped_rate'] for r in rows]))}
    atomic_json(out/'gradient_diagnostics.json',{'aux_ratio_target':opt.aux_grad_ratio,
                                                'gradient_matching_per_batch':True,
                                                'gradient_matches_base_rank_plus_protect_not_rank_alone':True,
                                                'methods':diagnostics})
    report={'experiment':EXPERIMENT_NAME,'evaluation_protocol':'STANDARD_SINGLE_TEXT_VAL',
            'model_init_protocol':'Shared rank warmup weights AND AdamW optimizer state for each seed',
            'baseline':baseline,'aggregate':aggregate,'per_seed':records,'paired':paired,
            'teacher':json.loads((out/'teacher_diagnostics.json').read_text(encoding='utf-8')),
            'gradient':json.loads((out/'gradient_diagnostics.json').read_text(encoding='utf-8')),
            'source_control_truth':'Different-PID random support count equals Cross; matched version aligns angular distance per query.',
            'val_pid_support_used':False,'train_val_pid_intersection':0,
            'settings':{k:v for k,v in vars(opt).items() if k not in ('fp_train','fp_val')},
            'limits':['Training loss gradient norm matching does not equalize gradient direction, optimizer dynamics, or effective feature drift.',
                      'PyTorch AdamW optimizer-state matched continuation; seed comparisons use same val data.',
                      'Support texts are found with TRAIN PID only. Val PIDs used only by final metrics.',
                      'Cross-image does not guarantee confirmed camera change.',
                      'Exploratory validation; reserve unmodified test split for finalized method.']}
    atomic_json(out/'experiment04_report.json',report)
    lines=['# 实验04：诊断A（梯度强度）与诊断C（身份语义负对照）','','评估为标准单文本 Val；所有分支共享同种子 Rank warmup 权重及优化器状态。','',
           '| 方法 | R1 (均值±SD) | mAP (均值±SD) | 相对 Rank-Continue R1 |',
           '|---|---:|---:|---:|']
    for r in aggregate:
        lines.append(f'| {r["method"]} | {r["R1_mean"]:.4f} ± {r["R1_sd"]:.4f} | {r["mAP_mean"]:.4f} ± {r["mAP_sd"]:.4f} | {r["delta_R1_vs_rank_continue_mean"] if r["delta_R1_vs_rank_continue_mean"] is not None else 0:+.4f} |')
    lines.extend(['','判断依赖 Cross-Relation 是否稳定超过 Rank-Continue、Self-Relation、Extra-Protect、WrongPID-Matched，不能只和原始 IRRA 比较。',
                  '即使总体提高也不等于证明语义补全；配套教师角度匹配误差与梯度方向必须一起阅读。'])
    (out/'experiment04_readme.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    msg('EXP04 full metrics, paired controls, teacher angles and gradient diagnostics exported.')


def self_test04():
    import torch
    rng=np.random.default_rng(401)
    with tempfile.TemporaryDirectory(prefix='irra_exp04_selftest_') as tmp:
        out=Path(tmp);dim=24;annotations=[];train_txt=[];train_img=[];val_txt=[];val_img=[]
        for pid in range(1,10):
            center=normalize(rng.normal(size=dim))
            for im in range(3):
                annotations.append({'split':'train','id':pid,'file_path':f'train/{pid}_{im}.jpg',
                                    'captions':[f'train pid {pid} view {im} t 0',f'train pid {pid} view {im} t 1']})
                train_img.append(normalize(center+.27*rng.normal(size=dim)))
                for j in range(2):train_txt.append(normalize(center+.34*rng.normal(size=dim)))
        for pid in range(21,25):
            center=normalize(rng.normal(size=dim))
            for im in range(2):
                annotations.append({'split':'val','id':pid,'file_path':f'val/{pid}_{im}.jpg',
                                    'captions':[f'val pid {pid} img {im} t 0',f'val pid {pid} img {im} t 1']})
                val_img.append(normalize(center+.23*rng.normal(size=dim)))
                for j in range(2):val_txt.append(normalize(center+.29*rng.normal(size=dim)))
        atomic_json(out/'synthetic_annotations.json',annotations)
        atomic_json(out/'selected_train_pids.json',list(range(1,10)))
        for name,values in [('train_text',train_txt),('train_image',train_img),('text',val_txt),('image',val_img)]:
            np.save(out/(name+'_features.npy'),np.asarray(values,dtype=np.float32))
        fp_train={'synthetic':'train04'};fp_val={'synthetic':'val04'}
        fake=SimpleNamespace(fp_train=fp_train)
        fpi=train_img_fingerprint(fake,list(range(len(train_img))))
        for name,fp,rows in [('train_text',fp_train,len(train_txt)),('train_image',fpi,len(train_img)),
                             ('text',fp_val,len(val_txt)),('image',fp_val,len(val_img))]:
            atomic_json(out/(name+'_cache_meta.json'),{'fingerprint':fp,'done':rows,'shape':[rows,dim]})
        settings={'annotation_file':str(out/'synthetic_annotations.json'),
                  'project_root':str(out),'config_file':str(out/'fake.yaml'),
                  'checkpoint':str(out/'fake.pth'),'images_dir':str(out),
                  'train_text_cache_dir':'','train_image_cache_dir':'','val_cache_dir':'',
                  'device':'cpu','seeds':[42,43],'selected_pids':list(range(1,10)),
                  'warmup_epochs':2,'continue_epochs':2,'train_batch':16,'hidden':32,
                  'gain':.3,'lr':.0003,'rank_weight':.2,'protect_weight':.5,
                  'relation_weight':4.,'aux_grad_ratio':.3,'min_grad_norm':1e-12,
                  'max_aux_scale':100000.,'margin':.05,'temperature':.07,
                  'hard_k':2,'mine_batch':16,'max_support':5,'teacher_strength':.35,
                  'random_teacher_seed':2003,'fp_train':fp_train,'fp_val':fp_val,
                  'expected_baseline_r1':None,'settings_hash':'synthetic04'}
        atomic_json(out/'run_info.json',{'settings':settings,'status':'running'})
        script=Path(__file__).resolve()
        for stage in ('encode','evidence','train','eval'):
            env=os.environ.copy();env.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
            p=subprocess.run([sys.executable,str(script),'--_worker',stage,'--_run-dir',str(out)],
                text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300,env=env)
            if p.returncode:raise AssertionError(f'{stage} failed:\n{p.stdout}')
        report=json.loads((out/'experiment04_report.json').read_text(encoding='utf-8'))
        assert len(report['per_seed'])==len(METHODS)*2+1
        assert report['train_val_pid_intersection']==0
        assert report['teacher']['angular_distance_match_max_abs_error']<2e-4
        assert report['gradient']['methods']['relation_cross']['realized_aux_grad_ratio_mean']>0
        assert len(list(out.glob('*_training_meta.json')))==len(METHODS)*2
        for s in (42,43):
            a=torch.load(out/f'rank_warmup_s{s}_student.pt',map_location='cpu',weights_only=False)
            b=torch.load(out/f'rank_continue_s{s}_student.pt',map_location='cpu',weights_only=False)
            assert a['settings_hash']==b['settings_hash']
        p=subprocess.run([sys.executable,str(script),'--_worker','train','--_run-dir',str(out)],
                         capture_output=True,text=True,timeout=90,
                         env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        if p.returncode or p.stdout.count('validated completed checkpoint')!=len(METHODS)*2:
            raise AssertionError('Resume failed '+p.stdout+' '+p.stderr)
        shutil.copy2(script,out/script.name)
        package_for_chat(out)
        with zipfile.ZipFile(out/'发送给ChatGPT.zip') as z:
            for name in ('aggregate_summary.csv','experiment04_report.json','gradient_diagnostics.json',
                         'teacher_diagnostics.json','training_history.csv',script.name):
                assert name in z.namelist()
            assert not any(f.endswith(('.npy','.pt')) for f in z.namelist())
    msg('SELF-TEST PASSED: 14 models, matched wrong-PID angles, shared optimizer warmup, gradient matching, standard evaluation, resume, ZIP.')


def worker04(stage,out):
    if stage=='encode':stage_encode03(Path(out))
    elif stage=='evidence':stage_evidence04(Path(out))
    elif stage=='train':stage_train04(Path(out))
    elif stage=='eval':evaluate04(Path(out))
    else:raise ValueError('Unknown stage '+str(stage))


def main():
    global LOG_FILE
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root',default=DEFAULT_ROOT)
    parser.add_argument('--config-file',default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint',default=DEFAULT_CKPT)
    parser.add_argument('--dataset-root',default='')
    parser.add_argument('--annotation-file',default='')
    parser.add_argument('--images-root',default='')
    parser.add_argument('--val-cache-dir',default='')
    parser.add_argument('--train-text-cache-dir',default='')
    parser.add_argument('--train-image-cache-dir',default='')
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--max-train-pids',type=int,default=2200)
    parser.add_argument('--subset-seed',type=int,default=42)
    parser.add_argument('--seeds',default='42,43,44')
    parser.add_argument('--warmup-epochs',type=int,default=5)
    parser.add_argument('--continue-epochs',type=int,default=3)
    parser.add_argument('--text-batch',type=int,default=64)
    parser.add_argument('--image-batch',type=int,default=8)
    parser.add_argument('--train-batch',type=int,default=256)
    parser.add_argument('--hidden',type=int,default=256)
    parser.add_argument('--gain',type=float,default=.3)
    parser.add_argument('--lr',type=float,default=.0003)
    parser.add_argument('--rank-weight',type=float,default=.2)
    parser.add_argument('--relation-weight',type=float,default=4.,
                        help='Used only if non-matched training is explicitly enabled by code; matched mode default.')
    parser.add_argument('--protect-weight',type=float,default=.5)
    parser.add_argument('--margin',type=float,default=.05)
    parser.add_argument('--temperature',type=float,default=.07)
    parser.add_argument('--aux-grad-ratio',type=float,default=.3,
                        help='Target ratio norm(weighted aux gradient) / norm(rank+protect gradient).')
    parser.add_argument('--max-aux-scale',type=float,default=100000.)
    parser.add_argument('--min-grad-norm',type=float,default=1e-12)
    parser.add_argument('--teacher-strength',type=float,default=.35)
    parser.add_argument('--hard-k',type=int,default=4)
    parser.add_argument('--mine-batch',type=int,default=96)
    parser.add_argument('--max-support',type=int,default=8)
    parser.add_argument('--random-teacher-seed',type=int,default=2003)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--_worker',default='',help=argparse.SUPPRESS)
    parser.add_argument('--_run-dir',default='',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args._worker:
        try:worker04(args._worker,args._run_dir)
        except Exception:traceback.print_exc();sys.exit(1)
        return
    if args.self_test:
        self_test04();return
    try:seeds=[int(x.strip()) for x in args.seeds.split(',') if x.strip()]
    except ValueError:parser.error('--seeds must be comma-separated integers')
    if not seeds or len(set(seeds))!=len(seeds) or any(x<0 for x in seeds):
        parser.error('Seed list must contain unique nonnegative integers')
    for k in ('warmup_epochs','continue_epochs','train_batch','hidden','text_batch',
              'image_batch','hard_k','mine_batch','max_support'):
        if getattr(args,k)<=0:parser.error(k+' must be > 0')
    if args.max_train_pids<0:parser.error('--max-train-pids cannot be negative')
    if not(0<args.teacher_strength<1 and args.lr>0 and args.gain>0 and
           args.aux_grad_ratio>0 and args.max_aux_scale>0 and args.min_grad_norm>0 and
           args.temperature>0 and args.rank_weight>=0 and args.protect_weight>=0):
        parser.error('Invalid floating point training parameter')
    root=Path(args.project_root).resolve();cfg=Path(args.config_file).resolve();ckpt=Path(args.checkpoint).resolve()
    for f in (root/'model'/'build.py',cfg,ckpt):
        if not f.is_file():raise FileNotFoundError('Missing IRRA file '+str(f))
    _,ann,imgs,dataset_source=resolve_dataset(root,cfg,args.dataset_root,args.annotation_file,args.images_root)
    full_im,full_tx,_=read_annotations(ann,'train')
    val_im,val_tx,_=read_annotations(ann,'val')
    allpids=sorted({r['pid'] for r in full_tx})
    if set(allpids)&{r['pid'] for r in val_tx}:
        raise RuntimeError('Train-val PID overlap detected')
    if args.max_train_pids==0 or args.max_train_pids>=len(allpids):selected=allpids
    else:selected=sorted(map(int,np.random.default_rng(args.subset_seed).choice(allpids,args.max_train_pids,replace=False)))
    verify_image_paths(imgs,[a for a in full_im if a['pid'] in set(selected)])
    fp_train=fingerprint(root,cfg,ckpt,ann,'train')
    fp_val=fingerprint(root,cfg,ckpt,ann,'val')
    parent=root/PROJECT_FOLDER_NAME
    valcache=source_val_cache(root,parent,args.val_cache_dir,fp_val,len(val_tx),len(val_im))
    textcache=source_train_text_cache(root,parent,args.train_text_cache_dir,fp_train,len(full_tx))
    ids=[i for i,a in enumerate(full_im) if a['pid'] in set(selected)]
    cacher=SimpleNamespace(fp_train=fp_train,selected_pids=selected)
    imagecache=source_train_image_cache(parent,args.train_image_cache_dir,cacher,ids)
    settings={'project_root':str(root),'config_file':str(cfg),'checkpoint':str(ckpt),
              'annotation_file':str(ann),'images_dir':str(imgs),'dataset_source':dataset_source,
              'val_cache_dir':str(valcache or ''),'train_text_cache_dir':str(textcache or ''),
              'train_image_cache_dir':str(imagecache or ''),'selected_pids':selected,
              'device':args.device,'seeds':seeds,'subset_seed':args.subset_seed,
              'warmup_epochs':args.warmup_epochs,'continue_epochs':args.continue_epochs,
              'text_batch':args.text_batch,'image_batch':args.image_batch,
              'train_batch':args.train_batch,'hidden':args.hidden,'gain':args.gain,
              'lr':args.lr,'rank_weight':args.rank_weight,'relation_weight':args.relation_weight,
              'protect_weight':args.protect_weight,'margin':args.margin,
              'temperature':args.temperature,'aux_grad_ratio':args.aux_grad_ratio,
              'max_aux_scale':args.max_aux_scale,'min_grad_norm':args.min_grad_norm,
              'teacher_strength':args.teacher_strength,'hard_k':args.hard_k,
              'mine_batch':args.mine_batch,'max_support':args.max_support,
              'random_teacher_seed':args.random_teacher_seed,
              'fp_train':fp_train,'fp_val':fp_val,
              'expected_baseline_r1':56.83663527119195 if (
                  len(val_tx)==6158 and len(val_im)==3078 and
                  fp_val.get('checkpoint_sha256')=='3bf21f54271d0e7bd0cfd545daa2c2b5367ee2d734419bead417e70c2ec0a36c') else None}
    settings['settings_hash']=hashlib.sha256(json.dumps(settings,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()
    out,resumed=new_or_resume(parent,args,settings)
    LOG_FILE=out/'experiment_run.log'
    shutil.copy2(Path(__file__).resolve(),out/Path(__file__).name)
    if resumed:
        previous=json.loads((out/'selected_train_pids.json').read_text(encoding='utf-8'))
        if previous!=selected:raise RuntimeError('Resume PID subset mismatch')
    else:atomic_json(out/'selected_train_pids.json',selected)
    info_path=out/'run_info.json'
    info=json.loads(info_path.read_text(encoding='utf-8')) if resumed else {}
    info.update({'experiment_name':EXPERIMENT_NAME,'settings':settings,'status':'running',
                 'resumed':resumed,'started_at':datetime.now().isoformat()})
    atomic_json(info_path,info)
    msg(f'EXP04 folder: {out}')
    msg(f'Train identities {len(selected)}, val images {len(val_im)}, val texts {len(val_tx)}')
    msg(f'Cache source: val={valcache or out}, train_txt={textcache or out}, train_image={imagecache or out}')
    msg(f'Seeds={seeds}, warmup={args.warmup_epochs}, continue={args.continue_epochs}, methods={METHODS}')
    try:
        for stage in ('encode','evidence','train','eval'):
            run_subprocess(stage,out,root)
        info['status']='completed';info['finished_at']=datetime.now().isoformat()
        atomic_json(info_path,info)
        msg('EXP04 COMPLETED: experiment reports, gradients, teacher angles, metrics, and ZIP ready.')
    except Exception as exc:
        info['status']='failed';info['error']=repr(exc)
        info['failed_at']=datetime.now().isoformat()
        atomic_json(info_path,info)
        (out/'error_traceback.log').write_text(traceback.format_exc(),encoding='utf-8')
        msg('EXP04 FAILED: '+repr(exc))
        raise
    finally:
        try:package_for_chat(out)
        except Exception as exc:msg('Packaging failed: '+repr(exc))

if __name__=='__main__':
    main()
