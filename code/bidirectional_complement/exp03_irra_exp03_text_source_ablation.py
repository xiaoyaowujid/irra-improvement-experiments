# -*- coding: utf-8 -*-
"""实验03：多文本来源因果消融（冻结 IRRA，三个训练随机种子）

Train-only PID multi-caption teachers + frozen train images + hard negatives;
five identically initialized students per seed: rank, same, cross, random-PID, selective.
Validation: EXACTLY one caption/query, original fixed IRRA image gallery;
validation PIDs are used exclusively by evaluator AFTER prediction.

PyCharm: put this .py under IRRA-main; default run with no parameters.
--self-test: synthetic end-to-end verification, no real dataset/model needed.
--resume: resume a matching timestamped run, skip completed encodings/methods.
--max-train-pids 0: use ALL training identities (slower; default 2200).
--val-cache-dir / --train-text-cache-dir: optional explicit EXP00/EXP01 caches.
Every run emits 图文双向互补实验/实验03_..._YYYYMMDD_HHMMSS/发送给ChatGPT.zip.
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
EXPERIMENT_NAME = '实验03_多文本来源消融与多种子验证'
METHODS = ('rank','rank_same','rank_cross','rank_random','rank_selective')
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
        raise FileNotFoundError('No matching experiment03 run to resume. Remove --resume for a new run.')
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


# ========================= EXP03 implementation =========================

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


# ========================= EXP03: source ablation =========================

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


def make_teacher03(out,opt,xf,vf,texts,images,pos,neg):
    n,d=xf.shape
    pids=np.asarray([t['pid'] for t in texts],dtype=np.int64)
    image_idx=np.asarray([t['img_idx'] for t in texts],dtype=np.int64)
    bypid=defaultdict(list)
    byimage=defaultdict(list)
    for i, (p,im) in enumerate(zip(pids,image_idx)):
        bypid[int(p)].append(i)
        byimage[int(im)].append(i)
    tpaths={m:out/(m+'_teacher.npy') for m in METHODS if m!='rank'}
    matrices={m:np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(n,d))
              for m,path in tpaths.items()}
    accept=np.lib.format.open_memmap(out/'rank_selective_accept.npy',mode='w+',
                                    dtype='uint8',shape=(n,))
    # random other-PID descriptions, matched to the number of cross-image supporters
    rng=np.random.default_rng(opt.random_teacher_seed)
    random_by_pid={int(p):np.flatnonzero(pids!=p) for p in np.unique(pids)}
    samples=[]
    stats={m:{'covered':0,'selected':0,'sum_training_gain':0.,'sum_holdout_gain':0.,
              'training_gain_positive':0,'holdout_gain_positive':0,'sum_cos':0.}
           for m in matrices}
    k=opt.hard_k
    for i in range(n):
        anchor=xf[i]; own_img=vf[pos[i]]
        train_neg=vf[neg[i,:k]]
        hold_neg=vf[neg[i,k:]]
        base_gap=float(anchor@own_img-np.max(train_neg@anchor))
        base_hold=float(anchor@own_img-np.max(hold_neg@anchor)) if len(hold_neg) else float('nan')
        same=[j for j in byimage[int(image_idx[i])] if j!=i]
        cross=[j for j in bypid[int(pids[i])] if image_idx[j]!=image_idx[i]]
        same=capped_peer_indices(same,anchor,xf,opt.max_support)
        cross=capped_peer_indices(cross,anchor,xf,opt.max_support)
        random_pool=random_by_pid[int(pids[i])]
        rand=list(map(int,rng.choice(random_pool,len(cross),replace=False))) if cross else []
        candidates={'rank_same':same,'rank_cross':cross,'rank_random':rand}
        result={}
        for m,support in candidates.items():
            if support:
                teacher=normalize((1-opt.teacher_strength)*anchor +
                                  opt.teacher_strength*xf[support].mean(axis=0))
            else:
                teacher=anchor
            result[m]=teacher
        best=anchor; selected_idx=-1; best_gain=opt.min_gain
        for j in cross:
            cand=normalize((1-opt.teacher_strength)*anchor+opt.teacher_strength*xf[j])
            if float(cand@own_img-anchor@own_img)<-opt.max_pos_drop:
                continue
            improvement=float(cand@own_img-np.max(train_neg@cand))-base_gap
            if improvement>best_gain:
                best_gain=improvement;best=cand;selected_idx=j
        result['rank_selective']=best
        accept[i]=int(selected_idx>=0)
        for m,t in result.items():
            matrices[m][i]=t
            st=stats[m]
            st['covered']+=int(bool(cross if m=='rank_selective' else candidates[m]))
            st['selected']+=int(selected_idx>=0) if m=='rank_selective' else int(bool(candidates[m]))
            gain=float(t@own_img-np.max(train_neg@t))-base_gap
            hg=float(t@own_img-np.max(hold_neg@t))-base_hold if len(hold_neg) else 0.
            st['sum_training_gain']+=gain
            st['sum_holdout_gain']+=hg
            st['training_gain_positive']+=int(gain>0)
            st['holdout_gain_positive']+=int(hg>0)
            st['sum_cos']+=float(t@anchor)
        if len(samples)<180 and (i%max(1,n//180)==0 or len(samples)<36):
            samples.append({'query_idx':i,'pid_train':int(pids[i]),'anchor':texts[i]['text'],
                'same_texts':' || '.join(texts[j]['text'] for j in same),
                'cross_texts':' || '.join(texts[j]['text'] for j in cross),
                'random_texts':' || '.join(texts[j]['text'] for j in rand),
                'selected_text':texts[selected_idx]['text'] if selected_idx>=0 else '',
                'selected':int(selected_idx>=0),
                'selected_train_gain':best_gain if selected_idx>=0 else 0.,
                'selected_holdout_gain':float(best@own_img-np.max(hold_neg@best))-base_hold
                                            if len(hold_neg) else ''})
        if (i+1)%5000==0:msg(f'Teacher supports built: {i+1}/{n}')
    accept.flush()
    for a in matrices.values():a.flush()
    write_csv(out/'teacher_examples.csv',samples,list(samples[0]))
    for m,st in stats.items():
        st.update({
            'coverage_rate':st['covered']/n,'accept_rate':st['selected']/n,
            'average_train_margin_gain':st.pop('sum_training_gain')/n,
            'average_holdout_negative_margin_gain':st.pop('sum_holdout_gain')/n,
            'training_margin_improved_rate':st.pop('training_gain_positive')/n,
            'holdout_negative_margin_improved_rate':st.pop('holdout_gain_positive')/n,
            'mean_cos_to_anchor':st.pop('sum_cos')/n})
    atomic_json(out/'teacher_diagnostics.json',{
        'num_train_queries':n,'num_train_images':len(vf),'train_pids':len(bypid),
        'mined_negatives':int(neg.shape[1]),'train_mining_negatives':k,
        'holdout_mined_negatives':int(neg.shape[1]-k),
        'teacher_sources':stats,'selective_search_uses_train_pid':True,
        'random_control_matches_cross_support_count':True,
        'important_limit':'Holdout negatives are from same frozen train gallery, not independent identities or untouched validation. Teacher diagnostics are NOT standard retrieval results.',
        'cross_image_means_different_image_not_camera_verified':True})
    msg('Teacher source diagnostics: '+json.dumps({m:{'train':round(s['average_train_margin_gain'],5),
         'holdout':round(s['average_holdout_negative_margin_gain'],5),
         'coverage':round(s['coverage_rate'],3)} for m,s in stats.items()},ensure_ascii=False))


def stage_evidence03(out):
    _,opt,out=manifest_for_stage(out)
    needed=['hard_negative_indices.npy','positive_gallery_idx.npy','rank_same_teacher.npy',
            'rank_cross_teacher.npy','rank_random_teacher.npy','rank_selective_teacher.npy',
            'rank_selective_accept.npy','teacher_diagnostics.json']
    completion=out/'evidence_complete.json'
    if completion.is_file():
        meta=json.loads(completion.read_text(encoding='utf-8'))
        if meta.get('settings_hash')==opt.settings_hash and all((out/x).is_file() for x in needed):
            msg('Verified completed teacher evidence stage, skip.');return
    xf,vf,txt,img,pos=train_arrays03(out,opt)
    pids=np.asarray([r['pid'] for r in txt]); ipids=np.asarray([r['pid'] for r in img])
    if len(np.unique(pids))<2:raise RuntimeError('Need >=2 training PIDs')
    msg(f'Train evidence: {len(txt)} queries, {len(img)} images, {len(set(pids))} PIDs.')
    neg=hard_negative_mine(xf,vf,pids,ipids,2*opt.hard_k,opt.mine_batch)
    assert neg.shape==(len(txt),2*opt.hard_k)
    if np.any(pids[:,None]==ipids[neg]):
        raise AssertionError('Training negatives contain the anchor PID')
    np.save(out/'hard_negative_indices.npy',neg)
    np.save(out/'positive_gallery_idx.npy',pos)
    make_teacher03(out,opt,xf,vf,txt,img,pos,neg)
    atomic_json(completion,{'settings_hash':opt.settings_hash,'completed_at':datetime.now().isoformat()})
    msg('All train-only teacher sources and held-out negative diagnostics complete.')


def all_training_records(out):
    records=[]
    for f in sorted(out.glob('*_training_meta.json')):
        entry=json.loads(f.read_text(encoding='utf-8'))
        if entry.get('complete'):records.extend(entry['history'])
    return records


def stage_train03(out):
    _,opt,out=manifest_for_stage(out)
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    device=torch.device('cuda' if opt.device=='cuda' or
                        (opt.device=='auto' and torch.cuda.is_available()) else 'cpu')
    if opt.device=='cuda' and device.type!='cuda':raise RuntimeError('CUDA requested but unavailable')
    xf,vf,txt,imgs,pos_idx=train_arrays03(out,opt)
    n,d=xf.shape
    valdir=Path(opt.val_cache_dir) if opt.val_cache_dir else out
    xval=normalize(np.load(valdir/'text_features.npy',mmap_mode='r'))
    negidx=np.load(out/'hard_negative_indices.npy',allow_pickle=False)
    if negidx.shape!=(n,2*opt.hard_k):raise ValueError('Wrong hard negative matrix shape')
    negidx=negidx[:,:opt.hard_k]
    tensors={m:torch.from_numpy(np.asarray(np.load(out/(m+'_teacher.npy'),mmap_mode='r')).copy())
             for m in METHODS if m!='rank'}
    isselected=torch.from_numpy(np.asarray(np.load(out/'rank_selective_accept.npy'),dtype=np.float32).copy())
    trainx=torch.from_numpy(xf.copy())
    valx=torch.from_numpy(xval.copy())
    gallery=torch.from_numpy(vf.copy()).to(device)
    pos=torch.from_numpy(pos_idx.copy())
    neg=torch.from_numpy(negidx.astype(np.int64))
    cls=get_adapter_exp02()
    for seed in opt.seeds:
        for method in METHODS:
            stem=f'{method}_s{seed}'
            ck=out/(stem+'_student.pt')
            feat=out/(stem+'_val_features.npy')
            meta=out/(stem+'_training_meta.json')
            if meta.is_file() and ck.is_file() and feat.is_file():
                saved=json.loads(meta.read_text(encoding='utf-8'))
                try:
                    shape=tuple(np.load(feat,mmap_mode='r').shape)
                except Exception:shape=()
                if saved.get('complete') and saved.get('settings_hash')==opt.settings_hash and shape==tuple(valx.shape):
                    msg(f'{stem}: validated completed training, skip.');continue
            torch.manual_seed(seed)
            if device.type=='cuda':torch.cuda.manual_seed_all(seed)
            adapter=cls(d,opt.hidden,opt.gain).to(device)
            optim=torch.optim.AdamW(adapter.parameters(),lr=opt.lr,weight_decay=1e-4)
            gen=torch.Generator().manual_seed(seed)
            history=[]
            msg(f'--- Train method={method}, seed={seed}, queries={n}, epochs={opt.epochs} ---')
            for epoch in range(1,opt.epochs+1):
                adapter.train()
                perm=torch.randperm(n,generator=gen)
                totals=np.zeros(4,dtype=np.float64)
                for start in range(0,n,opt.train_batch):
                    idx=perm[start:start+opt.train_batch]
                    x=trainx[idx].to(device)
                    z=adapter(x)
                    choices=torch.cat([pos[idx,None],neg[idx]],dim=1).to(device)
                    g=gallery[choices]
                    pred=torch.einsum('bkd,bd->bk',g,z)
                    l_rank=F.softplus((pred[:,1:]-pred[:,:1]+opt.margin)/opt.temperature).mean()
                    l_protect=(1-(z*x).sum(-1)).mean()
                    if method=='rank':
                        l_relation=pred.sum()*0.
                    else:
                        t=tensors[method][idx].to(device)
                        target=torch.einsum('bkd,bd->bk',g,t)
                        dg=pred[:,:1]-pred[:,1:]
                        tg=(target[:,:1]-target[:,1:]).detach()
                        errors=F.smooth_l1_loss(dg,tg,reduction='none').mean(dim=1)
                        if method=='rank_selective':
                            mask=isselected[idx].to(device)
                            l_relation=(errors*mask).sum()/mask.sum().clamp_min(1.)
                        else:
                            l_relation=errors.mean()
                    loss=opt.rank_weight*l_rank+opt.protect_weight*l_protect+opt.relation_weight*l_relation
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(),5.)
                    optim.step()
                    totals+=np.array([l_rank.item(),l_relation.item(),l_protect.item(),loss.item()])*len(idx)
                avg=totals/n
                row={'method':method,'seed':int(seed),'epoch':epoch,'rank_loss':avg[0],
                     'relation_loss':avg[1],'protect_loss':avg[2],'total_loss':avg[3],
                     'train_queries':n}
                history.append(row)
                msg(f'{stem} epoch {epoch}/{opt.epochs}: rank={avg[0]:.5f} relation={avg[1]:.6f} protect={avg[2]:.6f}')
            adapter.eval()
            tmp_feat=out/(stem+'_val_features.partial.npy')
            arr=np.lib.format.open_memmap(tmp_feat,mode='w+',dtype='float32',shape=tuple(valx.shape))
            cos_total=0.
            with torch.inference_mode():
                for start in range(0,len(valx),opt.train_batch):
                    x=valx[start:start+opt.train_batch].to(device)
                    z=adapter(x)
                    arr[start:start+len(z)]=z.cpu().numpy()
                    cos_total+=float((z*x).sum().item())
            arr.flush();del arr
            _replace_with_retries(tmp_feat,feat)
            history[-1]['val_cos_to_original']=cos_total/len(valx)
            tmp_ck=ck.with_name(ck.name+'.tmp')
            torch.save({'state_dict':adapter.cpu().state_dict(),
                        'dim':d,'hidden':opt.hidden,'gain':opt.gain,'seed':seed,
                        'method':method,'settings_hash':opt.settings_hash},tmp_ck)
            _replace_with_retries(tmp_ck,ck)
            atomic_json(meta,{'complete':True,'settings_hash':opt.settings_hash,
                              'method':method,'seed':seed,'history':history})
            # A fully completed model is checkpointed in *_training_meta.json above.
            # Never rewrite the shared history CSV per model: Windows may lock it
            # while PyCharm/Excel indexes or previews the file.
            msg(f'{stem} checkpoint, metadata and validation features committed; '
                f'feature cosine={history[-1]["val_cos_to_original"]:.6f}')
            del adapter,optim
            if device.type=='cuda':torch.cuda.empty_cache()
    # Summary is derived from independent per-model JSON records. Any failure
    # to replace this one CSV must not invalidate completed training results.
    all_history=all_training_records(out)
    if all_history:
        write_csv(out/'training_history.csv',all_history,list(all_history[0]),
                  allow_locked_fallback=True)
    msg('Completed all student methods and seeds.')

def evaluate03(out):
    _,opt,out=manifest_for_stage(out)
    val_img,val_text,_=read_annotations(Path(opt.annotation_file),'val')
    train_img,train_text,_=read_annotations(Path(opt.annotation_file),'train')
    gpid=np.asarray([r['pid'] for r in val_img],dtype=np.int64)
    qpid=np.asarray([r['pid'] for r in val_text],dtype=np.int64)
    overlap=set(qpid.tolist())&{r['pid'] for r in train_text}
    if overlap:raise AssertionError(f'Train/Val identity overlap: {list(overlap)[:6]}')
    vdir=Path(opt.val_cache_dir) if opt.val_cache_dir else out
    if not cache_is_ready(vdir,'text',opt.fp_val,len(val_text)) or not cache_is_ready(vdir,'image',opt.fp_val,len(val_img)):
        raise RuntimeError('Missing valid val caches')
    original=normalize(np.load(vdir/'text_features.npy',mmap_mode='r'))
    gallery=normalize(np.load(vdir/'image_features.npy',mmap_mode='r'))
    baseline,br=metrics(original @ gallery.T,qpid,gpid)
    if opt.expected_baseline_r1 is not None and abs(baseline['R1']-opt.expected_baseline_r1)>0.011:
        raise RuntimeError('Baseline IRRA does not match expected model; no silently mismatched comparison')
    bcorrect=br['rank']==1
    records=[];qrows=[];cases=[];paired=[]
    msg('ORIGINAL IRRA baseline: '+json.dumps(baseline))
    base_line={'method':'baseline_IRRA','seed':'fixed','evaluation_type':'STANDARD_SINGLE_TEXT',
               **baseline,**{'delta_'+k:0.0 for k in baseline},
               'repaired_vs_irra':0,'harmed_vs_irra':0,'r1_delta_vs_rank':0.,
               'fixed_gallery':True}
    records.append(base_line)
    for seed in opt.seeds:
        per_seed={}
        for method in METHODS:
            file=out/f'{method}_s{seed}_val_features.npy'
            if not file.is_file():raise FileNotFoundError(f'Student features missing {file}')
            feat=normalize(np.load(file,mmap_mode='r'))
            if feat.shape!=original.shape:raise ValueError(f'Incorrect shape {file}')
            stats,res=metrics(feat @ gallery.T,qpid,gpid)
            per_seed[method]=(stats,res)
            right=res['rank']==1
            repaired=int(np.count_nonzero(~bcorrect & right))
            harmed=int(np.count_nonzero(bcorrect & ~right))
            mean_cos=float(np.mean(np.sum(feat*original,axis=1)))
            row={'method':method,'seed':seed,'evaluation_type':'STANDARD_SINGLE_TEXT',
                 **stats,**{'delta_'+k:stats[k]-baseline[k] for k in baseline},
                 'repaired_vs_irra':repaired,'harmed_vs_irra':harmed,
                 'margin_improvement_rate':float(np.mean(res['margin']>br['margin']+1e-7)),
                 'mean_margin_delta':float(np.mean(res['margin']-br['margin'])),
                 'mean_feature_cos_to_irra':mean_cos,'fixed_gallery':True}
            records.append(row)
            msg(f'seed={seed} {method}: R1={stats["R1"]:.4f} (ΔIRRA {stats["R1"]-baseline["R1"]:+.4f}), '
                f'mAP={stats["mAP"]:.4f}, repaired={repaired}, harmed={harmed}')
            for i,q in enumerate(val_text):
                item={'seed':seed,'method':method,'query_idx':i,'pid':int(qpid[i]),
                      'text':q['text'],'original_rank':int(br['rank'][i]),
                      'new_rank':int(res['rank'][i]),
                      'original_margin':float(br['margin'][i]),
                      'new_margin':float(res['margin'][i]),
                      'delta_margin':float(res['margin'][i]-br['margin'][i]),
                      'original_R1_correct':int(bcorrect[i]),
                      'new_R1_correct':int(right[i]),
                      'original_pos_score':float(br['positive_score'][i]),
                      'new_pos_score':float(res['positive_score'][i]),
                      'new_hard_negative_score':float(res['negative_score'][i]),
                      'top1_gallery_index':int(res['top1_image'][i]),
                      'top1_pid':int(gpid[res['top1_image'][i]]),
                      'feature_cos_to_irra':float(np.sum(feat[i]*original[i]))}
                qrows.append(item)
                if bcorrect[i]!=right[i]:
                    cases.append({**item,'change_type_vs_irra':'repaired' if right[i] else 'harmed'})
            del feat
        rankstats,rankres=per_seed['rank']
        rankcorrect=rankres['rank']==1
        for method in METHODS:
            st,rr=per_seed[method]
            rr_correct=rr['rank']==1
            better=int(np.count_nonzero(~rankcorrect & rr_correct))
            worse=int(np.count_nonzero(rankcorrect & ~rr_correct))
            records[[k for k,x in enumerate(records) if x.get('seed')==seed and x['method']==method][0]][
                'r1_delta_vs_rank']=st['R1']-rankstats['R1']
            paired.append({'seed':seed,'method':method,'ref':'rank',
                           'r1_delta_pp':st['R1']-rankstats['R1'],
                           'mAP_delta_pp':st['mAP']-rankstats['mAP'],
                           'queries_fixed_vs_rank':better,'queries_harmed_vs_rank':worse,
                           'net_improved_queries':better-worse})
    write_csv(out/'method_summary.csv',records,list(records[1]))
    write_csv(out/'query_deltas.csv',qrows,list(qrows[0]))
    write_csv(out/'case_changes.csv',cases,list(cases[0]) if cases else
              list(qrows[0])+['change_type_vs_irra'])
    write_csv(out/'paired_rank_comparisons.csv',paired,list(paired[0]))
    summary=[]
    for method in ('baseline_IRRA',)+METHODS:
        r=[x for x in records if x['method']==method]
        entry={'method':method,'seeds':len(r)}
        for metric in ('R1','R5','R10','mAP','mINP'):
            vals=np.asarray([x[metric] for x in r],dtype=float)
            entry[metric+'_mean']=float(vals.mean())
            entry[metric+'_sd']=float(vals.std(ddof=1)) if len(vals)>1 else 0.
            entry['delta_'+metric+'_vs_IRRA_mean']=float(vals.mean()-baseline[metric])
        if method!='baseline_IRRA':
            delta=np.asarray([x['r1_delta_pp'] for x in paired if x['method']==method],dtype=float)
            entry['delta_R1_vs_Rank_mean']=float(delta.mean())
            entry['delta_R1_vs_Rank_sd']=float(delta.std(ddof=1)) if len(delta)>1 else 0.
            entry['wins_vs_Rank_seeds']=int(np.count_nonzero(delta>0))
        summary.append(entry)
    write_csv(out/'aggregate_summary.csv',summary,list(summary[1]))
    report={'experiment':EXPERIMENT_NAME,'evaluation_protocol':'STANDARD_SINGLE_TEXT_VAL',
            'original_irra_encoders_frozen':True,'val_support_texts_used':False,
            'train_val_pid_intersection':0,'num_val_queries':len(val_text),
            'num_val_images':len(val_img),'num_train_pids_selected':len(opt.selected_pids),
            'training_seeds':opt.seeds,'same_pid_support_train_only':True,
            'baseline':baseline,'per_seed':records,'aggregate':summary,
            'rank_pair_comparisons':paired,
            'teacher_diagnostics':json.loads((out/'teacher_diagnostics.json').read_text(encoding='utf-8')),
            'settings':{k:v for k,v in vars(opt).items() if k not in ('fp_train','fp_val')},
            'limitations':[
                'Multi-seed evaluation uses the same fixed val identities; seed SD is not a multi-dataset confidence interval.',
                'Random-PID controls for extra teacher loss, not all conceivable regularization effects.',
                'Selective teacher uses TRAIN ground-truth PID and TRAIN gallery only; train teacher gain is not a generalization metric.',
                'Cross-image description means different photos, not necessarily different camera views.',
                'All model selection is exploratory on val; reserve untouched test until methods are fixed.',
                'All students see a single caption and unchanged original image gallery at inference.'
            ]}
    atomic_json(out/'experiment03_report.json',report)
    lines=['# 实验03：同身份文本互补因果消融（3 seeds）','',
           '统一测试：一条验证文本、一套冻结 IRRA 图像特征、评估阶段仅使用 PID 计算正确率。','',
           '| 方法 | R1 Mean±SD | ΔIRRA R1 | ΔRank R1 | mAP Mean±SD |','|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["method"]} | {r["R1_mean"]:.4f}±{r["R1_sd"]:.4f} | '
                     f'{r["delta_R1_vs_IRRA_mean"]:+.4f} | '
                     f'{r.get("delta_R1_vs_Rank_mean",0.):+.4f} | '
                     f'{r["mAP_mean"]:.4f}±{r["mAP_sd"]:.4f} |')
    lines.extend(['','解释标准：跨图 Cross/Selective 只有在稳定优于 Rank、Random，'
                   '且多个随机种子表现一致时，才可作为多文本互补正面证据。',
                   '教师筛选/多文本使用的训练 PID 与验证 PID 严格分离。'])
    (out/'experiment03_readme.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    msg('EXP03 complete standard single-text validation plus multi-seed aggregate saved.')


def self_test03():
    rng=np.random.default_rng(83)
    with tempfile.TemporaryDirectory(prefix='irra_exp03_test_') as tmp:
        out=Path(tmp);d=32
        ann=[];tx=[];im=[];tv=[];iv=[]
        for p in range(1,13):
            center=normalize(rng.normal(size=d))
            for i in range(3):
                ann.append({'split':'train','id':p,'file_path':f'train/{p}_{i}.jpg',
                            'captions':[f'person {p} image {i} text {q}' for q in range(2)]})
                im.append(normalize(center+.27*rng.normal(size=d)))
                for q in range(2):tx.append(normalize(center+.32*rng.normal(size=d)))
        for p in range(31,35):
            center=normalize(rng.normal(size=d))
            for i in range(2):
                ann.append({'split':'val','id':p,'file_path':f'val/{p}_{i}.jpg',
                            'captions':[f'val person {p} view {i} text {q}' for q in range(2)]})
                iv.append(normalize(center+.25*rng.normal(size=d)))
                for q in range(2):tv.append(normalize(center+.24*rng.normal(size=d)))
        (out/'synthetic_annotations.json').write_text(json.dumps(ann),encoding='utf-8')
        atomic_json(out/'selected_train_pids.json',list(range(1,13)))
        np.save(out/'train_text_features.npy',np.asarray(tx,dtype=np.float32))
        np.save(out/'train_image_features.npy',np.asarray(im,dtype=np.float32))
        np.save(out/'text_features.npy',np.asarray(tv,dtype=np.float32))
        np.save(out/'image_features.npy',np.asarray(iv,dtype=np.float32))
        fpt={'synthetic':'train'};fpv={'synthetic':'val'}
        fake=SimpleNamespace(fp_train=fpt)
        iii=list(range(36)); fpi=train_img_fingerprint(fake,iii)
        for pfx,fp,n in [('train_text',fpt,len(tx)),('train_image',fpi,len(im)),
                         ('text',fpv,len(tv)),('image',fpv,len(iv))]:
            atomic_json(out/(pfx+'_cache_meta.json'),{'fingerprint':fp,
                         'shape':[n,d],'done':n})
        settings={'annotation_file':str(out/'synthetic_annotations.json'),
            'project_root':str(out),'config_file':str(out/'fake.yaml'),
            'checkpoint':str(out/'fake.pth'),'images_dir':str(out),
            'train_text_cache_dir':'','train_image_cache_dir':'','val_cache_dir':'',
            'device':'cpu','seeds':[42,43],'selected_pids':list(range(1,13)),
            'epochs':2,'train_batch':16,'hidden':32,'gain':.3,'lr':.0003,
            'rank_weight':.2,'relation_weight':4.,'protect_weight':.5,
            'margin':.05,'temperature':.07,'hard_k':2,'mine_batch':16,
            'max_support':5,'teacher_strength':.35,'min_gain':.003,
            'max_pos_drop':.025,'random_teacher_seed':2003,
            'fp_train':fpt,'fp_val':fpv,'expected_baseline_r1':None,
            'settings_hash':'synthetic03'}
        atomic_json(out/'run_info.json',{'settings':settings,'status':'running'})
        script=Path(__file__).resolve()
        for stage in ('encode','evidence','train','eval'):
            p=subprocess.run([sys.executable,str(script),'--_worker',stage,'--_run-dir',str(out)],
                capture_output=True,text=True,timeout=240,
                env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                     'OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
            if p.returncode:
                raise AssertionError(f'Selftest {stage} failed: {p.stdout}\n{p.stderr}')
        report=json.loads((out/'experiment03_report.json').read_text(encoding='utf-8'))
        assert len(report['per_seed'])==1+2*5
        assert report['train_val_pid_intersection']==0
        assert {r['method'] for r in report['aggregate']}==set(METHODS)|{'baseline_IRRA'}
        assert len(list(csv.DictReader(open(out/'training_history.csv',encoding='utf-8-sig'))))==20
        assert len(list(csv.DictReader(open(out/'query_deltas.csv',encoding='utf-8-sig'))))==160
        assert all((out/f'{m}_s{s}_student.pt').exists() for s in [42,43] for m in METHODS)
        # Validate resume at stage granularity; skip all completed students, avoid extra work.
        test=subprocess.run([sys.executable,str(script),'--_worker','train','--_run-dir',str(out)],
            capture_output=True,text=True,timeout=80,
            env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                 'OPENBLAS_NUM_THREADS':'1'})
        if test.returncode or test.stdout.count('validated completed training')!=len(METHODS)*2:
            raise AssertionError('Resume training stage failed: '+test.stdout+'\n'+test.stderr)
        (out/'experiment_run.log').write_text('Synthetic pipeline passed',encoding='utf-8')
        shutil.copy2(script,out/script.name)
        package_for_chat(out)
        with zipfile.ZipFile(out/'发送给ChatGPT.zip') as z:
            for f in ('method_summary.csv','aggregate_summary.csv','experiment03_report.json',
                      'teacher_diagnostics.json',script.name):
                assert f in z.namelist()
            assert not any(zname.endswith(('.npy','.pt')) for zname in z.namelist())
    msg('SELF-TEST PASSED: caches, train-only controls, held-out negatives, 10 students, '
        'full validation metrics, resume, all reports and ZIP.')


def worker03(stage,out):
    if stage=='encode':stage_encode03(Path(out))
    elif stage=='evidence':stage_evidence03(Path(out))
    elif stage=='train':stage_train03(Path(out))
    elif stage=='eval':evaluate03(Path(out))
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
    parser.add_argument('--max-train-pids',type=int,default=2200,
                        help='Explore on fixed train identity subset (0 = all train identities).')
    parser.add_argument('--subset-seed',type=int,default=42,
                        help='Keep experiment02 identity subset unchanged by default.')
    parser.add_argument('--seeds',default='42,43,44',
                        help='Comma-separated independent model seeds (default 42,43,44).')
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--text-batch',type=int,default=64)
    parser.add_argument('--image-batch',type=int,default=8)
    parser.add_argument('--train-batch',type=int,default=256)
    parser.add_argument('--hidden',type=int,default=256)
    parser.add_argument('--gain',type=float,default=.3)
    parser.add_argument('--lr',type=float,default=.0003)
    parser.add_argument('--rank-weight',type=float,default=.2)
    parser.add_argument('--relation-weight',type=float,default=4.)
    parser.add_argument('--protect-weight',type=float,default=.5)
    parser.add_argument('--margin',type=float,default=.05)
    parser.add_argument('--temperature',type=float,default=.07)
    parser.add_argument('--teacher-strength',type=float,default=.35)
    parser.add_argument('--min-gain',type=float,default=.003)
    parser.add_argument('--max-pos-drop',type=float,default=.025)
    parser.add_argument('--hard-k',type=int,default=4,
                        help='Train negatives K; a further K holdout negatives reserved for teacher diagnostics.')
    parser.add_argument('--mine-batch',type=int,default=96)
    parser.add_argument('--max-support',type=int,default=8)
    parser.add_argument('--random-teacher-seed',type=int,default=2003)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--_worker',default='',help=argparse.SUPPRESS)
    parser.add_argument('--_run-dir',default='',help=argparse.SUPPRESS)
    a=parser.parse_args()
    if a._worker:
        try:worker03(a._worker,a._run_dir)
        except Exception:
            traceback.print_exc()
            sys.exit(1)
        return
    if a.self_test:
        self_test03();return
    try:
        seeds=[int(s.strip()) for s in a.seeds.split(',') if s.strip()]
    except ValueError:parser.error('--seeds must be comma-separated integers')
    if not seeds or len(set(seeds))!=len(seeds) or any(s<0 for s in seeds):
        parser.error('seeds must be nonempty, unique nonnegative integers')
    for name in ('epochs','text_batch','image_batch','train_batch','hidden','hard_k',
                 'mine_batch','max_support'):
        if getattr(a,name)<=0:parser.error(name+' must be positive')
    if a.max_train_pids<0:parser.error('--max-train-pids must be nonnegative')
    if not (0<a.teacher_strength<1 and a.lr>0 and a.gain>0 and a.temperature>0 and
            a.min_gain>=0 and a.max_pos_drop>=0 and a.margin>=0 and a.rank_weight>=0 and
            a.relation_weight>=0 and a.protect_weight>=0):
        parser.error('Invalid teacher-strength or nonnegative training parameter')
    root=Path(a.project_root).resolve();cfg=Path(a.config_file).resolve()
    checkpoint=Path(a.checkpoint).resolve()
    for f in (root/'model'/'build.py',cfg,checkpoint):
        if not f.is_file():raise FileNotFoundError(f'Missing required IRRA file: {f}')
    _,ann,imgs,detected=resolve_dataset(root,cfg,a.dataset_root,a.annotation_file,a.images_root)
    all_train_img,all_train_txt,nclasses=read_annotations(ann,'train')
    val_img,val_txt,_=read_annotations(ann,'val')
    train_pids=sorted({t['pid'] for t in all_train_txt})
    if {t['pid'] for t in val_txt}&set(train_pids):
        raise RuntimeError('Train/val PID overlap; stop to prevent leakage')
    if a.max_train_pids==0 or a.max_train_pids>=len(train_pids):
        chosen=train_pids
    else:
        rng=np.random.default_rng(a.subset_seed)
        chosen=sorted(map(int,rng.choice(train_pids,a.max_train_pids,replace=False)))
    selected=set(chosen)
    count_img=sum(1 for z in all_train_img if z['pid'] in selected)
    count_txt=sum(1 for z in all_train_txt if z['pid'] in selected)
    msg(f'EXP03: train PID subset={len(chosen)}/{len(train_pids)}, '
        f'train images={count_img}, train captions={count_txt}, '
        f'val images={len(val_img)}, val captions={len(val_txt)}')
    verify_image_paths(imgs,[item for item in all_train_img if item['pid'] in selected])
    fp_t=fingerprint(root,cfg,checkpoint,ann,'train')
    fp_v=fingerprint(root,cfg,checkpoint,ann,'val')
    project_folder=root/PROJECT_FOLDER_NAME
    valcache=source_val_cache(root,project_folder,a.val_cache_dir,fp_v,len(val_txt),len(val_img))
    textcache=source_train_text_cache(root,project_folder,a.train_text_cache_dir,fp_t,len(all_train_txt))
    imgidx=[i for i,x in enumerate(all_train_img) if x['pid'] in selected]
    checker=SimpleNamespace(fp_train=fp_t,selected_pids=chosen)
    imagecache=source_train_image_cache(project_folder,a.train_image_cache_dir,checker,imgidx)
    settings={'project_root':str(root),'config_file':str(cfg),'checkpoint':str(checkpoint),
        'annotation_file':str(ann),'images_dir':str(imgs),'dataset_source':detected,
        'val_cache_dir':str(valcache or ''),'train_text_cache_dir':str(textcache or ''),
        'train_image_cache_dir':str(imagecache or ''),'selected_pids':chosen,
        'device':a.device,'seeds':seeds,'subset_seed':a.subset_seed,'epochs':a.epochs,
        'text_batch':a.text_batch,'image_batch':a.image_batch,'train_batch':a.train_batch,
        'hidden':a.hidden,'gain':a.gain,'lr':a.lr,'rank_weight':a.rank_weight,
        'relation_weight':a.relation_weight,'protect_weight':a.protect_weight,
        'margin':a.margin,'temperature':a.temperature,
        'teacher_strength':a.teacher_strength,'min_gain':a.min_gain,
        'max_pos_drop':a.max_pos_drop,'hard_k':a.hard_k,'mine_batch':a.mine_batch,
        'max_support':a.max_support,'random_teacher_seed':a.random_teacher_seed,
        'fp_train':fp_t,'fp_val':fp_v,
        'expected_baseline_r1':56.83663527119195 if (
            len(val_txt)==6158 and len(val_img)==3078 and
            fp_v.get('checkpoint_sha256')=='3bf21f54271d0e7bd0cfd545daa2c2b5367ee2d734419bead417e70c2ec0a36c') else None}
    settings['settings_hash']=hashlib.sha256(json.dumps(settings,sort_keys=True,
                                     ensure_ascii=False).encode('utf-8')).hexdigest()
    out,resumed=new_or_resume(project_folder,a,settings)
    LOG_FILE=out/'experiment_run.log'
    shutil.copy2(Path(__file__).resolve(),out/Path(__file__).name)
    if resumed:
        old=json.loads((out/'selected_train_pids.json').read_text(encoding='utf-8'))
        if old!=chosen:raise RuntimeError('Selected identity subset changed on resume')
    else:atomic_json(out/'selected_train_pids.json',chosen)
    info_path=out/'run_info.json'
    info=json.loads(info_path.read_text(encoding='utf-8')) if resumed else {}
    info.update({'experiment_name':EXPERIMENT_NAME,'settings':settings,'status':'running',
                 'resumed':resumed,'started_at':datetime.now().isoformat()})
    atomic_json(info_path,info)
    msg('Outputs: '+str(out))
    msg(f'Cache folders: val={valcache or out}; train_text={textcache or out}; train_image={imagecache or out}')
    msg(f'Seeds={seeds}, methods={METHODS}; original IRRA baseline always reported.')
    try:
        for stage in ('encode','evidence','train','eval'):
            run_subprocess(stage,out,root)
        info['status']='completed';info['finished_at']=datetime.now().isoformat()
        atomic_json(info_path,info)
        msg('EXP03 COMPLETED: per-seed results, IRRA baseline, aggregate and case files ready.')
    except Exception as e:
        info['status']='failed';info['error']=repr(e)
        info['failed_at']=datetime.now().isoformat()
        atomic_json(info_path,info)
        (out/'error_traceback.log').write_text(traceback.format_exc(),encoding='utf-8')
        msg('EXP03 FAILED: '+repr(e))
        raise
    finally:
        try:package_for_chat(out)
        except Exception as e:msg('ZIP packaging failed: '+repr(e))


if __name__=='__main__':
    main()
