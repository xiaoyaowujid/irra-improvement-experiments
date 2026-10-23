# -*- coding: utf-8 -*-
"""实验07：可靠性驱动的条件式文本互补（冻结 IRRA，多随机种子）

Research aims: (1) learn to ACCEPT useful text support and REJECT harmful
support with equally weighted evaluation, (2) transfer useful support selection
to a candidate-image-conditioned single-caption student, with no Val PID use.
Evaluate identity-isolated support triage on 20% heldout TRAIN PIDs, then train
matched-architecture ranking/student controls on remaining 80% TRAIN PIDs.
Do not mistake teacher PID-assisted evidence diagnostics for standard retrieval.
Inference only accesses current caption and each gallery image; reranks a fixed
original-IRRA top-K shortlist. Standard retrieval metrics cover the FULL gallery.
Stages run in separated subprocesses; NumPy BLAS Top-K search is isolated
from torch training to prevent Windows libiomp5md.dll duplication.
Runs in IRRA-main with default paths, --self-test, --resume. Outputs ZIP.
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
EXPERIMENT_NAME = '实验07_条件式文本互补与有害信息拒绝'
METHODS = ('warmup','rank_continue','reg','reg_shuffle','unconditional_true','conditional_true','conditional_wrong','reg_conditional_true','reg_conditional_wrong')
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


def triage_features(t,s,p,sp,negative):
    # Gate inputs are train-time visual evidence, never PID or label.
    # No negative margin / correctness label is fed to the classifier.
    return np.asarray([float(t@s),float(s@p),float(s@sp),float(t@sp),
                       float(t@p),float(s@negative),float((s-t)@p),
                       float((s-t)@sp)],dtype=np.float32)


def triage_label(t,s,p,neg, strength, helpful_threshold, harm_threshold, instance_floor):
    new=normalize((1-strength)*t+strength*s)
    score_before=float(t@p-np.max(neg@t))
    score_after=float(new@p-np.max(neg@new))
    delta=score_after-score_before
    instance_delta=float(new@p-t@p)
    # Classes: 0 harmful, 1 uncertain/duplicate, 2 useful.
    if delta <= -harm_threshold or instance_delta < -instance_floor: y=0
    elif delta >= helpful_threshold and instance_delta >= -instance_floor: y=2
    else:y=1
    return y,delta,instance_delta


def build_triage_rows(x,v,texts,pos,neg,opt,limit=None):
    byid=defaultdict(list)
    for i,r in enumerate(texts):byid[int(r['pid'])].append(i)
    feat=[];labels=[];deltas=[];records=[]
    for i in range(len(x)):
        peers=[j for j in byid[int(texts[i]['pid'])]
               if int(texts[j]['img_idx'])!=int(texts[i]['img_idx'])]
        peers=capped_peer_indices(peers,x[i],x,opt.max_support)
        for j in peers:
            vector=triage_features(x[i],x[j],v[pos[i]],v[pos[j]],v[neg[i,0]])
            y,d,ins=triage_label(x[i],x[j],v[pos[i]],v[neg[i]],
                    opt.teacher_strength,opt.help_threshold,opt.harm_threshold,opt.instance_floor)
            feat.append(vector);labels.append(y);deltas.append(d)
            if limit is None or len(records)<limit:
                records.append({'query_idx':i,'support_idx':j,'pid':int(texts[i]['pid']),
                                'query_text':texts[i]['text'],'support_text':texts[j]['text'],
                                'proxy_label':('harmful','uncertain','useful')[y],
                                'margin_delta':d,'positive_instance_delta':ins})
    if not feat:raise RuntimeError('No cross-image same-PID caption supports in selected identities')
    return np.stack(feat),np.asarray(labels,np.int32),np.asarray(deltas,np.float32),records


def fit_three_way_gate(rows,labels,seed):
    f=np.asarray(rows,np.float64); y=np.asarray(labels,np.int64)
    mu=f.mean(0);sd=np.maximum(f.std(0),1e-5)
    z=np.column_stack([np.clip((f-mu)/sd,-8,8),np.ones(len(f))])
    freq=np.bincount(y,minlength=3)
    rng=np.random.default_rng(seed)
    w=np.zeros((z.shape[1],3),np.float64)
    prior=np.maximum(freq,1)/max(1,len(y));w[-1]=np.log(prior)
    # Class-balancing is training-only. No quotas force the final gate to accept.
    weights=len(y)/(3*np.maximum(freq,1))
    weights=np.clip(weights,.1,20)
    for _ in range(100):
        for indices in np.array_split(rng.permutation(len(y)), max(1,(len(y)+2047)//2048)):
            logits=z[indices]@w;logits-=np.max(logits,axis=1,keepdims=True)
            pp=np.exp(logits);pp/=pp.sum(1,keepdims=True)
            diff=pp;diff[np.arange(len(indices)),y[indices]]-=1
            diff*=weights[y[indices],None]
            grad=z[indices].T@diff/len(indices)
            grad[:-1]+=.02*w[:-1]
            w-=.045*grad
    return {'weights':w.tolist(),'mean':mu.tolist(),'std':sd.tolist(),
            'train_class_counts':freq.tolist(),'class_labels':['harmful','uncertain','useful'],
            'features':['text_text','support_anchor_visual','support_own_visual','anchor_support_visual',
                        'anchor_own_visual','support_negative_visual','support_delta_anchor_visual',
                        'support_delta_own_visual']}


def gate_predict(rows, gate):
    z=np.column_stack([np.clip((np.asarray(rows,np.float64)-gate['mean'])/gate['std'],-8,8),
                        np.ones(len(rows))])
    logit=z@np.asarray(gate['weights']);logit-=logit.max(axis=1,keepdims=True)
    e=np.exp(logit);return e/e.sum(axis=1,keepdims=True)


def triage_diag(y,pp,delta,opt):
    pred=np.argmax(pp,axis=1);counts=np.bincount(y,minlength=3)
    cm=np.zeros((3,3),dtype=np.int64)
    for a,b in zip(y,pred):cm[int(a),int(b)]+=1
    recalls=[float(cm[i,i]/counts[i]) if counts[i] else None for i in range(3)]
    precisions=[float(cm[i,i]/max(1,cm[:,i].sum())) if cm[:,i].sum() else None for i in range(3)]
    accept=(pp[:,2]>=opt.use_threshold)&(pp[:,2]>=pp[:,0]+opt.use_vs_harm_margin)
    reject=(pp[:,0]>=opt.harm_threshold_prob)&(pp[:,0]>=pp[:,2]+opt.use_vs_harm_margin)
    both=(recalls[0] is not None and recalls[2] is not None)
    dual_score=(2*recalls[0]*recalls[2]/max(recalls[0]+recalls[2],1e-12)) if both else None
    return {'n_support_pairs':len(y),'class_counts':counts.tolist(),
            'dual_objective_harmful_recall_useful_recall_harmonic':dual_score,
            'both_classes_present':bool(both),
            'confusion_rows_true_cols_pred':cm.tolist(),
            'recall_harmful_uncertain_useful':recalls,'precision_harmful_uncertain_useful':precisions,
            'macro_recall_present':float(np.mean([v for v in recalls if v is not None])),
            'reject_harmful_rate':float(np.mean(reject[y==0])) if np.any(y==0) else None,
            'accept_useful_rate':float(np.mean(accept[y==2])) if np.any(y==2) else None,
            'accepted_fraction_all':float(accept.mean()),'rejected_fraction_all':float(reject.mean()),
            'unresolved_fraction_all':float((~(accept|reject)).mean()),
            'accepted_true_useful_precision':float(np.mean(y[accept]==2)) if np.any(accept) else None,
            'accepted_proxy_margin_mean':float(delta[accept].mean()) if np.any(accept) else None,
            'rejected_proxy_margin_mean':float(delta[reject].mean()) if np.any(reject) else None,
            'warning':'Labels are image-and-hard-negative based utility PROXIES, not verified natural-language truth. '
                      'Identity-assisted triage is ORACLE diagnostics; no Val PID use in inference.'}


def wrong_direction_matched(anchor,wrong,reference,rng):
    a=np.asarray(anchor,np.float64);a/=max(np.linalg.norm(a),1e-30)
    w=np.asarray(wrong,np.float64);w/=max(np.linalg.norm(w),1e-30)
    r=np.asarray(reference,np.float64);r/=max(np.linalg.norm(r),1e-30)
    d=min(float(np.linalg.norm(r-a)),2.)
    if d<1e-12:return normalize(a.astype(np.float32))
    tangent=w-a*np.dot(w,a)
    if np.linalg.norm(tangent)<1e-10:
        tangent=rng.normal(size=len(a));tangent-=a*np.dot(tangent,a)
    tangent/=max(np.linalg.norm(tangent),1e-30)
    result=(1-.5*d*d)*a+d*math.sqrt(max(0,1-d*d/4))*tangent
    return normalize(result.astype(np.float32))


def construct_teachers(x,v,texts,pos,neg,gate,opt):
    byid=defaultdict(list)
    for k,r in enumerate(texts):byid[int(r['pid'])].append(k)
    rng=np.random.default_rng(opt.random_teacher_seed)
    n=len(x);true=np.empty_like(x);uncond=np.empty_like(x);wrong=np.empty_like(x)
    mask=np.zeros(n,dtype=np.uint8);counts=defaultdict(int);examples=[];angle_err=[]
    pids=np.asarray([r['pid'] for r in texts])
    pid_pools={int(p):np.flatnonzero(pids!=p) for p in np.unique(pids)}
    for i in range(n):
        candidates=[j for j in byid[int(pids[i])]
                    if int(texts[j]['img_idx'])!=int(texts[i]['img_idx'])]
        candidates=capped_peer_indices(candidates,x[i],x,opt.max_support)
        if not candidates:
            true[i]=uncond[i]=wrong[i]=x[i];counts['no_support']+=1;continue
        uncond[i]=normalize((1-opt.teacher_strength)*x[i]+opt.teacher_strength*np.mean(x[candidates],axis=0))
        rows=np.stack([triage_features(x[i],x[j],v[pos[i]],v[pos[j]],v[neg[i,0]])
                       for j in candidates])
        pp=gate_predict(rows,gate)
        use=(pp[:,2]>=opt.use_threshold)&(pp[:,2]>=pp[:,0]+opt.use_vs_harm_margin)
        reject=(pp[:,0]>=opt.harm_threshold_prob)&(pp[:,0]>=pp[:,2]+opt.use_vs_harm_margin)
        counts['available_queries']+=1;counts['total_supports']+=len(candidates)
        counts['accepted_supports']+=int(np.sum(use));counts['rejected_supports']+=int(np.sum(reject))
        if np.any(use):
            selected=np.asarray(candidates)[use]
            weight=np.maximum(pp[use,2]-pp[use,0],1e-5);weight/=weight.sum()
            source=np.average(x[selected],axis=0,weights=weight)
            true[i]=normalize((1-opt.teacher_strength)*x[i]+opt.teacher_strength*source)
            mask[i]=2;counts['accepted_queries']+=1
        elif np.any(reject):
            true[i]=x[i];mask[i]=0;counts['rejected_queries']+=1
        else:
            true[i]=x[i];mask[i]=1;counts['uncertain_queries']+=1
        # Exactly matched displacement magnitude, no real PID information in Wrong direction.
        pool=pid_pools[int(pids[i])]
        others=rng.choice(pool,size=min(len(candidates),len(pool)),replace=False)
        wd=np.mean(x[others],axis=0)
        wrong[i]=wrong_direction_matched(x[i],wd,true[i],rng)
        err=abs(np.linalg.norm(wrong[i].astype(np.float64)-x[i].astype(np.float64))-
                np.linalg.norm(true[i].astype(np.float64)-x[i].astype(np.float64)))
        angle_err.append(err)
        if i<1000:
            examples.append({'query_idx':i,'pid':int(pids[i]),'text':texts[i]['text'],
                             'available':len(candidates),'accepted':int(np.sum(use)),
                             'rejected':int(np.sum(reject)),
                             'status':('reject','uncertain','accept')[int(mask[i])],
                             'true_teacher_delta_angle':float(np.linalg.norm(true[i]-x[i])),
                             'wrong_angle_error':float(err)})
    if max(angle_err,default=0)>3e-4:raise AssertionError('WrongPID angle-matching error')
    return {'true':normalize(true),'unconditional':normalize(uncond),'wrong':normalize(wrong),
            'mask':mask}, dict(counts),examples,max(angle_err,default=0)


def stage_evidence07(out):
    _,opt,out=manifest_for_stage(out)
    needed=['triage_gate.json','gate_diagnostics.json','true_teacher.npy',
            'unconditional_teacher.npy','wrong_teacher.npy','support_mask.npy',
            'fit_negative_indices.npy','fit_text_indices.npy','fit_gallery_indices.npy']
    mark=out/'evidence_complete.json'
    if mark.is_file():
        try:
            state=json.loads(mark.read_text(encoding='utf-8'))
            if state.get('settings_hash')==opt.settings_hash and all((out/name).is_file() for name in needed):
                shapes=[np.load(out/(name+'.npy'),mmap_mode='r').shape[0]
                        for name in ('true_teacher','unconditional_teacher','wrong_teacher','support_mask','fit_negative_indices')]
                if len(set(shapes))==1 and shapes[0]==state.get('n_fit'):
                    msg('Complete EXP07 evidence artifacts found, skip.');return
        except (ValueError,KeyError,OSError):pass
    x,v,tx,im,pos=train_arrays03(out,opt)
    xf,vf,tf,gf,pf,ti,gi=partition(x,v,tx,im,pos,opt.fit_pids)
    xh,vh,th,gh,ph,_,_=partition(x,v,tx,im,pos,opt.holdout_pids)
    if set(r['pid'] for r in tf)&set(r['pid'] for r in th):raise AssertionError('Fit/holdout identity overlap')
    negf=mined_cache(out,'fit',xf,vf,tf,gf,opt.hard_k*2,opt.mine_batch,opt.settings_hash)
    hold_gallery_pids=np.asarray([r['pid'] for r in gh])
    available_hold=min(int(np.sum(hold_gallery_pids!=p)) for p in set(hold_gallery_pids))
    hold_k=min(opt.hard_k*2,available_hold)
    if hold_k<opt.hard_k:raise ValueError('Internal unseen identity gallery lacks sufficient different-PID negatives')
    negh=mined_cache(out,'holdout',xh,vh,th,gh,hold_k,opt.mine_batch,opt.settings_hash)
    if hold_k<2*opt.hard_k:
        msg('WARNING: internal holdout negative pool too small for disjoint selection/diagnosis halves; '+
            'PIDs are still entirely disjoint from mechanism training')
    # Split negatives for supervision vs a disjoint verification pool. Holdout
    # identities are also distinct (the stronger generalization diagnostic).
    f,y,d,ex=build_triage_rows(xf,vf,tf,pf,negf[:,:opt.hard_k],opt,1200)
    hold_eval_neg=(negh[:,opt.hard_k:] if hold_k>=2*opt.hard_k else negh[:,:opt.hard_k])
    h,hy,hd,h_ex=build_triage_rows(xh,vh,th,ph,hold_eval_neg,opt,300)
    gate=fit_three_way_gate(f,y,opt.split_seed)
    fp=gate_predict(f,gate);hp=gate_predict(h,gate)
    fit_diag=triage_diag(y,fp,d,opt);hold_diag=triage_diag(hy,hp,hd,opt)
    atomic_json(out/'triage_gate.json',gate)
    diag={'fit_identities':len(opt.fit_pids),'heldout_identities':len(opt.holdout_pids),
          'fit':fit_diag,'identity_disjoint_holdout':hold_diag,
          'identity_disjoint':True,'equal_priority_outcomes':['accept_useful_rate','reject_harmful_rate'],
          'no_failure_is_interpreted_as_proof_of_no_semantic_value':True,
          'evaluation_warning':'Heldout support labels require identity metadata: ORACLE mechanism diagnostic ONLY.'}
    hold_diag_status=hold_diag['reject_harmful_rate'] is not None and hold_diag['accept_useful_rate'] is not None
    if not hold_diag_status:
        msg('WARNING: heldout identity test lacks useful or harmful proxy examples; triage inconclusive')
    atomic_json(out/'gate_diagnostics.json',diag)
    write_csv(out/'triage_examples.csv',ex+h_ex,list((ex or h_ex)[0]) if ex or h_ex else [])
    teachers,counts,examples,maxerr=construct_teachers(xf,vf,tf,pf,negf[:,:opt.hard_k],gate,opt)
    for name,array in teachers.items():atomic_npy(out/(('support_mask' if name=='mask' else name+'_teacher')+'.npy'),array)
    atomic_npy(out/'fit_negative_indices.npy',negf[:,:opt.hard_k])
    atomic_npy(out/'fit_text_indices.npy',ti);atomic_npy(out/'fit_gallery_indices.npy',gi)
    write_csv(out/'teacher_examples.csv',examples,list(examples[0]) if examples else ['query_idx'])
    diag['student_teacher']={'counts':counts,'total_fit_queries':len(tf),'max_wrong_chord_error':maxerr,
        'support_coverage':float(np.mean(teachers['mask']!=1)),'accepted_query_fraction':float(np.mean(teachers['mask']==2)),
        'rejected_query_fraction':float(np.mean(teachers['mask']==0)),
        'uncertain_query_fraction':float(np.mean(teachers['mask']==1)),
        'warning':'Reject/accept without both classes is a degenerate support gate; report as failure.'}
    atomic_json(out/'gate_diagnostics.json',diag)
    atomic_json(mark,{'settings_hash':opt.settings_hash,'n_fit':len(tf)})
    msg('Heldout useful acceptance recall: '+str(hold_diag['accept_useful_rate'])+
        ' / harmful rejection recall: '+str(hold_diag['reject_harmful_rate']))
    msg('Teachers created: accepted '+str(counts.get('accepted_queries',0))+
        ', rejected '+str(counts.get('rejected_queries',0))+
        ', uncertain '+str(counts.get('uncertain_queries',0)))
# ========== EXP07: candidate-image-conditioned inference without PID ==========

def get_pair_model_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    class PairReranker(nn.Module):
        """Every control uses identical trainable capacity. No identity lookup.

        score(T,I) = cosine(T,I) + max_delta * sigmoid(gate(T,I)) * tanh(delta(T,I))
        Only IRRA embeddings are inputs. T and I are independently available at test.
        """
        def __init__(self,dim,hidden,max_delta):
            super().__init__()
            self.fc1=nn.Linear(2*dim+1,hidden)
            self.fc2=nn.Linear(hidden,hidden)
            self.gate=nn.Linear(hidden,1)
            self.delta=nn.Linear(hidden,1)
            self.max_delta=float(max_delta)
            nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
            nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,-.5)
        def forward(self,t,image,mode='normal'):
            import torch
            import torch.nn.functional as F
            # t: [B,D], image: [B,K,D].
            t=t[:,None,:].expand_as(image)
            cosine=(t*image).sum(-1)
            h=F.gelu(self.fc1(torch.cat([t*image,torch.abs(t-image),cosine[...,None]],-1)))
            h=F.gelu(self.fc2(h))
            gate_logit=self.gate(h).squeeze(-1)
            raw=self.max_delta*torch.tanh(self.delta(h).squeeze(-1))
            p=torch.sigmoid(gate_logit)
            if mode=='nogate':p=torch.ones_like(p)
            if mode=='noresidual':raw=torch.zeros_like(raw)
            correction=p*raw
            return cosine+correction, correction, gate_logit, p
    return PairReranker


def make_val_topk(out,opt):
    vd=Path(opt.val_cache_dir) if opt.val_cache_dir else out
    ims,txt,_=read_annotations(Path(opt.annotation_file),'val')
    x=normalize(np.load(vd/'text_features.npy',mmap_mode='r'))
    v=normalize(np.load(vd/'image_features.npy',mmap_mode='r'))
    topk=min(opt.eval_topk,len(v))
    meta=out/'val_topk_meta.json';path=out/'val_topk_indices.npy'
    stamp={'settings_hash':opt.settings_hash,'shape':[len(x),topk]}
    if meta.is_file() and path.is_file():
        try:
            p=np.load(path,mmap_mode='r',allow_pickle=False)
            if json.loads(meta.read_text(encoding='utf-8'))==stamp and p.shape==(len(x),topk):
                return np.asarray(p,dtype=np.int32)
        except (ValueError,OSError):pass
    out_idx=np.empty((len(x),topk),np.int32)
    for k in range(0,len(x),128):
        scores=x[k:k+128]@v.T
        ind=np.argpartition(-scores,topk-1,axis=1)[:,:topk]
        order=np.argsort(-np.take_along_axis(scores,ind,axis=1),axis=1,kind='stable')
        out_idx[k:k+len(ind)]=np.take_along_axis(ind,order,axis=1)
    atomic_npy(path,out_idx);atomic_json(meta,stamp)
    return out_idx


def load_val_topk_cached(out, opt):
    """Only loads already-computed Top-K. Never invokes NumPy BLAS in torch train worker.

    A Windows MKL/Intel-OpenMP collision can be triggered by performing NumPy
    matrix multiplication after torch has initialized its OpenMP runtime. The
    candidate search is therefore done by the dedicated NumPy-only topk stage.
    """
    vd=Path(opt.val_cache_dir) if opt.val_cache_dir else out
    ims,txt,_=read_annotations(Path(opt.annotation_file),'val')
    n=len(txt)
    topk=min(opt.eval_topk,len(ims))
    meta=out/'val_topk_meta.json'
    path=out/'val_topk_indices.npy'
    expected={'settings_hash':opt.settings_hash,'shape':[n,topk]}
    if not meta.is_file() or not path.is_file():
        raise RuntimeError('Top-K candidates not ready. Run the separate NumPy-only topk stage before training.')
    saved=json.loads(meta.read_text(encoding='utf-8'))
    if saved!=expected:
        raise RuntimeError('Top-K candidate fingerprint mismatch; do not reuse stale retrieval indices.')
    candidate=np.load(path,mmap_mode='r',allow_pickle=False)
    if candidate.shape!=(n,topk) or candidate.dtype.kind not in 'iu':
        raise RuntimeError('Top-K candidate shape or dtype mismatch.')
    if np.any(candidate<0) or np.any(candidate>=len(ims)):
        raise RuntimeError('Top-K candidate index outside the gallery.')
    if topk>1 and np.any(np.diff(np.sort(candidate,axis=1),axis=1)==0):
        raise RuntimeError('Top-K candidate indices are duplicated within a query.')
    return np.array(candidate,dtype=np.int32,copy=True)


def stage_topk07(out):
    """Compute full-gallery fixed shortlist ONLY in a NumPy-only process."""
    _,opt,out=manifest_for_stage(out)
    # Crucially, do not import torch in this stage.
    make_val_topk(out,opt)
    indices=load_val_topk_cached(out,opt)
    msg(f'EXP07 fixed top-{indices.shape[1]} validated: {indices.shape[0]} queries; torch not loaded.')


def model_record_done(out,opt,stem,n,k):
    ck=out/(stem+'_student.pt');sc=out/(stem+'_topk_score.npy')
    gate=out/(stem+'_topk_gate.npy');meta=out/(stem+'_training_meta.json')
    if not all(z.is_file() for z in (ck,sc,gate,meta)):return False
    try:
        m=json.loads(meta.read_text(encoding='utf-8'))
        a=np.load(sc,mmap_mode='r',allow_pickle=False);b=np.load(gate,mmap_mode='r',allow_pickle=False)
        return (m.get('complete') and m.get('settings_hash')==opt.settings_hash and
                m.get('checkpoint_sha256')==sha256_small_file(ck) and
                a.shape==(n,k) and b.shape==(n,k) and
                a.dtype==np.float32 and b.dtype==np.float32 and
                np.isfinite(a).all() and np.isfinite(b).all())
    except (OSError,ValueError,KeyError):return False


def cache_model_predictions(model, xv,vg,indices,device,opt,out,stem):
    import torch
    n,k=indices.shape
    s=np.lib.format.open_memmap(out/(stem+'_topk_score.partial.npy'),mode='w+',dtype='float32',shape=(n,k))
    g=np.lib.format.open_memmap(out/(stem+'_topk_gate.partial.npy'),mode='w+',dtype='float32',shape=(n,k))
    model.eval()
    # Dynamic chunk bound for 6 GiB GPU: <=128*topK*D temporary vectors.
    qbatch=max(1,min(opt.eval_query_batch,8192//max(1,k)))
    with torch.inference_mode():
        for start in range(0,n,qbatch):
            end=min(start+qbatch,n)
            t=xv[start:end].to(device)
            idx=torch.from_numpy(indices[start:end].astype(np.int64)).to(device)
            imgs=vg[idx]
            score,_,_,gate=model(t,imgs)
            s[start:end]=score.float().cpu().numpy();g[start:end]=gate.float().cpu().numpy()
    s.flush();g.flush();del s,g
    _replace_with_retries(out/(stem+'_topk_score.partial.npy'),out/(stem+'_topk_score.npy'))
    _replace_with_retries(out/(stem+'_topk_gate.partial.npy'),out/(stem+'_topk_gate.npy'))


def train_one_epoch(model,optimizer,order,tx,imgs,pos,neg,teachers,masks,method,opt,device,seed):
    import torch
    import torch.nn.functional as F
    params=list(model.parameters());model.train()
    use_reg=method in ('reg','reg_shuffle','reg_conditional_true','reg_conditional_wrong')
    source={'unconditional_true':'unconditional','conditional_true':'true',
            'conditional_wrong':'wrong','reg_conditional_true':'true',
            'reg_conditional_wrong':'wrong'}.get(method)
    conditional=source in ('true','wrong')
    sums=defaultdict(float);count=defaultdict(int)
    # Same random decisions across branches per seed; no PID at inference.
    for start in range(0,len(order),opt.train_batch):
        inds=order[start:start+opt.train_batch]
        original=tx[inds].to(device)
        samples=torch.cat((pos[inds,None],neg[inds]),1).to(device)
        candidates=imgs[samples]
        score,correction,gate_logits,gates=model(original,candidates)
        base=torch.sum(original[:,None,:]*candidates,dim=-1)
        rank_loss=F.softplus((score[:,1:]-score[:,:1]+opt.margin)/opt.temperature).mean()
        # Small shared preservation across all branches, including Rank baseline.
        protect=(correction**2).mean()
        primary=opt.rank_weight*rank_loss+opt.protect_weight*protect
        extra=[]
        if use_reg:
            # Same-image relation preservation, no other caption or identity data.
            reg_loss=F.smooth_l1_loss(score[:,:1]-score[:,1:],base[:,:1]-base[:,1:])
            extra.append(('reg',reg_loss))
        if source:
            teacher=teachers[source][inds].to(device)
            delta_t=torch.einsum('bd,bkd->bk',teacher-original,candidates).detach()
            if conditional:
                status=masks[inds].to(device)
                ok=status==2;bad=status==0;unknown=status==1
                # Explicit *both* positive (useful) and negative (reject) objectives;
                # normalized separately so class imbalance does not erase either.
                benefit=(F.smooth_l1_loss(correction[ok],delta_t[ok]) if bool(ok.any())
                         else torch.zeros((),device=device))
                refusal=(F.smooth_l1_loss(correction[bad],torch.zeros_like(correction[bad]))
                         if bool(bad.any()) else torch.zeros((),device=device))
                uncertain=(F.smooth_l1_loss(correction[unknown],torch.zeros_like(correction[unknown]))
                           if bool(unknown.any()) else torch.zeros((),device=device))
                known=ok|bad
                if bool(known.any()):
                    labels=ok[known,None].float().expand(-1,correction.shape[1])
                    gate_loss=F.binary_cross_entropy_with_logits(gate_logits[known],labels)
                else:gate_loss=torch.zeros((),device=device)
                sem_loss=benefit+refusal+.25*uncertain+opt.gate_bce_weight*gate_loss
                sums['useful_batch_fraction']+=float(ok.float().mean())*len(inds)
                sums['rejected_batch_fraction']+=float(bad.float().mean())*len(inds)
                sums['uncertain_batch_fraction']+=float(unknown.float().mean())*len(inds)
            else:
                sem_loss=F.smooth_l1_loss(correction,delta_t)
            extra.append(('semantic',sem_loss))
        if method=='reg_shuffle':
            # Distribution-matched artificial accept/reject supervision, no other text.
            # The seed/mask permutation is shared across epochs independent of labels.
            m=masks[inds].to(device)
            labels=(m==2).float()
            gen=torch.Generator().manual_seed(seed+start+104729)
            shuffled=labels[torch.randperm(len(labels),generator=gen)].to(device)
            gate_loss=F.binary_cross_entropy_with_logits(gate_logits,
                        shuffled[:,None].expand_as(gate_logits))
            extra.append(('shuffle_gate',gate_loss))
        # Norm-matched auxiliary gradients. Log gradient cosine so apparent
        # semantic gains are not automatically attributed to identity semantics.
        gr=torch.autograd.grad(primary,params,retain_graph=bool(extra),allow_unused=True)
        main_g=[torch.zeros_like(p) if h is None else h.detach() for p,h in zip(params,gr)]
        norm=math.sqrt(sum(float(g.double().square().sum()) for g in main_g))
        update=[g.clone() for g in main_g]
        for ei,(name,loss) in enumerate(extra):
            grads=torch.autograd.grad(loss,params,retain_graph=(ei<len(extra)-1),allow_unused=True)
            gg=[torch.zeros_like(p) if h is None else h.detach() for p,h in zip(params,grads)]
            gn=math.sqrt(sum(float(g.double().square().sum()) for g in gg))
            scale=(min(opt.aux_grad_ratio*norm/max(gn,1e-30),opt.max_aux_scale)
                   if gn>1e-12 and norm>1e-12 else 0.)
            dot=sum(float((u.double()*w.double()).sum()) for u,w in zip(main_g,gg))
            cosine=dot/max(norm*gn,1e-30)
            for i in range(len(update)):update[i].add_(gg[i],alpha=scale)
            sums[name+'_grad_ratio']+=scale*gn/max(norm,1e-30)*len(inds)
            sums[name+'_grad_cosine']+=cosine*len(inds)
            sums[name+'_loss']+=float(loss.detach())*len(inds)
            count[name+'_capped']+=int(scale>=opt.max_aux_scale-1e-10)
        optimizer.zero_grad(set_to_none=True)
        for p,g in zip(params,update):p.grad=g
        grad_norm=math.sqrt(sum(float(v.double().square().sum()) for v in update))
        clipped=torch.nn.utils.clip_grad_norm_(params,5.)
        before=[p.detach().clone() for p in params]
        optimizer.step()
        actual_step=math.sqrt(sum(float((p.detach()-q).double().square().sum()) for p,q in zip(params,before)))
        if not math.isfinite(actual_step):raise FloatingPointError('Nonfinite AdamW update')
        sums['rank_loss']+=float(rank_loss.detach())*len(inds)
        sums['protect_loss']+=float(protect.detach())*len(inds)
        sums['main_grad_norm']+=norm*len(inds)
        sums['applied_grad_norm']+=grad_norm*len(inds)
        sums['adamw_step_norm']+=actual_step*len(inds)
        sums['student_gate_mean']+=float(gates.detach().mean())*len(inds)
        count['clipped']+=int(clipped>5);count['batches']+=1
    return {**{k:float(v)/len(order) for k,v in sums.items()},**dict(count)}


def commit_pair_model(model,optimizer,out,stem,history,seed,method, xv,vg,topk,opt,device):
    import torch
    cache_model_predictions(model,xv,vg,topk,device,opt,out,stem)
    ck=out/(stem+'_student.pt');temp=out/(stem+'_student.partial.pt')
    torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                'settings_hash':opt.settings_hash,'seed':seed,'method':method,
                'dim':int(xv.shape[1]),'max_delta':opt.max_delta},temp)
    _replace_with_retries(temp,ck)
    atomic_json(out/(stem+'_training_meta.json'),{
        'complete':True,'settings_hash':opt.settings_hash,'checkpoint_sha256':sha256_small_file(ck),
        'method':method,'seed':seed,'history':history,'n_queries':len(xv),'topk':topk.shape[1]})


def stage_train07(out):
    import torch
    torch.set_num_threads(1)
    _,opt,out=manifest_for_stage(out)
    device=torch.device('cuda' if opt.device=='cuda' or (opt.device=='auto' and torch.cuda.is_available()) else 'cpu')
    if opt.device=='cuda' and device.type!='cuda':raise RuntimeError('CUDA requested but unavailable')
    x,v,tx,ims,pos=train_arrays03(out,opt)
    xf,vf,tf,gf,pf,_,_=partition(x,v,tx,ims,pos,opt.fit_pids)
    if not (out/'evidence_complete.json').is_file():raise RuntimeError('Evidence stage incomplete')
    mark=json.loads((out/'evidence_complete.json').read_text(encoding='utf-8'))
    if mark.get('settings_hash')!=opt.settings_hash:raise RuntimeError('Evidence fingerprint mismatch')
    negatives=np.load(out/'fit_negative_indices.npy',allow_pickle=False)
    if negatives.shape!=(len(xf),opt.hard_k):raise RuntimeError('Mined negative shape mismatch')
    support_mask=np.load(out/'support_mask.npy',allow_pickle=False)
    if support_mask.shape!=(len(xf),) or not np.all(np.isin(support_mask,[0,1,2])):
        raise RuntimeError('Invalid support triage mask')
    teachers={k:torch.from_numpy(np.array(np.load(out/(k+'_teacher.npy'),allow_pickle=False),copy=True))
              for k in ('true','unconditional','wrong')}
    valdir=Path(opt.val_cache_dir) if opt.val_cache_dir else out
    xx=normalize(np.load(valdir/'text_features.npy',mmap_mode='r'))
    vv=normalize(np.load(valdir/'image_features.npy',mmap_mode='r'))
    # Top-K is computed and validated by the separate NumPy-only worker.
    # Never call make_val_topk() in this torch training subprocess: on Windows,
    # NumPy BLAS after torch may load a second libiomp5md.dll and abort process.
    topk=load_val_topk_cached(out,opt)
    msg(f'EXP07 train: loaded precomputed top-{topk.shape[1]} (no NumPy BLAS candidate search).')
    data=torch.from_numpy(np.array(xf,copy=True));gallery=torch.from_numpy(np.array(vf,copy=True)).to(device)
    posit=torch.from_numpy(pf.astype(np.int64));neg=torch.from_numpy(negatives.astype(np.int64))
    mask=torch.from_numpy(support_mask.astype(np.int64))
    vt=torch.from_numpy(np.array(xx,copy=True));vi=torch.from_numpy(np.array(vv,copy=True)).to(device)
    model_class=get_pair_model_class()
    for seed in opt.seeds:
        rootstem=f'warmup_s{seed}'
        if not model_record_done(out,opt,rootstem,len(xx),topk.shape[1]):
            torch.manual_seed(seed)
            if device.type=='cuda':torch.cuda.manual_seed_all(seed)
            net=model_class(xf.shape[1],opt.hidden,opt.max_delta).to(device)
            optim=torch.optim.AdamW(net.parameters(),lr=opt.lr,weight_decay=1e-4)
            generator=torch.Generator().manual_seed(seed)
            history=[]
            for epoch in range(1,opt.warmup_epochs+1):
                order=torch.randperm(len(xf),generator=generator)
                stats=train_one_epoch(net,optim,order,data,gallery,posit,neg,teachers,mask,'warmup',opt,device,seed)
                history.append({'method':'warmup','seed':seed,'epoch':epoch,'phase':'warmup',**stats})
                msg(f'warmup_s{seed} epoch {epoch}/{opt.warmup_epochs}: rank={stats["rank_loss"]:.5f}')
            commit_pair_model(net,optim,out,rootstem,history,seed,'warmup',vt,vi,topk,opt,device)
            del net,optim
        else:msg(rootstem+': validated completed, skip')
        saved=torch.load(out/(rootstem+'_student.pt'),map_location='cpu',weights_only=False)
        if saved['settings_hash']!=opt.settings_hash:raise RuntimeError('Warmup fingerprint mismatch')
        for method in METHODS[1:]:
            stem=f'{method}_s{seed}'
            if model_record_done(out,opt,stem,len(xx),topk.shape[1]):
                msg(stem+': validated completed, skip');continue
            torch.manual_seed(seed)
            net=model_class(xf.shape[1],opt.hidden,opt.max_delta).to(device)
            net.load_state_dict(saved['model_state_dict'])
            optim=torch.optim.AdamW(net.parameters(),lr=opt.lr,weight_decay=1e-4)
            optim.load_state_dict(saved['optimizer_state_dict'])
            gen=torch.Generator().manual_seed(seed+10000)
            history=[]
            for epoch in range(1,opt.continue_epochs+1):
                order=torch.randperm(len(xf),generator=gen)
                stats=train_one_epoch(net,optim,order,data,gallery,posit,neg,teachers,mask,method,opt,device,seed)
                history.append({'method':method,'seed':seed,'epoch':epoch,'phase':'continue',**stats})
                msg(f'{stem} epoch {epoch}/{opt.continue_epochs}: rank={stats["rank_loss"]:.5f}, '
                    f'gate={stats["student_gate_mean"]:.3f}')
            commit_pair_model(net,optim,out,stem,history,seed,method,vt,vi,topk,opt,device)
            del net,optim
            if device.type=='cuda':torch.cuda.empty_cache()
    history=all_training_records(out)
    if history:
        cols=list(dict.fromkeys(k for r in history for k in r))
        write_csv(out/'training_history.csv',history,cols,allow_locked_fallback=True)
    msg('EXP07 training complete: shared warmup, matched student architectures and branches.')
# ================= EXP07 eval in a separate NumPy-only process ===============

def evaluate07(out):
    _,opt,out=manifest_for_stage(out)
    ims,txt,_=read_annotations(Path(opt.annotation_file),'val')
    vd=Path(opt.val_cache_dir) if opt.val_cache_dir else out
    x=normalize(np.load(vd/'text_features.npy',mmap_mode='r'))
    v=normalize(np.load(vd/'image_features.npy',mmap_mode='r'))
    # Construct scores using only (caption, image); PID arrays BELOW this point
    # are evaluation labels only. Never used by inference or candidate selection.
    base_scores=np.asarray(x@v.T,dtype=np.float32)
    topk=np.load(out/'val_topk_indices.npy',allow_pickle=False)
    n,k=topk.shape;row=np.arange(n)[:,None]
    base_topk=base_scores[row,topk]
    pids=np.asarray([r['pid'] for r in txt]);gpids=np.asarray([r['pid'] for r in ims])
    if set(pids)&(set(opt.fit_pids)|set(opt.holdout_pids)):
        raise RuntimeError('Train/Val PID leakage detected')
    baseline,br=metrics(base_scores,pids,gpids)
    if opt.expected_baseline_r1 is not None and abs(baseline['R1']-opt.expected_baseline_r1)>.05:
        raise RuntimeError('Original IRRA baseline mismatch: wrong data, cache or checkpoint')
    metrics_rows=[{'method':'original_IRRA','seed':-1,'protocol':'STANDARD_SINGLE_TEXT_FIXED_IRRA_TOPK_RERANK',**baseline}]
    byseed={};cases=[];queries=[];pair=[];gate_abl=[]
    base_ok=br['rank']==1
    for seed in opt.seeds:
        results={}
        for method in METHODS:
            stem=f'{method}_s{seed}'
            if not model_record_done(out,opt,stem,n,k):raise RuntimeError('Incomplete student: '+stem)
            pair_score=np.load(out/(stem+'_topk_score.npy'),mmap_mode='r',allow_pickle=False)
            pair_gate=np.load(out/(stem+'_topk_gate.npy'),mmap_mode='r',allow_pickle=False)
            scores=base_scores.copy();scores[row,topk]=pair_score
            result,r=metrics(scores,pids,gpids)
            ok=r['rank']==1
            results[method]=(result,r,ok)
            # Gate effectiveness against a constant gate with equal average activation.
            const=float(np.mean(pair_gate))
            raw=(np.asarray(pair_score)-base_topk)/np.maximum(pair_gate,1e-6)
            const_scores=base_scores.copy();const_scores[row,topk]=base_topk+const*raw
            cg,cr=metrics(const_scores,pids,gpids)
            gate_abl.append({'seed':seed,'method':method,'gate_mean':const,
                'gate_gt_05_fraction':float(np.mean(pair_gate>.5)),
                'gate_gt_08_fraction':float(np.mean(pair_gate>.8)),
                'adaptive_R1':result['R1'],'constant_gate_R1':cg['R1'],
                'adaptive_minus_constant_R1_pp':result['R1']-cg['R1'],
                'adaptive_mAP':result['mAP'],'constant_gate_mAP':cg['mAP'],
                'adaptive_minus_constant_mAP_pp':result['mAP']-cg['mAP']})
            metrics_rows.append({'method':method,'seed':seed,
                  'protocol':'STANDARD_SINGLE_TEXT_FIXED_IRRA_TOPK_RERANK',
                  'candidate_topk':k,**result,
                  'delta_R1_vs_IRRA_pp':result['R1']-baseline['R1'],
                  'delta_mAP_vs_IRRA_pp':result['mAP']-baseline['mAP'],
                  'repaired_vs_IRRA':int(np.sum(~base_ok&ok)),
                  'harmed_vs_IRRA':int(np.sum(base_ok&~ok)),
                  'mean_margin_change':float(np.mean(r['margin']-br['margin'])),
                  'gate_mean':const})
            msg(f'seed {seed} {method}: R1={result["R1"]:.4f} '
                f'(ΔIRRA {result["R1"]-baseline["R1"]:+.4f}), mAP={result["mAP"]:.4f}, '
                f'gate={const:.3f}')
            for j,t in enumerate(txt):
                rec={'seed':seed,'method':method,'query_idx':j,'pid':int(pids[j]),
                     'text':t['text'],'baseline_rank':int(br['rank'][j]),
                     'rank':int(r['rank'][j]),'baseline_margin':float(br['margin'][j]),
                     'margin':float(r['margin'][j]),'delta_margin':float(r['margin'][j]-br['margin'][j]),
                     'baseline_top1':int(base_ok[j]),'correct_top1':int(ok[j]),
                     'query_gate_mean':float(np.mean(pair_gate[j]))}
                queries.append(rec)
                if base_ok[j]!=ok[j]:cases.append({**rec,'change':'repaired' if ok[j] else 'harmed'})
        byseed[seed]=results
        def diff(model,reference):
            a,_,ao=results[model];b,_,bo=results[reference]
            pair.append({'seed':seed,'method':model,'reference':reference,
                         'delta_R1_pp':a['R1']-b['R1'],'delta_mAP_pp':a['mAP']-b['mAP'],
                         'newly_correct':int(np.sum(ao&~bo)),
                         'newly_wrong':int(np.sum(~ao&bo))})
        for meth in METHODS[1:]:diff(meth,'rank_continue')
        for meth,ref in [('reg_conditional_true','reg'),
                         ('reg_conditional_true','reg_conditional_wrong'),
                         ('conditional_true','conditional_wrong'),
                         ('conditional_true','unconditional_true'),
                         ('reg_conditional_true','reg_shuffle'),
                         ('reg_conditional_true','conditional_true')]:diff(meth,ref)
    def make_csv(filename,rows):
        if rows:write_csv(out/filename,rows,list(dict.fromkeys(k for r in rows for k in r)))
    make_csv('method_summary.csv',metrics_rows)
    make_csv('gate_ablation.csv',gate_abl)
    make_csv('paired_rank_comparisons.csv',pair)
    make_csv('query_deltas.csv',queries)
    make_csv('case_changes.csv',cases)
    agg=[]
    for meth in ('original_IRRA',)+METHODS:
        this=[r for r in metrics_rows if r['method']==meth]
        e={'method':meth,'n_seeds':len(this)}
        for key in ('R1','R5','R10','mAP','mINP'):
            a=np.asarray([q[key] for q in this],np.float64)
            e[key+'_mean']=float(a.mean());e[key+'_sd']=float(a.std(ddof=1)) if len(a)>1 else 0.
            e['delta_'+key+'_IRRA_pp']=float(a.mean()-baseline[key])
        agg.append(e)
    make_csv('aggregate_summary.csv',agg)
    factorial=[]
    for seed in opt.seeds:
        rr=byseed[seed]
        row={'seed':seed}
        for key in ('R1','mAP'):
            get=lambda m:rr[m][0][key]
            a=get('rank_continue');b=get('reg');c=get('conditional_true');d=get('reg_conditional_true')
            row[key+'_reg_gain_pp']=b-a
            row[key+'_semantic_gain_unregularized_pp']=c-a
            row[key+'_semantic_gain_over_reg_pp']=d-b
            row[key+'_interaction_pp']=d-b-c+a
            row[key+'_true_vs_wrong_pp']=c-get('conditional_wrong')
            row[key+'_reg_true_vs_reg_wrong_pp']=d-get('reg_conditional_wrong')
        factorial.append(row)
    make_csv('factorial_effects.csv',factorial)
    history=all_training_records(out)
    if history:
        cols=list(dict.fromkeys(k for r in history for k in r))
        write_csv(out/'training_history.csv',history,cols,allow_locked_fallback=True)
    grad={}
    for method in METHODS:
        rows=[r for r in history if r.get('method')==method]
        grad[method]={key:float(np.mean([r.get(key,0.) for r in rows])) if rows else None
                      for key in ('reg_grad_ratio','semantic_grad_ratio','semantic_grad_cosine',
                                  'reg_grad_cosine','shuffle_gate_grad_ratio','adamw_step_norm')}
    atomic_json(out/'gradient_diagnostics.json',{'methods':grad,'aux_ratio':opt.aux_grad_ratio,
         'warning':'Norm matching does not match gradient direction or AdamW effective parameter update.'})
    gate_diag=json.loads((out/'gate_diagnostics.json').read_text(encoding='utf-8'))
    report={'experiment':'EXP07 conditional text complementarity / reject harmful vs use useful',
            'protocol':'Single original query + candidate image only, fixed original IRRA Top-K reranking of FULL gallery',
            'fixed_topk':k,'original_irra_baseline':baseline,'methods':agg,
            'identity_split':{'fit_pids':len(opt.fit_pids),'internal_unseen_train_pids':len(opt.holdout_pids),
                              'train_val_pid_overlap':0},
            'gate_diagnostics':gate_diag,'factorial_effects':factorial,
            'paired_controls':pair,'gate_ablation':gate_abl,
            'research_objectives':[
              '1. Reliable, nontrivial SAME-IDENTITY MULTI-TEXT complementarity beyond matched regularization and wrong PID',
              '2. Reliable SAME-IDENTITY MULTI-VIEW IMAGE complementarity',
              '3. Rigorous joint benefits of both mechanisms'],
            'limitations':[
              'No Val PID is used for candidate generation, inference or support selection; PID used to compute evaluation metrics ONLY.',
              'Gate utility classes come from image-level ranking effects, not human verified text semantic truth.',
              'Top-K is fixed by original IRRA and reranked; no unrestricted full-gallery pair model claim.',
              'Teacher reliability measured on heldout TRAIN identities is PID-assisted ORACLE diagnosis, not deployment.',
              'Repeated Val development and only one dataset do not prove independent test generalization.',
              'Rank, regularization, semantic knowledge and their interaction may each contribute; do not force exclusive attribution.',
              'If useful acceptance or harmful rejection is degenerate, do not claim conditional gating succeeds.',
              'Cross identity teacher advantages must exceed both same-architecture no-support and wrong-PID controls.']}
    atomic_json(out/'experiment07_report.json',report)
    atomic_json(out/'mechanism_decision.json',{'automatic_success_declared':False,
       'requirements':['Useful acceptance recall on identity-disjoint train holdout',
                       'Harmful rejection recall on same holdout',
                       'Conditional-True > Conditional-Wrong and no-support on multiple seeds',
                       'Reg+Conditional-True > Reg and Reg+Conditional-Wrong',
                       'R1 and mAP comparisons plus repair/harm counts and gate ablation',
                       'Method locking before independent TEST evaluation'],
       'warning':'This is an evidence report, not an automatic scientific success claim.'})
    msg('EXP07 complete: full-gallery retrieval metrics, triage and gated controls exported.')
# ================ EXP07 integration tests / orchestration ====================

def self_test07():
    rng=np.random.default_rng(707)
    # Separate intentionally separable triage labels verify that the 3-class
    # learner can ACCEPT useful and REJECT harmful, not just return uncertain.
    toy=np.vstack([rng.normal(loc=[-2]+[0]*7,scale=.35,size=(150,8)),
                   rng.normal(loc=[0]*8,scale=.35,size=(150,8)),
                   rng.normal(loc=[2]+[0]*7,scale=.35,size=(150,8))])
    toy_y=np.repeat([0,1,2],150)
    test_gate=fit_three_way_gate(toy,toy_y,707)
    predicted=gate_predict(toy,test_gate)
    assert np.mean(np.argmax(predicted,axis=1)[toy_y==0]==0)>.9
    assert np.mean(np.argmax(predicted,axis=1)[toy_y==2]==2)>.9
    with tempfile.TemporaryDirectory(prefix='irra_exp07_test_') as tmp:
        out=Path(tmp);d=24
        records=[];tr_txt=[];tr_img=[];val_txt=[];val_img=[]
        for pid in range(1,13):
            center=normalize(rng.normal(size=d))
            for view in range(3):
                records.append({'split':'train','id':pid,'file_path':f'train/{pid}_{view}.jpg',
                                'captions':[f'person {pid} view {view} shirt',
                                            f'person {pid} view {view} shoes']})
                tr_img.append(normalize(center+.2*rng.normal(size=d)))
                for j in range(2):tr_txt.append(normalize(center+.3*rng.normal(size=d)))
        for pid in range(50,55):
            center=normalize(rng.normal(size=d))
            for view in range(2):
                records.append({'split':'val','id':pid,'file_path':f'val/{pid}_{view}.jpg',
                                'captions':[f'val person {pid} view {view} one',
                                            f'val person {pid} view {view} two']})
                val_img.append(normalize(center+.2*rng.normal(size=d)))
                for j in range(2):val_txt.append(normalize(center+.3*rng.normal(size=d)))
        atomic_json(out/'synthetic_annotations.json',records)
        selected=list(range(1,13)); fit,hold=pid_split(selected,.75,776)
        atomic_json(out/'selected_train_pids.json',selected)
        for name,z in [('train_text',tr_txt),('train_image',tr_img),('text',val_txt),('image',val_img)]:
            np.save(out/(name+'_features.npy'),np.asarray(z,dtype=np.float32))
        ft={'synthetic':'train07'};fv={'synthetic':'val07'}
        fp_img=train_img_fingerprint(SimpleNamespace(fp_train=ft),list(range(len(tr_img))))
        for name,f,n in [('train_text',ft,len(tr_txt)),('train_image',fp_img,len(tr_img)),
                          ('text',fv,len(val_txt)),('image',fv,len(val_img))]:
            atomic_json(out/(name+'_cache_meta.json'),{'fingerprint':f,'done':n,'shape':[n,d]})
        settings={
            'project_root':str(out),'annotation_file':str(out/'synthetic_annotations.json'),
            'config_file':str(out/'fake.yaml'),'checkpoint':str(out/'fake.pt'),'images_dir':str(out),
            'val_cache_dir':'','train_text_cache_dir':'','train_image_cache_dir':'',
            'selected_pids':selected,'fit_pids':fit,'holdout_pids':hold,
            'fp_train':ft,'fp_val':fv,'device':'cpu','seeds':[42,43],
            'warmup_epochs':1,'continue_epochs':1,'max_support':4,'hard_k':2,
            'mine_batch':30,'teacher_strength':.35,'help_threshold':.003,
            'harm_threshold':.003,'instance_floor':.015,'use_threshold':.42,
            'harm_threshold_prob':.42,'use_vs_harm_margin':.03,
            'wrong_pool':32,'random_teacher_seed':2003,'split_seed':776,
            'train_batch':12,'hidden':32,'max_delta':.08,
            'lr':.0004,'rank_weight':.2,'protect_weight':.5,
            'margin':.05,'temperature':.07,'gate_bce_weight':.1,
            'aux_grad_ratio':.2,'max_aux_scale':100000.,
            'eval_topk':6,'eval_query_batch':16,'expected_baseline_r1':None,
            'settings_hash':'synthetic07'}
        atomic_json(out/'run_info.json',{'settings':settings,'status':'self_test'})
        src=Path(__file__).resolve()
        env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
        def run(stage):
            p=subprocess.run([sys.executable,str(src),'--_worker',stage,'--_run-dir',str(out)],
                             text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             env=env,timeout=240)
            if p.returncode:raise AssertionError(f'EXP07 selftest {stage} FAILED:\n{p.stdout[-10000:]}')
            return p.stdout
        for st in ('encode','evidence','topk','train','eval'):run(st)
        r=json.loads((out/'experiment07_report.json').read_text(encoding='utf-8'))
        assert len(r['methods'])==len(METHODS)+1
        assert r['identity_split']['train_val_pid_overlap']==0
        assert r['gate_diagnostics']['identity_disjoint']
        assert len(list(out.glob('*_training_meta.json')))==len(METHODS)*2
        assert r['fixed_topk']==6
        g=r['gate_diagnostics']['identity_disjoint_holdout']
        assert all(len(row)==3 for row in g['confusion_rows_true_cols_pred'])
        for method in METHODS:
            for seed in (42,43):
                assert np.isfinite(np.load(out/f'{method}_s{seed}_topk_score.npy')).all()
        # Verify independently generated shortlist matches a NumPy reference.
        cache=load_val_topk_cached(out,SimpleNamespace(**settings))
        base_x=normalize(np.load(out/'text_features.npy'))
        base_v=normalize(np.load(out/'image_features.npy'))
        ref=base_x @ base_v.T
        expected=np.argsort(-ref,axis=1,kind='stable')[:,:cache.shape[1]]
        assert np.all(np.take_along_axis(ref,cache,axis=1) >=
                      np.take_along_axis(ref,expected[:,-1:],axis=1)-1e-6)
        # Verify restart skips completed models and evidence, keeping outputs untouched.
        oldhash=sha256_small_file(out/'conditional_true_s42_student.pt')
        output=run('train')
        assert output.count('validated completed, skip')==len(METHODS)*2
        assert sha256_small_file(out/'conditional_true_s42_student.pt')==oldhash
        assert 'skip' in run('evidence')
        assert 'validated' in run('topk')
        shutil.copy2(src,out/src.name);package_for_chat(out)
        with zipfile.ZipFile(out/'发送给ChatGPT.zip') as z:
            for req in ('gate_diagnostics.json','teacher_examples.csv','gate_ablation.csv',
                        'method_summary.csv','experiment07_report.json','mechanism_decision.json',
                        'training_history.csv',src.name):
                assert req in z.namelist(),req
            assert not any(s.endswith(('.pt','.npy')) for s in z.namelist())
    msg('EXP07 SELF-TEST PASSED: 3-class gate, identity holdout, 18 students, full gallery, resume, ZIP.')


def worker07(stage,out):
    out=Path(out)
    if stage=='encode':stage_encode03(out)
    elif stage=='evidence':stage_evidence07(out)
    elif stage=='topk':stage_topk07(out)
    elif stage=='train':stage_train07(out)
    elif stage=='eval':evaluate07(out)
    else:raise ValueError('Unknown worker stage '+stage)


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
    parser.add_argument('--mechanism-fit-fraction',type=float,default=.8)
    parser.add_argument('--split-seed',type=int,default=776)
    parser.add_argument('--seeds',default='42,43,44')
    parser.add_argument('--warmup-epochs',type=int,default=2)
    parser.add_argument('--continue-epochs',type=int,default=2)
    parser.add_argument('--text-batch',type=int,default=64)
    parser.add_argument('--image-batch',type=int,default=8)
    parser.add_argument('--train-batch',type=int,default=128)
    parser.add_argument('--hidden',type=int,default=64)
    parser.add_argument('--lr',type=float,default=.0003)
    parser.add_argument('--rank-weight',type=float,default=.2)
    parser.add_argument('--protect-weight',type=float,default=.5)
    parser.add_argument('--margin',type=float,default=.05)
    parser.add_argument('--temperature',type=float,default=.07)
    parser.add_argument('--max-delta',type=float,default=.08)
    parser.add_argument('--teacher-strength',type=float,default=.35)
    parser.add_argument('--max-support',type=int,default=8)
    parser.add_argument('--hard-k',type=int,default=4)
    parser.add_argument('--mine-batch',type=int,default=96)
    parser.add_argument('--help-threshold',type=float,default=.005)
    parser.add_argument('--harm-threshold',type=float,default=.005)
    parser.add_argument('--instance-floor',type=float,default=.015)
    parser.add_argument('--use-threshold',type=float,default=.50)
    parser.add_argument('--harm-threshold-prob',type=float,default=.50)
    parser.add_argument('--use-vs-harm-margin',type=float,default=.10)
    parser.add_argument('--wrong-pool',type=int,default=256)
    parser.add_argument('--random-teacher-seed',type=int,default=2003)
    parser.add_argument('--gate-bce-weight',type=float,default=.1)
    parser.add_argument('--aux-grad-ratio',type=float,default=.2)
    parser.add_argument('--max-aux-scale',type=float,default=100000.)
    parser.add_argument('--eval-topk',type=int,default=100)
    parser.add_argument('--eval-query-batch',type=int,default=64)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--_worker',default='',help=argparse.SUPPRESS)
    parser.add_argument('--_run-dir',default='',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args._worker:
        try:worker07(args._worker,args._run_dir)
        except Exception:traceback.print_exc();sys.exit(1)
        return
    if args.self_test:self_test07();return
    try:seeds=[int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    except ValueError:parser.error('Invalid --seeds')
    if not seeds or len(set(seeds))!=len(seeds) or min(seeds)<0:
        parser.error('Seeds must be distinct nonnegative integers')
    for key in ('warmup_epochs','continue_epochs','train_batch','hidden','text_batch','image_batch',
                'hard_k','mine_batch','max_support','wrong_pool','eval_topk','eval_query_batch'):
        if getattr(args,key)<1:parser.error('--'+key.replace('_','-')+' must be >=1')
    if (args.max_train_pids<0 or not .5<=args.mechanism_fit_fraction<=.95 or
        not 0<args.teacher_strength<1 or args.max_delta<=0 or args.lr<=0 or
        args.aux_grad_ratio<=0 or args.max_aux_scale<=0 or args.rank_weight<=0 or
        args.protect_weight<0 or args.temperature<=0 or args.gate_bce_weight<0 or
        args.help_threshold<=0 or args.harm_threshold<=0 or args.instance_floor<0 or
        not 0<args.use_threshold<1 or not 0<args.harm_threshold_prob<1 or
        not 0<=args.use_vs_harm_margin<1):parser.error('Invalid numerical experiment settings')
    root=Path(args.project_root).resolve();cfg=Path(args.config_file).resolve();ck=Path(args.checkpoint).resolve()
    for p in (root/'model'/'build.py',cfg,ck):
        if not p.is_file():raise FileNotFoundError('Missing required IRRA file: '+str(p))
    _,ann,imgs,source=resolve_dataset(root,cfg,args.dataset_root,args.annotation_file,args.images_root)
    ims,txt,_=read_annotations(ann,'train');vim,vtxt,_=read_annotations(ann,'val')
    ids=sorted({int(r['pid']) for r in txt})
    if set(ids)&{int(r['pid']) for r in vtxt}:raise RuntimeError('Train/Val PID overlap')
    selected=(ids if args.max_train_pids==0 or args.max_train_pids>=len(ids)
              else sorted(map(int,np.random.default_rng(args.subset_seed).choice(
                   ids,args.max_train_pids,replace=False))))
    fit,hold=pid_split(selected,args.mechanism_fit_fraction,args.split_seed)
    verify_image_paths(imgs,[im for im in ims if im['pid'] in set(selected)])
    fpt=fingerprint(root,cfg,ck,ann,'train');fpv=fingerprint(root,cfg,ck,ann,'val')
    parent=root/PROJECT_FOLDER_NAME
    valcache=source_val_cache(root,parent,args.val_cache_dir,fpv,len(vtxt),len(vim))
    traintextcache=source_train_text_cache(root,parent,args.train_text_cache_dir,fpt,len(txt))
    imgidx=[j for j,im in enumerate(ims) if im['pid'] in set(selected)]
    imcache=source_train_image_cache(parent,args.train_image_cache_dir,
              SimpleNamespace(fp_train=fpt,selected_pids=selected),imgidx)
    settings={'project_root':str(root),'config_file':str(cfg),'checkpoint':str(ck),
              'annotation_file':str(ann),'images_dir':str(imgs),'dataset_source':source,
              'val_cache_dir':str(valcache or ''),'train_text_cache_dir':str(traintextcache or ''),
              'train_image_cache_dir':str(imcache or ''),
              'selected_pids':selected,'fit_pids':fit,'holdout_pids':hold,
              'fp_train':fpt,'fp_val':fpv,'device':args.device,'seeds':seeds,
              'subset_seed':args.subset_seed,'split_seed':args.split_seed,
              'mechanism_fit_fraction':args.mechanism_fit_fraction,
              'warmup_epochs':args.warmup_epochs,'continue_epochs':args.continue_epochs,
              'text_batch':args.text_batch,'image_batch':args.image_batch,
              'train_batch':args.train_batch,'hidden':args.hidden,
              'lr':args.lr,'rank_weight':args.rank_weight,'protect_weight':args.protect_weight,
              'margin':args.margin,'temperature':args.temperature,'max_delta':args.max_delta,
              'teacher_strength':args.teacher_strength,'max_support':args.max_support,
              'hard_k':args.hard_k,'mine_batch':args.mine_batch,
              'help_threshold':args.help_threshold,'harm_threshold':args.harm_threshold,
              'instance_floor':args.instance_floor,'use_threshold':args.use_threshold,
              'harm_threshold_prob':args.harm_threshold_prob,
              'use_vs_harm_margin':args.use_vs_harm_margin,'wrong_pool':args.wrong_pool,
              'random_teacher_seed':args.random_teacher_seed,
              'gate_bce_weight':args.gate_bce_weight,'aux_grad_ratio':args.aux_grad_ratio,
              'max_aux_scale':args.max_aux_scale,'eval_topk':args.eval_topk,
              'eval_query_batch':args.eval_query_batch,
              'expected_baseline_r1':56.83663527119195 if (
                  len(vtxt)==6158 and len(vim)==3078 and
                  fpv.get('checkpoint_sha256')=='3bf21f54271d0e7bd0cfd545daa2c2b5367ee2d734419bead417e70c2ec0a36c') else None}
    settings['settings_hash']=hashlib.sha256(json.dumps(settings,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    out,resumed=new_or_resume(parent,args,settings)
    LOG_FILE=out/'experiment_run.log'
    shutil.copy2(Path(__file__).resolve(),out/Path(__file__).name)
    (out/'实验07_使用说明与研究边界.md').write_text(
        '# 实验07：条件式多文本互补与有害信息拒绝\n\n'
        '## 两个同等重要的目标\n'
        '1. 对确实有用的同身份补充文本，学会接受和使用。\n'
        '2. 对损害检索匹配的支持文本，学会拒绝；不确定时不强行补全。\n\n'
        '## 协议\n'
        '- Train PID: 支持集构建、教师评分器及监督，可使用真实身份。\n'
        '- 内部未见 Train PID: 只做标明 Oracle 的教师三分类诊断。\n'
        '- Val: 当前查询和每张候选图像独立打分，禁止使用 Val PID 或其他同身份文本。\n'
        '- 先在独立 NumPy 进程生成固定 Top-K，训练进程仅读取索引；完整图库计算 R1/R5/R10/mAP/mINP。\n'
        '- 不与原始 IRRA 的单向量检索效率混为一谈：本研究为候选图像条件重排。\n\n'
        '## 代理标签与限制\n'
        '- useful / harmful / uncertain 来自带有训练 PID 的正图与困难负图上的分数变化。\n'
        '- 这些不是人工确认的语义属性真值；教师门控的分类分数不能当最终检索能力。\n'
        '- WrongPID 角度匹配教师复用了真实教师的接受/拒绝掩码：隔离教师更新方向，不是彻底无身份信息的独立门控。\n'
        '- reg_shuffle 则打乱接受/拒绝标记，作为模型容量和门控监督的额外负对照。\n'
        '- 未通过有益接受与有害拒绝的两项指标，则不要宣称可靠性门控成功。\n'
        '- 任何新机制的检索收益都应与无支持监督、普通正则化、错误身份进行比较。\n'
        '- Val 已反复用于研究；固定机制后才在独立 Test 评估。\n\n'
        '## 运行\n'
        '首次：在 PyCharm 直接运行；检查：--self-test；中断恢复：--resume。\n'
        '全部输出放在本实验目录，发送给ChatGPT.zip 排除大型 npy 与权重。\n',
        encoding='utf-8')
    file=out/'selected_train_pids.json'
    if resumed:
        if not file.is_file() or json.loads(file.read_text(encoding='utf-8'))!=selected:
            raise RuntimeError('Resume PID selection mismatch')
    else:atomic_json(file,selected)
    atomic_json(out/'identity_partition.json',{
        'fit_pids':fit,'internal_unseen_train_pids':hold,'val_overlap':0,
        'warning':'Train identities only; holdout Oracle diagnostics and Val retrieval are different.'})
    meta=out/'run_info.json';info=json.loads(meta.read_text(encoding='utf-8')) if resumed else {}
    info.update({'experiment':EXPERIMENT_NAME,'settings':settings,'started_at':datetime.now().isoformat(),
                 'resumed':resumed,'status':'running'})
    atomic_json(meta,info)
    msg('EXP07 folder: '+str(out))
    msg(f'Fit IDs={len(fit)}, unseen TRAIN IDs={len(hold)}, Val captions={len(vtxt)}')
    msg('METHODS: '+str(METHODS))
    try:
        # A dedicated NumPy-only Top-K stage must run BEFORE the torch worker.
        # This also upgrades interrupted runs created by older EXP07 versions.
        for st in ('encode','evidence','topk','train','eval'):run_subprocess(st,out,root)
        info['status']='completed';info['finished_at']=datetime.now().isoformat()
        atomic_json(meta,info)
        msg('EXP07 COMPLETED: gate diagnostics, full-gallery metrics and automatic ZIP.')
    except Exception as exc:
        info['status']='failed';info['error']=repr(exc);info['failed_at']=datetime.now().isoformat()
        atomic_json(meta,info)
        (out/'error_traceback.log').write_text(traceback.format_exc(),encoding='utf-8')
        msg('EXP07 FAILED: '+repr(exc));raise
    finally:
        try:package_for_chat(out)
        except Exception as e:msg('ZIP packaging failed: '+repr(e))

if __name__=='__main__':
    main()
