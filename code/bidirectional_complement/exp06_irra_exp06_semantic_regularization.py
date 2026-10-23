# -*- coding: utf-8 -*-
"""实验06：真实身份语义与普通正则化贡献分离（冻结 IRRA，多随机种子）

Study plan: freeze IRRA, separate mechanism-fit and heldout identities within Train,
learn a light support reliability model ONLY on fit identities, evaluate it on unseen
Train IDs without fitting, then train single-caption students on fit identities.
Factorial 2x2: additional self-relation regularization (R) x identity-specific
semantic teacher (S); matched wrong-PID/random and hard semantic wrong-PID controls
with identical per-anchor angular displacement; report complementary interaction.
Wrong-PID matching uses float64 chord-distance geometry to handle nearly
identical text representations correctly; verified mining caches resume safely.
Validation: one caption to frozen gallery, NO Val PID used for model/teacher inputs.

Run without args in IRRA-main; --self-test for synthetic full pipeline; --resume to
continue a matching timestamped run. Output in 图文双向互补实验/实验06_.../;
automatically creates 发送给ChatGPT.zip excluding weights and large caches.
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
EXPERIMENT_NAME = '实验06_身份语义与正则化分离'
METHODS = ('rank_warmup','rank','reg','sem_true','reg_sem_true','sem_wrong_matched','reg_sem_wrong_matched','sem_wrong_nearest','reg_sem_wrong_nearest')
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
        raise FileNotFoundError('No matching experiment06 run to resume. Remove --resume for a new run.')
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



# =================== EXP06 semantic-evidence study ===================

def split_mechanism_pids(all_pids, ratio, seed):
    ids=np.asarray(sorted(map(int,all_pids)),dtype=np.int64)
    rng=np.random.default_rng(seed)
    shuffled=rng.permutation(ids)
    k=min(len(ids)-1,max(2,int(round(len(ids)*ratio))))
    if k>=len(ids): raise ValueError('Need at least one unseen training identity')
    a,b=sorted(map(int,shuffled[:k])),sorted(map(int,shuffled[k:]))
    if set(a)&set(b):raise AssertionError('Mechanism fit/holdout PID overlap')
    return a,b


def split_feature_arrays(xf,vf,texts,images,pos,allowed_pids):
    allowed=set(map(int,allowed_pids))
    ti=np.asarray([i for i,t in enumerate(texts) if int(t['pid']) in allowed],dtype=np.int64)
    gi=np.asarray([i for i,g in enumerate(images) if int(g['pid']) in allowed],dtype=np.int64)
    lookup={int(orig):j for j,orig in enumerate(gi)}
    local_pos=np.asarray([lookup[int(pos[i])] for i in ti],dtype=np.int64)
    tx=[texts[int(i)] for i in ti];gs=[images[int(i)] for i in gi]
    assert all(tx[j]['pid']==gs[int(local_pos[j])]['pid'] for j in range(len(tx)))
    return normalize(xf[ti]),normalize(vf[gi]),tx,gs,local_pos


def support_features(anchor, candidate, current_image, support_image, negatives):
    # Every feature is observable from Train-only frozen text/image embeddings;
    # no true margin label appears in the gate's input.
    return np.asarray([
        float(anchor@candidate),float(candidate@current_image),
        float(candidate@support_image),float(anchor@support_image),
        float(candidate@negatives[0]),float(anchor@current_image),
        float((candidate-anchor)@current_image),
    ],dtype=np.float32)


def sigmoid_np(z):
    z=np.clip(z,-25,25)
    return 1.0/(1.0+np.exp(-z))


def fit_reliability_gate(features, labels, seed=6):
    # Fit on mechanism-fit PIDs only; heldout PIDs are never used to train it.
    f=np.asarray(features,dtype=np.float64)
    y=np.asarray(labels,dtype=np.float64)
    mu=f.mean(0);std=np.maximum(f.std(0),1e-5)
    z=np.clip((f-mu)/std,-8,8)
    z=np.column_stack([z,np.ones(len(z))])
    weights=np.zeros(z.shape[1],dtype=np.float64)
    rng=np.random.default_rng(seed)
    # deterministic regularized batch gradient logistic regression; no sklearn.
    for _ in range(200):
        ids=rng.permutation(len(y))
        for ii in np.array_split(ids,max(1,(len(y)+2047)//2048)):
            error=sigmoid_np(z[ii]@weights)-y[ii]
            grad=(z[ii].T@error)/len(ii)
            grad[:-1]+=0.02*weights[:-1]
            weights-=0.08*grad
    return {'mean':mu.tolist(),'std':std.tolist(),'weights':weights.tolist(),
            'training_positive_fraction':float(y.mean()),'n_training_pairs':int(len(y)),
            'features':['text_text_cos','candidate_anchor_image','candidate_own_image',
                        'anchor_candidate_image','candidate_hard_negative','anchor_own_image',
                        'candidate_minus_anchor_on_image']}


def reliability_prob(row,gate):
    z=np.clip((np.asarray(row,dtype=np.float64)-np.asarray(gate['mean']))/
              np.asarray(gate['std']),-8,8)
    return float(sigmoid_np(np.r_[z,1.]@np.asarray(gate['weights'])))


def positive_negative_margin(t,pos_image,negative_gallery):
    return float(t@pos_image-np.max(negative_gallery@t))


def choose_supports(i,xf,vf,texts,pos,neg,by_pid,opt,gate=None,collect_pairs=False):
    pid=int(texts[i]['pid']);image_id=int(texts[i]['img_idx'])
    candidates=[j for j in by_pid[pid] if int(texts[j]['img_idx'])!=image_id]
    candidates=capped_peer_indices(candidates,xf[i],xf,opt.max_support)
    rows=[];labels=[];support_indices=[];probs=[]
    if not candidates:return support_indices,probs,rows,labels
    a=xf[i];p=vf[pos[i]]; negatives=vf[neg[i]]
    base=positive_negative_margin(a,p,negatives)
    for j in candidates:
        feat=support_features(a,xf[j],p,vf[pos[j]],negatives)
        blend=normalize((1-opt.teacher_strength)*a+opt.teacher_strength*xf[j])
        lab=int(positive_negative_margin(blend,p,negatives)>base+1e-6)
        rows.append(feat);labels.append(lab)
        support_indices.append(j)
        if gate is not None:probs.append(reliability_prob(feat,gate))
    return support_indices,probs,rows,labels


def aggregate_true_teacher(i,x,v,texts,pos,neg,bypid,opt,gate):
    candidates,probs,feats,labels=choose_supports(i,x,v,texts,pos,neg,bypid,opt,gate)
    if not candidates:return x[i],candidates,probs,labels
    prob=np.asarray(probs,dtype=np.float32)
    # Conservative evidence gate: when no support reaches 0.5, retain original.
    pick=prob>=0.5
    if not np.any(pick):return x[i],candidates,probs,labels
    inds=np.asarray(candidates,dtype=np.int64)[pick]
    weights=np.maximum(prob[pick]-0.45,1e-4)
    weights/=weights.sum()
    source=np.average(x[inds],axis=0,weights=weights)
    target=normalize((1-opt.teacher_strength)*x[i]+opt.teacher_strength*source)
    return target,candidates,probs,labels


def wrong_direction_matched(anchor, other, reference, rng):
    """Match the True teacher's *chord distance* with a WrongPID direction.

    The previous implementation evaluated float32 ``anchor @ reference`` and
    ``sqrt(1-cos**2)``. Near-parallel float32 vectors can round this cosine
    to exactly 1 (sometimes >1), losing a real nonzero displacement. Compute
    the distance and spherical rotation in float64; cast only the final unit
    embedding to float32. The identity/direction of ``reference`` is NEVER
    used to construct the WrongPID tangent (only its displacement magnitude).
    """
    a = np.asarray(anchor, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    a /= max(float(np.linalg.norm(a)), 1e-30)
    ref /= max(float(np.linalg.norm(ref)), 1e-30)
    other /= max(float(np.linalg.norm(other)), 1e-30)

    # Chord length is numerically stable for nearly coincident unit vectors.
    d = float(np.linalg.norm(ref - a))
    if d <= 1e-12:
        return normalize(np.asarray(a, dtype=np.float32))
    d = min(d, 2.)

    tangent = other - float(np.dot(other, a)) * a
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm < 1e-10:
        # Collinear wrong-PID candidates do not define a stable tangent.
        # Choose a fallback independent of the True teacher DIRECTION.
        tangent = rng.normal(size=len(a))
        tangent -= float(np.dot(tangent, a)) * a
        tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm < 1e-12:
        raise AssertionError('WrongPID tangent degeneracy')
    tangent /= tangent_norm

    # For unit vectors, chord distance d = 2 sin(theta/2).  Avoid
    # cancellation from computing sin(theta) via sqrt(1-cos(theta)**2).
    cosine = 1. - .5 * d * d
    sine = d * math.sqrt(max(0., 1. - .25 * d * d))
    result = cosine * a + sine * tangent
    return normalize(np.asarray(result, dtype=np.float32))


def diagnostic_group(xf,vf,texts,pos,neg,gate,opt,part,make_controls=False,out=None):
    pids=np.asarray([t['pid'] for t in texts],dtype=np.int64)
    bypid=defaultdict(list)
    for i,p in enumerate(pids):bypid[int(p)].append(i)
    rng=np.random.default_rng(opt.random_teacher_seed+(0 if part=='fit' else 100))
    base_rows=[];gated=[];teacher_arrays={}
    if make_controls:
        teacher_arrays={name:np.lib.format.open_memmap(out/(name+'_teacher.npy'),mode='w+',
                 dtype='float32',shape=xf.shape)
                        for name in ('sem_true','sem_wrong_matched','sem_wrong_nearest')}
    diag=defaultdict(float);examples=[];n=len(xf);angle_err=[]
    # A 256-caption proposal pool avoids O(n^2) all-pairs semantic-Wrong search;
    # per-query nearest is nearest within this fixed-size wrong-PID proposal pool.
    ids_by_pid={int(p):np.flatnonzero(pids!=p) for p in np.unique(pids)} if make_controls else {}
    for i in range(n):
        true,candidates,probs,labels=aggregate_true_teacher(i,xf,vf,texts,pos,neg,bypid,opt,gate)
        base=positive_negative_margin(xf[i],vf[pos[i]],vf[neg[i]])
        gain=positive_negative_margin(true,vf[pos[i]],vf[neg[i]])-base
        diag['num_queries']+=1
        diag['supported_queries']+=bool(candidates)
        diag['accepted_queries']+=int(np.linalg.norm(true-xf[i])>1e-5)
        diag['selected_true_supports']+=sum(p>=.5 for p in probs)
        diag['gain_sum']+=gain
        diag['improved']+=int(gain>1e-6)
        diag['correct_predictions']+=sum(int((p>=.5)==bool(y)) for p,y in zip(probs,labels))
        diag['num_candidate_pairs']+=len(labels)
        diag['labels_positive']+=sum(labels)
        if make_controls:
            pool=ids_by_pid[int(pids[i])]
            count=max(1,len(candidates))
            if len(pool)<count:raise ValueError('Not enough wrong-PID training texts')
            other=np.asarray(rng.choice(pool,size=count,replace=False),dtype=np.int64)
            raw=normalize((1-opt.teacher_strength)*xf[i]+opt.teacher_strength*np.mean(xf[other],axis=0))
            # Semantic-matched wrong control: most similar wrong-PID text among a
            # bounded, randomly chosen proposal set; DO NOT use validation labels.
            pool2=np.asarray(rng.choice(pool,size=min(opt.wrong_pool,len(pool)),replace=False),dtype=np.int64)
            selected=np.asarray(pool2[np.argsort(-(xf[pool2]@xf[i]),kind='stable')[:count]],dtype=np.int64)
            close=normalize((1-opt.teacher_strength)*xf[i]+opt.teacher_strength*np.mean(xf[selected],axis=0))
            wrong=wrong_direction_matched(xf[i],raw,true,rng)
            wrongclose=wrong_direction_matched(xf[i],close,true,rng)
            for name,t in [('sem_true',true),('sem_wrong_matched',wrong),('sem_wrong_nearest',wrongclose)]:
                teacher_arrays[name][i]=t
                diag[name+'_gain']+=positive_negative_margin(t,vf[pos[i]],vf[neg[i]])-base
                diag[name+'_angle']+=1-float(xf[i]@t)
            # Compute the *check* in float64 as well; this verifies the final
            # float32 vectors without losing small but real displacements.
            target=float(np.linalg.norm(np.asarray(true,dtype=np.float64)-
                                        np.asarray(xf[i],dtype=np.float64)))
            e1=abs(float(np.linalg.norm(np.asarray(wrong,dtype=np.float64)-
                                        np.asarray(xf[i],dtype=np.float64)))-target)
            e2=abs(float(np.linalg.norm(np.asarray(wrongclose,dtype=np.float64)-
                                        np.asarray(xf[i],dtype=np.float64)))-target)
            angle_err.extend((e1,e2))
            if max(e1,e2)>3e-4:
                raise AssertionError('Matched WrongPID teacher distance mismatch '
                        f'at query={i}, pid={int(pids[i])}, target={target:.8g}, '
                        f'err_random={e1:.8g}, err_nearest={e2:.8g}; '
                        'not safe to compare unmatched controls')
            if i<100:
                examples.append({'query_idx':i,'pid':int(pids[i]),'text':texts[i]['text'],
                    'same_identity_texts':' || '.join(texts[j]['text'] for j in candidates),
                    'wrong_texts':' || '.join(texts[j]['text'] for j in other),
                    'semantic_wrong_texts':' || '.join(texts[j]['text'] for j in selected),
                    'gate_probs':' || '.join(f'{p:.4f}' for p in probs),
                    'true_margin_gain':gain,
                    'wrong_matched_margin_gain':positive_negative_margin(wrong,vf[pos[i]],vf[neg[i]])-base,
                    'wrong_semantic_margin_gain':positive_negative_margin(wrongclose,vf[pos[i]],vf[neg[i]])-base,
                    'same_identity_count':len(candidates),
                    'angular_match_error':max(e1,e2)})
        if (i+1)%5000==0:msg(f'EXP06 teacher {part}: {i+1}/{n}')
    for ar in teacher_arrays.values():ar.flush()
    if examples:write_csv(out/'teacher_examples.csv',examples,list(examples[0]))
    return {'identity_partition':part,'n_queries':n,'n_pid':len(bypid),
            'support_coverage':diag['supported_queries']/n,
            'accepted_rate':diag['accepted_queries']/n,
            'candidate_pair_count':int(diag['num_candidate_pairs']),
            'pair_positive_fraction':diag['labels_positive']/max(1,diag['num_candidate_pairs']),
            'gate_label_accuracy':diag['correct_predictions']/max(1,diag['num_candidate_pairs']),
            'true_mean_margin_gain':diag['gain_sum']/n,
            'true_improved_rate':diag['improved']/n,
            'mean_selected_supports':diag['selected_true_supports']/n,
            'control_mean_margin_gain':{name:diag[name+'_gain']/n for name in teacher_arrays},
            'control_mean_one_minus_cos':{name:diag[name+'_angle']/n for name in teacher_arrays},
            'max_wrong_teacher_angle_error':max(angle_err) if angle_err else None}



def _valid_mined_negatives(path, text_pids, gallery_pids, hard_k):
    """Accept only completed, correctly-shaped PID-safe negative arrays."""
    if not Path(path).is_file():
        return None
    try:
        arr = np.load(path, mmap_mode='r', allow_pickle=False)
        if arr.shape != (len(text_pids), hard_k * 2) or arr.dtype.kind not in 'iu':
            return None
        if not np.all((arr >= 0) & (arr < len(gallery_pids))):
            return None
        # Cheap enough for the selected Train subset; avoids consuming a
        # half-written or incompatible cache after a terminated run.
        if np.any(gallery_pids[arr] == np.asarray(text_pids)[:, None]):
            return None
        return arr
    except (OSError, ValueError, IndexError, EOFError):
        return None


def _validated_fit_mining_cache(out, texts, images, fit, opt):
    """Also recovers the arrays saved by the older EXP06 on failed runs."""
    paths = [out/'hard_negative_indices.npy', out/'positive_gallery_idx.npy',
             out/'fit_text_indices.npy', out/'fit_gallery_indices.npy']
    if not all(p.is_file() for p in paths):
        return None
    try:
        selected = set(map(int, fit))
        exp_t = np.fromiter((i for i,t in enumerate(texts) if int(t['pid']) in selected),
                            dtype=np.int64)
        exp_g = np.fromiter((i for i,g in enumerate(images) if int(g['pid']) in selected),
                            dtype=np.int64)
        idx_t = np.load(paths[2], allow_pickle=False)
        idx_g = np.load(paths[3], allow_pickle=False)
        if not (np.array_equal(exp_t,idx_t) and np.array_equal(exp_g,idx_g)):
            return None
        pos = np.load(paths[1], allow_pickle=False)
        if pos.shape != (len(exp_t),) or pos.dtype.kind not in 'iu':
            return None
        lookup = {int(original): j for j, original in enumerate(exp_g)}
        expected_pos = np.asarray([lookup[int(texts[i]['img_idx'])] for i in exp_t],
                                  dtype=np.int64)
        if not np.array_equal(pos,expected_pos):
            return None
        tx = np.asarray([texts[i]['pid'] for i in exp_t])
        gp = np.asarray([images[i]['pid'] for i in exp_g])
        neg = _valid_mined_negatives(paths[0],tx,gp,opt.hard_k)
        return neg
    except (OSError, ValueError, IndexError, KeyError, EOFError):
        return None


def stage_evidence06(out):
    _,opt,out=manifest_for_stage(out)
    completion=out/'evidence_complete.json'
    required=[out/'hard_negative_indices.npy',out/'positive_gallery_idx.npy',
              out/'fit_text_indices.npy',out/'fit_gallery_indices.npy',
              out/'sem_true_teacher.npy',out/'sem_wrong_matched_teacher.npy',
              out/'sem_wrong_nearest_teacher.npy',out/'teacher_diagnostics.json']
    if completion.is_file() and all(p.is_file() for p in required):
        try:
            mark=json.loads(completion.read_text(encoding='utf-8'))
            if mark.get('settings_hash')==opt.settings_hash:
                n=int(mark['n_fit'])
                neg_shape=np.load(out/'hard_negative_indices.npy',mmap_mode='r',allow_pickle=False).shape
                teacher_shapes=[np.load(out/(key+'_teacher.npy'),mmap_mode='r',allow_pickle=False).shape
                    for key in ('sem_true','sem_wrong_matched','sem_wrong_nearest')]
                if (neg_shape==(n,2*opt.hard_k) and
                    all(shape[0]==n and len(shape)==2 for shape in teacher_shapes)):
                    msg('Complete EXP06 evidence artifacts found; skip');return
        except (OSError,ValueError,KeyError):
            pass
        msg('WARNING: stale/corrupted EXP06 evidence cache; rebuilding evidence stage')
    x,v,texts,images,pos=train_arrays03(out,opt)
    fit,hold=opt.fit_pids,opt.holdout_pids
    xf,vf,tf,gf,pf=split_feature_arrays(x,v,texts,images,pos,fit)
    xh,vh,th,gh,ph=split_feature_arrays(x,v,texts,images,pos,hold)
    tx=np.asarray([t['pid'] for t in tf]);gp=np.asarray([g['pid'] for g in gf])
    hx=np.asarray([t['pid'] for t in th]);hg=np.asarray([g['pid'] for g in gh])
    if set(tx)&set(hx):raise RuntimeError('Fit/heldout train identity overlap')
    neg=_validated_fit_mining_cache(out,texts,images,fit,opt)
    if neg is not None:
        msg('Recovered validated fit hard-negative cache from the interrupted run; '
            'skip mining fit negatives.')
    else:
        msg('No validated fit hard-negative cache; mine fit negatives.')
        neg=hard_negative_mine(xf,vf,tx,gp,opt.hard_k*2,opt.mine_batch)
        np.save(out/'hard_negative_indices.npy',neg)
        np.save(out/'positive_gallery_idx.npy',pf)
        fit_set=set(fit)
        np.save(out/'fit_text_indices.npy',np.asarray([i for i,t in enumerate(texts)
                      if t['pid'] in fit_set],dtype=np.int32))
        np.save(out/'fit_gallery_indices.npy',np.asarray([i for i,g in enumerate(images)
                      if g['pid'] in fit_set],dtype=np.int32))
    # Cache internal-unseen negatives too; holdout identities never enter training.
    heldout_path=out/'internal_unseen_negative_indices.npy'
    hneg=_valid_mined_negatives(heldout_path,hx,hg,opt.hard_k)
    if hneg is not None:
        msg('Recovered validated internal-unseen hard-negative cache.')
    else:
        hneg=hard_negative_mine(xh,vh,hx,hg,opt.hard_k*2,opt.mine_batch)
        np.save(heldout_path,hneg)
    # Train the text-reliability scorer using ONLY fit IDs, not heldout IDs.
    bypid=defaultdict(list)
    for i,t in enumerate(tf):bypid[int(t['pid'])].append(i)
    rows=[];labels=[]
    for i in range(len(tf)):
        _,_,a,b=choose_supports(i,xf,vf,tf,pf,neg[:,:opt.hard_k],bypid,opt)
        rows.extend(a);labels.extend(b)
    if len(rows)<5:
        raise RuntimeError('Too few support pairs for reliability diagnostic')
    if len(set(labels))<2:
        # Degenerate synthetic or weak data: cannot estimate discriminative
        # reliability; use a constant empirical prior and disclose it.
        prior=float(np.clip(np.mean(labels),1e-4,1-1e-4))
        gate={'mean':np.mean(rows,axis=0).tolist(),
              'std':np.maximum(np.std(rows,axis=0),1e-5).tolist(),
              'weights':[0.]*len(rows[0])+[math.log(prior/(1-prior))],
              'training_positive_fraction':float(np.mean(labels)),
              'n_training_pairs':len(labels),'degenerate_single_class':True,
              'warning':'No discriminative gate was learned; this run is diagnostic only.'}
        msg('WARNING: evidence labels are single-class; gate uses empirical prior')
    else:
        gate=fit_reliability_gate(np.asarray(rows),np.asarray(labels),opt.split_seed)
        gate['degenerate_single_class']=False
    atomic_json(out/'learned_reliability_gate.json',gate)
    fit_diag=diagnostic_group(xf,vf,tf,pf,neg[:,opt.hard_k:],gate,opt,'fit_heldout_negatives',True,out)
    hold_diag=diagnostic_group(xh,vh,th,ph,hneg[:,opt.hard_k:],gate,opt,'INTERNAL_UNSEEN_PIDS',False,None)
    atomic_json(out/'teacher_diagnostics.json',{
        'gate_training_partition':'mechanism_fit_pids_only',
        'internal_holdout_pids_never_used_to_fit_gate_or_student':True,
        'fit_pid_count':len(fit),'holdout_pid_count':len(hold),
        'fit':fit_diag,'internal_unseen':hold_diag,
        'holdout_uses_pid_for_oracle_diagnostic_ONLY':True,
        'support_selection_uses_predicted_reliability_not_ground_truth_gain':True,
        'heldout_negatives_unseen_in_fit_gate_training':True,
        'wrong_control_proposal_pool':opt.wrong_pool,
        'warning':'Internal unseen teacher measures use heldout training PID; NOT standard retrieval or deployable.'})
    atomic_json(completion,{'settings_hash':opt.settings_hash,'time':datetime.now().isoformat(),
                            'n_fit':len(tf),'n_heldout':len(th)})
    msg('EXP06 internal unseen identity teacher gain: '+str(round(hold_diag['true_mean_margin_gain'],6)))


# =================== EXP06 controlled student training ===================

def all_training_records(out):
    records=[]
    for file in sorted(Path(out).glob('*_training_meta.json')):
        try:
            item=json.loads(file.read_text(encoding='utf-8'))
            if item.get('complete'):records.extend(item.get('history',[]))
        except (OSError,ValueError):continue
    return records


def _model_done06(out,opt,stem,val_shape):
    paths=[out/(stem+'_student.pt'),out/(stem+'_training_meta.json'),out/(stem+'_val_features.npy')]
    if not all(p.is_file() for p in paths):return False
    try:
        meta=json.loads(paths[1].read_text(encoding='utf-8'))
        return bool(meta.get('complete') and meta['settings_hash']==opt.settings_hash and
                meta['features_shape']==list(val_shape) and
                np.load(paths[2],mmap_mode='r',allow_pickle=False).shape==tuple(val_shape) and
                meta['checkpoint_sha256']==sha256_small_file(paths[0]))
    except (OSError,ValueError,KeyError):return False


def commit_model06(out,model,optimizer,stem,history,seed,method,val_text,opt,device):
    import torch
    target=out/(stem+'_val_features.npy');tmp=out/(stem+'_val_features.partial.npy')
    mm=np.lib.format.open_memmap(tmp,mode='w+',dtype='float32',shape=tuple(val_text.shape))
    model.eval()
    with torch.inference_mode():
        for j in range(0,len(val_text),opt.train_batch):
            x=val_text[j:j+opt.train_batch].to(device)
            mm[j:j+len(x)]=model(x).cpu().numpy()
    mm.flush();del mm
    _replace_with_retries(tmp,target)
    ck=out/(stem+'_student.pt');ctmp=out/(stem+'_student.partial.pt')
    torch.save({'state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                'settings_hash':opt.settings_hash,'seed':seed,'method':method,
                'hidden':opt.hidden,'gain':opt.gain,'dim':val_text.shape[1]},ctmp)
    _replace_with_retries(ctmp,ck)
    atomic_json(out/(stem+'_training_meta.json'),{
        'complete':True,'settings_hash':opt.settings_hash,'method':method,'seed':seed,
        'features_shape':list(val_text.shape),'checkpoint_sha256':sha256_small_file(ck),
        'history':history})


def train_epoch06(model,optimizer,order,tx,gallery,pos,neg,teachers,method,opt,device):
    import torch
    import torch.nn.functional as F
    model.train();params=list(model.parameters())
    # R and S are independent experimental factors. Matched gradient norm PER
    # ACTIVE FACTOR, not an artificial identity-equivalent total-step guarantee.
    reg_on=method in ('reg','reg_sem_true','reg_sem_wrong_matched','reg_sem_wrong_nearest')
    sources={'sem_true':'sem_true','reg_sem_true':'sem_true',
             'sem_wrong_matched':'sem_wrong_matched','reg_sem_wrong_matched':'sem_wrong_matched',
             'sem_wrong_nearest':'sem_wrong_nearest','reg_sem_wrong_nearest':'sem_wrong_nearest'}
    sem_name=sources.get(method)
    sums=defaultdict(float);counts=defaultdict(int)
    N=len(order)
    for start in range(0,N,opt.train_batch):
        inds=order[start:start+opt.train_batch]
        original=tx[inds].to(device)
        z=model(original)
        idx=torch.cat([pos[inds,None],neg[inds]],dim=1).to(device)
        imgs=gallery[idx]
        pred=torch.einsum('bkd,bd->bk',imgs,z)
        lrank=F.softplus((pred[:,1:]-pred[:,:1]+opt.margin)/opt.temperature).mean()
        lprotect=(1-(z*original).sum(dim=-1)).mean()
        main=opt.rank_weight*lrank+opt.protect_weight*lprotect
        reg_loss=torch.zeros((),device=device);sem_loss=torch.zeros((),device=device)
        base_grads=torch.autograd.grad(main,params,retain_graph=(reg_on or bool(sem_name)),allow_unused=True)
        base_grads=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,base_grads)]
        base_norm=math.sqrt(sum(float(g.double().square().sum()) for g in base_grads))
        accum=[g.clone() for g in base_grads]
        extras=[]
        if reg_on:
            # Self-relation = regularization without new identity/caption information.
            orig_pred=torch.einsum('bkd,bd->bk',imgs,original).detach()
            reg_loss=F.smooth_l1_loss(pred[:,:1]-pred[:,1:],orig_pred[:,:1]-orig_pred[:,1:])
            extras.append(('reg',reg_loss,reg_on and bool(sem_name)))
        if sem_name:
            tg=teachers[sem_name][inds].to(device)
            expected=torch.einsum('bkd,bd->bk',imgs,tg).detach()
            sem_loss=F.smooth_l1_loss(pred[:,:1]-pred[:,1:],expected[:,:1]-expected[:,1:])
            extras.append(('sem',sem_loss,False))
        for name,aux,keep in extras:
            g=torch.autograd.grad(aux,params,retain_graph=keep,allow_unused=True)
            g=[torch.zeros_like(p) if v is None else v.detach() for p,v in zip(params,g)]
            grad_norm=math.sqrt(sum(float(v.double().square().sum()) for v in g))
            scale=(min(opt.aux_grad_ratio*base_norm/max(grad_norm,1e-30),opt.max_aux_scale)
                   if base_norm>1e-12 and grad_norm>1e-12 else 0.)
            cos=(sum(float((a.double()*b.double()).sum()) for a,b in zip(base_grads,g))/
                  (base_norm*grad_norm) if base_norm>1e-12 and grad_norm>1e-12 else 0.)
            for k in range(len(accum)):accum[k].add_(g[k],alpha=scale)
            sums[name+'_gradient_ratio']+=scale*grad_norm/max(base_norm,1e-30)*len(inds)
            sums[name+'_gradient_cosine']+=cos*len(inds)
            sums[name+'_loss']+=float(aux.detach().item())*len(inds)
            counts[name+'_active_batches']+=int(scale>0)
            counts[name+'_capped_batches']+=int(scale>=opt.max_aux_scale-1e-10)
        optimizer.zero_grad(set_to_none=True)
        for p,g in zip(params,accum):p.grad=g
        grad_norm=float(sum(float(g.double().square().sum()) for g in accum)**.5)
        actual_clip=torch.nn.utils.clip_grad_norm_(params,5.)
        counts['clipped_batches']+=int(actual_clip>5.)
        before=[p.detach().clone() for p in params]
        optimizer.step()
        step=math.sqrt(sum(float((p.detach()-b).double().square().sum()) for p,b in zip(params,before)))
        if not math.isfinite(step):raise FloatingPointError('Optimizer generated a nonfinite update')
        w=len(inds)
        for k,value in [('rank_loss',lrank.detach().item()),('protect_loss',lprotect.detach().item()),
                        ('base_gradient_norm',base_norm),('applied_gradient_norm',grad_norm),
                        ('adamw_actual_step_norm',step)]:sums[k]+=float(value)*w
        counts['processed']+=w;counts['batches']+=1
    stats={key:float(v)/N for key,v in sums.items()}
    stats.update({k:int(v) for k,v in counts.items()})
    return stats


def stage_train06(out):
    import torch
    torch.set_num_threads(1)
    _,opt,out=manifest_for_stage(out)
    device=torch.device('cuda' if opt.device=='cuda' or (opt.device=='auto' and torch.cuda.is_available()) else 'cpu')
    if opt.device=='cuda' and device.type!='cuda':raise RuntimeError('CUDA requested but unavailable')
    x,v,texts,images,pos0=train_arrays03(out,opt)
    xf,vf,tf,gf,posf=split_feature_arrays(x,v,texts,images,pos0,opt.fit_pids)
    negf=np.asarray(np.load(out/'hard_negative_indices.npy',allow_pickle=False)[:,:opt.hard_k],dtype=np.int64)
    assert negf.shape==(len(xf),opt.hard_k)
    vv=Path(opt.val_cache_dir) if opt.val_cache_dir else out
    val=normalize(np.load(vv/'text_features.npy',mmap_mode='r'))
    assert val.shape[1]==xf.shape[1]
    features=torch.from_numpy(np.array(xf,copy=True))
    gallery=torch.from_numpy(np.array(vf,copy=True)).to(device)
    pos=torch.from_numpy(np.array(posf,copy=True));neg=torch.from_numpy(negf.copy())
    vval=torch.from_numpy(np.array(val,copy=True))
    teachers={name:torch.from_numpy(np.array(np.load(out/(name+'_teacher.npy'),mmap_mode='r'),copy=True))
              for name in ('sem_true','sem_wrong_matched','sem_wrong_nearest')}
    cls=get_adapter_exp02()
    for seed in opt.seeds:
        warm=f'rank_warmup_s{seed}'
        if not _model_done06(out,opt,warm,val.shape):
            torch.manual_seed(seed)
            if device.type=='cuda':torch.cuda.manual_seed_all(seed)
            model=cls(xf.shape[1],opt.hidden,opt.gain).to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=opt.lr,weight_decay=1e-4)
            rng=torch.Generator().manual_seed(seed)
            history=[]
            for epoch in range(1,opt.warmup_epochs+1):
                order=torch.randperm(len(xf),generator=rng)
                stats=train_epoch06(model,optimizer,order,features,gallery,pos,neg,teachers,'rank',opt,device)
                history.append({'method':'rank_warmup','seed':seed,'epoch':epoch,'phase':'warmup',**stats})
                msg(f'{warm} {epoch}/{opt.warmup_epochs}: rank={stats["rank_loss"]:.5f}')
            commit_model06(out,model,optimizer,warm,history,seed,'rank_warmup',vval,opt,device)
            del model,optimizer
        else:msg(warm+': completed + verified, skip')
        saved=torch.load(out/(warm+'_student.pt'),map_location='cpu',weights_only=False)
        if saved['settings_hash']!=opt.settings_hash:raise RuntimeError('Warmup model fingerprint error')
        for method in METHODS[1:]:
            stem=method+'_s'+str(seed)
            if _model_done06(out,opt,stem,val.shape):
                msg(stem+': completed + verified, skip');continue
            torch.manual_seed(seed)
            model=cls(xf.shape[1],opt.hidden,opt.gain).to(device)
            model.load_state_dict(saved['state_dict'])
            optimizer=torch.optim.AdamW(model.parameters(),lr=opt.lr,weight_decay=1e-4)
            optimizer.load_state_dict(saved['optimizer_state_dict'])
            # Match shuffle across branches, including regularization-only and negative controls.
            rng=torch.Generator().manual_seed(seed+10000)
            history=[]
            for epoch in range(1,opt.continue_epochs+1):
                order=torch.randperm(len(xf),generator=rng)
                stats=train_epoch06(model,optimizer,order,features,gallery,pos,neg,teachers,method,opt,device)
                history.append({'method':method,'seed':seed,'epoch':epoch,'phase':'continue',**stats})
                msg(f'{stem} {epoch}/{opt.continue_epochs}: rank={stats["rank_loss"]:.5f} '
                    f'reg_ratio={stats.get("reg_gradient_ratio",0):.3f} '
                    f'sem_ratio={stats.get("sem_gradient_ratio",0):.3f}')
            commit_model06(out,model,optimizer,stem,history,seed,method,vval,opt,device)
            del model,optimizer
            if device.type=='cuda':torch.cuda.empty_cache()
    hist=all_training_records(out)
    if hist:write_csv(out/'training_history.csv',hist,list(dict.fromkeys(k for r in hist for k in r)),allow_locked_fallback=True)
    msg('EXP06 complete: shared-Rank warmup + 8 controlled branches each seed')


def evaluate06(out):
    _,opt,out=manifest_for_stage(out)
    ims,texts,_=read_annotations(Path(opt.annotation_file),'val')
    vv=Path(opt.val_cache_dir) if opt.val_cache_dir else out
    x=normalize(np.load(vv/'text_features.npy',mmap_mode='r'))
    v=normalize(np.load(vv/'image_features.npy',mmap_mode='r'))
    pids=np.asarray([t['pid'] for t in texts],dtype=np.int64)
    gpids=np.asarray([i['pid'] for i in ims],dtype=np.int64)
    if (set(pids)&set(opt.selected_pids)) or (set(pids)&set(opt.holdout_pids)):
        raise RuntimeError('Train/Val PID contamination')
    base,br=metrics(x@v.T,pids,gpids)
    if opt.expected_baseline_r1 is not None and abs(base['R1']-opt.expected_baseline_r1)>.05:
        raise RuntimeError('Unexpected original IRRA checkpoint/data baseline')
    rows=[{'method':'original_IRRA','seed':-1,'protocol':'STANDARD_SINGLE_TEXT',**base}]
    queries=[];cases=[];pairs=[];byseed={}
    baseok=np.asarray(br['rank'])==1
    for seed in opt.seeds:
        pred={}
        for method in METHODS:
            stem=f'{method}_s{seed}'
            if not _model_done06(out,opt,stem,x.shape):raise RuntimeError('Incomplete model '+stem)
            emb=normalize(np.load(out/(stem+'_val_features.npy'),mmap_mode='r'))
            scores,r=metrics(emb@v.T,pids,gpids)
            ok=np.asarray(r['rank'])==1
            pred[method]=(scores,ok,r)
            rows.append({'method':method,'seed':seed,'protocol':'STANDARD_SINGLE_TEXT',**scores,
                         'delta_R1_vs_IRRA':scores['R1']-base['R1'],
                         'delta_mAP_vs_IRRA':scores['mAP']-base['mAP'],
                         'repaired_vs_IRRA':int(np.count_nonzero(~baseok&ok)),
                         'harmed_vs_IRRA':int(np.count_nonzero(baseok&~ok)),
                         'mean_margin_change':float(np.mean(r['margin']-br['margin'])),
                         'mean_cos_to_IRRA':float(np.mean((emb*x).sum(1)))})
            msg(f'seed {seed} {method}: R1={scores["R1"]:.4f} '
                f'(ΔIRRA {scores["R1"]-base["R1"]:+.4f}), mAP={scores["mAP"]:.4f}')
            for i,t in enumerate(texts):
                rec={'seed':seed,'method':method,'query_idx':i,'pid':int(pids[i]),
                     'text':t['text'],'baseline_rank':int(br['rank'][i]),
                     'rank':int(r['rank'][i]),'baseline_margin':float(br['margin'][i]),
                     'margin':float(r['margin'][i]),'delta_margin':float(r['margin'][i]-br['margin'][i]),
                     'correct_top1':int(ok[i]),'baseline_top1':int(baseok[i])}
                queries.append(rec)
                if bool(ok[i])!=bool(baseok[i]):cases.append({**rec,'change':'repaired' if ok[i] else 'harmed'})
        byseed[seed]=pred
        def contrast(m,ref):
            a,ao,_=pred[m];b,bo,_=pred[ref]
            pairs.append({'seed':seed,'method':m,'reference':ref,
                          'delta_R1_pp':a['R1']-b['R1'],'delta_mAP_pp':a['mAP']-b['mAP'],
                          'newly_correct':int(np.count_nonzero(ao&~bo)),
                          'newly_wrong':int(np.count_nonzero(~ao&bo))})
        for m in METHODS[1:]:contrast(m,'rank')
        for m,ref in [('sem_true','sem_wrong_matched'),('sem_true','sem_wrong_nearest'),
                      ('reg_sem_true','reg_sem_wrong_matched'),('reg_sem_true','reg_sem_wrong_nearest'),
                      ('reg_sem_true','reg'),('reg_sem_true','sem_true'),('reg','rank')]:contrast(m,ref)
    write_csv(out/'method_summary.csv',rows,list(dict.fromkeys(k for r in rows for k in r)))
    write_csv(out/'paired_rank_comparisons.csv',pairs,list(pairs[0]))
    write_csv(out/'query_deltas.csv',queries,list(queries[0]))
    write_csv(out/'case_changes.csv',cases,list(cases[0]) if cases else list(queries[0])+['change'])
    aggregate=[]
    for method in ('original_IRRA',)+METHODS:
        this=[r for r in rows if r['method']==method]
        z={'method':method,'n_seeds':len(this)}
        for m in ('R1','R5','R10','mAP','mINP'):
            a=np.asarray([r[m] for r in this],dtype=np.float64)
            z[m+'_mean']=float(np.mean(a));z[m+'_sd']=float(np.std(a,ddof=1)) if len(a)>1 else 0.
            z['delta_'+m+'_vs_IRRA_mean']=float(np.mean(a)-base[m])
        aggregate.append(z)
    write_csv(out/'aggregate_summary.csv',aggregate,list(aggregate[0]))
    # True 2x2 factorial differences computed ON EACH SEED, not from maximum model.
    factorial=[]
    for seed in opt.seeds:
        ps=byseed[seed]
        def metric(m,k):return ps[m][0][k]
        entry={'seed':seed}
        for m in ('R1','mAP'):
            a,b,c,d=(metric(k,m) for k in ('rank','reg','sem_true','reg_sem_true'))
            entry[m+'_regularization_gain']=b-a
            entry[m+'_semantic_gain_without_extra_reg']=c-a
            entry[m+'_semantic_gain_with_reg']=d-b
            entry[m+'_interaction']=d-b-c+a
            for name in ('sem_wrong_matched','sem_wrong_nearest'):
                entry[m+'_true_vs_'+name]=c-metric(name,m)
                entry[m+'_joint_true_vs_reg_'+name]=d-metric('reg_'+name,m)
        factorial.append(entry)
    write_csv(out/'factorial_effects.csv',factorial,list(factorial[0]))
    hist=all_training_records(out)
    grad={}
    for method in METHODS[1:]:
        sub=[r for r in hist if r['method']==method]
        grad[method]={key:float(np.mean([r.get(key,0) for r in sub]))
                      for key in ('reg_gradient_ratio','sem_gradient_ratio','reg_gradient_cosine',
                                  'sem_gradient_cosine','applied_gradient_norm','adamw_actual_step_norm')}
        grad[method]['num_epoch_records']=len(sub)
    atomic_json(out/'gradient_diagnostics.json',{
        'main_loss':'rank plus original feature protect',
        'auxiliary_gradient_ratio_per_active_factor':opt.aux_grad_ratio,
        'warning':'Combining regularization and semantics raises total auxiliary gradient budget. '
                  'Different AdamW trajectories still confound exact effect attribution.',
        'methods':grad})
    if not (out/'teacher_diagnostics.json').is_file():raise RuntimeError('Teacher evidence absent')
    report={'experiment':'EXP06_semantic_regularization_factorial',
            'protocol':'Original IRRA frozen, SINGLE text per Val query, no Val PID features/support',
            'baseline':base,'train_pids_for_student':len(opt.fit_pids),
            'internal_unseen_train_pids':len(opt.holdout_pids),
            'training_val_pid_overlap':0,
            'methods':rows,'aggregate':aggregate,'factorial':factorial,
            'teacher_diagnostics':json.loads((out/'teacher_diagnostics.json').read_text(encoding='utf-8')),
            'gradient_diagnostics':grad,
            'disclaimer':['Internal heldout teacher diagnostic uses true identity labels ONLY for its oracle assessment.',
                'True semantic effect is not proven by joint model improvement alone; compare both wrong-PID controls.',
                'One validation split is repeatedly used in the wider research project; keep Test untouched until locked.',
                'This experiment does not train or enhance the image encoder.',
                'Regularization and semantic teaching may both help, interact or conflict; no exclusive attribution.']}
    atomic_json(out/'experiment06_report.json',report)
    atomic_json(out/'research_objectives.json',{
       '1':'Identify stable, real same-identity multi-caption semantic complementarity beyond regularization',
       '2':'Independently identify stable cross-view multi-image complementarity',
       '3':'Demonstrate robust interaction and improvement from fusion of both mechanisms',
       'current':'Goal 1 only. Do not promote Oracle results to normal retrieval metrics.'})
    atomic_json(out/'experiment06_decision_support.json',{
       'factorial_effects_per_seed':factorial,
       'required_controls':['rank','reg','sem_true','reg_sem_true',
                            'sem_wrong_matched','reg_sem_wrong_matched','sem_wrong_nearest','reg_sem_wrong_nearest'],
       'decision':'Do not auto-declare semantic success; inspect each seed, mAP, rank, two wrong-ID controls and internal-heldout reliability first.'})
    msg('EXP06 full single-text metrics and factorial contributions exported.')

# =================== EXP06 validation / orchestration ===================

def self_test06():
    import torch
    rng=np.random.default_rng(606)
    with tempfile.TemporaryDirectory(prefix='irra_exp06_selftest_') as directory:
        out=Path(directory);dim=24;data=[];ttrain=[];itrain=[];tval=[];ival=[]
        for pid in range(1,13):
            center=normalize(rng.normal(size=dim))
            for im in range(3):
                data.append({'split':'train','id':pid,'file_path':f'train/{pid}_{im}.jpg',
                             'captions':[f'person {pid} view {im} description 0',
                                         f'person {pid} view {im} description 1']})
                itrain.append(normalize(center+.22*rng.normal(size=dim)))
                for _ in range(2):ttrain.append(normalize(center+.27*rng.normal(size=dim)))
        for pid in range(51,56):
            center=normalize(rng.normal(size=dim))
            for im in range(2):
                data.append({'split':'val','id':pid,'file_path':f'val/{pid}_{im}.jpg',
                             'captions':[f'val person {pid} view {im} 0',f'val person {pid} view {im} 1']})
                ival.append(normalize(center+.22*rng.normal(size=dim)))
                for _ in range(2):tval.append(normalize(center+.24*rng.normal(size=dim)))
        atomic_json(out/'synthetic_annotations.json',data)
        selected=list(range(1,13));fit,hold=split_mechanism_pids(selected,.75,776)
        atomic_json(out/'selected_train_pids.json',selected)
        for key,ar in [('train_text',ttrain),('train_image',itrain),('text',tval),('image',ival)]:
            np.save(out/(key+'_features.npy'),np.asarray(ar,dtype=np.float32))
        fp_train={'synthetic':'train06'};fp_val={'synthetic':'val06'}
        fake=SimpleNamespace(fp_train=fp_train)
        fp_img=train_img_fingerprint(fake,list(range(len(itrain))))
        for key,fp,n in [('train_text',fp_train,len(ttrain)),('train_image',fp_img,len(itrain)),
                          ('text',fp_val,len(tval)),('image',fp_val,len(ival))]:
            atomic_json(out/(key+'_cache_meta.json'),{'fingerprint':fp,'done':n,'shape':[n,dim]})
        settings={
            'annotation_file':str(out/'synthetic_annotations.json'),'project_root':str(out),
            'config_file':str(out/'fake.yaml'),'checkpoint':str(out/'fake.pth'),
            'images_dir':str(out),'train_text_cache_dir':'','train_image_cache_dir':'',
            'val_cache_dir':'','fp_train':fp_train,'fp_val':fp_val,
            'selected_pids':selected,'fit_pids':fit,'holdout_pids':hold,
            'device':'cpu','seeds':[42,43],'warmup_epochs':1,'continue_epochs':1,
            'train_batch':12,'hidden':32,'gain':.3,'lr':.0003,
            'rank_weight':.2,'protect_weight':.5,'margin':.05,'temperature':.07,
            'hard_k':2,'mine_batch':32,'max_support':5,'teacher_strength':.35,
            'random_teacher_seed':2003,'split_seed':776,'wrong_pool':20,
            'aux_grad_ratio':.2,'max_aux_scale':100000.,'expected_baseline_r1':None,
            'settings_hash':'synthetic06'}
        atomic_json(out/'run_info.json',{'settings':settings,'status':'testing'})
        script=Path(__file__).resolve()
        env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        for stage in ('encode','evidence','train','eval'):
            p=subprocess.run([sys.executable,str(script),'--_worker',stage,'--_run-dir',str(out)],
                             text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,timeout=300)
            if p.returncode:raise AssertionError(f'Self-test {stage} failed:\n{p.stdout}')
        report=json.loads((out/'experiment06_report.json').read_text(encoding='utf-8'))
        assert len(report['methods'])==1+len(METHODS)*2
        assert report['training_val_pid_overlap']==0
        assert report['teacher_diagnostics']['internal_holdout_pids_never_used_to_fit_gate_or_student']
        assert report['teacher_diagnostics']['fit_pid_count']==len(fit)
        assert report['teacher_diagnostics']['internal_unseen']['n_pid']==len(hold)
        assert report['teacher_diagnostics']['fit']['max_wrong_teacher_angle_error']<3e-4
        assert len(report['factorial'])==2
        assert len(list(out.glob('*_training_meta.json')))==len(METHODS)*2
        # Emulate the user's interrupted evidence stage: valid fit and
        # internal-unseen negative caches survive, but the final completion
        # marker is absent. The worker must reuse the caches and reproduce
        # exactly the same teachers rather than mine negatives again.
        teacher_names=('sem_true','sem_wrong_matched','sem_wrong_nearest')
        original_teacher_hashes={k:sha256_small_file(out/(k+'_teacher.npy'))
                                 for k in teacher_names}
        (out/'evidence_complete.json').unlink()
        p=subprocess.run([sys.executable,str(script),'--_worker','evidence','--_run-dir',str(out)],
                         text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                         env=env,timeout=90)
        if (p.returncode or 'Recovered validated fit hard-negative cache' not in p.stdout
                or 'Recovered validated internal-unseen hard-negative cache' not in p.stdout):
            raise AssertionError('Interrupted evidence resume failed:\n'+p.stdout)
        for k in teacher_names:
            assert sha256_small_file(out/(k+'_teacher.npy'))==original_teacher_hashes[k],k
        p=subprocess.run([sys.executable,str(script),'--_worker','train','--_run-dir',str(out)],
                         text=True,capture_output=True,timeout=90,env=env)
        if p.returncode or p.stdout.count('completed + verified, skip')!=len(METHODS)*2:
            raise AssertionError('Resume self-test failed:\n'+p.stdout+'\n'+p.stderr)
        shutil.copy2(script,out/script.name)
        package_for_chat(out)
        with zipfile.ZipFile(out/'发送给ChatGPT.zip') as z:
            for required in ('method_summary.csv','aggregate_summary.csv','factorial_effects.csv',
                             'teacher_diagnostics.json','experiment06_report.json',
                             'gradient_diagnostics.json','training_history.csv',script.name):
                assert required in z.namelist(),required
            assert not any(k.endswith(('.pt','.npy')) for k in z.namelist())
    msg('SELF-TEST PASSED: PID separation, reliability gate, matched wrong-PID teachers, 18 students, metrics, resume, ZIP.')


def worker06(stage,out):
    if stage=='encode':stage_encode03(Path(out))
    elif stage=='evidence':stage_evidence06(Path(out))
    elif stage=='train':stage_train06(Path(out))
    elif stage=='eval':evaluate06(Path(out))
    else:raise ValueError('Unrecognized worker stage: '+str(stage))


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
    parser.add_argument('--warmup-epochs',type=int,default=5)
    parser.add_argument('--continue-epochs',type=int,default=3)
    parser.add_argument('--text-batch',type=int,default=64)
    parser.add_argument('--image-batch',type=int,default=8)
    parser.add_argument('--train-batch',type=int,default=256)
    parser.add_argument('--hidden',type=int,default=256)
    parser.add_argument('--gain',type=float,default=.3)
    parser.add_argument('--lr',type=float,default=.0003)
    parser.add_argument('--rank-weight',type=float,default=.2)
    parser.add_argument('--protect-weight',type=float,default=.5)
    parser.add_argument('--margin',type=float,default=.05)
    parser.add_argument('--temperature',type=float,default=.07)
    parser.add_argument('--teacher-strength',type=float,default=.35)
    parser.add_argument('--hard-k',type=int,default=4)
    parser.add_argument('--mine-batch',type=int,default=96)
    parser.add_argument('--max-support',type=int,default=8)
    parser.add_argument('--random-teacher-seed',type=int,default=2003)
    parser.add_argument('--wrong-pool',type=int,default=256)
    parser.add_argument('--aux-grad-ratio',type=float,default=.2)
    parser.add_argument('--max-aux-scale',type=float,default=100000.)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--_worker',default='',help=argparse.SUPPRESS)
    parser.add_argument('--_run-dir',default='',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args._worker:
        try:worker06(args._worker,args._run_dir)
        except Exception:traceback.print_exc();sys.exit(1)
        return
    if args.self_test:
        self_test06();return
    try:seeds=[int(x.strip()) for x in args.seeds.split(',') if x.strip()]
    except ValueError:parser.error('Seeds must be comma-separated integers')
    if not seeds or len(seeds)!=len(set(seeds)) or any(x<0 for x in seeds):
        parser.error('Specify unique nonnegative seeds')
    for key in ('warmup_epochs','continue_epochs','train_batch','hidden','text_batch',
                'image_batch','hard_k','mine_batch','max_support','wrong_pool'):
        if getattr(args,key)<=0:parser.error(key+' must be positive')
    if args.max_train_pids<0 or not .5<=args.mechanism_fit_fraction<=.95:
        parser.error('Invalid identity selection or --mechanism-fit-fraction (0.5 to 0.95)')
    if not (0<args.teacher_strength<1 and args.lr>0 and args.gain>0 and
            args.aux_grad_ratio>0 and args.max_aux_scale>0 and args.temperature>0 and
            args.rank_weight>0 and args.protect_weight>=0):parser.error('Invalid optimization parameters')
    root=Path(args.project_root).resolve();cfg=Path(args.config_file).resolve();ckpt=Path(args.checkpoint).resolve()
    for f in (root/'model'/'build.py',cfg,ckpt):
        if not f.is_file():raise FileNotFoundError('Missing IRRA project file: '+str(f))
    _,ann,imgs,dataset_source=resolve_dataset(root,cfg,args.dataset_root,args.annotation_file,args.images_root)
    all_img,all_txt,_=read_annotations(ann,'train')
    val_img,val_txt,_=read_annotations(ann,'val')
    allpids=sorted({r['pid'] for r in all_txt})
    if set(allpids)&{r['pid'] for r in val_txt}:raise RuntimeError('Train / Val identity overlap')
    selected=(allpids if args.max_train_pids==0 or args.max_train_pids>=len(allpids)
             else sorted(map(int,np.random.default_rng(args.subset_seed).choice(
                  allpids,args.max_train_pids,replace=False))))
    if len(selected)<8:parser.error('At least eight Train identities are needed for split and controls')
    fit,hold=split_mechanism_pids(selected,args.mechanism_fit_fraction,args.split_seed)
    if len(hold)<2:parser.error('The internal unseen identity partition needs at least two PIDs')
    verify_image_paths(imgs,[im for im in all_img if im['pid'] in set(selected)])
    fp_train=fingerprint(root,cfg,ckpt,ann,'train');fp_val=fingerprint(root,cfg,ckpt,ann,'val')
    parent=root/PROJECT_FOLDER_NAME
    valcache=source_val_cache(root,parent,args.val_cache_dir,fp_val,len(val_txt),len(val_img))
    traintextcache=source_train_text_cache(root,parent,args.train_text_cache_dir,fp_train,len(all_txt))
    image_indices=[i for i,r in enumerate(all_img) if r['pid'] in set(selected)]
    cache_args=SimpleNamespace(fp_train=fp_train,selected_pids=selected)
    trainimagecache=source_train_image_cache(parent,args.train_image_cache_dir,cache_args,image_indices)
    settings={
        'project_root':str(root),'config_file':str(cfg),'checkpoint':str(ckpt),
        'annotation_file':str(ann),'images_dir':str(imgs),'dataset_source':dataset_source,
        'val_cache_dir':str(valcache or ''),'train_text_cache_dir':str(traintextcache or ''),
        'train_image_cache_dir':str(trainimagecache or ''),
        'selected_pids':selected,'fit_pids':fit,'holdout_pids':hold,
        'device':args.device,'seeds':seeds,'subset_seed':args.subset_seed,
        'split_seed':args.split_seed,'mechanism_fit_fraction':args.mechanism_fit_fraction,
        'warmup_epochs':args.warmup_epochs,'continue_epochs':args.continue_epochs,
        'text_batch':args.text_batch,'image_batch':args.image_batch,
        'train_batch':args.train_batch,'hidden':args.hidden,'gain':args.gain,'lr':args.lr,
        'rank_weight':args.rank_weight,'protect_weight':args.protect_weight,
        'margin':args.margin,'temperature':args.temperature,
        'teacher_strength':args.teacher_strength,'hard_k':args.hard_k,'mine_batch':args.mine_batch,
        'max_support':args.max_support,'random_teacher_seed':args.random_teacher_seed,
        'wrong_pool':args.wrong_pool,'aux_grad_ratio':args.aux_grad_ratio,
        'max_aux_scale':args.max_aux_scale,'fp_train':fp_train,'fp_val':fp_val,
        'expected_baseline_r1':56.83663527119195 if (
             len(val_txt)==6158 and len(val_img)==3078 and
             fp_val.get('checkpoint_sha256')=='3bf21f54271d0e7bd0cfd545daa2c2b5367ee2d734419bead417e70c2ec0a36c') else None}
    settings['settings_hash']=hashlib.sha256(json.dumps(settings,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()
    out,resumed=new_or_resume(parent,args,settings)
    LOG_FILE=out/'experiment_run.log'
    shutil.copy2(Path(__file__).resolve(),out/Path(__file__).name)
    pids_path=out/'selected_train_pids.json'
    if resumed:
        if not pids_path.is_file() or json.loads(pids_path.read_text(encoding='utf-8'))!=selected:
            raise RuntimeError('Resume identity selection mismatch')
    else:atomic_json(pids_path,selected)
    atomic_json(out/'identity_partition.json',{
        'fit_pid_count':len(fit),'holdout_pid_count':len(hold),
        'fit_pids':fit,'internal_unseen_pids':hold,'train_val_pid_overlap':0})
    meta_path=out/'run_info.json'
    meta=json.loads(meta_path.read_text(encoding='utf-8')) if resumed else {}
    meta.update({'experiment_name':EXPERIMENT_NAME,'settings':settings,'status':'running',
                 'resumed':resumed,'started_at':datetime.now().isoformat()})
    atomic_json(meta_path,meta)
    msg('EXP06 folder: '+str(out))
    msg(f'Identity split: {len(fit)} fit, {len(hold)} internal unseen, Val {len(val_txt)} queries')
    msg(f'Seeds {seeds}; model groups {METHODS}')
    try:
        for stage in ('encode','evidence','train','eval'):
            run_subprocess(stage,out,root)
        meta['status']='completed';meta['finished_at']=datetime.now().isoformat()
        atomic_json(meta_path,meta)
        msg('EXP06 COMPLETED: see factorial_effects.csv and 发送给ChatGPT.zip')
    except Exception as e:
        meta['status']='failed';meta['error']=repr(e);meta['failed_at']=datetime.now().isoformat()
        atomic_json(meta_path,meta)
        (out/'error_traceback.log').write_text(traceback.format_exc(),encoding='utf-8')
        msg('EXP06 FAILED: '+repr(e))
        raise
    finally:
        try:package_for_chat(out)
        except Exception as exc:msg('EXP06 ZIP creation error: '+repr(exc))

if __name__=='__main__':
    main()
