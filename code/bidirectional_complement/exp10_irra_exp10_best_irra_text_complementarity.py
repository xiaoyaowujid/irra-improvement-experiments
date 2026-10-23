# -*- coding: utf-8 -*-
"""实验10：正确的IRRA Epoch58最佳权重下，重跑已保留的文本互补实验09。

只训练（1）重新建立的EXP07 Warmup、（2）No-Support、（3）Selected-Support。
同一固定Top100候选与完整Val图库、6个修正系数；Train PID内部拆分。
原IRRA编码器及best.pth完全只读，绝不optimizer.step()或保存覆盖原模型。
脚本具备Windows分进程、断点恢复、特征权重指纹检验、ZIP报告。
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
DEFAULT_CONFIG = r"G:\PyCharmProjects\IRRA\IRRA-main\logs\CUHK-PEDES\20261009_095215_irra_baseline_b8_resume20\configs.yaml"
DEFAULT_CKPT = r"G:\PyCharmProjects\IRRA\IRRA-main\logs\CUHK-PEDES\20261009_095215_irra_baseline_b8_resume20\best.pth"
EXPECTED_SHA = "f3ab12028e3406546b836ed6a75c8fc5e02a42146c5da13af423e7f5cf0e7ea6"
EXPECTED_CONFIG_SHA = "ec2fae46ec4e0cce06372789afe1c877a5cebeddf73654781e11b0484fc30fac"
EXPECTED_R1 = 66.43390655517578
EXPECTED_METRICS = {'R1':66.43390655517578,'R5':85.28743743896484,
                    'R10':90.58135986328125,'mAP':61.86301040649414,
                    'mINP':47.24529266357422}
EXPECTED_ANNOTATION_SHA = 'fc3d15542f91dd70e23028c2cb40171a8f41734a39afddb9505e352e396cabf0'
PROJECT_FOLDER_NAME = '图文双向互补实验'
EXPERIMENT_NAME = '实验10_最佳IRRA重跑文本互补选择'
METHODS = ('no_support','selected_support')
VERSION09 = 'exp10_best58_locked_20261021_v1'
ALPHAS09 = (0., .1, .25, .5, .75, 1.)
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
            raise RuntimeError(f'Cache metadata conflict at {meta_path}; rerun without --resume to create a fresh output folder')
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
                    if {k:v for k,v in x.get('settings',{}).items() if k!='run_dir'}==settings:
                        return d,True
                except (OSError,ValueError):
                    pass
        raise FileNotFoundError('No matching experiment10 run to resume. This output folder may have been deleted. Run WITHOUT --resume to create a fresh experiment10 folder.')
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


def _teacher_pairs09(x,v,tx,im,pos,neg,gate,opt,sourceopt,save_examples):
    """Create a support teacher using TRAIN PID only. No inference-time support."""
    byid=defaultdict(list)
    for j,r in enumerate(tx):byid[int(r['pid'])].append(j)
    selected=np.array(x,copy=True)
    mask=np.zeros(len(x),dtype=np.uint8)
    cnt=defaultdict(int);rows=[]
    for i in range(len(x)):
        peers=[j for j in byid[int(tx[i]['pid'])]
               if int(tx[j]['img_idx'])!=int(tx[i]['img_idx'])]
        peers=capped_peer_indices(peers,x[i],x,sourceopt.max_support)
        if not peers:
            cnt['no_support']+=1;continue
        features=np.stack([triage_features(x[i],x[j],v[pos[i]],v[pos[j]],v[neg[i,0]]) for j in peers])
        prob=gate_predict(features,gate)
        use=(prob[:,2]>=opt.use_threshold)&(prob[:,2]>=prob[:,0]+opt.prob_margin)
        rej=(prob[:,0]>=opt.reject_threshold)&(prob[:,0]>=prob[:,2]+opt.prob_margin)
        cnt['possible_support']+=len(peers)
        cnt['accepted_support']+=int(use.sum())
        cnt['rejected_support']+=int(rej.sum())
        cnt['uncertain_support']+=int(len(peers)-int(use.sum())-int(rej.sum()))
        if use.any():
            chosen=np.asarray(peers)[use]
            w=np.maximum(prob[use,2]-prob[use,0],1e-8);w=w/w.sum()
            peer=np.average(x[chosen],axis=0,weights=w)
            selected[i]=normalize((1-sourceopt.teacher_strength)*x[i]+sourceopt.teacher_strength*peer)
            mask[i]=2;cnt['queries_accepted']+=1
        elif rej.any():
            mask[i]=0;cnt['queries_rejected']+=1
        else:mask[i]=1;cnt['queries_uncertain']+=1
        if i<save_examples:
            rows.append({'query_idx':i,'text':tx[i]['text'],'pid_train_only':int(tx[i]['pid']),
              'candidates':len(peers),'accepted':int(use.sum()),'rejected':int(rej.sum()),
              'uncertain':int(len(peers)-use.sum()-rej.sum()),
              'result':['reject','uncertain','accept'][mask[i]],
              'teacher_chord_delta':float(np.linalg.norm(selected[i]-x[i]))})
    return normalize(selected),mask,dict(cnt),rows


def evidence09(out):
    _,opt,_=manifest_for_stage(out)
    mark=out/'evidence_complete.json'
    requirements=('fit_teacher.npy','fit_mask.npy','selector.json','selector_diagnostics.json')
    if mark.is_file():
        try:
            m=json.loads(mark.read_text(encoding='utf-8'))
            src,so,fit,hold=source_data09(out)
            if (m.get('settings_hash')==opt.settings_hash and all((out/f).is_file() for f in requirements)
                and np.load(out/'fit_teacher.npy',mmap_mode='r').shape==fit[0].shape
                and np.load(out/'fit_mask.npy',mmap_mode='r').shape==(len(fit[0]),)):
                msg('EXP09 valid evidence cache found; skip');return
        except (ValueError,OSError,KeyError):pass
    src,so,fit,hold=source_data09(out)
    xf,vf,tf,gf,pf,negf=fit
    xh,vh,th,gh,ph=hold
    nh_path=src/'holdout_negatives.npy'
    if not nh_path.is_file():raise RuntimeError('Missing EXP07 internal unseen negatives')
    nh=np.load(nh_path,allow_pickle=False)
    if nh.shape[0]!=len(xh) or nh.shape[1]<1:raise ValueError('Holdout negatives incompatible')
    hpids=np.asarray([z['pid'] for z in th]);gpids=np.asarray([z['pid'] for z in gh])
    if np.any(nh<0) or np.any(nh>=len(vh)) or np.any(gpids[nh]==hpids[:,None]):
        raise RuntimeError('Holdout negatives contain illegal identity')
    # Same usefulness proxy as EXP07; holding out identities, not just captions.
    f,y,d,fe=build_triage_rows(xf,vf,tf,pf,negf,so,1200)
    h,hy,hd,he=build_triage_rows(xh,vh,th,ph,nh[:,-min(int(so.hard_k),nh.shape[1]):],so,450)
    if not (np.any(y==0) and np.any(y==2) and np.any(hy==0) and np.any(hy==2)):
        raise RuntimeError('No useful AND harmful support on fit/holdout; cannot certify both objectives')
    g=fit_three_way_gate(f,y,opt.selector_seed)
    fp=gate_predict(f,g);hp=gate_predict(h,g)
    fitstat=triage_diag(y,fp,d,SimpleNamespace(use_threshold=opt.use_threshold,
      harm_threshold_prob=opt.reject_threshold,use_vs_harm_margin=opt.prob_margin))
    holdstat=triage_diag(hy,hp,hd,SimpleNamespace(use_threshold=opt.use_threshold,
      harm_threshold_prob=opt.reject_threshold,use_vs_harm_margin=opt.prob_margin))
    teacher,mask,counts,examples=_teacher_pairs09(xf,vf,tf,gf,pf,negf,g,opt,so,1200)
    atomic_npy(out/'fit_teacher.npy',teacher)
    atomic_npy(out/'fit_mask.npy',mask)
    atomic_json(out/'selector.json',g)
    diag={'identity_split':{'fit_pids':len(so.fit_pids),'internal_unseen_train_pids':len(so.holdout_pids)},
      'fit':fitstat,'identity_disjoint_holdout':holdstat,'selection_counts':counts,
      'equal_priority':['accept_useful_rate','reject_harmful_rate'],
      'no_leakage':'PID used only for teacher support grouping; validation inference never loads this support set',
      'limits':'Ranking-utility proxy, not a ground-truth claim about textual factual correctness'}
    atomic_json(out/'selector_diagnostics.json',diag)
    if examples:write_csv(out/'support_selection_cases.csv',examples,list(examples[0]),allow_locked_fallback=True)
    if fe or he:write_csv(out/'triage_examples.csv',[dict(e,subset='fit') for e in fe]+[dict(e,subset='internal_unseen') for e in he],list((fe or he)[0])+['subset'],allow_locked_fallback=True)
    atomic_json(mark,{'settings_hash':opt.settings_hash,'n_fit':len(xf),
                      'selector_sha256':sha256_small_file(out/'selector.json')})
    msg(f"EXP09 holdout ACCEPT useful={holdstat['accept_useful_rate']:.4f}, "
        f"REJECT harmful={holdstat['reject_harmful_rate']:.4f}, "
        f"accepted_fit={counts.get('queries_accepted',0)}/{len(xf)}")


def get_pair09():
    import torch
    from torch import nn
    from torch.nn import functional as F
    class PairReranker(nn.Module):
        def __init__(self,dim,hidden,max_delta):
            super().__init__()
            self.fc1=nn.Linear(2*dim+1,hidden)
            self.fc2=nn.Linear(hidden,hidden)
            self.gate=nn.Linear(hidden,1)
            self.delta=nn.Linear(hidden,1)
            self.max_delta=float(max_delta)
        def forward(self,t,image):
            t=t[:,None,:].expand_as(image)
            cos=(t*image).sum(-1)
            h=F.gelu(self.fc1(torch.cat((t*image,torch.abs(t-image),cos[...,None]),dim=-1)))
            h=F.gelu(self.fc2(h))
            logits=self.gate(h).squeeze(-1)
            gate=torch.sigmoid(logits)
            raw=self.max_delta*torch.tanh(self.delta(h).squeeze(-1))
            return cos+gate*raw,gate*raw,gate
    return PairReranker


def prediction09(net,tx,imgs,indices,device,qbatch):
    import torch
    n,k=indices.shape
    arr=np.empty((n,k),np.float32)
    net.eval()
    with torch.inference_mode():
        for st in range(0,n,max(1,min(qbatch,8192//max(1,k)))):
            end=min(n,st+max(1,min(qbatch,8192//max(1,k))))
            q=tx[st:end].to(device)
            ids=torch.as_tensor(np.array(indices[st:end],copy=True).astype(np.int64),device=device)
            v=imgs[ids]
            out,_,_=net(q,v)
            arr[st:end]=out.float().cpu().numpy()
    return arr


def check_student09(out,stem,opt,n,k):
    ck=out/(stem+'_student.pt');meta=out/(stem+'_training_meta.json')
    pred=out/(stem+'_topk_score.npy');hold=out/(stem+'_holdout_topk_score.npy')
    if not all(z.is_file() for z in (ck,meta,pred,hold)):return False
    try:
        m=json.loads(meta.read_text(encoding='utf-8'))
        a=np.load(pred,mmap_mode='r',allow_pickle=False)
        b=np.load(hold,mmap_mode='r',allow_pickle=False)
        return (m.get('complete') and m.get('settings_hash')==opt.settings_hash
          and m.get('checkpoint_sha256')==sha256_small_file(ck)
          and a.shape==(n,k) and a.dtype==np.float32 and np.isfinite(a).all()
          and b.ndim==2 and b.dtype==np.float32 and np.isfinite(b).all())
    except (ValueError,KeyError,OSError):return False


def holdout_topk09(out):
    """Precompute holdout scores and candidates in NumPy-only evidence worker."""
    _,opt,_=manifest_for_stage(out)
    src,so,fit,hold=source_data09(out)
    xh,vh,th,gh,ph=hold
    path=out/'holdout_topk.npy'
    meta=out/'holdout_topk_meta.json'
    expected={'settings_hash':opt.settings_hash,'n_queries':len(xh),
              'n_gallery':len(vh),'k':min(int(so.eval_topk),len(vh))}
    if path.is_file() and meta.is_file():
        try:
            a=np.load(path,mmap_mode='r',allow_pickle=False)
            if json.loads(meta.read_text(encoding='utf-8'))==expected and a.shape==(len(xh),expected['k']):
                return np.asarray(a,dtype=np.int32)
        except (OSError,ValueError,KeyError):pass
    k=expected['k'];res=np.empty((len(xh),k),np.int32)
    for st in range(0,len(xh),96):
        scores=xh[st:st+96]@vh.T
        idx=np.argpartition(-scores,k-1,axis=1)[:,:k]
        idx=np.take_along_axis(idx,np.argsort(-np.take_along_axis(scores,idx,axis=1),axis=1,kind='stable'),axis=1)
        res[st:st+len(idx)]=idx
    atomic_npy(path,res);atomic_json(meta,expected)
    return res


def stage_prepare09(out):
    evidence09(out)
    tk=holdout_topk09(out)
    msg(f'EXP09 internal heldout top-{tk.shape[1]} cached by NumPy-only process')


def train_epoch09(net,warm,optimizer,order,tx,gv,pos,neg,teacher,mask,mode,so,opt,device):
    import torch
    import torch.nn.functional as F
    net.train();warm.eval()
    sums=defaultdict(float);nt=0
    for st in range(0,len(order),int(so.train_batch)):
        sub=order[st:st+int(so.train_batch)];bs=len(sub)
        t=tx[sub].to(device)
        ids=torch.cat([pos[sub,None],neg[sub]],dim=1).to(device)
        images=gv[ids]
        scores,corr,gate=net(t,images)
        with torch.no_grad():baseline,_,_=warm(t,images)
        # Controlled base ranking objective in BOTH branches. Always preserve Warmup.
        rank=F.softplus((scores[:,1:]-scores[:,:1]+float(so.margin))/float(so.temperature)).mean()
        warm_preserve=(scores-baseline).square().mean()
        target=baseline.detach().clone()
        if mode=='selected_support':
            usable=(mask[sub]==2).to(device)
            if usable.any():
                counterpart=teacher[sub].to(device)
                original=(t[:,None,:]*images).sum(-1)
                new=(counterpart[:,None,:]*images).sum(-1)
                # Teacher offers only bounded, evidence-screened correction.
                additional=torch.clamp(new-original,-opt.teacher_delta_cap,opt.teacher_delta_cap)
                target=target+additional*usable[:,None]
            sums['selected_query_fraction']+=float(usable.float().mean())*bs
        distill=F.smooth_l1_loss(scores,target,beta=.02)
        loss=(opt.rank_weight*rank+opt.preserve_weight*warm_preserve+
              opt.distill_weight*distill)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
        optimizer.step()
        sums['rank_loss']+=float(rank.detach())*bs
        sums['warmup_preserve_loss']+=float(warm_preserve.detach())*bs
        sums['distill_loss']+=float(distill.detach())*bs
        sums['absolute_student_warmup_diff']+=float((scores-baseline).detach().abs().mean())*bs
        sums['gate_mean']+=float(gate.detach().mean())*bs
        nt+=bs
    return {key:float(val)/max(1,nt) for key,val in sums.items()}


def train09(out):
    import torch
    torch.set_num_threads(1)
    _,opt,_=manifest_for_stage(out)
    src,so,fit,hold=source_data09(out)
    _,(xv,vv,vi,vt,topk)=source_data09(out,True)
    xf,vf,tf,gf,pf,neg=fit
    xh,vh,th,gh,ph=hold
    if not (out/'evidence_complete.json').is_file() or not (out/'holdout_topk.npy').is_file():
        raise RuntimeError('Run NumPy-only preparation before PyTorch training')
    teacher=np.load(out/'fit_teacher.npy',mmap_mode='r')
    mask=np.load(out/'fit_mask.npy',mmap_mode='r')
    if teacher.shape!=xf.shape or mask.shape!=(len(xf),):raise RuntimeError('Teacher evidence mismatch')
    heldtk=np.load(out/'holdout_topk.npy',mmap_mode='r',allow_pickle=False)
    if heldtk.shape[0]!=len(xh):raise RuntimeError('Holdout indices shape mismatch')
    tk=np.asarray(topk,dtype=np.int64)
    device=torch.device('cuda' if opt.device=='cuda' or (opt.device=='auto' and torch.cuda.is_available()) else 'cpu')
    if opt.device=='cuda' and device.type!='cuda':raise RuntimeError('CUDA was requested but unavailable')
    tr=torch.from_numpy(np.array(xf,copy=True));images=torch.from_numpy(np.array(vf,copy=True)).to(device)
    pos=torch.from_numpy(np.asarray(pf,dtype=np.int64));negs=torch.from_numpy(np.asarray(neg,dtype=np.int64))
    tt=torch.from_numpy(np.array(teacher,copy=True));mm=torch.from_numpy(np.array(mask,dtype=np.int64,copy=True))
    vt_tensor=torch.from_numpy(np.array(xv,copy=True));vg_tensor=torch.from_numpy(np.array(vv,copy=True)).to(device)
    ht_tensor=torch.from_numpy(np.array(xh,copy=True));hg_tensor=torch.from_numpy(np.array(vh,copy=True)).to(device)
    class_ = get_pair09()
    for seed in opt.seeds:
        validate_warmup09(src,so,seed,len(xv),tk.shape[1])
        saved=torch.load(src/f'warmup_s{seed}_student.pt',map_location='cpu',weights_only=False)
        if saved['settings_hash']!=so.settings_hash:raise RuntimeError('Source checkpoint settings changed')
        frozen=class_(xf.shape[1],int(so.hidden),float(so.max_delta)).to(device)
        frozen.load_state_dict(saved['model_state_dict']);frozen.eval()
        for param in frozen.parameters():param.requires_grad_(False)
        # Verify the source Warmup on Val BEFORE training any student.
        reconstructed=prediction09(frozen,vt_tensor,vg_tensor,tk,device,opt.eval_query_batch)
        original=np.load(src/f'warmup_s{seed}_topk_score.npy',mmap_mode='r',allow_pickle=False)
        warm_error=float(np.max(np.abs(reconstructed-original)))
        if warm_error>2e-4:
            raise RuntimeError(f'Warmup verification failed for seed {seed}: {warm_error:.7f}; refuse this source checkpoint')
        warm_holdout=out/f'warmup_s{seed}_holdout_topk_score.npy'
        if warm_holdout.is_file():
            try:
                arr=np.load(warm_holdout,mmap_mode='r',allow_pickle=False)
                if arr.shape!=heldtk.shape or not np.isfinite(arr).all():
                    raise ValueError('Invalid warmup holdout score')
            except (ValueError,OSError):
                atomic_npy(warm_holdout,prediction09(frozen,ht_tensor,hg_tensor,np.array(heldtk,dtype=np.int32),device,opt.eval_query_batch))
        else:
            atomic_npy(warm_holdout,prediction09(frozen,ht_tensor,hg_tensor,np.array(heldtk,dtype=np.int32),device,opt.eval_query_batch))
        msg(f'EXP09 seed {seed} original Warmup reproduction max error={warm_error:.2g}')
        for method in METHODS:
            stem=f'{method}_s{seed}'
            if check_student09(out,stem,opt,len(xv),tk.shape[1]):
                msg(stem+': validated complete; skip');continue
            torch.manual_seed(seed)
            model=class_(xf.shape[1],int(so.hidden),float(so.max_delta)).to(device)
            model.load_state_dict(saved['model_state_dict'])
            optim=torch.optim.AdamW(model.parameters(),lr=float(opt.lr),weight_decay=1e-4)
            gen=torch.Generator().manual_seed(seed+23001)
            hist=[]
            for epoch in range(1,opt.epochs+1):
                order=torch.randperm(len(xf),generator=gen)
                stats=train_epoch09(model,frozen,optim,order,tr,images,pos,negs,tt,mm,method,so,opt,device)
                hist.append({'seed':seed,'method':method,'epoch':epoch,**stats})
                msg(f'{stem} epoch {epoch}/{opt.epochs}: rank={stats["rank_loss"]:.5f}; preserve={stats["warmup_preserve_loss"]:.6f}')
            prediction=prediction09(model,vt_tensor,vg_tensor,tk,device,opt.eval_query_batch)
            hold_prediction=prediction09(model,ht_tensor,hg_tensor,np.array(heldtk,dtype=np.int32),device,opt.eval_query_batch)
            # Save arrays with closed file descriptors to prevent WinError32.
            atomic_npy(out/(stem+'_topk_score.npy'),prediction)
            atomic_npy(out/(stem+'_holdout_topk_score.npy'),hold_prediction)
            saved_path=out/(stem+'_student.pt');partial=out/(stem+'_student.partial.pt')
            torch.save({'model_state_dict':model.state_dict(),'settings_hash':opt.settings_hash,
                        'seed':seed,'method':method,'epoch':opt.epochs},partial)
            _replace_with_retries(partial,saved_path)
            atomic_json(out/(stem+'_training_meta.json'),{
               'complete':True,'settings_hash':opt.settings_hash,'checkpoint_sha256':sha256_small_file(saved_path),
               'seed':seed,'method':method,'history':hist,'n_val_queries':len(xv),
               'n_holdout_queries':len(xh),'k_val':tk.shape[1],'k_holdout':heldtk.shape[1]})
            del model,optim
        del frozen
        if device.type=='cuda':torch.cuda.empty_cache()
    msg('EXP10 student training completed: only no_support and selected_support.')


def read_training_hist09(out,opt):
    hist=[]
    for seed in opt.seeds:
        for meth in METHODS:
            m=out/(f'{meth}_s{seed}_training_meta.json')
            if m.is_file():
                d=json.loads(m.read_text(encoding='utf-8'))
                if d.get('settings_hash')==opt.settings_hash and d.get('complete'):
                    hist.extend(d.get('history',[]))
    return hist


def _rerank09(base,x,v,idx,pair,alpha):
    n,k=idx.shape;rows=np.arange(n)[:,None]
    out=base.copy()
    original=base[rows,idx]
    out[rows,idx]=original+float(alpha)*(pair-original)
    return out


def selection09(out):
    """NumPy-only: select ONE common alpha using heldout TRAIN identities."""
    _,opt,_=manifest_for_stage(out)
    src,so,fit,(xh,vh,th,gh,ph)=source_data09(out)
    tk=np.load(out/'holdout_topk.npy',allow_pickle=False)
    pids=np.asarray([a['pid'] for a in th]);gpids=np.asarray([a['pid'] for a in gh])
    full=xh@vh.T
    base_metric,_=metrics(full,pids,gpids)
    rows=[]
    for seed in opt.seeds:
        warm=np.load(out/f'warmup_s{seed}_holdout_topk_score.npy',mmap_mode='r',allow_pickle=False)
        if warm.shape!=tk.shape:raise RuntimeError('Warmup heldout scores missing')
        for meth in ('warmup',)+METHODS:
            p=warm if meth=='warmup' else np.load(out/(f'{meth}_s{seed}_holdout_topk_score.npy'),mmap_mode='r')
            for alpha in ALPHAS09:
                sim=_rerank09(full,xh,vh,tk,p,alpha)
                met,_=metrics(sim,pids,gpids)
                rows.append({'seed':seed,'method':meth,'alpha':alpha,'R1':met['R1'],'mAP':met['mAP'],
                             'baseline_R1':base_metric['R1'],'baseline_mAP':base_metric['mAP'],
                             'protocol':'INTERNAL_TRAIN_HOLDOUT_FIXED_TOPK_VALID_ALPHA_SELECTION'})
    write_csv(out/'internal_alpha_grid.csv',rows,list(rows[0]),allow_locked_fallback=True)
    # One COMMON alpha chosen based on WARMUP only, with prespecified mAP criterion.
    options=[]
    for alpha in ALPHAS09:
        vals=[r['mAP'] for r in rows if r['method']=='warmup' and r['alpha']==alpha]
        options.append((float(np.mean(vals)),float(alpha)))
    best=sorted(options,key=lambda x:(-x[0],x[1]))[0][1]
    atomic_json(out/'alpha_selection.json',{'selected_common_alpha':best,
      'selection_method':'Warmup heldout TRAIN identities mAP, averaged across seeds; tie breaks toward lower alpha',
      'alpha_candidates':list(ALPHAS09),'holdout_train_identity_count':len(so.holdout_pids),
      'warning':'Internal holdout was also used to report selector diagnostics: diagnostic and calibration roles must not be conflated.'})
    msg(f'EXP09 common alpha locked from TRAIN holdout Warmup: {best}')


def eval09(out):
    _,opt,_=manifest_for_stage(out)
    src,so,fit,hold=source_data09(out)
    _,(x,v,images,texts,tk)=source_data09(out,True)
    val_pids=np.asarray([r['pid'] for r in texts]);gallery_pids=np.asarray([r['pid'] for r in images])
    sim=x@v.T
    baseline,bd=metrics(sim,val_pids,gallery_pids)
    if opt.expected_baseline_r1 is not None and abs(baseline['R1']-opt.expected_baseline_r1)>.05:
        raise RuntimeError('Original IRRA baseline mismatch')
    locked=json.loads((out/'alpha_selection.json').read_text(encoding='utf-8'))['selected_common_alpha']
    results=[];cases=[];pairwise=[]
    b_ok=bd['rank']==1
    for seed in opt.seeds:
        cases_by_meth={}
        source_warm=src/f'warmup_s{seed}_topk_score.npy'
        warm=np.load(source_warm,mmap_mode='r',allow_pickle=False)
        if warm.shape!=tk.shape:raise RuntimeError('EXP10 Warmup prediction mismatch')
        for meth in ('warmup',)+METHODS:
            if meth=='warmup':sc=warm
            else:
                stem=f'{meth}_s{seed}'
                if not check_student09(out,stem,opt,len(x),tk.shape[1]):
                    raise RuntimeError('Student result missing/corrupted: '+stem)
                sc=np.load(out/(stem+'_topk_score.npy'),mmap_mode='r',allow_pickle=False)
            for alpha in ALPHAS09:
                s=_rerank09(sim,x,v,tk,sc,alpha)
                met,details=metrics(s,val_pids,gallery_pids)
                ok=details['rank']==1
                results.append({'method':meth,'seed':seed,'alpha':alpha,'selected_alpha':locked,
                   'alpha_is_locked':int(alpha==locked),**met,
                   'R1_minus_IRRA_pp':met['R1']-baseline['R1'],
                   'repaired_vs_IRRA':int((~b_ok&ok).sum()),
                   'harmed_vs_IRRA':int((b_ok&~ok).sum()),
                   'protocol':'SINGLE_TEXT_PID_BLIND_FIXED_ORIGINAL_IRRA_TOPK_FULL_GALLERY'})
                if alpha==locked:
                    cases_by_meth[meth]=ok
                    # Record query changes against BOTH fixed baselines.
                    for j in np.where(ok!=b_ok)[0]:
                        cases.append({'seed':seed,'method':meth,'query_idx':int(j),
                         'text':texts[j]['text'],'pid_METRICS_ONLY':int(val_pids[j]),
                         'IRRA_rank':int(bd['rank'][j]),'new_rank':int(details['rank'][j]),
                         'repair_vs_IRRA':int(ok[j] and not b_ok[j]),
                         'harm_vs_IRRA':int(not ok[j] and b_ok[j])})
        # The actual EXP07 Warmup baseline is alpha=1 regardless of selected alpha.
        # When selected alpha is zero, it is STILL reported separately from IRRA.
        src_warm=next(r for r in results if r['seed']==seed and r['method']=='warmup' and r['alpha']==1.0)
        original_warm_scores=_rerank09(sim,x,v,tk,warm,1.0)
        _,source_details=metrics(original_warm_scores,val_pids,gallery_pids)
        source_ok=source_details['rank']==1
        for meth in METHODS:
            for reference in ('warmup_original','warmup_locked','no_support') if meth=='selected_support' else ('warmup_original','warmup_locked'):
                if reference=='no_support':
                    ref=cases_by_meth['no_support']
                    met2=next(r for r in results if r['seed']==seed and r['method']=='no_support' and r['alpha']==locked)
                elif reference=='warmup_locked':
                    ref=cases_by_meth['warmup']
                    met2=next(r for r in results if r['seed']==seed and r['method']=='warmup' and r['alpha']==locked)
                else:
                    ref=source_ok
                    met2=src_warm
                here=cases_by_meth[meth]
                met1=next(r for r in results if r['seed']==seed and r['method']==meth and r['alpha']==locked)
                pairwise.append({'seed':seed,'method':meth,'reference':reference,
                    'delta_R1_pp':met1['R1']-met2['R1'],
                    'delta_mAP_pp':met1['mAP']-met2['mAP'],
                    'newly_correct':int((here&~ref).sum()),'newly_wrong':int((~here&ref).sum())})
    results.insert(0,{'method':'original_IRRA','seed':-1,'alpha':0.,'selected_alpha':locked,
      'alpha_is_locked':1,**baseline,'R1_minus_IRRA_pp':0.,'repaired_vs_IRRA':0,
      'harmed_vs_IRRA':0,'protocol':'FROZEN_BEST_EPOCH58_IRRA_FULL_GALLERY'})
    write_csv(out/'method_summary.csv',results,list(dict.fromkeys(k for r in results for k in r)),allow_locked_fallback=True)
    write_csv(out/'paired_rank_comparisons.csv',pairwise,list(pairwise[0]) if pairwise else [],allow_locked_fallback=True)
    if cases:write_csv(out/'case_changes.csv',cases,list(cases[0]),allow_locked_fallback=True)
    agg=[]
    for method in ('original_IRRA','warmup_original','warmup_locked')+METHODS:
        if method=='warmup_original':
            chosen=[r for r in results if r['method']=='warmup' and r['alpha']==1.0]
        elif method=='warmup_locked':
            chosen=[r for r in results if r['method']=='warmup' and r['alpha']==locked]
        else:
            chosen=[r for r in results if r['method']==method and r['alpha_is_locked']==1]
        if not chosen:raise RuntimeError('Empty locked-alpha aggregate')
        item={'method':method,'selected_alpha':locked,'seeds':len(chosen)}
        for key in ('R1','R5','R10','mAP','mINP'):
            a=np.array([r[key] for r in chosen],dtype=np.float64)
            item[key+'_mean']=float(a.mean());item[key+'_sd']=float(a.std(ddof=1)) if len(a)>1 else 0.
            item['delta_'+key+'_vs_IRRA_pp']=float(a.mean()-baseline[key])
        agg.append(item)
    write_csv(out/'aggregate_summary.csv',agg,list(agg[0]),allow_locked_fallback=True)
    hist=read_training_hist09(out,opt)
    if hist:write_csv(out/'training_history.csv',hist,list(dict.fromkeys(k for r in hist for k in r)),allow_locked_fallback=True)
    gate_diag=json.loads((out/'selector_diagnostics.json').read_text(encoding='utf-8'))
    report={'experiment':'EXP10 new BEST Epoch58 IRRA rerun of retained EXP09 with matched fresh Warmup',
      'model_branches':list(METHODS),'baseline_methods':['best_epoch58_original_IRRA','new_best_checkpoint_warmup_alpha1'],
      'alpha_lock':locked,'alpha_choice':'Training holdout Warmup mAP ONLY; not Val',
      'results':agg,'selector_identity_holdout':gate_diag['identity_disjoint_holdout'],
      'no_val_pid_in_scoring':True,
      'limitations':['Support truth/usefulness is defined by frozen IRRA ranking-utility proxy, not natural language factual labels.',
       'The student cannot select unavailable support at inference; it transfers supervision from train-time selected support.',
       'Heldout TRAIN PID used to lock alpha; not independent test of hyperparameter generalization.',
       'Validation repeatedly used in prior experiments; all alpha sweep results there are descriptive, never use Val optimum to make claims.',
       'Both students have identical architecture and same training protocol; no_support controls extra optimization but not all conceivable regularization mechanisms.',
       'Scoring is original IRRA fixed Top-K full-gallery reranking; not independent-index vector retrieval.'],
      'success_criteria':['Both useful acceptance and harmful rejection must be nondegenerate on unseen TRAIN identities',
       'selected_support must exceed best IRRA Epoch58, freshly retrained Warmup (alpha=1), and no_support under locked alpha across seeds',
       'R1 and mAP, paired repairs and harms are both analyzed; no success claim from one seed or Val-tuned alpha'],
      'research_objectives':['1: reliable multi-text complementarity','2: reliable multi-view image complementarity','3: strict positive joint fusion']}
    atomic_json(out/'experiment10_report.json',report)
    msg('EXP10 finished: best Epoch58 IRRA full-gallery Val metrics, locked alpha, selector diagnostics.')


def package09(out):
    out=Path(out);z=out/'发送给ChatGPT.zip'
    fd,tmp=_new_temp_path(z);os.close(fd)
    try:
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
            for p in sorted(out.iterdir()):
                if p.is_file() and p.suffix.lower() in ('.csv','.json','.md','.py','.log','.txt'):
                    zf.write(p,p.name)
        try:_replace_with_retries(tmp,z)
        except PermissionError:
            z=out/('发送给ChatGPT_恢复版_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.zip')
            _replace_with_retries(tmp,z)
    finally:
        try:tmp.unlink(missing_ok=True)
        except OSError:pass
    msg(f'EXP09 analysis ZIP saved: {z} ({z.stat().st_size/1048576:.2f} MiB)')


def validate_warmup09(src,so,seed,n=None,k=None):
    stem=f'warmup_s{seed}'
    ck=src/(stem+'_student.pt');meta=src/(stem+'_training_meta.json')
    sc=src/(stem+'_topk_score.npy');gate=src/(stem+'_topk_gate.npy')
    if not all(p.is_file() for p in (ck,meta,sc,gate)):
        raise FileNotFoundError('Incomplete EXP07 Warmup: '+stem)
    m=json.loads(meta.read_text(encoding='utf-8'))
    if m.get('settings_hash')!=so.settings_hash or m.get('checkpoint_sha256')!=sha256_small_file(ck):
        raise RuntimeError('Warmup checkpoint integrity check failed: '+stem)
    if n is not None:
        a=np.load(sc,mmap_mode='r');g=np.load(gate,mmap_mode='r')
        if a.shape!=(n,k) or g.shape!=(n,k) or not np.isfinite(a).all():
            raise RuntimeError('Warmup score / gate artifact corrupted: '+stem)
    return ck


# ================= EXP10: correct best-checkpoint standalone source ==============

def source09(opt):
    """Never read or import stale EXP07 files: this EXP10 run owns all data."""
    p=Path(opt.run_dir)
    if not (p/'run_info.json').is_file():
        raise FileNotFoundError('EXP10 manifest missing: '+str(p))
    state=json.loads((p/'run_info.json').read_text(encoding='utf-8'))['settings']
    if state['settings_hash']!=opt.settings_hash:
        raise RuntimeError('Source manifest fingerprint changed')
    return p,SimpleNamespace(**state)


def source_data09(out, include_val=False):
    _,opt,out=manifest_for_stage(out)
    x,v,tx,im,pos=train_arrays03(out,opt)
    xf,vf,tf,gf,pf,_,_=partition(x,v,tx,im,pos,opt.fit_pids)
    xh,vh,th,gh,ph,_,_=partition(x,v,tx,im,pos,opt.holdout_pids)
    if set(q['pid'] for q in tf)&set(q['pid'] for q in th):
        raise AssertionError('Fit / heldout TRAIN PIDs overlap')
    neg_path=out/'fit_negative_indices.npy'
    if not neg_path.is_file():raise FileNotFoundError('Run mine stage before evidence/training')
    negatives=np.load(neg_path,allow_pickle=False)
    if negatives.shape!=(len(xf),int(opt.hard_k)) or negatives.dtype.kind not in 'iu':
        raise RuntimeError('Fit negative cache shape invalid')
    gpids=np.array([q['pid'] for q in gf]);tpids=np.array([q['pid'] for q in tf])
    if np.any(negatives<0) or np.any(negatives>=len(vf)) or np.any(gpids[negatives]==tpids[:,None]):
        raise RuntimeError('Illegal negative PID in training cache')
    src=(out,opt,(xf,vf,tf,gf,pf,negatives),(xh,vh,th,gh,ph))
    if not include_val:return src
    val_imgs,val_texts,_=read_annotations(opt.annotation_file,'val')
    vfolder=Path(opt.val_cache_dir) if opt.val_cache_dir else out
    if not (cache_is_ready(vfolder,'text',opt.fp_val,len(val_texts)) and
            cache_is_ready(vfolder,'image',opt.fp_val,len(val_imgs))):
        raise RuntimeError('Best checkpoint Validation features not ready')
    vx=normalize(np.load(vfolder/'text_features.npy',mmap_mode='r'))
    vg=normalize(np.load(vfolder/'image_features.npy',mmap_mode='r'))
    tk=load_val_topk10(out,opt,len(vx),len(vg))
    if set(q['pid'] for q in val_texts) & set(q['pid'] for q in tx):
        raise RuntimeError('Validation/train PID overlap')
    return src,(vx,vg,val_imgs,val_texts,tk)


def stage_mine10(out):
    _,opt,out=manifest_for_stage(out)
    x,v,tx,im,pos=train_arrays03(out,opt)
    xf,vf,tf,gf,pf,_,_=partition(x,v,tx,im,pos,opt.fit_pids)
    xh,vh,th,gh,ph,_,_=partition(x,v,tx,im,pos,opt.holdout_pids)
    for name,xx,vv,tt,ii,n in [('fit_negative_indices',xf,vf,tf,gf,int(opt.hard_k)),
                               ('holdout_negatives',xh,vh,th,gh,int(opt.hard_k)*2)]:
        path=out/(name+'.npy');meta=out/(name+'_meta.json')
        p1=np.array([q['pid'] for q in tt]);p2=np.array([q['pid'] for q in ii])
        available=int(np.min([(p2!=p).sum() for p in np.unique(p1)]))
        n=min(n,available)
        if n<int(opt.hard_k):raise RuntimeError('Insufficient different-PID negatives')
        stamp={'settings_hash':opt.settings_hash,'query_n':len(xx),'gallery_n':len(vv),
               'k':n,'pid_sha256':hashlib.sha256(p1.tobytes()+p2.tobytes()).hexdigest()}
        if path.is_file() and meta.is_file():
            try:
                prev=json.loads(meta.read_text(encoding='utf-8'))
                arr=np.load(path,mmap_mode='r',allow_pickle=False)
                if (prev==stamp and arr.shape==(len(xx),n) and arr.dtype.kind in 'iu' and
                    np.all(arr>=0) and np.all(arr<len(vv)) and np.all(p2[arr]!=p1[:,None])):
                    msg('Validated '+name+' cache: skip mining');continue
            except (OSError,ValueError,KeyError):pass
        arr=hard_negative_mine(xx,vv,p1,p2,n,opt.mine_batch)
        atomic_npy(path,arr);atomic_json(meta,stamp)
        msg('Mined '+name+': '+str(arr.shape))


def load_val_topk10(out,opt,nq,ng):
    path=out/'val_topk_indices.npy';meta=out/'val_topk_meta.json'
    expected={'settings_hash':opt.settings_hash,'n_queries':nq,
              'n_gallery':ng,'topk':min(ng,int(opt.eval_topk))}
    if not path.is_file() or not meta.is_file() or json.loads(meta.read_text(encoding='utf-8'))!=expected:
        raise RuntimeError('TopK missing or incorrect fingerprint')
    a=np.load(path,mmap_mode='r',allow_pickle=False)
    if a.shape!=(nq,expected['topk']) or a.dtype.kind not in 'iu':
        raise RuntimeError('TopK shape mismatch')
    if np.any(a<0) or np.any(a>=ng) or (a.shape[1]>1 and
            np.any(np.diff(np.sort(a,axis=1),axis=1)==0)):
        raise RuntimeError('TopK contains duplicate/out-of-bounds IDs')
    return np.asarray(a,dtype=np.int32)


def stage_topk10(out):
    """Only NumPy BLAS, no torch imported; audit true baseline BEFORE training."""
    _,opt,out=manifest_for_stage(out)
    imgs,txt,_=read_annotations(opt.annotation_file,'val')
    folder=Path(opt.val_cache_dir) if opt.val_cache_dir else out
    x=normalize(np.load(folder/'text_features.npy',mmap_mode='r'))
    v=normalize(np.load(folder/'image_features.npy',mmap_mode='r'))
    if x.shape[0]!=len(txt) or v.shape[0]!=len(imgs):raise RuntimeError('Val cache dimensions wrong')
    m,detail=metrics(x@v.T,[a['pid'] for a in txt],[a['pid'] for a in imgs])
    expected=opt.expected_baseline_r1
    msg('EXP10 NEW BEST original IRRA: '+str(m))
    if expected is not None:
        diffs={key:float(m[key]-ref) for key,ref in EXPECTED_METRICS.items()}
        if any(abs(d)>opt.baseline_tolerance for d in diffs.values()):
            raise RuntimeError('BASELINE REPRODUCTION FAILED: metrics differences (percentage points) '+repr(diffs)+'; STOP before training.')
    atomic_json(out/'new_best_baseline_audit.json',
        {'metrics':m,'expected_log_epoch':58,'expected_R1':expected,
         'expected_checkpoint_SHA256':EXPECTED_SHA,'actual_checkpoint_SHA256':opt.fp_val.get('checkpoint_sha256','SYNTHETIC_TEST'),
         'validation_queries':len(txt),'validation_gallery':len(imgs),
         'validation_identities':len(set(a['pid'] for a in txt)),
         'comparison_tolerance_pp':opt.baseline_tolerance,
         'metric_protocol':'full Val gallery; query PID only in metrics()',
         'frozen_original_checkpoint':opt.checkpoint})
    path=out/'val_topk_indices.npy';meta=out/'val_topk_meta.json'
    stamp={'settings_hash':opt.settings_hash,'n_queries':len(x),
           'n_gallery':len(v),'topk':min(len(v),int(opt.eval_topk))}
    if path.is_file() and meta.is_file():
        try:
            if json.loads(meta.read_text(encoding='utf-8'))==stamp:
                load_val_topk10(out,opt,len(x),len(v))
                msg('Validated fixed topK cache: skip');return
        except (OSError,ValueError,KeyError,RuntimeError):pass
    k=stamp['topk'];out_idx=np.empty((len(x),k),dtype=np.int32)
    for st in range(0,len(x),96):
        logits=x[st:st+96]@v.T
        ind=np.argpartition(-logits,k-1,axis=1)[:,:k]
        order=np.argsort(-np.take_along_axis(logits,ind,axis=1),axis=1,kind='stable')
        out_idx[st:st+len(ind)]=np.take_along_axis(ind,order,axis=1)
    atomic_npy(path,out_idx);atomic_json(meta,stamp)
    msg('Cached original IRRA fixed Top-%d, only best checkpoint.'%k)


def _warmup_step10(net,optimizer,order,x,v,pos,neg,opt,device):
    """Same main-gradient update rule as EXP07 Warmup, no auxiliary branch."""
    import torch
    from torch.nn import functional as F
    net.train();params=list(net.parameters());sums=defaultdict(float)
    for st in range(0,len(order),int(opt.train_batch)):
        sub=order[st:st+int(opt.train_batch)];bs=len(sub)
        t=x[sub].to(device)
        ids=torch.cat((pos[sub,None],neg[sub]),dim=1).to(device)
        z=v[ids]
        score,corr,gate=net(t,z)
        rank=F.softplus((score[:,1:]-score[:,:1]+opt.margin)/opt.temperature).mean()
        protect=corr.square().mean()
        primary=opt.warmup_rank_weight*rank+opt.warmup_protect_weight*protect
        grads=torch.autograd.grad(primary,params,retain_graph=False,allow_unused=True)
        main_g=[torch.zeros_like(p) if h is None else h.detach() for p,h in zip(params,grads)]
        update=[g.clone() for g in main_g]
        optimizer.zero_grad(set_to_none=True)
        for p,g in zip(params,update):p.grad=g
        torch.nn.utils.clip_grad_norm_(params,5.)
        optimizer.step()
        sums['rank_loss']+=float(rank.detach())*bs
        sums['protect_loss']+=float(protect.detach())*bs
        sums['gate_mean']+=float(gate.detach().mean())*bs
    return {k:v/max(1,len(order)) for k,v in sums.items()}


def stage_warmup10(out):
    import torch
    torch.set_num_threads(1)
    _,opt,out=manifest_for_stage(out)
    src,so,fit,hold=source_data09(out)
    _,(valx,valv,im,tx,tk)=source_data09(out,True)
    xf,vf,tf,gf,pf,neg=fit
    device=torch.device('cuda' if opt.device=='cuda' or (opt.device=='auto' and torch.cuda.is_available()) else 'cpu')
    if opt.device=='cuda' and device.type!='cuda':raise RuntimeError('GPU requested but unavailable')
    xt=torch.from_numpy(np.array(xf,copy=True));gv=torch.from_numpy(np.array(vf,copy=True)).to(device)
    ps=torch.from_numpy(np.array(pf,dtype=np.int64));ns=torch.from_numpy(np.array(neg,dtype=np.int64))
    vx=torch.from_numpy(np.array(valx,copy=True));vv=torch.from_numpy(np.array(valv,copy=True)).to(device)
    for seed in opt.seeds:
        stem=f'warmup_s{seed}'
        if _warmup_valid10(out,opt,stem,len(valx),tk.shape[1]):
            msg(stem+': complete and hash checked; skip');continue
        torch.manual_seed(seed)
        if device.type=='cuda':torch.cuda.manual_seed_all(seed)
        net=get_pair09()(xf.shape[1],int(opt.hidden),float(opt.max_delta)).to(device)
        # EXACT EXP07 Warmup initialization. No custom adaptation.
        torch.nn.init.zeros_(net.delta.weight);torch.nn.init.zeros_(net.delta.bias)
        torch.nn.init.zeros_(net.gate.weight);torch.nn.init.constant_(net.gate.bias,-.5)
        optimizer=torch.optim.AdamW(net.parameters(),lr=float(opt.warmup_lr),weight_decay=1e-4)
        generator=torch.Generator().manual_seed(seed)
        hist=[]
        for epoch in range(1,int(opt.warmup_epochs)+1):
            order=torch.randperm(len(xf),generator=generator)
            stats=_warmup_step10(net,optimizer,order,xt,gv,ps,ns,opt,device)
            hist.append({'method':'warmup','seed':seed,'epoch':epoch,'phase':'warmup',**stats})
            msg('%s epoch %d/%d: rank %.5f gate %.4f'%(stem,epoch,opt.warmup_epochs,stats['rank_loss'],stats['gate_mean']))
        scores=prediction09(net,vx,vv,tk,device,opt.eval_query_batch)
        # Gate file written for compatibility and reproducibility checks.
        gates=np.empty_like(scores)
        net.eval()
        with torch.inference_mode():
            batch=max(1,min(opt.eval_query_batch,8192//max(1,tk.shape[1])))
            for st in range(0,len(vx),batch):
                ed=min(len(vx),st+batch)
                ids=torch.as_tensor(np.array(tk[st:ed],dtype=np.int64,copy=True),device=device)
                _,_,g=net(vx[st:ed].to(device),vv[ids])
                gates[st:ed]=g.float().cpu().numpy()
        path=out/(stem+'_student.pt');tmp=out/(stem+'_student.partial.pt')
        torch.save({'model_state_dict':net.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'settings_hash':opt.settings_hash,'seed':seed,'method':'warmup',
                    'epoch':int(opt.warmup_epochs),'source_checkpoint_sha256':opt.fp_val.get('checkpoint_sha256','SYNTHETIC_TEST')},tmp)
        _replace_with_retries(tmp,path)
        atomic_npy(out/(stem+'_topk_score.npy'),scores)
        atomic_npy(out/(stem+'_topk_gate.npy'),gates)
        atomic_json(out/(stem+'_training_meta.json'),
           {'complete':True,'settings_hash':opt.settings_hash,
            'checkpoint_sha256':sha256_small_file(path),'history':hist,
            'source_checkpoint_sha256':opt.fp_val.get('checkpoint_sha256','SYNTHETIC_TEST')})
        msg('Best model Warmup saved: '+stem)
        del net,optimizer
        if device.type=='cuda':torch.cuda.empty_cache()


def _warmup_valid10(out,opt,stem,n,k):
    try:
        validate_warmup09(out,opt,int(stem.split('s')[-1]),n,k)
        meta=json.loads((out/(stem+'_training_meta.json')).read_text(encoding='utf-8'))
        return meta['source_checkpoint_sha256']==opt.fp_val.get('checkpoint_sha256','SYNTHETIC_TEST')
    except (FileNotFoundError,RuntimeError,ValueError,KeyError):return False


def audit_sources10(out):
    _,opt,out=manifest_for_stage(out)
    if not (out/'new_best_baseline_audit.json').is_file():raise RuntimeError('Baseline audit absent')
    m=json.loads((out/'new_best_baseline_audit.json').read_text(encoding='utf-8'))
    if abs(m['metrics']['R1']-opt.expected_baseline_r1)>opt.baseline_tolerance:
        raise RuntimeError('Baseline R1 audit not passed')
    if m['actual_checkpoint_SHA256']!=EXPECTED_SHA and opt.expected_baseline_r1 is not None:
        raise RuntimeError('Wrong best checkpoint provenance')


def worker10(stage,out):
    out=Path(out)
    if stage=='encode':stage_encode03(out)
    elif stage=='mine':stage_mine10(out)
    elif stage=='topk':stage_topk10(out)
    elif stage=='prepare':stage_prepare09(out)
    elif stage=='warmup':stage_warmup10(out)
    elif stage=='train':train09(out)
    elif stage=='select':selection09(out)
    elif stage=='eval':eval09(out)
    else:raise ValueError('Invalid EXP10 worker stage: '+stage)


def main10():
    global LOG_FILE
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root',default=DEFAULT_ROOT)
    p.add_argument('--config-file',default=DEFAULT_CONFIG)
    p.add_argument('--checkpoint',default=DEFAULT_CKPT)
    p.add_argument('--dataset-root',default='')
    p.add_argument('--annotation-file',default='')
    p.add_argument('--images-root',default='')
    p.add_argument('--val-cache-dir',default='')
    p.add_argument('--train-text-cache-dir',default='')
    p.add_argument('--train-image-cache-dir',default='')
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    p.add_argument('--max-train-pids',type=int,default=2200)
    p.add_argument('--subset-seed',type=int,default=42)
    p.add_argument('--fit-fraction',type=float,default=.8)
    p.add_argument('--split-seed',type=int,default=776)
    p.add_argument('--seeds',default='42,43,44')
    p.add_argument('--warmup-epochs',type=int,default=2)
    p.add_argument('--epochs',type=int,default=1,help='EXP09 no_support/selected_support continuation epochs')
    p.add_argument('--warmup-lr',type=float,default=3e-4)
    p.add_argument('--lr',type=float,default=5e-5)
    p.add_argument('--warmup-rank-weight',type=float,default=.2)
    p.add_argument('--warmup-protect-weight',type=float,default=.5)
    p.add_argument('--rank-weight',type=float,default=.08)
    p.add_argument('--preserve-weight',type=float,default=20.)
    p.add_argument('--distill-weight',type=float,default=1.)
    p.add_argument('--teacher-delta-cap',type=float,default=.02)
    p.add_argument('--margin',type=float,default=.05)
    p.add_argument('--temperature',type=float,default=.07)
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--max-delta',type=float,default=.08)
    p.add_argument('--train-batch',type=int,default=128)
    p.add_argument('--text-batch',type=int,default=64)
    p.add_argument('--image-batch',type=int,default=8)
    p.add_argument('--hard-k',type=int,default=4)
    p.add_argument('--mine-batch',type=int,default=96)
    p.add_argument('--max-support',type=int,default=8)
    p.add_argument('--teacher-strength',type=float,default=.35)
    p.add_argument('--help-threshold',type=float,default=.005)
    p.add_argument('--harm-threshold',type=float,default=.005)
    p.add_argument('--instance-floor',type=float,default=.015)
    p.add_argument('--use-threshold',type=float,default=.5)
    p.add_argument('--reject-threshold',type=float,default=.5)
    p.add_argument('--prob-margin',type=float,default=.1)
    p.add_argument('--selector-seed',type=int,default=909)
    p.add_argument('--eval-topk',type=int,default=100)
    p.add_argument('--eval-query-batch',type=int,default=64)
    p.add_argument('--baseline-tolerance',type=float,default=.02)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--_worker',default='',help=argparse.SUPPRESS)
    p.add_argument('--_run-dir',default='',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a._worker:
        try:worker10(a._worker,a._run_dir)
        except Exception:traceback.print_exc();sys.exit(1)
        return
    if a.self_test:self_test10();return
    if not (a.max_train_pids>=0 and .5<=a.fit_fraction<=.95 and
            a.warmup_epochs>=1 and a.epochs>=1 and a.hidden>=1 and a.hard_k>=1 and
            a.train_batch>=1 and a.image_batch>=1 and a.text_batch>=1 and
            a.eval_topk>=1 and a.mine_batch>=1 and a.max_support>=1 and
            a.warmup_lr>0 and a.lr>0 and a.preserve_weight>0 and a.distill_weight>0 and
            a.teacher_delta_cap>0 and a.max_delta>0 and a.baseline_tolerance>0):
        p.error('Invalid hyperparameters')
    root=Path(a.project_root).resolve();cfg=Path(a.config_file).resolve();ck=Path(a.checkpoint).resolve()
    for file in (root/'model'/'build.py',cfg,ck):
        if not file.is_file():p.error('Missing IRRA file: '+str(file))
    if sha256_small_file(ck)!=EXPECTED_SHA:
        p.error('Checkpoint SHA mismatch! Refuse old/different model: '+str(ck))
    if sha256_small_file(cfg)!=EXPECTED_CONFIG_SHA:
        p.error('Configs.yaml differs from validated best model configuration')
    _,ann,imgs,data_source=resolve_dataset(root,cfg,a.dataset_root,a.annotation_file,a.images_root)
    if sha256_small_file(ann)!=EXPECTED_ANNOTATION_SHA:
        p.error('Annotation sha256 differs from verified best-baseline protocol')
    it,tt,_=read_annotations(ann,'train');vi,vt,_=read_annotations(ann,'val')
    if len(vt)!=6158 or len(vi)!=3078:
        p.error('Best model known Validation gallery/query sizes do not match: '+str((len(vi),len(vt))))
    ids=sorted({int(r['pid']) for r in tt})
    if set(ids)&{int(r['pid']) for r in vt}:p.error('Train/Val PID overlap')
    selected=(ids if a.max_train_pids==0 or a.max_train_pids>=len(ids)
      else sorted(map(int,np.random.default_rng(a.subset_seed).choice(ids,a.max_train_pids,replace=False))))
    fit,hold=pid_split(selected,a.fit_fraction,a.split_seed)
    verify_image_paths(imgs,[r for r in it if r['pid'] in set(selected)])
    fp_train=fingerprint(root,cfg,ck,ann,'train');fp_val=fingerprint(root,cfg,ck,ann,'val')
    folder=root/PROJECT_FOLDER_NAME
    valdir=source_val_cache(root,folder,a.val_cache_dir,fp_val,len(vt),len(vi))
    txtdir=source_train_text_cache(root,folder,a.train_text_cache_dir,fp_train,len(tt))
    indexes=[j for j,r in enumerate(it) if r['pid'] in set(selected)]
    imgdir=source_train_image_cache(folder,a.train_image_cache_dir,
                SimpleNamespace(fp_train=fp_train,selected_pids=selected),indexes)
    try:seeds=[int(s.strip()) for s in a.seeds.split(',') if s.strip()]
    except ValueError:p.error('Invalid seeds')
    if not seeds or len(set(seeds))!=len(seeds) or min(seeds)<0:p.error('Seeds must be unique >=0')
    settings={'project_root':str(root),'config_file':str(cfg),'checkpoint':str(ck),
       'annotation_file':str(ann),'images_dir':str(imgs),'dataset_source':data_source,
       'val_cache_dir':str(valdir or ''),'train_text_cache_dir':str(txtdir or ''),
       'train_image_cache_dir':str(imgdir or ''),'selected_pids':selected,
       'fit_pids':fit,'holdout_pids':hold,'fp_train':fp_train,'fp_val':fp_val,
       'device':a.device,'seeds':seeds,'epochs':a.epochs,'warmup_epochs':a.warmup_epochs,
       'warmup_lr':a.warmup_lr,'lr':a.lr,'warmup_rank_weight':a.warmup_rank_weight,
       'warmup_protect_weight':a.warmup_protect_weight,'rank_weight':a.rank_weight,
       'preserve_weight':a.preserve_weight,'distill_weight':a.distill_weight,
       'teacher_delta_cap':a.teacher_delta_cap,'margin':a.margin,
       'temperature':a.temperature,'hidden':a.hidden,'max_delta':a.max_delta,
       'train_batch':a.train_batch,'text_batch':a.text_batch,'image_batch':a.image_batch,
       'hard_k':a.hard_k,'mine_batch':a.mine_batch,'max_support':a.max_support,
       'teacher_strength':a.teacher_strength,'help_threshold':a.help_threshold,
       'harm_threshold':a.harm_threshold,'instance_floor':a.instance_floor,
       'use_threshold':a.use_threshold,'reject_threshold':a.reject_threshold,
       'prob_margin':a.prob_margin,'selector_seed':a.selector_seed,
       'eval_topk':a.eval_topk,'eval_query_batch':a.eval_query_batch,
       'expected_baseline_r1':EXPECTED_R1,'baseline_tolerance':a.baseline_tolerance,
       'version':VERSION09}
    settings['settings_hash']=hashlib.sha256(json.dumps(settings,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    out,resumed=new_or_resume(folder,a,settings)
    settings['run_dir']=str(out)
    # Keep settings_hash identical for resumability: run_dir is excluded from the hash.
    LOG_FILE=out/'experiment_run.log'
    pid_file=out/'selected_train_pids.json'
    if resumed and pid_file.is_file() and json.loads(pid_file.read_text(encoding='utf-8'))!=selected:
        raise RuntimeError('Resume PID selection changed; refuse reuse')
    if not pid_file.is_file():atomic_json(pid_file,selected)
    if not resumed:atomic_json(out/'run_info.json',{'experiment':EXPERIMENT_NAME,'settings':settings})
    else:
        original=json.loads((out/'run_info.json').read_text(encoding='utf-8'))['settings']
        if original['run_dir']!=str(out):raise RuntimeError('Resume run dir tampering')
    shutil.copy2(Path(__file__).resolve(),out/Path(__file__).name)
    (out/'实验10_目标与边界.md').write_text(
       '# 实验10：最佳模型与最小文本互补复验\n\n'
       '主基线为Epoch58权重的原始IRRA，先严格复现66.4339066%。\n'
       '新权重重新建立warmup、no_support、selected_support，旧权重缓存不得复用。\n'
       '原始IRRA不训练不修改、不覆盖best.pth，所有优化只针对外置reranker。\n'
       'Train 2200PID中80%机制学习/20%未见诊断；Val PID只用于指标。\n'
       '仅保留已约定的3模型及6个推理系数；不新增其他分支。\n'
       '原版Warmup alpha1始终单列，调整alpha不等于新增多文本增益。\n'
       '不依靠Val选择最佳alpha；所有Val alpha扫描仅作诊断。\n',encoding='utf-8')
    try:
        for stage in ('encode','mine','topk','prepare','warmup','train','select','eval'):
            run_subprocess(stage,out,root)
        msg('EXP10 complete: trained only warmup, no_support, selected_support on BEST IRRA.')
    finally:package09(out)


def self_test10():
    """Synthetic complete validation of 3 model types, isolated subprocesses, resume and ZIP."""
    p=Path(__file__).resolve()
    m=sys.modules[__name__]
    rng=np.random.default_rng(20261021)
    with tempfile.TemporaryDirectory() as td:
     out=Path(td);ann=out/'synthetic.json';data=[];x=[];v=[];xval=[];vval=[];dim=24
     for pid in range(1,19):
      center=rng.standard_normal(dim);center/=np.linalg.norm(center)
      for view in range(2):
       data.append({'id':pid,'split':'train','file_path':f'{pid}_{view}.png','captions':[f'p{pid} query {view} a',f'p{pid} query {view} b']})
       v.append(m.normalize(center+.3*rng.standard_normal(dim)))
       for c in range(2):x.append(m.normalize(center+.4*rng.standard_normal(dim)))
     for pid in range(101,107):
      center=rng.standard_normal(dim);center/=np.linalg.norm(center)
      for view in range(2):
       data.append({'id':pid,'split':'val','file_path':f'{pid}_{view}.png','captions':['query a','query b']})
       vval.append(m.normalize(center+.3*rng.standard_normal(dim)))
       for c in range(2):xval.append(m.normalize(center+.4*rng.standard_normal(dim)))
     m.atomic_json(ann,data)
     selected=list(range(1,19));fit,hold=m.pid_split(selected,.78,776)
     settings=dict(run_dir=str(out), settings_hash='exp10_fixture',annotation_file=str(ann),selected_pids=selected,
      fit_pids=fit,holdout_pids=hold,fp_train={'fake':'train'},fp_val={'fake':'val'},
      val_cache_dir='',train_text_cache_dir='',train_image_cache_dir='',seeds=[42,43],
      device='cpu',hard_k=2,mine_batch=32,eval_topk=5,train_batch=16,hidden=16,max_delta=.08,
      warmup_lr=.0003,warmup_epochs=2,warmup_rank_weight=.2,warmup_protect_weight=.5,
      margin=.05,temperature=.07,epochs=1,lr=5e-5,rank_weight=.08,preserve_weight=20.,
      distill_weight=1.,teacher_delta_cap=.02,eval_query_batch=8,teacher_strength=.35,
      max_support=4,help_threshold=.002,harm_threshold=.002,instance_floor=.01,
      use_threshold=.5,reject_threshold=.5,prob_margin=.1,selector_seed=909,
      expected_baseline_r1=None,baseline_tolerance=.02,checkpoint='fake_checkpoint',
      project_root=str(out),config_file='fake_config',images_dir=str(out))
     m.atomic_json(out/'run_info.json',{'settings':settings});m.atomic_json(out/'selected_train_pids.json',selected)
     x=np.array(x,dtype=np.float32);v=np.array(v,dtype=np.float32)
     xval=np.array(xval,dtype=np.float32);vval=np.array(vval,dtype=np.float32)
     for name,a,fp in [('train_text',x,settings['fp_train']),('train_image',v,m.train_img_fingerprint(m.SimpleNamespace(fp_train=settings['fp_train']),list(range(len(v))))),('text',xval,settings['fp_val']),('image',vval,settings['fp_val'])]:
      m.atomic_npy(out/(name+'_features.npy'),a)
      m.atomic_json(out/(name+'_cache_meta.json'),{'fingerprint':fp,'done':len(a),'shape':list(a.shape)})
     env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
     for stage in ('mine','topk','prepare','warmup','train','select','eval'):
      r=subprocess.run([sys.executable,str(p),'--_worker',stage,'--_run-dir',str(out)],capture_output=True,text=True,env=env,timeout=150)
      print('STAGE',stage,'CODE',r.returncode,'TAIL',r.stdout[-1500:],'ERR',r.stderr[-3000:])
      if r.returncode:raise RuntimeError(stage)
     orig=m.sha256_small_file(out/'warmup_s42_student.pt')
     for stage in ('mine','topk','prepare','warmup','train'):
      r=subprocess.run([sys.executable,str(p),'--_worker',stage,'--_run-dir',str(out)],capture_output=True,text=True,env=env,timeout=150)
      print('RESUME',stage,r.returncode)
      if r.returncode:raise RuntimeError('resume '+stage+r.stderr)
     assert m.sha256_small_file(out/'warmup_s42_student.pt')==orig
     m.package09(out)
     with zipfile.ZipFile(out/'发送给ChatGPT.zip') as z:
      print('ZIP',len(z.namelist()),z.namelist()[:16])
      assert 'aggregate_summary.csv' in z.namelist()
      assert not any(n.endswith('.pt') or n.endswith('.npy') for n in z.namelist())
     print('OK synthetic end to end')

if __name__=='__main__':main10()
