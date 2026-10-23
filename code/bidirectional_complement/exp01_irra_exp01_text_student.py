# -*- coding: utf-8 -*-
"""
实验01：同身份多描述蒸馏到单文本学生（原始 IRRA 冻结）
Experiment 01: Text-only cross-description complementary distillation.

核心评估原则：
  * 训练集 PID 仅用于构造同身份描述教师；验证集 PID 只用于统计指标。
  * 验证时每条查询单独编码；完全不使用验证集支持文本或 PID。
  * 原始 IRRA 图像和文本编码器保持冻结，只训练小型 512D 文本残差适配器。
  * 所有模型用同一个网络、损失、训练轮数；比较 same-mean、cross-mean、cross-gate。
  * 这是检验“互补可迁移性”的实验，不承诺检索精度必然提升。

默认工程：G:/PyCharmProjects/IRRA/IRRA-main
默认输出：IRRA-main\图文双向互补实验\实验01_单文本互补蒸馏_时间戳
运行：在 PyCharm 中直接运行，或 python irra_exp01_text_student.py --resume
自测：python irra_exp01_text_student.py --self-test

为避免 Windows OpenMP 重复加载，特征提取 / 教师构造 / PyTorch训练 /
NumPy 排名评估在独立子进程完成。禁止以 KMP_DUPLICATE_LIB_OK 掩盖冲突。
只依赖标准库、numpy、torch、torchvision、PyYAML 和现有 IRRA 工程。
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
EXPERIMENT_NAME = '实验01_单文本互补蒸馏'
METHODS = ('same_mean','cross_mean','cross_gate')
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

def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with open(temp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(temp, path)

def write_csv(path, rows, cols):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with open(temp, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    os.replace(temp, path)

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


def find_val_cache(root, project_folder, explicit, valfp, nt, ni):
    if explicit:
        options=[Path(explicit).resolve()]
    else:
        options=sorted((p for p in project_folder.iterdir() if p.is_dir()
                        and p.name.startswith('实验00_')),
                       key=lambda p:p.name,reverse=True) if project_folder.exists() else []
    for folder in options:
        if (cache_is_ready(folder,'text',valfp,nt) and
                cache_is_ready(folder,'image',valfp,ni)):
            msg('Found compatible, completed EXP00 validation cache: '+str(folder))
            return folder
    if explicit:
        raise ValueError('The specified --val-cache-dir was not a complete, fingerprint-matching '
                         'IRRA feature cache (same checkpoint/config/annotations/split required).')
    msg('No compatible EXP00 cache. Validation features will be encoded inside experiment01.')
    return None


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
        raise FileNotFoundError('No matching experiment01 run to resume. Remove --resume for a new run.')
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
    """Always generate a small, portable bundle including failure logs."""
    out=Path(out)
    zip_path=out/'发送给ChatGPT.zip'
    # Exclude raw embeddings, checkpoints, JSON annotations and giant training matrices.
    allow_ext={'.csv','.json','.txt','.log','.md','.py','.png'}
    files=[p for p in out.iterdir() if p.is_file() and p.suffix.lower() in allow_ext]
    temp=zip_path.with_name(zip_path.name+'.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):
            z.write(p,p.name)
    os.replace(temp,zip_path)
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


def stage_encode(out):
    info,opt,out=manifest_for_stage(out)
    root=Path(opt.project_root)
    ann=Path(opt.annotation_file)
    imroot=Path(opt.images_dir)
    train_images,train_texts,ntrain=read_annotations(ann,'train')
    val_images,val_texts,_=read_annotations(ann,'val')
    import torch
    device=torch.device('cuda' if opt.device=='cuda' or
                        (opt.device=='auto' and torch.cuda.is_available()) else 'cpu')
    if opt.device=='cuda' and device.type!='cuda':
        raise RuntimeError('CUDA explicitly required, but not available')
    valcache=Path(opt.val_cache_dir) if opt.val_cache_dir else out
    ready_train=cache_is_ready(out,'train_text',opt.fp_train,len(train_texts))
    ready_val=(cache_is_ready(valcache,'text',opt.fp_val,len(val_texts)) and
               cache_is_ready(valcache,'image',opt.fp_val,len(val_images)))
    if ready_train and ready_val:
        msg('Both train text and val image/text caches already complete; skip model load.')
        return
    if not ready_train:
        model,irra_args,dim=load_irra_model(root,Path(opt.config_file),
                                           Path(opt.checkpoint),ntrain,device)
        encode_text_only(model,device,train_texts,out,opt.fp_train,opt.text_batch,dim)
        del model
        if device.type=='cuda':
            torch.cuda.empty_cache()
    if not ready_val:
        optval=SimpleNamespace(text_batch=opt.text_batch,image_batch=opt.image_batch,workers=0)
        encode_and_cache(root,Path(opt.config_file),Path(opt.checkpoint),ann,
                         imroot,val_images,val_texts,ntrain,opt.fp_val,out,optval,device)
    msg('Encoding complete. All required cache files available.')


def capped_peer_indices(candidates, x, feats, cap):
    """Fixed deterministic support subsampling for large IDs; no evaluation IDs."""
    if len(candidates)<=cap:
        return candidates
    similarities=feats[candidates]@x
    ordered=np.argsort(similarities,kind='stable')
    pos=np.unique(np.linspace(0,len(candidates)-1,cap,dtype=int))
    return [candidates[int(ordered[k])] for k in pos]


def build_targets(train_texts,features,method,strength,max_support,out):
    """Teacher is a TRAIN-only label-assisted upper view; val not used here."""
    n,d=features.shape
    pids=np.array([t['pid'] for t in train_texts],dtype=np.int64)
    images=np.array([t['img_idx'] for t in train_texts],dtype=np.int64)
    by_pid=defaultdict(list);by_img=defaultdict(list)
    for i,(p,img) in enumerate(zip(pids,images)):
        by_pid[p].append(i)
        by_img[img].append(i)
    target=np.lib.format.open_memmap(out/(method+'_targets.npy'),mode='w+',dtype='float32',shape=(n,d))
    examples=[]
    coverage=0
    sum_cos=0.
    sum_move=0.
    pos_example = set(np.linspace(0,n-1,80,dtype=int).tolist())
    for i in range(n):
        x=features[i]
        if method=='same_mean':
            options=[k for k in by_img[images[i]] if k!=i]
        else:
            options=[k for k in by_pid[pids[i]] if images[k]!=images[i]]
        options=capped_peer_indices(options,x,features,max_support)
        if not options:
            target[i]=x
            continue
        coverage+=1
        peer=features[options]
        c=np.clip(peer@x,-1.,1.)
        if method in ('cross_mean','same_mean'):
            pool=peer.mean(axis=0)
            target[i]=normalize((1-strength)*x+strength*pool)
            reliability=np.ones(len(options),dtype=float)
        else:
            # Novelty x consistency: ignore near-duplicates and strongly contradictory peers.
            # No learned thresholds or val-driven choice: this is a *hypothesis* for experiment01.
            trust=np.clip((c-0.20)/0.50,0,1)
            novelty=np.clip((0.95-c)/0.40,0,1)
            reliability=trust*novelty
            if reliability.sum()<1e-8:
                target[i]=x
            else:
                w=reliability/reliability.sum()
                direction=(w[:,None]*(peer-c[:,None]*x[None,:])).sum(axis=0)
                target[i]=normalize(x+strength*direction)
        new=target[i]
        sim=float(x@new)
        sum_cos+=sim
        sum_move+=float(np.linalg.norm(new-x))
        if i in pos_example:
            examples.append({'method':method,'index':i,'train_pid':int(pids[i]),
                             'anchor_text':train_texts[i]['text'],
                             'support_count':len(options),
                             'supports':' || '.join(train_texts[k]['text'] for k in options),
                             'anchor_support_cosines':json.dumps([round(float(v),4) for v in c]),
                             'support_reliability':json.dumps([round(float(v),4) for v in reliability]),
                             'teacher_cos_to_anchor':round(sim,6)})
        if (i+1)%5000==0:
            msg(f'{method}: generated {i+1}/{n}')
    target.flush()
    return {'method':method,'num_texts':n,'with_support':coverage,'support_rate':coverage/n,
            'mean_teacher_input_cos':float(sum_cos/max(coverage,1)),
            'mean_teacher_delta_l2':float(sum_move/max(coverage,1))},examples


def stage_teacher(out):
    info,opt,out=manifest_for_stage(out)
    _,texts,_=read_annotations(Path(opt.annotation_file),'train')
    inp=np.load(out/'train_text_features.npy',mmap_mode='r')
    if len(texts)!=len(inp):
        raise ValueError('Train text features mismatched with annotation ordering')
    csvrows=[]
    summaries=[]
    for method in METHODS:
        targetfile=out/(method+'_targets.npy')
        marker=out/(method+'_targets_meta.json')
        if targetfile.is_file() and marker.is_file():
            m=json.loads(marker.read_text(encoding='utf-8'))
            if m.get('fp')==opt.fp_train and m.get('strength')==opt.strength and m.get('max_support')==opt.max_support:
                ar=np.load(targetfile,mmap_mode='r')
                if ar.shape==inp.shape and ar.dtype==np.float32 and m.get('status')=='complete':
                    msg(method+': teacher target cache already complete.')
                    summaries.append(m['summary'])
                    continue
        summary,examples=build_targets(texts,inp,method,opt.strength,opt.max_support,out)
        summaries.append(summary);csvrows.extend(examples)
        atomic_json(marker,{'fp':opt.fp_train,'strength':opt.strength,
                            'max_support':opt.max_support,'status':'complete','summary':summary})
    if csvrows:
        write_csv(out/'teacher_examples.csv',csvrows,list(csvrows[0]))
    atomic_json(out/'teacher_diagnostics.json',{'train_only':True,'strategies':summaries})
    msg('All teachers generated, entirely from train PIDs; no val labels accessed.')


# Residual adapter is created on demand to prevent importing torch in NumPy workers.
def get_adapter_class():
    import torch
    from torch import nn
    import torch.nn.functional as F
    class TextAdapter(nn.Module):
        def __init__(self,dim,hidden,gain):
            super().__init__()
            self.fc1=nn.Linear(dim,hidden)
            self.fc2=nn.Linear(hidden,dim)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
            self.gain=float(gain)
        def forward(self,x):
            residual=self.fc2(F.gelu(self.fc1(x)))
            return F.normalize(x+self.gain*residual,dim=-1)
    return TextAdapter


def stage_train(out):
    info,opt,out=manifest_for_stage(out)
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    device=torch.device('cuda' if opt.device=='cuda' or
                        (opt.device=='auto' and torch.cuda.is_available()) else 'cpu')
    if opt.device=='cuda' and device.type!='cuda':
        raise RuntimeError('CUDA explicitly required, but unavailable')
    train=np.load(out/'train_text_features.npy',mmap_mode='r')
    valdir=Path(opt.val_cache_dir) if opt.val_cache_dir else out
    val=np.load(valdir/'text_features.npy',mmap_mode='r')
    n,d=train.shape
    train_x=torch.from_numpy(np.array(train,copy=True))
    val_x=torch.from_numpy(np.array(val,copy=True))
    del train
    adapter_class=get_adapter_class()
    fieldnames=['method','epoch','train_distill_loss','train_protect_loss','train_total_loss',
                'mean_student_input_cos','num_train_queries','training_seed']
    history=[]
    for meth in METHODS:
        path=out/(meth+'_student.pt')
        prediction=out/(meth+'_val_features.npy')
        completed=out/(meth+'_training_meta.json')
        if completed.is_file() and path.is_file() and prediction.is_file():
            status=json.loads(completed.read_text(encoding='utf-8'))
            if (status.get('settings_hash')==opt.settings_hash and
                    status.get('status')=='complete' and
                    tuple(np.load(prediction,mmap_mode='r').shape)==tuple(val_x.shape)):
                msg(meth+': matching trained model + val features found. Skip.')
                history.extend(status.get('history',[]))
                continue
        torch.manual_seed(opt.seed)
        if device.type=='cuda':torch.cuda.manual_seed_all(opt.seed)
        adapter=adapter_class(d,opt.hidden,opt.gain).to(device)
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=opt.lr,weight_decay=1e-4)
        tgt=torch.from_numpy(np.array(np.load(out/(meth+'_targets.npy'),mmap_mode='r'),copy=True))
        g=torch.Generator().manual_seed(opt.seed)
        hist=[]
        msg(f'Training {meth}: n={n}, dim={d}, epochs={opt.epochs}, device={device}')
        for epoch in range(1,opt.epochs+1):
            adapter.train()
            perm=torch.randperm(n,generator=g)
            running=np.zeros(3,dtype=np.float64)
            number=0
            for j in range(0,n,opt.train_batch):
                inds=perm[j:j+opt.train_batch]
                a=train_x[inds].to(device)
                t=tgt[inds].to(device)
                z=adapter(a)
                l_teacher=(1-(z*t).sum(dim=1)).mean()
                l_protect=(1-(z*a).sum(dim=1)).mean()
                loss=l_teacher+opt.protect_weight*l_protect
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(),5.)
                optimizer.step()
                batch=len(inds)
                number+=batch
                running+=np.array([l_teacher.item(),l_protect.item(),loss.item()])*batch
            row={'method':meth,'epoch':epoch,'train_distill_loss':float(running[0]/number),
                 'train_protect_loss':float(running[1]/number),
                 'train_total_loss':float(running[2]/number),
                 'num_train_queries':n,'training_seed':opt.seed,
                 'mean_student_input_cos':None}
            hist.append(row)
            msg(f'{meth} epoch {epoch}/{opt.epochs}: teacher={row["train_distill_loss"]:.6f}; '
                f'protect={row["train_protect_loss"]:.6f}; total={row["train_total_loss"]:.6f}')
        adapter.eval()
        predictions=np.lib.format.open_memmap(prediction,mode='w+',dtype='float32',shape=tuple(val_x.shape))
        total_cos=0.
        with torch.inference_mode():
            for j in range(0,len(val_x),opt.train_batch):
                input_x=val_x[j:j+opt.train_batch].to(device)
                v=adapter(input_x)
                predictions[j:j+len(v)]=v.cpu().numpy()
                total_cos+=float((v*input_x).sum().item())
        predictions.flush()
        hist[-1]['mean_student_input_cos']=total_cos/len(val_x)
        tmp=path.with_suffix('.tmp')
        torch.save({'state_dict':adapter.cpu().state_dict(),
                    'dim':d,'hidden':opt.hidden,'gain':opt.gain,'method':meth,
                    'seed':opt.seed,'epoch':opt.epochs,
                    'settings_hash':opt.settings_hash},tmp)
        os.replace(tmp,path)
        atomic_json(completed,{'status':'complete','settings_hash':opt.settings_hash,
                               'history':hist,'validation_features':str(prediction),
                               'train_texts':n,'val_texts':len(val_x)})
        history.extend(hist)
        msg(f'{meth}: validation features exported; average cosine to IRRA={total_cos/len(val_x):.6f}')
        del adapter,tgt
        if device.type=='cuda': torch.cuda.empty_cache()
    write_csv(out/'training_history.csv',history,fieldnames)


def stage_eval(out):
    info,opt,out=manifest_for_stage(out)
    ann=Path(opt.annotation_file)
    imgs,txt,_=read_annotations(ann,'val')
    gallery_pid=np.asarray([z['pid'] for z in imgs])
    query_pid=np.asarray([z['pid'] for z in txt])
    train_pids={z['pid'] for z in read_annotations(ann,'train')[1]}
    overlap=train_pids.intersection(query_pid.tolist())
    if overlap:
        raise AssertionError(f'Data leak: train/val identities overlap! e.g. {list(overlap)[:5]}')
    valdir=Path(opt.val_cache_dir) if opt.val_cache_dir else out
    valtxt=normalize(np.load(valdir/'text_features.npy',mmap_mode='r'))
    valimg=normalize(np.load(valdir/'image_features.npy',mmap_mode='r'))
    if valtxt.shape!=(len(txt),valimg.shape[1]) or valimg.shape[0]!=len(imgs):
        raise ValueError('Val features inconsistent with annotations')
    base,bc=metrics(valtxt@valimg.T,query_pid,gallery_pid)
    # Hard guard against accidentally changing preprocessing/evaluator vs exp00.
    if opt.val_cache_dir and opt.expected_baseline_r1 is not None:
        if abs(base['R1']-opt.expected_baseline_r1)>0.011:
            raise RuntimeError('Baseline changed unexpectedly vs EXP00. Stop before reporting new model.')
    summaries=[]
    changes=[]
    query_rows=[]
    margin_baseline=bc['margin'];base_right=bc['rank']==1
    for meth in ('baseline',)+METHODS:
        features=valtxt if meth=='baseline' else normalize(np.load(out/(meth+'_val_features.npy'),mmap_mode='r'))
        if features.shape!=valtxt.shape:
            raise ValueError('Student val feature shape invalid: '+meth)
        valcos=np.sum(features*valtxt,axis=1)
        sim=features@valimg.T
        scores,cases=metrics(sim,query_pid,gallery_pid)
        correct=cases['rank']==1
        repaired=int((~base_right&correct).sum()) if meth!='baseline' else 0
        harmed=int((base_right&~correct).sum()) if meth!='baseline' else 0
        record={'method':meth,'kind':'valid_single_text_no_val_PID_support',
                'train_supervision':'none' if meth=='baseline' else meth,
                **scores,'delta_R1':scores['R1']-base['R1'],
                'delta_R5':scores['R5']-base['R5'],
                'delta_R10':scores['R10']-base['R10'],
                'delta_mAP':scores['mAP']-base['mAP'],
                'delta_mINP':scores['mINP']-base['mINP'],
                'recovered':repaired,'new_errors':harmed,
                'margin_improved_rate':float(np.mean(cases['margin']>margin_baseline+1e-7)),
                'mean_margin_delta':float(np.mean(cases['margin']-margin_baseline)),
                'mean_feature_cos_to_original':float(np.mean(valcos))}
        summaries.append(record)
        msg(f'{meth}: R1={scores["R1"]:.4f}, mAP={scores["mAP"]:.4f}; '
            f'R1 Δ={record["delta_R1"]:+.4f}; repaired={repaired}, harmed={harmed}')
        for i in range(len(txt)):
            r={'method':meth,'query_idx':i,'pid':int(query_pid[i]),
               'text':txt[i]['text'],'original_rank':int(bc['rank'][i]),
               'new_rank':int(cases['rank'][i]),
               'original_margin':float(bc['margin'][i]),'new_margin':float(cases['margin'][i]),
               'delta_margin':float(cases['margin'][i]-bc['margin'][i]),
               'original_R1_correct':int(base_right[i]),'new_R1_correct':int(correct[i]),
               'feature_cos_to_original':float(valcos[i]),
               'positive_score':float(cases['positive_score'][i]),
               'hard_negative_score':float(cases['negative_score'][i]),
               'top1_gallery_index':int(cases['top1_image'][i])}
            query_rows.append(r)
            if meth!='baseline' and base_right[i]!=correct[i]:
                changes.append({**r,'change_type':'recovered' if correct[i] else 'new_error',
                                'top1_pid':int(gallery_pid[cases['top1_image'][i]])})
    write_csv(out/'method_summary.csv',summaries,list(summaries[0]))
    write_csv(out/'query_deltas.csv',query_rows,list(query_rows[0]))
    if changes:
        write_csv(out/'case_changes.csv',changes,list(changes[0]))
    else:
        write_csv(out/'case_changes.csv',[],list(query_rows[0])+['change_type','top1_pid'])
    teacher=json.loads((out/'teacher_diagnostics.json').read_text(encoding='utf-8'))
    report={'experiment':'实验01_单文本互补蒸馏','evaluation_type':'STANDARD_SINGLE_TEXT_VAL',
            'split':'val','num_queries':len(txt),'num_images':len(imgs),
            'train_val_pid_intersection':len(overlap),
            'validation_support_texts_used':False,
            'original_irra_image_text_encoders_frozen':True,
            'val_cache_dir':str(valdir),'baseline':base,'methods':summaries,
            'teacher_train_diagnostics':teacher,'loss':'cosine_to_teacher + protect_weight * cosine_to_original',
            'limitations':['No train image features used; student has no direct image-alignment training.',
                           'Train identity group supervision does not guarantee transferable attributes.',
                           'One run/seed does not establish robust statistical gain.',
                           'Validation labels are used solely for metrics, not selecting support/query features.',
                           'Exp00 label-assisted fusion numbers cannot be compared as deployable performance.'],
            'settings':{k:v for k,v in vars(opt).items() if k not in ('fp_train','fp_val')}}
    atomic_json(out/'experiment01_report.json',report)
    rows=['# 实验 01：单文本互补蒸馏','',
          '验证采用单文本、完整图库。图像与原 IRRA 文本编码器冻结。',
          '同身份支持文本仅在训练集使用。',
          '', '| 方法 | R1 | mAP | ΔR1 (百分点) | 修复 | 新增错误 |',
          '|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        rows.append(f'| {r["method"]} | {r["R1"]:.4f} | {r["mAP"]:.4f} | '
                    f'{r["delta_R1"]:+.4f} | {r["recovered"]} | {r["new_errors"]} |')
    rows+=['','说明：cross_mean、cross_gate 使用训练 PID 构造教师；'
           '验证过程中模型只读取当前查询文本和图库图像特征。',
           '即使下降，也不能否定原始多文本信息的存在；只否定该蒸馏配置。']
    (out/'experiment01_readme.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    msg('Full standard single-query evaluation exported.')


def self_test():
    """Synthetic end-to-end stage test using disjoint train/val PIDs and cached features."""
    rng=np.random.default_rng(9)
    with tempfile.TemporaryDirectory(prefix='IRRA_EXP01_SELF_') as temp:
        out=Path(temp)
        nrpid=7;dv=16
        entries=[]; train_vec=[]
        for pid in range(1,nrpid+1):
            key=normalize(rng.normal(size=dv))
            for image_id in range(3):
                for j in range(2):
                    entries.append({'pid':pid,'img_idx':(pid-1)*3+image_id,
                                    'text':f'person {pid}, view {image_id}, caption {j}'})
                    train_vec.append(normalize(key+.22*rng.normal(size=dv)))
        feat=np.asarray(train_vec,dtype=np.float32)
        fake_info=SimpleNamespace(strength=0.35,max_support=8)
        teach=[]
        for method in METHODS:
            summary,examples=build_targets(entries,feat,method,.35,8,out)
            assert summary['with_support']==len(entries)
            a=np.load(out/(method+'_targets.npy'),mmap_mode='r')
            assert np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-5)
            teach.append(summary)
        # Real PyTorch adapter optimization with fake features and genuine child
        # subprocess eval, so we exercise full reporting and ZIP packaging.
        anno=[]
        for pid in range(1,nrpid+1):
            for image_id in range(3):
                anno.append({'split':'train','id':pid,'file_path':f'train/{pid}_{image_id}.jpg',
                             'captions':[f'person {pid}, view {image_id}, caption {j}' for j in range(2)]})
        val_texts=[]; val_imgs=[]
        for pid in [20,21,22,23]:
            base=normalize(rng.normal(size=dv))
            for image_id in range(2):
                anno.append({'split':'val','id':pid,'file_path':f'val/{pid}_{image_id}.jpg',
                             'captions':[f'val identity {pid} caption {j}' for j in range(2)]})
                val_imgs.append(normalize(base+.17*rng.normal(size=dv)))
                for j in range(2):
                    val_texts.append(normalize(base+.18*rng.normal(size=dv)))
        ann=out/'annot.json';ann.write_text(json.dumps(anno),encoding='utf-8')
        np.save(out/'train_text_features.npy',feat)
        np.save(out/'text_features.npy',np.asarray(val_texts,dtype=np.float32))
        np.save(out/'image_features.npy',np.asarray(val_imgs,dtype=np.float32))
        settings=dict(annotation_file=str(ann),project_root=str(out),config_file=str(out/'mock_cfg.yaml'),
                      checkpoint=str(out/'mock_best.pth'),images_dir=str(out),val_cache_dir='',
                      seed=42,device='cpu',epochs=2,train_batch=16,text_batch=8,image_batch=2,
                      hidden=32,gain=.3,lr=.005,protect_weight=.5,
                      strength=.35,max_support=8,fp_train={'self':True},fp_val={'self_val':True},
                      expected_baseline_r1=None,settings_hash='mockrun1')
        atomic_json(out/'run_info.json',{'status':'running','settings':settings})
        atomic_json(out/'teacher_diagnostics.json',{'train_only':True,'strategies':teach})
        root=Path(__file__).parent
        for stage in ('train','eval'):
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),
                                   '--_worker',stage,'--_run-dir',str(out)],
                                  cwd=str(root),capture_output=True,text=True,timeout=150,
                                  env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                                       'OPENBLAS_NUM_THREADS':'1'})
            if result.returncode:
                raise AssertionError(f'{stage} self-test failed:\n{result.stdout}\n{result.stderr}')
        assert (out/'method_summary.csv').is_file()
        report=json.loads((out/'experiment01_report.json').read_text(encoding='utf-8'))
        assert len(report['methods'])==4
        assert report['train_val_pid_intersection']==0
        (out/'experiment_run.log').write_text('Self-test log',encoding='utf-8')
        package_for_chat(out)
        with zipfile.ZipFile(out/'发送给ChatGPT.zip') as z:
            assert 'experiment01_report.json' in z.namelist()
            assert 'case_changes.csv' in z.namelist()
            assert 'run_info.json' in z.namelist()
    msg('SELF-TEST PASSED: teacher, three students, full ranking metrics, cases and ZIP.')


def worker_main(stage,out):
    global LOG_FILE
    # Parent captures worker stdout; workers do not append to log to avoid duplicates.
    if stage=='encode':stage_encode(out)
    elif stage=='teacher':stage_teacher(out)
    elif stage=='train':stage_train(out)
    elif stage=='eval':stage_eval(out)
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
    parser.add_argument('--val-cache-dir',default='',help='Optional completed experiment00 cache folder')
    parser.add_argument('--device',default='auto',choices=('auto','cpu','cuda'))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--text-batch',type=int,default=64)
    parser.add_argument('--image-batch',type=int,default=8)
    parser.add_argument('--train-batch',type=int,default=256)
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--hidden',type=int,default=256)
    parser.add_argument('--gain',type=float,default=0.3)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--strength',type=float,default=.35)
    parser.add_argument('--protect-weight',type=float,default=.5)
    parser.add_argument('--max-support',type=int,default=8)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--_worker',default='',help=argparse.SUPPRESS)
    parser.add_argument('--_run-dir',default='',help=argparse.SUPPRESS)
    opt=parser.parse_args()
    if opt._worker:
        try:
            worker_main(opt._worker,opt._run_dir)
        except Exception:
            traceback.print_exc()
            sys.exit(1)
        return
    if opt.self_test:
        self_test();return
    for k in ('text_batch','image_batch','train_batch','epochs','hidden','max_support'):
        if getattr(opt,k)<=0:parser.error(k+' must be positive')
    if not (0<opt.strength<1 and opt.gain>0 and opt.lr>0 and opt.protect_weight>=0):
        parser.error('strength in (0,1); gain,lr>0; protect_weight>=0')
    root=Path(opt.project_root).resolve()
    cfg=Path(opt.config_file).resolve();ck=Path(opt.checkpoint).resolve()
    for required in (root/'model'/'build.py',cfg,ck):
        if not required.is_file():raise FileNotFoundError(f'Required IRRA file missing: {required}')
    _,ann,imroot,detected=resolve_dataset(root,cfg,opt.dataset_root,opt.annotation_file,opt.images_root)
    valimgs,valtxt,_=read_annotations(ann,'val')
    trainimgs,traintxt,train_id_count=read_annotations(ann,'train')
    if {z['pid'] for z in traintxt}&{z['pid'] for z in valtxt}:
        raise ValueError('Train/val identity leak in annotations')
    fp_train=fingerprint(root,cfg,ck,ann,'train')
    fp_val=fingerprint(root,cfg,ck,ann,'val')
    output_root=root/PROJECT_FOLDER_NAME
    valcache=find_val_cache(root,output_root,opt.val_cache_dir,fp_val,len(valtxt),len(valimgs))
    args_settings={'project_root':str(root),'config_file':str(cfg),'checkpoint':str(ck),
                   'annotation_file':str(ann),'images_dir':str(imroot),'val_cache_dir':str(valcache or ''),
                   'device':opt.device,'text_batch':opt.text_batch,'image_batch':opt.image_batch,
                   'train_batch':opt.train_batch,'epochs':opt.epochs,'hidden':opt.hidden,
                   'gain':opt.gain,'lr':opt.lr,'strength':opt.strength,
                   'protect_weight':opt.protect_weight,'max_support':opt.max_support,
                   'seed':opt.seed,'fp_train':fp_train,'fp_val':fp_val,
                   'expected_baseline_r1':56.83663527119195 if
                   (fp_val.get('checkpoint_sha256')=='3bf21f54271d0e7bd0cfd545daa2c2b5367ee2d734419bead417e70c2ec0a36c'
                    and len(valtxt)==6158 and len(valimgs)==3078) else None}
    hashed=json.dumps(args_settings,sort_keys=True,ensure_ascii=False).encode('utf-8')
    args_settings['settings_hash']=hashlib.sha256(hashed).hexdigest()
    out,resumed=new_or_resume(output_root,opt,args_settings)
    LOG_FILE=out/'experiment_run.log'
    script_copy=out/'irra_exp01_text_student.py'
    shutil.copy2(Path(__file__).resolve(),script_copy)
    info_file=out/'run_info.json'
    prior=json.loads(info_file.read_text(encoding='utf-8')) if resumed else {}
    prior.update({'experiment_name':EXPERIMENT_NAME,'folder_name':out.name,
                  'settings':args_settings,'last_started_at':datetime.now().isoformat(),
                  'status':'running','resumed':resumed})
    atomic_json(info_file,prior)
    msg(f'EXPERIMENT: {out}')
    msg(f'Dataset source: {detected}; train texts={len(traintxt)}, val texts={len(valtxt)}, '
        f'val images={len(valimgs)}, train identities={train_id_count}.')
    msg(f'Val features: {valcache or "to be extracted in this run"}')
    try:
        for stage in ('encode','teacher','train','eval'):
            run_subprocess(stage,out,root)
        prior['status']='completed'
        prior['completed_at']=datetime.now().isoformat()
        atomic_json(info_file,prior)
        msg('EXP01 COMPLETED: standard single-text val metrics saved.')
    except Exception as e:
        prior['status']='failed'
        prior['error']=repr(e)
        prior['failed_at']=datetime.now().isoformat()
        atomic_json(info_file,prior)
        (out/'error_traceback.log').write_text(traceback.format_exc(),encoding='utf-8')
        msg('EXPERIMENT FAILED: '+repr(e))
        raise
    finally:
        try:package_for_chat(out)
        except Exception as ee:msg('Bundle creation failed: '+repr(ee))


if __name__=='__main__':
    main()
