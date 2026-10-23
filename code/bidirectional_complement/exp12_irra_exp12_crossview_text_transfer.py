# -*- coding: utf-8 -*-
"""实验12：同身份跨视角文本补充的真实性筛查与单文本迁移验证。

只读正确的Epoch58最佳IRRA；固定全文图库Validation；不改IRRA源码/权重。
三个相同结构学生：T1-only / T1+All genuine cross-view T2 / T1+Filtered genuine T2。
仅训练数据使用PID构造跨视角支持和负样本；Val真实PID仅用于最终指标。
不使用属性颜色解析、词替换、Oracle后验选择、修正系数扫参或TopK重排。
筛选器使用跨视角一致性（同PID vs 文本相似度匹配的错PID）训练；
"有害拒绝"仅指该受控错误身份代理，不代表同身份真实错误属性识别。
独立第三视角C只用于Train身份内部的教师诊断，不进入选择器特征/阈值。
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, random, re, sys, traceback
import tempfile, shutil, time, io, subprocess, zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np

DEFAULT_ROOT = r"G:\PyCharmProjects\IRRA\IRRA-main"
DEFAULT_CONFIG = DEFAULT_ROOT + r"\logs\CUHK-PEDES\20261009_095215_irra_baseline_b8_resume20\configs.yaml"
DEFAULT_CKPT = DEFAULT_ROOT + r"\logs\CUHK-PEDES\20261009_095215_irra_baseline_b8_resume20\best.pth"
EXPECTED_SHA = "f3ab12028e3406546b836ed6a75c8fc5e02a42146c5da13af423e7f5cf0e7ea6"
EXPECTED_CONFIG_SHA = "ec2fae46ec4e0cce06372789afe1c877a5cebeddf73654781e11b0484fc30fac"
EXPECTED_ANNOTATION_SHA = 'fc3d15542f91dd70e23028c2cb40171a8f41734a39afddb9505e352e396cabf0'
EXPECTED_METRICS = {'R1':66.43390655517578,'R5':85.28743743896484,
                    'R10':90.58135986328125,'mAP':61.86301040649414,
                    'mINP':47.24529266357422}
PROJECT_FOLDER_NAME = '图文双向互补实验'
EXPERIMENT_NAME = '实验12_最佳IRRA同身份文本互补迁移'
VERSION = 'exp12_v3_crossview_provenance_fixed_no_oracle'
METHODS = ('t1_only','all_same_pid','filtered_same_pid')
LOG_FILE = None

def msg(s):
    print(s,flush=True)
    if LOG_FILE is not None:
        with open(LOG_FILE,'a',encoding='utf-8') as f: f.write(str(s)+'\n')

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

def get_subset(ann, selected_pids):
    imgs, texts, _ = read_annotations(ann, 'train')
    allowed = set(map(int, selected_pids))
    img_ids = [i for i, r in enumerate(imgs) if r['pid'] in allowed]
    text_ids = [i for i, r in enumerate(texts) if r['pid'] in allowed]
    if not img_ids or not text_ids:
        raise ValueError('Training PID subset has no matching images/texts')
    return imgs, texts, img_ids, text_ids

def train_img_fingerprint(opt, img_idx):
    digest = hashlib.sha256(np.asarray(img_idx,dtype=np.int32).tobytes()).hexdigest()
    return {'train_base': opt.fp_train, 'selected_image_indices_sha256':digest,
            'selected_image_count':len(img_idx),'format_version':1}

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

# =============================================================
# Cross-view teacher and controlled student: no property keyword rules.
# =============================================================

def split_arrays(out, fit=True):
    _, opt, _ = manifest_for_stage(out)
    x, v, tx, im, pos = train_arrays03(out, opt)
    pids = opt.fit_pids if fit else opt.holdout_pids
    xx, vv, texts, imgs, pp, _, _ = partition(x, v, tx, im, pos, pids)
    return xx, vv, texts, imgs, pp


def construct_pairs(x, texts, pos, seed, wrong_pool=64):
    """Uniform cross-image supporter; wrong PID is text-similarity-matched.

    Source image is inferred from the caption's own image index, not PID at inference.
    Same-PID IDs here are TRAIN ONLY. No val/test PID enters this function.
    """
    rng = np.random.default_rng(seed)
    by_pid = defaultdict(list)
    by_image = defaultdict(list)
    pids = np.asarray([int(t['pid']) for t in texts], dtype=np.int64)
    for i, t in enumerate(texts):
        by_pid[int(t['pid'])].append(i)
        by_image[int(pos[i])].append(i)
    supporters = np.full(len(texts), -1, dtype=np.int32)
    impostors = np.full(len(texts), -1, dtype=np.int32)
    errors = np.full(len(texts), np.nan, dtype=np.float32)
    for i, t in enumerate(texts):
        cross = [j for j in by_pid[int(t['pid'])] if int(pos[j]) != int(pos[i])]
        if not cross:
            continue
        supporters[i] = int(cross[int(rng.integers(0, len(cross)))])
        target = float(x[i] @ x[supporters[i]])
        pool = rng.integers(0, len(x), size=wrong_pool * 2)
        pool = pool[pids[pool] != pids[i]][:wrong_pool]
        if len(pool) == 0:
            raise RuntimeError('Unable to draw different-PID matched control')
        sim = x[pool] @ x[i]
        z = int(np.argmin(np.abs(sim-target)))
        impostors[i] = int(pool[z])
        errors[i] = abs(float(sim[z]) - target)
    ok = supporters >= 0
    if np.any(pids[supporters[ok]] != pids[ok]) or np.any(pos[supporters[ok]] == pos[ok]):
        raise AssertionError('Same-PID, different-image cross-view pairing violated')
    if np.any(pids[impostors[ok]] == pids[ok]):
        raise AssertionError('Wrong-PID negative label leaked')
    return supporters, impostors, errors, by_image


def evidence_features(x, v, pos, anchors, supporters):
    """Only observed source T1,A and T2,B; no held-out C or PID features."""
    t = x[anchors]; a = v[pos[anchors]]
    u = x[supporters]; b = v[pos[supporters]]
    prod = lambda p, q: np.einsum('ij,ij->i', p, q, optimize=True)
    tt = prod(t,u); ta=prod(t,a); tb=prod(t,b)
    ua=prod(u,a); ub=prod(u,b); ab=prod(a,b)
    return np.stack([tt,ta,tb,ua,ub,ab,ub-ua,tb-ta,ub-tb],axis=1).astype(np.float32)


def fit_identity_gate(positive, negative, steps=400):
    """Deterministic small logistic gate; identity-compatibility *proxy*, not truth detector."""
    X = np.concatenate([positive,negative]).astype(np.float64)
    y = np.r_[np.ones(len(positive)),np.zeros(len(negative))]
    mu = X.mean(0); scale = np.maximum(X.std(0),1e-3)
    z = np.clip((X-mu)/scale,-10,10)
    w = np.zeros(z.shape[1],dtype=np.float64); b=0.
    for step in range(steps):
        score=np.clip(z@w+b,-35,35)
        pred=1./(1.+np.exp(-score))
        difference=pred-y
        gradient=z.T@difference/len(y)+0.02*w
        bias=difference.mean()
        lr=0.14/(1.0+step/300.)
        w-=lr*gradient;b-=lr*bias
    return {'mean':mu.tolist(),'scale':scale.tolist(),'weights':w.tolist(),
            'bias':float(b),'threshold':0.5,'label':'same-pid-vs-similarity-matched-wrong-pid'}


def gate_probability(features, gate):
    mu=np.asarray(gate['mean']); s=np.asarray(gate['scale']); w=np.asarray(gate['weights'])
    scores=np.clip(np.clip((np.asarray(features)-mu)/s,-10,10)@w+gate['bias'],-35,35)
    return (1./(1.+np.exp(-scores))).astype(np.float32)


def novelty_threshold(x, texts, pos, by_image):
    # Same-image captions provide a *redundancy reference*, not lexical ground truth.
    vals=[]
    for members in by_image.values():
        if len(members)>1:
            for j in members[1:]:
                vals.append(1.-float(x[members[0]]@x[j]))
    if len(vals)<20:
        raise RuntimeError('Insufficient same-image-caption pairs for novelty reference')
    return float(np.quantile(vals,.75)), len(vals)


def build_teacher_arrays(x,v,texts,pos,seed,gate=None,novelty_min=None):
    supporter,wrong,errors,by_image=construct_pairs(x,texts,pos,seed)
    valid=np.flatnonzero(supporter>=0)
    p=evidence_features(x,v,pos,valid,supporter[valid])
    n=evidence_features(x,v,pos,valid,wrong[valid])
    positive_prob = np.full(len(x),np.nan,dtype=np.float32)
    wrong_prob=np.full(len(x),np.nan,dtype=np.float32)
    if gate is not None:
        positive_prob[valid]=gate_probability(p,gate)
        wrong_prob[valid]=gate_probability(n,gate)
    novel = np.full(len(x),np.nan,dtype=np.float32)
    novel[valid]=1.-np.einsum('ij,ij->i',x[valid],x[supporter[valid]])
    selection=np.zeros(len(x),dtype=np.uint8)
    if gate is not None:
        selection[valid] = ((positive_prob[valid]>=float(gate['threshold'])) &
                            (novel[valid]>float(novelty_min))).astype(np.uint8)
    return {'support':supporter,'wrong':wrong,'matching_error':errors,
            'novelty':novel,'positive_probability':positive_prob,
            'wrong_probability':wrong_prob,'selection':selection,
            'pos_features':p,'neg_features':n,'valid_indices':valid,
            'by_image':by_image}


def teacher_target(x,support,selection,beta,mode):
    y=np.array(x,copy=True)
    valid=np.flatnonzero(support>=0)
    if mode=='t1_only': return y
    if mode=='filtered_same_pid':valid=valid[selection[valid]>0]
    if len(valid):y[valid]=normalize((1.-beta)*x[valid]+beta*x[support[valid]])
    return y


def teacher_third_view_diagnostic(x,v,tx,im,pos,pack,beta):
    """Third image C is ONLY an external diagnostic of teacher view transfer.

    C is never offered to evidence_features(), gate fitting or student target selection.
    In this diagnostic, negative candidates come from a fixed T1 gallery ranking;
    no test PID and no after-the-fact support selection by positive ranks.
    """
    images_by_pid=defaultdict(list)
    for i,row in enumerate(im):images_by_pid[int(row['pid'])].append(i)
    valid=pack['valid_indices'];support=pack['support'];wrong=pack['wrong']
    ids=[]; third=[]
    for i in valid:
        src=int(pos[i]); sec=int(pos[support[i]])
        remainder=[k for k in images_by_pid[int(tx[i]['pid'])] if k!=src and k!=sec]
        if remainder:
            ids.append(int(i));third.append(remainder[0])
    if not ids:return {'n_triple_view':0,'note':'No third-view cases, not evaluated'}
    idx=np.asarray(ids,np.int32);c=np.asarray(third,np.int32)
    t=x[idx]; val_t=normalize((1-beta)*t+beta*x[support[idx]])
    val_f=np.array(t,copy=True)
    chosen=(pack['selection'][idx]>0)
    val_f[chosen]=val_t[chosen]
    # Fixed distractor gallery selected WITHOUT observing the T2 target or C.
    gp=np.asarray([r['pid'] for r in im]); qp=np.asarray([tx[i]['pid'] for i in idx]);
    scores=t@v.T
    scores[qp[:,None]==gp[None,:]]=-np.inf
    top=np.argmax(scores,axis=1)
    vneg=v[top];vc=v[c]
    outcome={}
    for name,feature in [('t1_only',t),('all_same_pid',val_t),('filtered_same_pid',val_f)]:
        pos_score=np.einsum('ij,ij->i',feature,vc)
        neg_score=np.einsum('ij,ij->i',feature,vneg)
        margin=pos_score-neg_score
        outcome[name]={'n':len(idx),'third_view_pair_win_rate':float(np.mean(margin>0)),
                       'mean_margin':float(margin.mean())}
        if name!='t1_only':
            base=np.einsum('ij,ij->i',t,vc)-np.einsum('ij,ij->i',t,vneg)
            diff=margin-base
            outcome[name].update({'margin_improved_fraction':float(np.mean(diff>1e-7)),
                                  'margin_harmed_fraction':float(np.mean(diff< -1e-7)),
                                  'mean_margin_delta':float(diff.mean())})
    return {'n_triple_view':len(idx),'teacher_oracle_not_deployable':True,
            'do_not_use_to_fit_threshold':True,'methods':outcome}


def audit_baseline12(out):
    _,o,_=manifest_for_stage(out)
    if (out/'baseline_audit.json').is_file():
        old=json.loads((out/'baseline_audit.json').read_text(encoding='utf-8'))
        if old.get('settings_hash')==o.settings_hash and old.get('passed'):
            msg('Validated best-IRRA baseline audit exists; skip');return
    img,txt,_=read_annotations(o.annotation_file,'val')
    base=Path(o.val_cache_dir) if o.val_cache_dir else out
    if not cache_is_ready(base,'text',o.fp_val,len(txt)) or not cache_is_ready(base,'image',o.fp_val,len(img)):
        raise RuntimeError('Val cache fingerprint invalid')
    x=normalize(np.load(base/'text_features.npy',mmap_mode='r'))
    v=normalize(np.load(base/'image_features.npy',mmap_mode='r'))
    if x.shape != (6158,512) or v.shape != (3078,512):
        raise RuntimeError(f'Wrong best-model Val feature shapes: {x.shape} / {v.shape}')
    m,_=metrics(x@v.T,[a['pid'] for a in txt],[a['pid'] for a in img])
    delta={name:float(m[name]-ref) for name,ref in EXPECTED_METRICS.items()}
    passed=all(abs(v)<=o.baseline_tolerance for v in delta.values())
    atomic_json(out/'baseline_audit.json',{'metrics':m,'deviation_pp':delta,
                'checkpoint_sha256':o.fp_val['checkpoint_sha256'],
                'settings_hash':o.settings_hash,'passed':passed,
                'val_queries':6158,'val_gallery':3078,'protocol':'full-gallery Val, T2I'})
    msg('EXP12 BEST-IRRA baseline: '+str(m))
    if not passed:raise RuntimeError('Best-IRRA baseline mismatch: STOP before student training: '+repr(delta))


def stage_prepare12(out):
    _,o,_=manifest_for_stage(out)
    required=['fit_support.npy','fit_accepted.npy','fit_negative.npy',
              'gate_model.json','internal_teacher_diagnostic.json','preparation_complete.json']
    if all((out/x).is_file() for x in required):
        meta=json.loads((out/'preparation_complete.json').read_text(encoding='utf-8'))
        if meta.get('settings_hash')==o.settings_hash:
            xf, vf, _,_,_=split_arrays(out,True)
            if (np.load(out/'fit_support.npy').shape==(len(xf),) and
                np.load(out/'fit_negative.npy').shape==(len(xf),o.hard_k)):
                msg('Preparation caches validated; skip');return
    fitx,fitv,fitt,fitim,fitpos=split_arrays(out,True)
    holdx,holdv,holdt,holdim,holdpos=split_arrays(out,False)
    fitpre=build_teacher_arrays(fitx,fitv,fitt,fitpos,o.support_seed)
    if len(fitpre['valid_indices'])<100:
        raise RuntimeError('Too few valid different-image same-PID supports for this dataset')
    cut,den=novelty_threshold(fitx,fitt,fitpos,fitpre['by_image'])
    gate=fit_identity_gate(fitpre['pos_features'],fitpre['neg_features'])
    gate['novelty_threshold']=cut;gate['fit_redundancy_pair_count']=den
    fit=build_teacher_arrays(fitx,fitv,fitt,fitpos,o.support_seed,gate,cut)
    hold=build_teacher_arrays(holdx,holdv,holdt,holdpos,o.support_seed+1,gate,cut)
    # The only fitted classifier uses fit TRAIN identities. Holdout serves no threshold selection.
    def summary(p):
        ids=p['valid_indices']; t=p['positive_probability'][ids];n=p['wrong_probability'][ids]
        novelty=p['novelty'][ids];threshold=gate['threshold']
        wrong_nov=1.-np.einsum('ij,ij->i',
               holdx[ids] if p is hold else fitx[ids],
               (holdx if p is hold else fitx)[p['wrong'][ids]])
        true_accept=p['selection'][ids]>0
        wrong_accept=(n>=threshold)&(wrong_nov>cut)
        errors=p['matching_error'][ids]
        return {'pairs':len(ids),'total_captions':len(p['support']),
                'true_accept_rate_proxy':float(true_accept.mean()),
                'matched_wrong_support_rejection_rate_proxy':float((~wrong_accept).mean()),
                'wrong_support_false_accept_rate_proxy':float(wrong_accept.mean()),
                'true_identity_gate_accuracy':float((t>=threshold).mean()),
                'wrong_identity_gate_rejection':float((n<threshold).mean()),
                'novelty_cutoff':cut,'novelty_mean':float(novelty.mean()),
                'wrong_match_error_mean':float(errors.mean()),
                'wrong_match_error_p90':float(np.quantile(errors,.90)),
                'wrong_match_error_gt_0.05':float((errors>.05).mean())}
    diag={'fit':summary(fit),'holdout':summary(hold),
          'holdout_teacher_third_view':teacher_third_view_diagnostic(
               holdx,holdv,holdt,holdim,holdpos,hold,o.teacher_beta),
          'limitations':['True-PID is only used in TRAIN split to construct support/negative provenance',
            'The gate distinguishes same-PID from matched wrong-PID support, NOT actual factual errors in same-PID captions',
            'Novelty is embedding distance above same-image caption redundancy threshold, not human-verified semantic novelty',
            'Third image C is held out from support selection; C ranking only a diagnostic, NOT a deployable inference',
            'Val query never has a support T2 or PID-conditioned teacher']}
    neg=hard_negative_mine(fitx,fitv,np.asarray([z['pid'] for z in fitt]),
                      np.asarray([z['pid'] for z in fitim]),o.hard_k,o.mine_batch)
    atomic_npy(out/'fit_support.npy',fit['support']);atomic_npy(out/'fit_accepted.npy',fit['selection'])
    atomic_npy(out/'fit_negative.npy',neg)
    atomic_json(out/'gate_model.json',gate)
    atomic_json(out/'internal_teacher_diagnostic.json',diag)
    rows=[]
    # Balanced, fixed-seed case sample; avoid first-200 consecutive captions of the same PIDs.
    # Human labels remain EMPTY; these are real cross-view examples, not ground-truth judgments.
    review_rng=np.random.default_rng(20261021)
    yes=hold['valid_indices'][hold['selection'][hold['valid_indices']]>0]
    no=hold['valid_indices'][hold['selection'][hold['valid_indices']]==0]
    half=o.export_cases//2
    yes_part=review_rng.choice(yes,size=min(half,len(yes)),replace=False) if len(yes) else np.array([],dtype=int)
    no_part=review_rng.choice(no,size=min(o.export_cases-len(yes_part),len(no)),replace=False) if len(no) else np.array([],dtype=int)
    chosen=np.concatenate([yes_part,no_part])
    if len(chosen)<o.export_cases:
        remain=np.setdiff1d(hold['valid_indices'],chosen)
        extra=review_rng.choice(remain,size=min(o.export_cases-len(chosen),len(remain)),replace=False)
        chosen=np.concatenate([chosen,extra])
    chosen=review_rng.permutation(chosen)
    for i in chosen:
        j=int(hold['support'][i]);k=int(hold['wrong'][i])
        rows.append({'anchor_index':int(i),'anchor_pid':int(holdt[i]['pid']),
            'anchor_caption':holdt[i]['text'],'support_caption':holdt[j]['text'],
            'wrong_pid_caption':holdt[k]['text'],
            'anchor_image_path':holdim[int(holdpos[i])]['file_path'],
            'support_image_path':holdim[int(holdpos[j])]['file_path'],
            'wrong_image_path':holdim[int(holdpos[k])]['file_path'],
            'same_identity':1,'cross_image':int(holdpos[i]!=holdpos[j]),
            'novelty_proxy':float(hold['novelty'][i]),
            'same_pid_probability':float(hold['positive_probability'][i]),
            'matched_wrong_pid_probability':float(hold['wrong_probability'][i]),
            'accepted':int(hold['selection'][i]),
            'wrong_text_matching_error':float(hold['matching_error'][i]),
            '人工_真实新增':'','人工_可信/有害/未知':'','人工_说明':''})
    write_csv(out/'真实多文本人工核查样本.csv',rows,
         list(rows[0]) if rows else ['anchor_index'])
    atomic_json(out/'preparation_complete.json',{'settings_hash':o.settings_hash,
       'fit_samples':len(fitx),'fit_gallery':len(fitv),'holdout_samples':len(holdx),
       'holdout_gallery':len(holdv),'support_selection_ready':True})
    msg('Prepared cross-view support and matched negative controls; fit/holdout: '+str(diag['fit']['pairs'])+'/'+str(diag['holdout']['pairs']))


def get_text_student():
    import torch
    from torch import nn
    from torch.nn import functional as F
    class TextStudent(nn.Module):
        def __init__(self,dim,hidden,max_residual):
            super().__init__()
            self.fc1=nn.Linear(dim,hidden)
            self.fc2=nn.Linear(hidden,dim)
            self.max_residual=max_residual
            nn.init.zeros_(self.fc2.weight);nn.init.zeros_(self.fc2.bias)
        def forward(self,t):
            residual=self.max_residual*torch.tanh(self.fc2(F.gelu(self.fc1(t))))
            return F.normalize(t+residual,dim=-1)
    return TextStudent


def run_student_epoch(net,optimizer,x,v,pos,neg,target,order,o,device):
    import torch
    from torch.nn import functional as F
    net.train();tot=defaultdict(float);seen=0
    for start in range(0,len(order),o.train_batch):
        sub=order[start:start+o.train_batch];bs=len(sub)
        text=x[sub].to(device)
        student=net(text)
        imageids=torch.cat([pos[sub,None],neg[sub]],dim=1).to(device)
        candidates=v[imageids]
        score=(student[:,None,:]*candidates).sum(-1)
        rank=F.softplus((score[:,1:]-score[:,:1]+o.margin)/o.temperature).mean()
        distill=(1.-F.cosine_similarity(student,target[sub].to(device))).mean()
        protect=(1.-F.cosine_similarity(student,text)).mean()
        loss=o.rank_weight*rank+o.distill_weight*distill+o.protect_weight*protect
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
        optimizer.step()
        for name,value in [('rank_loss',rank),('distill_loss',distill),
                           ('protect_loss',protect),('total_loss',loss)]:
            tot[name]+=float(value.detach())*bs
        seen+=bs
    return {name:val/max(seen,1) for name,val in tot.items()}


def stage_train12(out):
    import torch
    torch.set_num_threads(1)
    _,o,_=manifest_for_stage(out)
    fitx,fitv,fitt,fitim,fitpos=split_arrays(out,True)
    support=np.load(out/'fit_support.npy',allow_pickle=False)
    selected=np.load(out/'fit_accepted.npy',allow_pickle=False)
    neg=np.load(out/'fit_negative.npy',allow_pickle=False)
    if not (len(fitx)==len(support)==len(selected)==len(neg)):
        raise RuntimeError('Training preparation shape mismatch')
    d=fitx.shape[1]
    device=torch.device('cuda' if o.device=='cuda' or (o.device=='auto' and torch.cuda.is_available()) else 'cpu')
    if o.device=='cuda' and device.type!='cuda':raise RuntimeError('CUDA requested but not found')
    xt=torch.tensor(np.array(fitx,copy=True),dtype=torch.float32)
    gv=torch.tensor(np.array(fitv,copy=True),device=device,dtype=torch.float32)
    pp=torch.tensor(np.asarray(fitpos,dtype=np.int64))
    nnx=torch.tensor(np.asarray(neg,dtype=np.int64))
    klass=get_text_student()
    for seed in o.seeds:
        # Same initialization / same minibatch order for all 3 treatments, fixed by seed.
        torch.manual_seed(seed)
        if device.type=='cuda':torch.cuda.manual_seed_all(seed)
        base=klass(d,o.hidden,o.max_residual)
        initial={k:val.detach().cpu().clone() for k,val in base.state_dict().items()}
        orders=[];generator=torch.Generator().manual_seed(seed)
        for ep in range(o.epochs):orders.append(torch.randperm(len(xt),generator=generator))
        for method in METHODS:
            stem=f'{method}_s{seed}';model_file=out/(stem+'.pt');meta_file=out/(stem+'_meta.json')
            if model_file.is_file() and meta_file.is_file():
                try:
                    meta=json.loads(meta_file.read_text(encoding='utf-8'))
                    if (meta['settings_hash']==o.settings_hash and
                        meta['checkpoint_sha256']==sha256_small_file(model_file) and
                        meta['source_checkpoint_sha256']==EXPECTED_SHA):
                        msg(stem+' validated complete; skip');continue
                except (OSError,KeyError,ValueError):pass
            targets=teacher_target(fitx,support,selected,o.teacher_beta,method)
            tt=torch.tensor(np.array(targets,copy=True),dtype=torch.float32)
            net=klass(d,o.hidden,o.max_residual).to(device)
            net.load_state_dict(initial)
            optimizer=torch.optim.AdamW(net.parameters(),lr=o.learning_rate,weight_decay=1e-4)
            history=[]
            for ep,order in enumerate(orders,1):
                st=run_student_epoch(net,optimizer,xt,gv,pp,nnx,tt,order,o,device)
                history.append({'method':method,'seed':seed,'epoch':ep,**st})
                msg(f'{stem} epoch {ep}/{o.epochs}: rank={st["rank_loss"]:.5f} distill={st["distill_loss"]:.5f}')
            state={'model_state_dict':net.cpu().state_dict(),'dim':d,'method':method,
                   'seed':seed,'settings_hash':o.settings_hash,
                   'source_checkpoint_sha256':EXPECTED_SHA,
                   'init_control_sha256':hashlib.sha256(b''.join(
                         val.numpy().tobytes() for val in initial.values())).hexdigest()}
            temp=model_file.with_suffix('.partial.pt')
            torch.save(state,temp);_replace_with_retries(temp,model_file)
            atomic_json(meta_file,{'settings_hash':o.settings_hash,
                 'checkpoint_sha256':sha256_small_file(model_file),
                 'source_checkpoint_sha256':EXPECTED_SHA,'history':history,
                 'selected_training_fraction':float(selected.mean()) if method=='filtered_same_pid' else None})
            del net,optimizer,tt
            if device.type=='cuda':torch.cuda.empty_cache()


def stage_predict12(out):
    """PyTorch-only process: create deployable, single-caption student embeddings."""
    import torch
    torch.set_num_threads(1)
    _,o,_=manifest_for_stage(out)
    valimg,valtxt,_=read_annotations(o.annotation_file,'val')
    base=Path(o.val_cache_dir) if o.val_cache_dir else out
    if not cache_is_ready(base,'text',o.fp_val,len(valtxt)):
        raise RuntimeError('Val query embedding cache not fingerprint validated')
    x=np.load(base/'text_features.npy',mmap_mode='r',allow_pickle=False)
    device=torch.device('cuda' if o.device=='cuda' or (o.device=='auto' and torch.cuda.is_available()) else 'cpu')
    if o.device=='cuda' and device.type!='cuda':raise RuntimeError('CUDA requested but unavailable')
    model_class=get_text_student()
    t=torch.tensor(np.array(x,copy=True),dtype=torch.float32)
    for seed in o.seeds:
        for method in METHODS:
            stem=f'{method}_s{seed}'
            ckpt=out/(stem+'.pt')
            meta=json.loads((out/(stem+'_meta.json')).read_text(encoding='utf-8'))
            cksha=sha256_small_file(ckpt)
            if meta.get('checkpoint_sha256')!=cksha:
                raise RuntimeError('Student checkpoint SHA mismatch: '+stem)
            fpath=out/(stem+'_val_feature.npy')
            fmeta=out/(stem+'_val_feature_meta.json')
            expected={'settings_hash':o.settings_hash,'model_sha256':cksha,
                      'n_queries':len(t),'dim':int(t.shape[1])}
            if fpath.is_file() and fmeta.is_file():
                try:
                    if json.loads(fmeta.read_text(encoding='utf-8'))==expected:
                        f=np.load(fpath,mmap_mode='r',allow_pickle=False)
                        if f.shape==tuple((len(t),t.shape[1])) and f.dtype==np.float32 and np.all(np.isfinite(f)):
                            msg('Validated student text-only prediction cached: '+stem)
                            continue
                except (OSError,ValueError):pass
            saved=torch.load(ckpt,map_location='cpu',weights_only=False)
            if (saved['settings_hash']!=o.settings_hash or
                 saved['source_checkpoint_sha256']!=EXPECTED_SHA or
                 saved['method']!=method or saved['seed']!=seed):
                raise RuntimeError('Student state provenance mismatch: '+stem)
            net=model_class(saved['dim'],o.hidden,o.max_residual).to(device)
            net.load_state_dict(saved['model_state_dict'],strict=True)
            net.eval();outputs=[]
            with torch.inference_mode():
                for start in range(0,len(t),o.eval_batch):
                    out_batch=net(t[start:start+o.eval_batch].to(device))
                    outputs.append(out_batch.cpu().numpy())
            features=np.concatenate(outputs).astype(np.float32)
            if not np.all(np.isfinite(features)):raise RuntimeError('Nonfinite prediction')
            atomic_npy(fpath,features)
            atomic_json(fmeta,expected)
            msg('Single-query Val features encoded: '+stem)
            del net
            if device.type=='cuda':torch.cuda.empty_cache()


def stage_score12(out):
    """NumPy-only process: full-gallery retrieval, never mixes Torch/MKL runtimes."""
    _,o,_=manifest_for_stage(out)
    valimg,valtxt,_=read_annotations(o.annotation_file,'val')
    base=Path(o.val_cache_dir) if o.val_cache_dir else out
    x=normalize(np.load(base/'text_features.npy',mmap_mode='r',allow_pickle=False))
    v=normalize(np.load(base/'image_features.npy',mmap_mode='r',allow_pickle=False))
    pid=np.asarray([t['pid'] for t in valtxt]);gp=np.asarray([a['pid'] for a in valimg])
    rows=[];case_rows=[]
    original,orig_details=metrics(x@v.T,pid,gp)
    rows.append({'seed':'base','method':'original_irra',**original,'feature_cos_to_base':1.0})
    for seed in o.seeds:
        for method in METHODS:
            stem=f'{method}_s{seed}'
            path=out/(stem+'_val_feature.npy')
            record=json.loads((out/(stem+'_val_feature_meta.json')).read_text(encoding='utf-8'))
            modelsha=sha256_small_file(out/(stem+'.pt'))
            if record!={'settings_hash':o.settings_hash,'model_sha256':modelsha,
                          'n_queries':len(x),'dim':x.shape[1]}:
                raise RuntimeError('Student embedding provenance mismatch: '+stem)
            student=normalize(np.load(path,allow_pickle=False))
            if student.shape!=x.shape or not np.all(np.isfinite(student)):
                raise RuntimeError('Invalid student text feature shape/values: '+stem)
            m,info=metrics(student@v.T,pid,gp)
            cosine=np.sum(student*x,axis=1)
            rows.append({'seed':seed,'method':method,**m,
                         'feature_cos_to_base':float(cosine.mean())})
            for i in range(len(x)):
                case_rows.append({'seed':seed,'method':method,'query_index':i,
                    'pid':int(pid[i]),'original_rank':int(orig_details['rank'][i]),
                    'student_rank':int(info['rank'][i]),
                    'original_r1_correct':int(orig_details['rank'][i]==1),
                    'student_r1_correct':int(info['rank'][i]==1),
                    'query_feature_cos_to_base':float(cosine[i])})
            msg(f'Val {stem}: R1={m["R1"]:.6f} mAP={m["mAP"]:.6f}')
    write_csv(out/'method_summary.csv',rows,list(rows[0]))
    aggregates=[]
    for method in METHODS:
        subset=[r for r in rows if r['method']==method]
        record={'method':method,'seeds':len(subset)}
        for metric in ('R1','R5','R10','mAP','mINP','feature_cos_to_base'):
            vals=np.asarray([r[metric] for r in subset])
            record[metric+'_mean']=float(vals.mean())
            record[metric+'_std']=float(vals.std(ddof=1)) if len(vals)>1 else 0.
            if metric!='feature_cos_to_base':
                record[metric+'_vs_irra_pp']=float(vals.mean()-original[metric])
        aggregates.append(record)
    write_csv(out/'aggregate_summary.csv',aggregates,list(aggregates[0]))
    write_csv(out/'case_changes.csv',case_rows,list(case_rows[0]))
    paired=[]
    for seed in o.seeds:
        sub=[r for r in case_rows if r['seed']==seed]
        lookup={(r['method'],r['query_index']):r for r in sub}
        for a,b in [('all_same_pid','t1_only'),('filtered_same_pid','t1_only'),
                    ('filtered_same_pid','all_same_pid')]:
            ar=[lookup[(a,i)] for i in range(len(x))]
            br=[lookup[(b,i)] for i in range(len(x))]
            corrections=sum(r['student_r1_correct'] and not q['student_r1_correct'] for r,q in zip(ar,br))
            harms=sum(not r['student_r1_correct'] and q['student_r1_correct'] for r,q in zip(ar,br))
            paired.append({'seed':seed,'comparison':a+' - '+b,
                           'newly_correct':corrections,'newly_wrong':harms,
                           'net_correct':corrections-harms,
                           'delta_R1_pp':100*(corrections-harms)/len(x)})
    write_csv(out/'paired_rank_comparisons.csv',paired,list(paired[0]))
    diag=json.loads((out/'internal_teacher_diagnostic.json').read_text(encoding='utf-8'))
    atomic_json(out/'experiment12_report.json',{'baseline':original,'aggregate':aggregates,
        'paired':paired,'teacher_diagnostic':diag,
        'checkpoint_sha256':EXPECTED_SHA,
        'evaluation':'Full Validation; one-caption student; no PID in inference',
        'limitations':diag['limitations']})

def package12(out):
    out=Path(out); dest=out/'发送给ChatGPT.zip'
    allowed={'.csv','.json','.py','.txt','.log','.md'}
    source=[p for p in out.iterdir() if p.is_file() and p.suffix.lower() in allowed]
    fd, tmp=_new_temp_path(dest);os.close(fd)
    try:
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in source:z.write(p,p.name)
        with zipfile.ZipFile(tmp,'r') as check_zip:
            if check_zip.testzip() is not None:raise RuntimeError('ZIP CRC verification failed')
        try:_replace_with_retries(tmp,dest)
        except PermissionError:
            dest=out/('发送给ChatGPT_恢复版_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.zip')
            _replace_with_retries(tmp,dest)
        msg('Analysis archive: '+str(dest))
    finally:
        try:tmp.unlink(missing_ok=True)
        except OSError:pass


def subprocess_stage(stage,out,root):
    cmd=[sys.executable,str(Path(__file__).resolve()),'--_stage',stage,'--_out',str(out)]
    env=os.environ.copy();env.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'})
    env.pop('KMP_DUPLICATE_LIB_OK',None)
    msg('\n==== Stage '+stage+' ====')
    with subprocess.Popen(cmd,cwd=str(root),env=env,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,text=True,encoding='utf-8',
                          errors='replace',bufsize=1) as proc:
        for line in proc.stdout:msg(line.rstrip('\r\n'))
        code=proc.wait()
    if code:raise RuntimeError(f'Stage {stage} failed exit code {code}; use --resume after correction')


def stage_worker(stage,out):
    if stage=='encode':stage_encode03(out)
    elif stage=='audit':audit_baseline12(out)
    elif stage=='prepare':stage_prepare12(out)
    elif stage=='train':stage_train12(out)
    elif stage=='predict':stage_predict12(out)
    elif stage=='score':stage_score12(out)
    else:raise ValueError('Unknown stage '+stage)


def source_cache_locations(folder,settings,all_img,all_txt,val_img,val_txt,img_idx):
    """Read-only reuse only after complete exact model+annotation+selection fingerprint checks."""
    for previous in sorted(folder.glob('实验10_*'),reverse=True):
        if not previous.is_dir():continue
        pids_file=previous/'selected_train_pids.json'
        if not pids_file.is_file():continue
        try:
            if json.loads(pids_file.read_text(encoding='utf-8'))!=settings['selected_pids']:continue
            if not (cache_is_ready(previous,'train_text',settings['fp_train'],len(all_txt)) and
                 cache_is_ready(previous,'train_image',train_img_fingerprint(SimpleNamespace(**settings),img_idx),len(img_idx)) and
                 cache_is_ready(previous,'text',settings['fp_val'],len(val_txt)) and
                 cache_is_ready(previous,'image',settings['fp_val'],len(val_img))):continue
            msg('Compatible BEST-IRRA EXP10 feature cache located: '+str(previous))
            return str(previous)
        except (OSError,KeyError,ValueError):continue
    return ''


def create_or_resume_folder(folder,settings,resume):
    folder.mkdir(parents=True,exist_ok=True)
    if resume:
        for directory in sorted(folder.glob(EXPERIMENT_NAME+'_*'),reverse=True):
            mf=directory/'run_info.json'
            if mf.is_file():
                try:
                    val=json.loads(mf.read_text(encoding='utf-8'))
                    old={k:v for k,v in val['settings'].items() if k!='run_dir'}
                    if old==settings:return directory,True
                except (OSError,KeyError,ValueError):pass
        raise RuntimeError('No matching experiment12 folder to resume. For a deleted folder, rerun WITHOUT --resume.')
    name=EXPERIMENT_NAME+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')
    out=folder/name;idx=2
    while out.exists():out=folder/(name+'_'+str(idx));idx+=1
    out.mkdir(parents=True);return out,False


def self_test12():
    """Synthetic causal and structural test. Does not pretend to test Windows/CUDA IRRA."""
    import torch
    rng=np.random.default_rng(91)
    nids=12;dim=32;per_pid=3
    v=[];txt=[];images=[];pos=[];t=[]
    for pid in range(nids):
        anchor=normalize(rng.standard_normal((1,dim)))[0]
        for imgno in range(per_pid):
            vec=normalize(anchor+.1*rng.standard_normal(dim))
            v.append(vec);images.append({'pid':pid,'file_path':f'{pid}_{imgno}.jpg'})
            for desc in range(2):
                text=normalize(vec+.2*rng.standard_normal(dim))
                t.append(text);txt.append({'pid':pid,'img_idx':len(v)-1,'text':f'person {pid}, view {imgno}, caption {desc}'})
                pos.append(len(v)-1)
    x=np.asarray(t,np.float32);v=np.asarray(v,np.float32);pos=np.asarray(pos,np.int32)
    fit=build_teacher_arrays(x,v,txt,pos,11)
    assert len(fit['valid_indices'])==len(x)
    th,_=novelty_threshold(x,txt,pos,fit['by_image'])
    gate=fit_identity_gate(fit['pos_features'],fit['neg_features'],steps=30)
    p=build_teacher_arrays(x,v,txt,pos,11,gate,th)
    wrong=construct_pairs(x,txt,pos,11)[1]
    assert np.all(np.asarray([txt[i]['pid'] for i in wrong])!=np.asarray([t['pid'] for t in txt]))
    diag=teacher_third_view_diagnostic(x,v,txt,images,pos,p,.25)
    assert diag['n_triple_view']>0
    negatives=hard_negative_mine(x,v,np.asarray([r['pid'] for r in txt]),
                               np.asarray([r['pid'] for r in images]),3,64)
    cls=get_text_student(); torch.set_num_threads(1)
    initial=cls(dim,16,.15).state_dict()
    order=torch.randperm(len(x),generator=torch.Generator().manual_seed(42))
    result={}
    for mode in METHODS:
        net=cls(dim,16,.15);net.load_state_dict(initial)
        optimizer=torch.optim.AdamW(net.parameters(),lr=.0002)
        target=teacher_target(x,p['support'],p['selection'],.25,mode)
        opts=SimpleNamespace(train_batch=16,margin=.05,temperature=.07,
                    rank_weight=.15,distill_weight=1.0,protect_weight=.5)
        st=run_student_epoch(net,optimizer,torch.from_numpy(x),torch.from_numpy(v),
                torch.from_numpy(pos.astype(np.int64)),torch.from_numpy(negatives.astype(np.int64)),
                torch.from_numpy(target),order,opts,torch.device('cpu'))
        assert np.isfinite(st['total_loss']) and st['total_loss']>0
        with torch.inference_mode():embedded=net(torch.from_numpy(x)).numpy()
        met,_=metrics(embedded@v.T,[z['pid'] for z in txt],[z['pid'] for z in images])
        result[mode]=met['R1']
    assert np.allclose(teacher_target(x,p['support'],p['selection'],.25,'t1_only'),x)
    assert np.allclose(teacher_target(x,p['support'],p['selection'],.25,'all_same_pid'),
                       teacher_target(x,p['support'],np.zeros_like(p['selection']),.25,'filtered_same_pid')) is False
    with tempfile.TemporaryDirectory() as td:
        out=Path(td)
        atomic_json(out/'self_test.json',{'ok':True,'R1':result,'third_view_count':diag['n_triple_view']})
        (out/'simple.py').write_text('# self test',encoding='utf-8')
        package12(out)
        with zipfile.ZipFile(out/'发送给ChatGPT.zip') as z:
            assert set(z.namelist())=={'self_test.json','simple.py'}
    msg('SELF TEST PASSED: supporter PID/view isolation, matched impostor, fixed gate, third-view diagnostics, 3 identical training branches, full-gallery metrics, atomic IO, ZIP. Real IRRA/CUDA still must be run locally.')


def main12():
    global LOG_FILE
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',default=DEFAULT_ROOT)
    ap.add_argument('--config-file',default=DEFAULT_CONFIG)
    ap.add_argument('--checkpoint',default=DEFAULT_CKPT)
    ap.add_argument('--dataset-root',default='')
    ap.add_argument('--annotation-file',default='')
    ap.add_argument('--images-root',default='')
    ap.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    ap.add_argument('--max-train-pids',type=int,default=2200)
    ap.add_argument('--subset-seed',type=int,default=42)
    ap.add_argument('--split-seed',type=int,default=776)
    ap.add_argument('--seeds',default='42,43,44')
    ap.add_argument('--support-seed',type=int,default=1212)
    ap.add_argument('--teacher-beta',type=float,default=.25)
    ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--hidden',type=int,default=128)
    ap.add_argument('--max-residual',type=float,default=.15)
    ap.add_argument('--learning-rate',type=float,default=.0002)
    ap.add_argument('--rank-weight',type=float,default=.15)
    ap.add_argument('--distill-weight',type=float,default=1.0)
    ap.add_argument('--protect-weight',type=float,default=.5)
    ap.add_argument('--margin',type=float,default=.05)
    ap.add_argument('--temperature',type=float,default=.07)
    ap.add_argument('--train-batch',type=int,default=128)
    ap.add_argument('--eval-batch',type=int,default=256)
    ap.add_argument('--text-batch',type=int,default=64)
    ap.add_argument('--image-batch',type=int,default=8)
    ap.add_argument('--hard-k',type=int,default=4)
    ap.add_argument('--mine-batch',type=int,default=96)
    ap.add_argument('--export-cases',type=int,default=200)
    ap.add_argument('--baseline-tolerance',type=float,default=.02)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--_stage',default='',help=argparse.SUPPRESS)
    ap.add_argument('--_out',default='',help=argparse.SUPPRESS)
    args=ap.parse_args()
    if args._stage:
        try:stage_worker(args._stage,Path(args._out))
        except Exception:traceback.print_exc();sys.exit(1)
        return
    if args.self_test:self_test12();return
    if (args.max_train_pids<200 or args.epochs<1 or args.hard_k<1 or
        args.train_batch<1 or args.eval_batch<1 or args.image_batch<1 or args.text_batch<1 or
        args.hidden<1 or not 0<args.teacher_beta<1 or not 0<args.max_residual<1 or
        args.learning_rate<=0 or args.baseline_tolerance<=0 or args.export_cases<1):
        ap.error('Invalid hyperparameters; check nonzero and in-range settings')
    try:seeds=[int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    except ValueError:ap.error('Invalid --seeds')
    if not seeds or len(set(seeds))!=len(seeds) or min(seeds)<0:ap.error('Seeds must be unique nonnegative')
    root=Path(args.project_root).resolve();config=Path(args.config_file).resolve();
    checkpoint=Path(args.checkpoint).resolve()
    for f in (root/'model'/'build.py',config,checkpoint):
        if not f.is_file():ap.error('Missing required IRRA file: '+str(f))
    if sha256_small_file(checkpoint)!=EXPECTED_SHA:ap.error('Incorrect best.pth SHA256; REFUSE old/wrong model')
    if sha256_small_file(config)!=EXPECTED_CONFIG_SHA:ap.error('Incorrect best-model configs.yaml SHA256')
    _,ann,imgroot,_=resolve_dataset(root,config,args.dataset_root,args.annotation_file,args.images_root)
    if sha256_small_file(ann)!=EXPECTED_ANNOTATION_SHA:ap.error('Annotation hash differs from best-model reproduction')
    all_img,all_txt,trainclasses=read_annotations(ann,'train')
    val_img,val_txt,_=read_annotations(ann,'val')
    if (len(val_img),len(val_txt))!=(3078,6158):ap.error('Validation split mismatch')
    trainpids=sorted({int(z['pid']) for z in all_txt})
    if set(trainpids)&{int(z['pid']) for z in val_txt}:ap.error('Train/Val PID overlap')
    rng=np.random.default_rng(args.subset_seed)
    selected=sorted(map(int,rng.choice(trainpids,min(args.max_train_pids,len(trainpids)),replace=False)))
    fit,hold=pid_split(selected,.8,args.split_seed)
    idx=[i for i,r in enumerate(all_img) if int(r['pid']) in set(selected)]
    verify_image_paths(imgroot,[all_img[i] for i in idx])
    fp_train=fingerprint(root,config,checkpoint,ann,'train')
    fp_val=fingerprint(root,config,checkpoint,ann,'val')
    folder=root/PROJECT_FOLDER_NAME
    settings={'project_root':str(root),'config_file':str(config),'checkpoint':str(checkpoint),
        'annotation_file':str(ann),'images_dir':str(imgroot),
        'selected_pids':selected,'fit_pids':fit,'holdout_pids':hold,
        'fp_train':fp_train,'fp_val':fp_val,'device':args.device,'seeds':seeds,
        'support_seed':args.support_seed,'teacher_beta':args.teacher_beta,
        'epochs':args.epochs,'hidden':args.hidden,'max_residual':args.max_residual,
        'learning_rate':args.learning_rate,'rank_weight':args.rank_weight,
        'distill_weight':args.distill_weight,'protect_weight':args.protect_weight,
        'margin':args.margin,'temperature':args.temperature,'train_batch':args.train_batch,
        'eval_batch':args.eval_batch,'text_batch':args.text_batch,
        'image_batch':args.image_batch,'hard_k':args.hard_k,
        'mine_batch':args.mine_batch,'export_cases':args.export_cases,
        'baseline_tolerance':args.baseline_tolerance,'version':VERSION}
    olddir=source_cache_locations(folder,settings,all_img,all_txt,val_img,val_txt,idx)
    settings['train_text_cache_dir']=olddir
    settings['train_image_cache_dir']=olddir
    settings['val_cache_dir']=olddir
    settings['settings_hash']=hashlib.sha256(json.dumps(settings,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    output,resumed=create_or_resume_folder(folder,settings,args.resume)
    LOG_FILE=output/'experiment_run.log'
    settings['run_dir']=str(output)
    if not resumed:
        atomic_json(output/'run_info.json',{'experiment':EXPERIMENT_NAME,'settings':settings})
        atomic_json(output/'selected_train_pids.json',selected)
    else:
        if json.loads((output/'selected_train_pids.json').read_text(encoding='utf-8'))!=selected:
            raise RuntimeError('Selected PID list changed on --resume')
    shutil.copy2(Path(__file__).resolve(),output/Path(__file__).name)
    (output/'实验12_目标与边界.md').write_text(
        '# 实验12：多文本可信支持与单文本迁移\n\n'
        '主基线：正确Epoch58最佳IRRA，Val R1 66.4339%，三种子固定。\n'
        '三个学生：T1-only、T1+全部真实同PID跨图支持、T1+筛选后支持。\n'
        '三种学生共享相同初始化、优化器、采样顺序、图像正负对及损失配比。\n'
        '筛选器由训练PID内真实/错误身份（文本相似度匹配）配对拟合，绝不读取Val PID。\n'
        '非冗余是向量差异高于同图重复描述参考的近似，并非人工语义真值。\n'
        '只有训练期和教师诊断可访问T2，正式Val仅单条查询文本与图库图像。\n'
        '第三视角C用于教师泛化诊断，绝不用于拟合筛选器或阈值。\n'
        '错误身份支持的拒绝率不等于同身份错误描述的真实有害拒绝率。\n'
        '绝不训练/修改IRRA本体；不做颜色词替换、属性规则修补、TopK重排或alpha扫描。\n',
        encoding='utf-8')
    try:
        for stage in ('encode','audit','prepare','train','predict','score'):
            subprocess_stage(stage,output,root)
        msg('EXP12 COMPLETED. Best IRRA preserved: '+EXPECTED_SHA)
    finally:
        package12(output)


if __name__=='__main__':
    main12()
