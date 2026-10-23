# -*- coding: utf-8 -*-
"""实验11：新增语义、视觉证据、拒绝有害补充——只读诊断。

将本脚本单独放入 IRRA-main 项目根目录，并使用 irra_clean 运行。
绝不训练原 IRRA / Adapter / 重排序器；绝不修改 best.pth、IRRA 源码或旧实验。
只使用 CUHK-PEDES Train PID 构造同身份和错误身份支持；不读取 Val/Test PID。

严格区分：
  (1) 规则解析到的文本属性差异（不是神经网络的语义理解）；
  (2) 原 IRRA 全局图像向量与属性提示的视觉近似证据（不是物体检测/真实标注）；
  (3) 人工可审查的案例；自动受控扰动是 sanity check，不是自然样本真值。

默认 2200 Train PID -> 1760 fit / 440 未见诊断（根据PID划分）。
包括真正同身份不同图描述、语义相似的不同身份描述、重复、受控删去属性的新增、
受控反事实属性的有害等；接受有益与拒绝有害等权统计。

本文件仅依赖官方 IRRA 工程、已训练的 Epoch58 best.pth 和常规库。
输出：CSV/JSON、200组离线HTML人工核查案例、发送给ChatGPT.zip。
"""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import html
import importlib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

DEFAULT_ROOT = r'G:\PyCharmProjects\IRRA\IRRA-main'
BEST_FOLDER = r'logs\CUHK-PEDES\20261009_095215_irra_baseline_b8_resume20'
BEST_SHA = 'f3ab12028e3406546b836ed6a75c8fc5e02a42146c5da13af423e7f5cf0e7ea6'
CONFIG_SHA = 'ec2fae46ec4e0cce06372789afe1c877a5cebeddf73654781e11b0484fc30fac'
ANNOTATION_SHA = 'fc3d15542f91dd70e23028c2cb40171a8f41734a39afddb9505e352e396cabf0'
VERSION = 'EXP11_semantic_visual_v1_20261021'
EXPERIMENT_NAME = '实验11_新增语义视觉证据与有害拒绝'

# Conservative text spans; order of alternatives follows longest first.
COLORS = ('dark blue','light blue','dark green','light green','dark gray','light gray',
          'navy blue','navy','blonde','blond','auburn','khaki','beige','maroon','purple','orange','yellow','brown',
          'green','black','white','gray','grey','blue','red','pink','gold','silver')
COLOR_CANON = {'grey':'gray','blond':'blonde','navy':'navy blue','dark grey':'dark gray','light grey':'light gray'}
COLOR_BASE = {'dark blue':'blue','light blue':'blue','navy blue':'blue','navy':'blue',
              'dark gray':'gray','light gray':'gray','grey':'gray',
              'dark green':'green','light green':'green'}
OBJECTS = {
    'upper': ('shirt','tshirt','t-shirt','tee','sweater','blouse','jersey','top','sweatshirt'),
    'outer': ('hoodie','coat','jacket','vest','cardigan'),
    'lower': ('pants','trousers','jeans','shorts','skirt','leggings'),
    'shoes': ('shoes','sneakers','boots','sandals','trainers','footwear'),
    'bag': ('backpack','handbag','rucksack','bag','purse'),
    'hat': ('hat','cap','beanie','helmet'),
    'hair': ('hair',),
}
SLOT_NOUN = {'outer':'jacket','upper':'shirt','lower':'pants','shoes':'shoes','bag':'backpack','hat':'hat','hair':'hair'}
SLOT_TEMPLATES = {
    'upper':'a person wearing a {value} shirt',
    'outer':'a person wearing a {value} jacket',
    'lower':'a person wearing {value} pants',
    'shoes':'a person wearing {value} shoes',
    'bag':'a person carrying a {value} backpack',
    'hat':'a person wearing a {value} hat',
    'hair':'a person with {value} hair',
}
SLOT_ORDER = tuple(OBJECTS)
ALL_COLOR_PAT = '|'.join(re.escape(x) for x in sorted(COLORS,key=lambda s:-len(s)))
NOUN_PAT = '|'.join(re.escape(x) for x in sorted({v for a in OBJECTS.values() for v in a},key=lambda s:-len(s)))
ATTR_RX = re.compile(r'(?<![a-z])(?P<color>'+ALL_COLOR_PAT+r')\s+(?:(?:long|short|sleeved|sleeve|dark|light|plain|striped|sports|tennis|denim|leather|running)\s+){0,3}(?P<noun>'+NOUN_PAT+r')\b',re.I)
NOUN_RX = re.compile(r'(?<![a-z])('+NOUN_PAT+r')\b',re.I)
WORD_RX = re.compile(r'[a-z]+',re.I)
NEG_RX = re.compile(r'\b(?:no|without|not\s+wearing|does\s+not\s+(?:have|carry|wear))\b',re.I)


def log(*args):
    print(*args,flush=True)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for ch in iter(lambda:f.read(4*1024*1024),b''):h.update(ch)
    return h.hexdigest()


def atomic_bytes(path, buf):
    dest=Path(path);dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_name(dest.name+'.partial')
    with tmp.open('wb') as f:
        f.write(buf);f.flush();os.fsync(f.fileno())
    # Source file is CLOSED before rename. Avoid open_memmap / WinError32.
    for trial in range(12):
        try:os.replace(tmp,dest);return
        except PermissionError:
            if trial==11:raise
            time.sleep(.2*(trial+1))


def atomic_json(path,obj):
    atomic_bytes(path,json.dumps(obj,ensure_ascii=False,indent=2,default=str).encode('utf-8'))


def atomic_npy(path,array):
    import numpy as np
    buf=io.BytesIO();np.save(buf,np.asarray(array),allow_pickle=False)
    atomic_bytes(path,buf.getvalue())


def csv_write(path,rows,cols=None):
    rows=list(rows);cols=list(cols or (rows[0].keys() if rows else ['status']))
    s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=cols,extrasaction='ignore')
    w.writeheader();w.writerows(rows)
    atomic_bytes(path,s.getvalue().encode('utf-8-sig'))


def parse_annotations(path,split):
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(obj,list):raise RuntimeError('reid_raw.json is not a list')
    images,texts=[],[]
    for r in obj:
        if r.get('split')!=split:continue
        if 'id' not in r or not r.get('captions') or not r.get('file_path'):
            raise RuntimeError('Incomplete CUHK-PEDES annotations')
        j=len(images);images.append({'pid':int(r['id']),'path':str(r['file_path'])})
        for txt in r['captions']:
            texts.append({'pid':int(r['id']),'image':j,'text':str(txt)})
    if not images or not texts:raise RuntimeError('Empty train split')
    return images,texts


def resolve_data(root,cfg,explicit):
    root=Path(root)
    import yaml
    config=yaml.load(Path(cfg).read_text(encoding='utf-8'),Loader=yaml.FullLoader)
    candidates=[]
    for loc in [explicit,config.get('root_dir',''),root/'datasets',root/'data',root.parent/'data']:
        if not loc:continue
        p=Path(loc)
        if p.name.lower()!='cuhk-pedes':p=p/'CUHK-PEDES'
        if p not in candidates:candidates.append(p)
    for p in candidates:
        ann=p/'reid_raw.json';images=p/'imgs'
        if ann.is_file() and images.is_dir():return ann,images,config
    raise FileNotFoundError('Cannot locate CUHK-PEDES/reid_raw.json and imgs. Checked: '+str(candidates))


def select_pids(texts,k,seed,fit_ratio,split_seed):
    import numpy as np
    ids=sorted({r['pid'] for r in texts})
    if len(ids)<8:raise RuntimeError('Fewer than 8 Train identities')
    rng=np.random.default_rng(seed)
    selected=sorted(map(int,rng.choice(ids,size=min(k,len(ids)),replace=False))) if k else ids
    shuffled=np.random.default_rng(split_seed).permutation(selected)
    count=max(4,min(len(selected)-2,round(len(selected)*fit_ratio)))
    fit=sorted(map(int,shuffled[:count]));hold=sorted(map(int,shuffled[count:]))
    if set(fit)&set(hold):raise AssertionError('PID overlap')
    return selected,fit,hold


def normal_np(x):
    import numpy as np
    x=np.asarray(x,dtype=np.float32)
    return x/np.maximum(np.sqrt((x*x).sum(axis=-1,keepdims=True)),1e-9)


def encode_model(root,cfgpath,ckptpath,nclasses,device):
    """Official IRRA architecture, no OpenAI pretrained download, all keys strict."""
    import torch
    import yaml
    root=Path(root)
    sys.path.insert(0,str(root)) if str(root) not in sys.path else None
    opt=SimpleNamespace(**yaml.load(Path(cfgpath).read_text(encoding='utf-8'),Loader=yaml.FullLoader))
    opt.img_size=tuple(opt.img_size)
    checkpoint=torch.load(str(ckptpath),map_location='cpu',weights_only=False)
    if 'model' not in checkpoint or not isinstance(checkpoint['model'],dict):
        raise RuntimeError('Not an official IRRA model checkpoint')
    state=checkpoint['model'];epoch=checkpoint.get('epoch')
    if epoch is not None and int(epoch)!=58:
        raise RuntimeError(f'Expected Epoch58; checkpoint epoch={epoch}')
    if state and all(k.startswith('module.') for k in state):
        state={k[7:]:v for k,v in state.items()}
    if 'base_model.text_projection' not in state or 'classifier.weight' not in state:
        raise RuntimeError('Checkpoint incompatible with official IRRA')
    if state['classifier.weight'].shape[0]!=nclasses:
        raise RuntimeError('Train IDs and checkpoint classifier shapes mismatch')
    def shape(name):return state['base_model.'+name].shape
    img_layers={int(m.group(1)) for k in state if (m:=re.match(r'base_model\.visual\.transformer\.resblocks\.(\d+)\.',k))}
    txt_layers={int(m.group(1)) for k in state if (m:=re.match(r'base_model\.transformer\.resblocks\.(\d+)\.',k))}
    shape_params={'embed_dim':int(shape('text_projection')[1]), 'image_resolution':opt.img_size,
        'vision_layers':len(img_layers),'vision_width':int(shape('visual.conv1.weight')[0]),
        'vision_patch_size':int(shape('visual.conv1.weight')[-1]),'stride_size':int(opt.stride_size),
        'context_length':int(shape('positional_embedding')[0]),
        'vocab_size':int(shape('token_embedding.weight')[0]),
        'transformer_width':int(shape('ln_final.weight')[0]),
        'transformer_heads':int(shape('ln_final.weight')[0])//64,
        'transformer_layers':len(txt_layers)}
    if len(img_layers)!=12 or len(txt_layers)!=12:raise RuntimeError('Not trained ViT-B/16')
    build=importlib.import_module('model.build')
    from model.clip_model import CLIP
    old=build.build_CLIP_from_openai_pretrained
    build.build_CLIP_from_openai_pretrained=lambda *_:(CLIP(**shape_params),{'embed_dim':shape_params['embed_dim']})
    try:model=build.build_model(opt,num_classes=nclasses)
    finally:build.build_CLIP_from_openai_pretrained=old
    if device.type=='cpu':model.float()
    model.load_state_dict(state,strict=True)
    del checkpoint,state
    model.to(device).eval()
    for param in model.parameters():param.requires_grad_(False)
    return model,opt,shape_params['embed_dim']


def tokenize_batch(tokenizer,texts,maxlen):
    import torch
    start=tokenizer.encoder['<|startoftext|>'];stop=tokenizer.encoder['<|endoftext|>']
    result=[]
    for txt in texts:
        seq=[start]+tokenizer.encode(txt)+[stop]
        if len(seq)>maxlen:seq=seq[:maxlen];seq[-1]=stop
        result.append(seq+[0]*(maxlen-len(seq)))
    return torch.tensor(result,dtype=torch.long)


def stage_encode(run):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader,Dataset
    from PIL import Image
    import torchvision.transforms as transforms
    from utils.simple_tokenizer import SimpleTokenizer
    cfg=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    imgs,tx=parse_annotations(cfg['annotation'],'train')
    ids=set(cfg['selected_pids'])
    picked_images=[j for j,r in enumerate(imgs) if r['pid'] in ids]
    mapper={original:local for local,original in enumerate(picked_images)}
    pi=[imgs[j] for j in picked_images]
    pt=[{'pid':r['pid'],'image':mapper[r['image']],'text':r['text']} for r in tx if r['pid'] in ids]
    if any(not (Path(cfg['imgs'])/r['path']).is_file() for r in pi):
        raise FileNotFoundError('Some selected images do not exist; check imgs path')
    # Cache only when both arrays + completion metadata + matching fingerprint exist.
    meta=run/'features_meta.json'
    if meta.is_file():
        info=json.loads(meta.read_text(encoding='utf-8'))
        paths=[run/'caption_features.npy',run/'image_features.npy',run/'prompt_features.npy',
               run/'prompt_metadata.json',run/'dataset_indices.json']
        if not all(p.is_file() for p in paths):
            log('Incomplete encoding after interruption; regenerate feature set')
        elif info=={'signature':cfg['signature'],'texts':len(pt),'images':len(pi),'dim':512,'complete':True}:
            a=np.load(paths[0],mmap_mode='r',allow_pickle=False)
            b=np.load(paths[1],mmap_mode='r',allow_pickle=False)
            c=np.load(paths[2],mmap_mode='r',allow_pickle=False)
            parsed=json.loads(paths[4].read_text(encoding='utf-8'))
            if (a.shape==(len(pt),512) and b.shape==(len(pi),512) and
                c.shape==(86,512) and parsed.get('captions')==pt and
                parsed.get('images')==pi and np.isfinite(a).all() and
                np.isfinite(b).all() and np.isfinite(c).all()):
                log('Validated complete feature cache: skip encoding');return
            raise RuntimeError('Existing feature cache shape/content mismatch')
        else:
            raise RuntimeError('Existing feature cache fingerprint mismatch')
    device=torch.device('cuda' if (cfg['device']=='auto' and torch.cuda.is_available()) or cfg['device']=='cuda' else 'cpu')
    if cfg['device']=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA not available')
    all_train_ids=len({r['pid'] for r in tx})
    model,opt,dim=encode_model(cfg['root'],cfg['config'],cfg['checkpoint'],all_train_ids,device)
    if dim!=512:raise RuntimeError(f'Unexpected feature dimension {dim}')
    tokenizer=SimpleTokenizer()
    tfeatures=np.empty((len(pt),dim),np.float32)
    for start in range(0,len(pt),cfg['text_batch']):
        part=pt[start:start+cfg['text_batch']]
        token=tokenize_batch(tokenizer,[r['text'] for r in part],int(opt.text_length)).to(device)
        with torch.inference_mode():f=model.encode_text(token)
        tfeatures[start:start+len(part)]=normal_np(f.float().cpu().numpy())
        if (start//cfg['text_batch'])%60==0:log(f'Encoded captions {start+len(part)}/{len(pt)}')
    class ImgDataset(Dataset):
        def __init__(self):
            self.tfm=transforms.Compose([transforms.Resize(tuple(opt.img_size)),transforms.ToTensor(),
                transforms.Normalize(mean=(.48145466,.4578275,.40821073),std=(.26862954,.26130258,.27577711))])
        def __len__(self):return len(pi)
        def __getitem__(self,j):
            with Image.open(Path(cfg['imgs'])/pi[j]['path']) as im:
                return self.tfm(im.convert('RGB'))
    loader=DataLoader(ImgDataset(),batch_size=cfg['image_batch'],shuffle=False,num_workers=0)
    vfeatures=np.empty((len(pi),dim),np.float32);count=0
    with torch.inference_mode():
        for pixels in loader:
            f=model.encode_image(pixels.to(device));a=normal_np(f.float().cpu().numpy())
            vfeatures[count:count+len(a)]=a;count+=len(a)
            if count%(cfg['image_batch']*90)<len(a) or count==len(pi):log(f'Encoded images {count}/{len(pi)}')
    if count!=len(pi):raise RuntimeError('Images truncated')
    # A fixed, public list of minimal attribute prompts; no PID or Val calibration.
    prompts=[];prompt_meta=[]
    for slot in SLOT_ORDER:
        for color in ('black','white','red','blue','green','yellow','gray','brown','pink','purple','orange','khaki') + (('blonde','auburn') if slot=='hair' else ()):
            prompts.append(SLOT_TEMPLATES[slot].format(value=color))
            prompt_meta.append({'slot':slot,'value':color})
    pf=[]
    for start in range(0,len(prompts),cfg['text_batch']):
        tokens=tokenize_batch(tokenizer,prompts[start:start+cfg['text_batch']],int(opt.text_length)).to(device)
        with torch.inference_mode():pf.append(normal_np(model.encode_text(tokens).float().cpu().numpy()))
    prompt_features=np.concatenate(pf,axis=0)
    for label,a in [('caption_features',tfeatures),('image_features',vfeatures),('prompt_features',prompt_features)]:
        if not np.isfinite(a).all():raise RuntimeError('Nonfinite '+label)
        atomic_npy(run/(label+'.npy'),a)
    atomic_json(run/'prompt_metadata.json',prompt_meta)
    atomic_json(run/'dataset_indices.json',{'images':pi,'captions':pt})
    atomic_json(meta,{'signature':cfg['signature'],'texts':len(pt),'images':len(pi),'dim':dim,'complete':True})
    log('Frozen IRRA feature encoding completed; best.pth was only read.')


def canonical(color):
    c=COLOR_CANON.get(color.lower(),color.lower())
    return COLOR_BASE.get(c,c)


def parse_attrs(text):
    """Conservative English lexical parse; unresolved/ambiguous -> unknown.

    Returns per-slot values, explicit positive/negative presence and spans.
    Does not mistake missing mention for explicit absence. If two inconsistent
    values occur in one caption, that slot is marked ambiguous and excluded.
    """
    norm=text.lower()
    slots={};spans={};ambiguous=[]
    for m in ATTR_RX.finditer(norm):
        color=canonical(m.group('color'));noun=m.group('noun')
        slot=next((k for k,terms in OBJECTS.items() if noun in terms),None)
        if slot is None:continue
        neg_context=norm[max(0,m.start()-24):m.start()]
        if NEG_RX.search(neg_context):continue
        if slot in slots and slots[slot]!=color:
            ambiguous.append(slot)
        else:
            slots[slot]=color;spans[slot]=(m.start(),m.end())
    for s in ambiguous:slots.pop(s,None);spans.pop(s,None)
    presence={}
    for m in NOUN_RX.finditer(norm):
        noun=m.group(1);slot=next((k for k,terms in OBJECTS.items() if noun in terms),None)
        if slot not in ('bag','hat'):continue
        ctx=norm[max(0,m.start()-28):m.start()]
        neg=bool(re.search(r'(?:\bno\s+|\bwithout\s+(?:a\s+)?|\bnot\s+wearing\s+(?:a\s+)?|\bdoes\s+not\s+(?:have|carry|wear)\s+(?:a\s+)?)$',ctx))
        val='absent' if neg else 'present'
        if slot in presence and presence[slot]!=val:presence[slot]='unknown'
        else:presence[slot]=val
    return {'values':slots,'presence':presence,'spans':spans,'ambiguous':sorted(set(ambiguous))}


def compare_attrs(anchor,support):
    a=parse_attrs(anchor);b=parse_attrs(support)
    novel=[];repeat=[];conflict=[];uncertain=[]
    for slot,val in b['values'].items():
        if slot in a['values']:
            av=a['values'][slot]
            (repeat if av==val else conflict).append({'slot':slot,'value':val,'anchor':av})
        else:novel.append({'slot':slot,'value':val})
    for slot,val in b['presence'].items():
        av=a['presence'].get(slot)
        if val=='unknown':uncertain.append({'slot':slot,'value':val});continue
        if av is not None and av!='unknown':
            (repeat if av==val else conflict).append({'slot':slot,'value':val,'anchor':av})
        elif val=='absent':
            uncertain.append({'slot':slot,'value':val})
    return {'novel':novel,'repeated':repeat,'conflict':conflict,'uncertain':uncertain,
            'anchor_parsed':a,'support_parsed':b}


def augment_caption(text,mode,rng):
    """Synthetic lexical controls, labeled only by construction."""
    p=parse_attrs(text)
    if not p['spans']:return None
    slot=rng.choice(sorted(p['spans']))
    st,en=p['spans'][slot]
    if mode=='remove':
        # Caption remains recognizable, but grammar may be imperfect: report caveat.
        new=(text[:st]+text[en:]).replace('  ',' ').strip()
        return (new,slot,p['values'][slot]) if new and new!=text else None
    if mode=='corrupt':
        old=p['values'][slot]
        pool=['red','blue','green','black','white','yellow','brown','gray']
        choices=[c for c in pool if c!=old]
        if not choices:return None
        alt=rng.choice(choices)
        span=text[st:en]
        m=ATTR_RX.search(span.lower())
        if not m:return None
        changed=span[:m.start('color')]+alt+span[m.end('color'):]
        return (text[:st]+changed+text[en:],slot,old)
    raise ValueError(mode)


def image_evidence(v, prompts, prompt_index, item, threshold):
    """IRRA global prompts: winner must equal requested color AND be confident.

    Calibration uses top-1 gap vs caption-mentioned ground proxy; it does NOT
    judge correctness using the same signed target-vs-other margin (tautology).
    """
    import numpy as np
    slot=item['slot'];color=item['value']
    choices=prompt_index.get(slot)
    if choices is None or color not in choices:
        return {'status':'unknown','margin':None,'winner':None,'top_gap':None}
    ids=list(choices)
    scores=np.asarray([float(v@prompts[j]) for j in choices.values()],dtype=np.float64)
    rank=np.argsort(-scores)
    winner=ids[int(rank[0])]
    target_idx=ids.index(color)
    target=float(scores[target_idx]);competitor=float(max(
        (s for j,s in enumerate(scores) if j!=target_idx),default=target))
    margin=target-competitor
    gap=float(scores[rank[0]]-scores[rank[1]]) if len(rank)>1 else 0.
    valid=winner==color and gap>=threshold
    return {'status':'supported' if valid else 'insufficient',
            'margin':round(margin,7),'top_gap':round(gap,7),
            'winner':winner,'target_score':round(target,6),
            'best_other_score':round(competitor,6)}


def pair_decision(anchor,support,anchor_vec,source_vec,prompts,prompt_idx,thresholds):
    d=compare_attrs(anchor,support)
    observations=[]
    for item in d['novel']:
        if item['value'] in ('present','absent'):
            target={'status':'unknown','margin':None,'winner':None}
            origin={'status':'unknown','margin':None,'winner':None}
        else:
            threshold=thresholds.get(item['slot'],1e9)
            target=image_evidence(anchor_vec,prompts,prompt_idx,item,threshold)
            origin=image_evidence(source_vec,prompts,prompt_idx,item,threshold)
        observations.append({**item,'target_A':target,'source_B':origin})
    # Same-PID different-view caption mismatch is incompatible for TRANSFER,
    # not proof that the source caption is factually false.
    if d['conflict']:
        decision='reject_conflicting_transfer'
    elif any(o['target_A']['status']=='supported' and
             o['source_B']['status']=='supported' for o in observations):
        decision='accept_evidenced_novelty'
    elif not d['novel'] and d['repeated'] and not d['uncertain']:
        decision='skip_redundant'
    else:
        decision='abstain_insufficient_evidence'
    return {'decision':decision,'novel':d['novel'],'repeated':d['repeated'],
            'conflict':d['conflict'],'uncertain':d['uncertain'],
            'observations':observations}


def learn_thresholds(image_feats,captions,fit_ids,prompts,prompt_idx,target_precision=.8,min_per_slot=20):
    """Fit-PID-only calibration from caption-explicit colors. Proxy labels only.

    If a slot cannot reach target caption-agreement precision, set +inf and
    abstain. Never select thresholds on holdout, Val or Test.
    """
    import numpy as np
    data=defaultdict(list)
    for r in captions:
        if r['pid'] not in fit_ids:continue
        p=parse_attrs(r['text'])
        for slot,col in p['values'].items():
            if col not in prompt_idx.get(slot,{}):continue
            spec={'slot':slot,'value':col}
            ev=image_evidence(image_feats[r['image']],prompts,prompt_idx,spec,-1e9)
            data[slot].append((ev['top_gap'],ev['winner']==col))
    out={};diag={}
    for slot in SLOT_ORDER:
        values=data[slot]
        if len(values)<min_per_slot:
            out[slot]=1e9;diag[slot]={'n':len(values),'usable':False,'reason':'not_enough_caption_proxy_samples'};continue
        ms=np.array([a for a,b in values]);ok=np.array([b for a,b in values])
        # Candidate thresholds fixed by fit data only; correctness is
        # independently measured against the caption-explicit value.
        # A high-confidence *wrong* winning color counts as an error.
        chosen=None
        for threshold in sorted(set([0.0]+[float(x) for x in np.quantile(ms,[.25,.50,.65,.75,.85,.90,.95,.98])])):
            sel=ms>=max(0.,threshold)
            if sel.sum()>=max(8,round(len(values)*.01)) and ok[sel].mean()>=target_precision:
                chosen=float(max(0.,threshold));break
        out[slot]=chosen if chosen is not None else 1e9
        active=ms>=out[slot]
        diag[slot]={'n':len(values),'threshold':out[slot] if chosen is not None else None,
                    'usable':chosen is not None,'selected':int(active.sum()),
                    'fit_caption_agreement':float(ok[active].mean()) if active.any() else None,
                    'all_caption_agreement':float(ok.mean()),
                    'warning':'Caption mentions are noisy proxies, not image truth'}
    return out,diag


def choose_wrong_support(anchor_idx, captions, features, pid_candidates, rng,
                         target_similarity, target_length, count=128):
    """Match wrong-PID candidate to a real same-PID support cosine and length.

    Identity only constructs the support pair; never used by the selector.
    Save residual cosine-matching error rather than claiming exact matching.
    """
    import numpy as np
    pool=pid_candidates
    if not len(pool):return None
    positions=rng.choice(pool,size=min(count,len(pool)),replace=False)
    positions=[int(j) for j in positions
               if captions[int(j)]['pid']!=captions[anchor_idx]['pid']]
    if not positions:return None
    similarities=features[np.array(positions)]@features[anchor_idx]
    lengths=np.array([len(captions[j]['text'].split()) for j in positions],dtype=float)
    score=np.abs(similarities-target_similarity)+.015*np.abs(lengths-target_length)/max(1,target_length)
    return positions[int(np.argmin(score))]


def build_rows(run):
    import numpy as np
    cfg=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    data=json.loads((run/'dataset_indices.json').read_text(encoding='utf-8'))
    imgs=data['images'];texts=data['captions'];fit=set(cfg['fit_pids']);hold=set(cfg['holdout_pids'])
    tf=np.load(run/'caption_features.npy',mmap_mode='r',allow_pickle=False)
    vf=np.load(run/'image_features.npy',mmap_mode='r',allow_pickle=False)
    pf=np.load(run/'prompt_features.npy',mmap_mode='r',allow_pickle=False)
    metas=json.loads((run/'prompt_metadata.json').read_text(encoding='utf-8'))
    if (tf.shape[0]!=len(texts) or vf.shape[0]!=len(imgs) or
        pf.shape[0]!=len(metas) or tf.ndim!=2 or vf.ndim!=2 or pf.ndim!=2 or
        not (tf.shape[1]==vf.shape[1]==pf.shape[1])):
        raise RuntimeError('Feature shape/manifest mismatch')
    pm=defaultdict(dict)
    for j,r in enumerate(metas):pm[r['slot']][r['value']]=j
    thresholds,diag=learn_thresholds(vf,texts,fit,pf,pm,cfg['precision_target'],min_per_slot=cfg['min_slot_samples'])
    atomic_json(run/'visual_calibration.json',{'fit_only':True,'target_precision':cfg['precision_target'],
        'fit_diagnostics':diag,'thresholds':{k:(v if v<1e8 else None) for k,v in thresholds.items()}})
    by_pid=defaultdict(list)
    for j,r in enumerate(texts):by_pid[r['pid']].append(j)
    # A deterministic, strict fit/holdout identity split.
    rows=[]
    rng=np.random.default_rng(cfg['pair_seed'])
    for group,pids,cap in [('fit',fit,cfg['fit_query_limit']),('holdout',hold,cfg['holdout_query_limit'])]:
        eligible=np.array([i for i,r in enumerate(texts) if r['pid'] in pids],dtype=int)
        if cap and len(eligible)>cap:eligible=rng.choice(eligible,cap,replace=False)
        wrong_pool=np.array([i for i,r in enumerate(texts) if r['pid'] in pids],dtype=int)
        for n,ai in enumerate(eligible):
            ai=int(ai);a=texts[ai];pid=a['pid'];anchor=a['text'];ivec=vf[a['image']]
            own=[j for j in by_pid[pid] if j!=ai and texts[j]['image']!=a['image']]
            if own:
                si=int(rng.choice(own));s=texts[si]
                true_cos=float(tf[ai]@tf[si])
                rows.append(_row(group,'real_same_pid_cross_image',ai,si,a,s,ivec,
                    vf[s['image']],pf,pm,thresholds,'unlabeled_real_pair',true_cos))
                wi=choose_wrong_support(ai,texts,tf,wrong_pool,rng,true_cos,len(s['text'].split()))
                if wi is not None:
                    w=texts[wi]
                    if w['pid']==pid:raise AssertionError('WrongPID construction leakage')
                    rows.append(_row(group,'near_wrong_pid',ai,wi,a,w,ivec,
                        vf[w['image']],pf,pm,thresholds,'unlabeled_real_pair',
                        float(tf[ai]@tf[wi]),matched_to_similarity=true_cos))
            # Synthetic controls do not make claims about independent human truth.
            rows.append(_row(group,'synthetic_redundant',ai,ai,a,a,ivec,ivec,pf,pm,thresholds,
                'redundant_by_construction',1.0))
            useful=augment_caption(anchor,'remove',rng)
            if useful:
                shortened,slot,value=useful
                acopy=dict(a);acopy['text']=shortened
                rows.append(_row(group,'synthetic_novel_caption_restore',ai,ai,acopy,a,ivec,ivec,pf,pm,thresholds,
                    'novel_by_construction',None,source_anchor=anchor,design_slot=slot))
            harmful=augment_caption(anchor,'corrupt',rng)
            if harmful:
                corrupted,slot,value=harmful
                scopy=dict(a);scopy['text']=corrupted
                rows.append(_row(group,'synthetic_conflicting_color',ai,ai,a,scopy,ivec,ivec,pf,pm,thresholds,
                    'conflict_by_construction',None,design_slot=slot))
            if (n+1)%700==0:log(f'Analyzed {group}: {n+1}/{len(eligible)} anchor texts')
    # The available 'near wrong' is a semantic similarity control, not automatically false properties.
    csvcols=['group','type','reference','anchor_idx','support_idx','anchor_pid_train_only',
        'support_pid_train_only','anchor_image','support_image','anchor_text','support_text',
        'original_anchor_before_synthetic','semantic_similarity','similarity_match_error','design_slot','n_novel','n_repeat',
        'n_conflict','n_uncertain','decision','novel','repeated','conflict','uncertain','visual_observations']
    csv_write(run/'all_pair_decisions.csv',rows,csvcols)
    stats=make_stats(rows,diag,cfg)
    atomic_json(run/'selector_diagnostics.json',stats)
    summary=[]
    for grp,group_stats in stats['groups'].items():
        for typ,t in group_stats['per_type'].items():
            summary.append({'group':grp,'type':typ,'count':t['n'],
                'accept_rate':t['accept_rate'],
                'explicit_reject_rate':t['explicit_reject_rate'],
                'refrain_from_accepting_rate':t['refrain_from_accepting_rate'],
                'mean_approx_similarity_match_error':t['mean_approx_similarity_match_error'],
                'controlled_dual_harmonic':group_stats['controlled_dual_objective_harmonic']})
    csv_write(run/'decision_summary.csv',summary)
    coverage=[]
    for grp,pidset in [('fit',fit),('holdout',hold)]:
        part=[r for r in texts if r['pid'] in pidset]
        pars=[parse_attrs(r['text']) for r in part]
        n=len(part)
        coverage.append({'group':grp,'caption_count':n,
            'captions_with_color_noun_slot':sum(bool(p['values']) for p in pars),
            'color_noun_parse_coverage':safe_ratio(sum(bool(p['values']) for p in pars),n),
            'captions_with_ambiguous_slot':sum(bool(p['ambiguous']) for p in pars),
            'captions_with_explicit_bag_or_hat':sum(bool(p['presence']) for p in pars)})
    csv_write(run/'semantic_parse_coverage.csv',coverage)
    log(f'Analyzed {len(rows)} pairs. Holdout controlled-useful accept and harmful reject recorded separately.')


def _row(group,typ,ai,si,a,s,ivec,source_vec,pf,pm,thresholds,ref,sim,
         source_anchor='',design_slot='',matched_to_similarity=None):
    result=pair_decision(a['text'],s['text'],ivec,source_vec,pf,pm,thresholds)
    return {'group':group,'type':typ,'reference':ref,'anchor_idx':ai,'support_idx':si,
            'anchor_pid_train_only':a['pid'],'support_pid_train_only':s['pid'],
            'anchor_image':a['image'],'support_image':s['image'],
            'anchor_text':a['text'],'support_text':s['text'],
            'original_anchor_before_synthetic':source_anchor,
            'semantic_similarity':round(float(sim),6) if sim is not None else '',
            'similarity_match_error':round(abs(float(sim)-float(matched_to_similarity)),6)
               if sim is not None and matched_to_similarity is not None else '',
            'design_slot':design_slot,'n_novel':len(result['novel']),
            'n_repeat':len(result['repeated']),'n_conflict':len(result['conflict']),
            'n_uncertain':len(result['uncertain']),'decision':result['decision'],
            'novel':json.dumps(result['novel'],ensure_ascii=False),
            'repeated':json.dumps(result['repeated'],ensure_ascii=False),
            'conflict':json.dumps(result['conflict'],ensure_ascii=False),
            'uncertain':json.dumps(result['uncertain'],ensure_ascii=False),
            'visual_observations':json.dumps(result['observations'],ensure_ascii=False)}


def safe_ratio(num,den):return float(num/den) if den else None


def make_stats(rows,calibration,config):
    groups={}
    for group in ('fit','holdout'):
        subset=[r for r in rows if r['group']==group]
        types={}
        for typ in ['real_same_pid_cross_image','near_wrong_pid', 'synthetic_redundant',
                    'synthetic_novel_caption_restore','synthetic_conflicting_color']:
            cur=[r for r in subset if r['type']==typ]
            counts=Counter(r['decision'] for r in cur)
            match_errors=[float(r['similarity_match_error']) for r in cur if r.get('similarity_match_error')!='']
            types[typ]={'n':len(cur),'decisions':dict(counts),
                'mean_approx_similarity_match_error':safe_ratio(sum(match_errors),len(match_errors)),
                'accept_rate':safe_ratio(counts['accept_evidenced_novelty'],len(cur)),
                'explicit_reject_rate':safe_ratio(counts['reject_conflicting_transfer'],len(cur)),
                'refrain_from_accepting_rate':safe_ratio(len(cur)-counts['accept_evidenced_novelty'],len(cur))}
        useful=types['synthetic_novel_caption_restore']['accept_rate']
        harmful=types['synthetic_conflicting_color']['explicit_reject_rate']
        harmonic=2*useful*harmful/(useful+harmful) if useful is not None and harmful is not None and useful+harmful>0 else None
        groups[group]={'pairs':len(subset),'per_type':types,
            'controlled_dual_objective_harmonic':harmonic,
            'disclaimer':'Controlled labels check rule behavior; not ground-truth accuracy for real same-identity complementarity'}
    return {'protocol':'Train-only PID split; no Val/Test support IDs; frozen best checkpoint',
        'checkpoint_sha256':BEST_SHA,'fit_identities':len(config['fit_pids']),
        'holdout_identities':len(config['holdout_pids']),
        'calibration':calibration,'groups':groups,
        'interpretation':{
            'semantic_novelty':'Conservative lexical color+noun bound slot difference, not learned general NLU',
            'visual_proxy':'IRRA global image/text prompt comparison; not localization nor manually labeled fact',
            'controlled_useful':'Original caption with one color+noun removed; restoration is semantically novel by construction only',
            'controlled_harmful':'Same caption with color replaced; explicit contradiction to original caption, not image truth',
            'real_pairs':'No correctness labels; wrong-PID is approximate cosine/length matched to same-PID support, not automatically false',
            'accept':'Requires novel attribute AND fit-PID calibrated global evidence on BOTH target image A and source image B',
            'reject':'Requires explicit same-slot contradiction in text; no-mention is not false',
            'uncertainty':'Does not turn absence of visual detection into conflict',
            'not_a_retrieval_gain_test':True}}


def stage_examples(run):
    from PIL import Image
    cfg=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    images=json.loads((run/'dataset_indices.json').read_text(encoding='utf-8'))['images']
    with (run/'all_pair_decisions.csv').open('r',encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    rng=random.Random(cfg['case_seed']);cases=[]
    # Stratify: genuine same, similar wrong, and difficult reject/accept examples.
    for typ,n in [('real_same_pid_cross_image',60),('near_wrong_pid',40),
                  ('synthetic_novel_caption_restore',45),('synthetic_conflicting_color',45),
                  ('synthetic_redundant',10)]:
        part=[r for r in rows if r['group']=='holdout' and r['type']==typ]
        rng.shuffle(part)
        cases.extend(part[:n])
    # Generate small self-contained HTML (no external image paths/URLs).
    def preview(idx):
        src=Path(cfg['imgs'])/images[int(idx)]['path']
        if not src.is_file():return '<span>IMAGE MISSING</span>'
        with Image.open(src) as im:
            im=im.convert('RGB');im.thumbnail((128,200))
            raw=io.BytesIO();im.save(raw,'JPEG',quality=67,optimize=True)
        return '<img loading="lazy" alt="train person" src="data:image/jpeg;base64,'+base64.b64encode(raw.getvalue()).decode('ascii')+'"/>'
    css='<style>body{font-family:Arial,sans-serif;max-width:1050px;margin:auto;background:#fafafa;color:#222}article{background:white;border:1px solid #ddd;border-radius:8px;margin:14px 0;padding:14px}figure{display:inline-block;margin:5px;vertical-align:top}img{height:170px}p{margin:6px 0}code{background:#eee;padding:2px}h1{font-size:22px}</style>'
    segments=['<!doctype html><html><head><meta charset="utf-8"><title>EXP11 Human Audit</title>'+css+'</head><body>',
      '<h1>实验11：内部未见训练身份人工核查（200例）</h1>',
      '<p>只使用Train PID构造；标签属于规则/受控扰动，非人工真值。请检查属性绑定、视觉证据、冲突、遮挡与错误接受。</p>']
    for n,r in enumerate(cases,1):
        copy=dict(r);copy['human_review']='';cases[n-1]=copy
        segments.append('<article><h3>'+str(n)+' — '+html.escape(r['type'])+' / '+html.escape(r['decision'])+'</h3>')
        segments.append('<figure>'+preview(r['anchor_image'])+'<figcaption>目标图像 A</figcaption></figure>')
        segments.append('<figure>'+preview(r['support_image'])+'<figcaption>来源图像 B</figcaption></figure>')
        for field in ['anchor_text','support_text','novel','repeated','conflict','visual_observations']:
            segments.append('<p><b>'+field+':</b> '+html.escape(str(r[field]))+'</p>')
        segments.append('<p>人工判断：□可靠新增 □有害 □重复 □不确定 □解析错误</p></article>')
    segments.append('</body></html>')
    atomic_bytes(run/'人工核查200例.html',''.join(segments).encode('utf-8'))
    csv_write(run/'人工核查200例.csv',cases)
    atomic_json(run/'experiment11_report.json',{
        'experiment':EXPERIMENT_NAME,'best_checkpoint_sha256':BEST_SHA,
        'original_irra_modified':False,'trainable_models':0,'selected_cases':len(cases),
        'split':'CUHK-PEDES Train exclusively','evaluation':'Teacher-side mechanism diagnostic only',
        'no_val_or_test_pid':True,
        'study_goals':['reliably identify true new semantic attributes',
                       'accept useful visually evidenced support',
                       'explicitly reject contradictory support while abstaining on uncertainty'],
        'limitation':'Global prompt differences and caption-based controls are proxies, not annotated visual truth.',
        'artifacts':['selector_diagnostics.json','all_pair_decisions.csv','visual_calibration.json','decision_summary.csv','semantic_parse_coverage.csv',
                     '人工核查200例.csv','人工核查200例.html']})
    log(f'Generated {len(cases)} manual review cases')


def package(run):
    include_ext={'.csv','.json','.md','.html','.py','.log'}
    file=run/'发送给ChatGPT.zip'
    pending=file.with_suffix('.partial.zip')
    with zipfile.ZipFile(pending,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(run.iterdir()):
            if p.is_file() and p.suffix in include_ext and p.name!=file.name and not p.name.endswith('.partial'):
                z.write(p,p.name)
    for i in range(10):
        try:os.replace(pending,file);break
        except PermissionError:
            if i==9:raise
            time.sleep(.2*(i+1))
    with zipfile.ZipFile(file) as z:
        assert z.testzip() is None
    log('Analysis ZIP saved:',file)


def launch_stage(stage,run):
    cmd=[sys.executable,str(Path(__file__).resolve()),'--_worker',stage,'--_run-dir',str(run)]
    env=os.environ.copy()
    env['OMP_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2'
    cwd=Path(json.loads((run/'manifest.json').read_text(encoding='utf-8'))['root'])
    env['PYTHONUNBUFFERED']='1'
    with (run/'experiment11_run.log').open('a',encoding='utf-8') as logfile:
        logfile.write('=== STAGE '+stage+' ===\n');logfile.flush()
        proc=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in proc.stdout:
            print(line,end='',flush=True);logfile.write(line);logfile.flush()
        code=proc.wait()
    if code:raise RuntimeError(f'Stage {stage} exited {code}; see experiment11_run.log')


def resume_folder(parent):
    folders=sorted([p for p in parent.glob(EXPERIMENT_NAME+'_20*') if (p/'manifest.json').is_file()],reverse=True)
    if not folders:raise FileNotFoundError('No experiment11 directory found for --resume; run without --resume')
    return folders[0]


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root',default=DEFAULT_ROOT)
    p.add_argument('--dataset-root',default='')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--max-train-pids',type=int,default=2200)
    p.add_argument('--fit-fraction',type=float,default=.8)
    p.add_argument('--selection-seed',type=int,default=42)
    p.add_argument('--split-seed',type=int,default=776)
    p.add_argument('--pair-seed',type=int,default=1109)
    p.add_argument('--case-seed',type=int,default=1111)
    p.add_argument('--fit-query-limit',type=int,default=3000)
    p.add_argument('--holdout-query-limit',type=int,default=2696)
    p.add_argument('--precision-target',type=float,default=.80)
    p.add_argument('--min-slot-samples',type=int,default=20)
    p.add_argument('--image-batch',type=int,default=8)
    p.add_argument('--text-batch',type=int,default=64)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--_worker',default='',help=argparse.SUPPRESS)
    p.add_argument('--_run-dir',default='',help=argparse.SUPPRESS)
    opt=p.parse_args()
    if opt._worker:
        try:
            if opt._worker=='encode':stage_encode(Path(opt._run_dir))
            elif opt._worker=='analyze':build_rows(Path(opt._run_dir))
            elif opt._worker=='cases':stage_examples(Path(opt._run_dir))
            else:raise RuntimeError('Unknown worker')
        except Exception:traceback.print_exc();sys.exit(1)
        return
    if opt.self_test:
        selftest();return
    if not (opt.max_train_pids>=8 and .5<=opt.fit_fraction<=.95 and opt.image_batch>=1
            and opt.text_batch>=1 and 0.5<=opt.precision_target<=1 and opt.min_slot_samples>=1
            and opt.fit_query_limit>=0 and opt.holdout_query_limit>=0):p.error('Invalid settings')
    root=Path(opt.project_root).resolve()
    cfg=root/BEST_FOLDER/'configs.yaml';ckpt=root/BEST_FOLDER/'best.pth'
    for f in [root/'model'/'build.py',root/'model'/'clip_model.py',cfg,ckpt]:
        if not f.is_file():p.error('IRRA file missing '+str(f))
    if sha(ckpt)!=BEST_SHA:p.error('Checkpoint mismatch; NOT the best Epoch58 model')
    if sha(cfg)!=CONFIG_SHA:p.error('Configuration SHA differs from reproduced best run')
    annotation,imgs,yaml_cfg=resolve_data(root,cfg,opt.dataset_root)
    if sha(annotation)!=ANNOTATION_SHA:p.error('Annotation SHA does not match validated CUHK-PEDES dataset')
    train_imgs,train_texts=parse_annotations(annotation,'train')
    ids,fit,hold=select_pids(train_texts,opt.max_train_pids,opt.selection_seed,
                             opt.fit_fraction,opt.split_seed)
    # Strict Train-only diagnostic; never parse Val/Test records or images.
    if len(ids)!=2200 and opt.max_train_pids==2200:raise RuntimeError('Expected 2200 train IDs')
    out_parent=root/'图文双向互补实验';out_parent.mkdir(parents=True,exist_ok=True)
    if opt.resume:
        run=resume_folder(out_parent)
        manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
        for key,want in [('root',str(root)),('checkpoint_sha256',BEST_SHA),('config_sha256',CONFIG_SHA),
                         ('selected_pids',ids),('fit_pids',fit),('holdout_pids',hold),
                         ('version',VERSION),('device',opt.device),
                         ('image_batch',opt.image_batch),('text_batch',opt.text_batch),
                         ('fit_query_limit',opt.fit_query_limit),
                         ('holdout_query_limit',opt.holdout_query_limit),
                         ('precision_target',opt.precision_target),
                         ('min_slot_samples',opt.min_slot_samples),
                         ('pair_seed',opt.pair_seed),('case_seed',opt.case_seed),
                         ('annotation',str(annotation)),('annotation_sha256',ANNOTATION_SHA),
                         ('imgs',str(imgs))]:
            if manifest[key]!=want:raise RuntimeError(f'Resume mismatch: {key}')
        check=dict(manifest);signature=check.pop('signature')
        if hashlib.sha256(json.dumps(check,sort_keys=True).encode()).hexdigest()!=signature:
            raise RuntimeError('Resume manifest signature corrupted')
        log('Resumed experiment11:',run)
    else:
        run=out_parent/(EXPERIMENT_NAME+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        if run.exists():raise RuntimeError('Output folder already exists')
        run.mkdir(parents=True)
        data={'version':VERSION,'root':str(root),'config':str(cfg),'checkpoint':str(ckpt),
              'checkpoint_sha256':BEST_SHA,'config_sha256':CONFIG_SHA,
              'annotation':str(annotation),'annotation_sha256':ANNOTATION_SHA,
              'imgs':str(imgs),'selected_pids':ids,'fit_pids':fit,'holdout_pids':hold,
              'device':opt.device,'image_batch':opt.image_batch,'text_batch':opt.text_batch,
              'fit_query_limit':opt.fit_query_limit,'holdout_query_limit':opt.holdout_query_limit,
              'precision_target':opt.precision_target,'min_slot_samples':opt.min_slot_samples,
              'pair_seed':opt.pair_seed,'case_seed':opt.case_seed}
        data['signature']=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
        atomic_json(run/'manifest.json',data)
        atomic_json(run/'selected_train_pids.json',{'selected':ids,'fit':fit,'internal_unseen':hold})
        log('New EXP11 folder:',run)
    shutil.copy2(Path(__file__).resolve(),run/'irra_exp11_semantic_novelty_visual_reliability.py')
    try:
        for st,finish in [('encode','features_meta.json'),('analyze','selector_diagnostics.json'),
                          ('cases','experiment11_report.json')]:
            # If a completed stage has invalid dependencies, its dedicated worker
            # validates the fingerprint before accepting a cache.
            log('=== Stage:',st,'===')
            if st=='analyze' and (run/finish).is_file() and (run/'all_pair_decisions.csv').is_file() and (run/'visual_calibration.json').is_file():
                log('Completed analysis files found; skip');continue
            if st=='cases' and (run/finish).is_file() and (run/'人工核查200例.html').is_file() and (run/'人工核查200例.csv').is_file():
                log('Completed case exports found; skip');continue
            launch_stage(st,run)
        log('EXP11 complete. No IRRA parameters modified.')
    finally:
        try:package(run)
        except Exception as err:log('Packaging error:',repr(err))


def selftest():
    """Synthetic end-to-end analysis and ZIP test; no Torch or original weights."""
    import numpy as np
    from PIL import Image
    from tempfile import TemporaryDirectory
    assert parse_attrs('a man in a black jacket and blue jeans')['values']=={'outer':'black','lower':'blue'}
    assert compare_attrs('wearing black pants','wearing red pants')['conflict']
    assert compare_attrs('wearing black pants','wearing a black backpack')['novel']
    assert not compare_attrs('a man wearing a black shirt','a man wearing black shirt')['novel']
    assert compare_attrs('a man in black pants','a man in blue jeans')['conflict']
    assert not compare_attrs('a man in black shirt','a man with a backpack')['conflict']
    assert parse_attrs('a man wearing no hat')['presence'].get('hat')=='absent'
    assert parse_attrs('a man wearing a hat')['presence'].get('hat')=='present'
    assert compare_attrs('wearing red shirt','wearing blue pants')['novel']
    assert parse_attrs('woman with blonde hair')['values']['hair']=='blonde'
    assert parse_attrs('black jacket over a white shirt')['values']=={'outer':'black','upper':'white'}
    assert parse_attrs('woman in a red t-shirt')['values']['upper']=='red'
    assert augment_caption('a man in a black shirt','corrupt',random.Random(2))
    with TemporaryDirectory(prefix='irra_exp11_test_') as folder:
        run=Path(folder);imgs=run/'imgs';imgs.mkdir()
        im_records=[];txt_records=[]
        captions=[
            'a man in a black shirt and blue jeans carrying a red backpack',
            'a man in a red jacket and black pants',
            'a woman wearing white shoes and green shirt',
            'a person in a blue shirt and gray trousers',
            'a person wearing yellow coat and black jeans',
            'a lady in pink shirt and white shorts',
            'a woman in black shirt and gray trousers',
            'a person wearing brown coat and blue jeans',
            'a man in green shirt and black shorts',
            'a person wearing white t-shirt and black pants',
        ]
        for pid in range(10):
            for j in range(2):
                name=f'{pid}_{j}.jpg';Image.new('RGB',(70,140),((pid*23)%256,90,140)).save(imgs/name)
                k=len(im_records);im_records.append({'pid':pid,'path':name})
                txt_records.append({'pid':pid,'image':k,'text':captions[pid]})
        d=32
        rng=np.random.default_rng(33)
        # Analysis does not assume feature_dim=512 except live encode cache validation.
        a=normal_np(rng.standard_normal((len(txt_records),d)).astype('float32'))
        b=normal_np(rng.standard_normal((len(im_records),d)).astype('float32'))
        prompt_meta=[{'slot':s,'value':c} for s in SLOT_ORDER for c in
                    ('black','white','red','blue','green','yellow','gray','brown','pink','purple','orange','khaki')+(('blonde','auburn') if s=='hair' else ())]
        pp=normal_np(rng.standard_normal((len(prompt_meta),d)).astype('float32'))
        conf={'version':VERSION,'selected_pids':list(range(10)),'fit_pids':list(range(8)),
              'holdout_pids':[8,9],'signature':'synthetic_only','precision_target':.8,
              'min_slot_samples':2,'fit_query_limit':16,'holdout_query_limit':4,
              'pair_seed':22,'case_seed':11,'imgs':str(imgs),'root':str(run),
              'annotation':str(run/'dummy.json')}
        atomic_json(run/'manifest.json',conf)
        atomic_json(run/'dataset_indices.json',{'images':im_records,'captions':txt_records})
        atomic_npy(run/'caption_features.npy',a);atomic_npy(run/'image_features.npy',b)
        atomic_npy(run/'prompt_features.npy',pp);atomic_json(run/'prompt_metadata.json',prompt_meta)
        # Tests use adjustable dimension in build_rows; live stage fixes it at 512.
        build_rows(run);stage_examples(run);package(run)
        with zipfile.ZipFile(run/'发送给ChatGPT.zip') as z:
            assert z.testzip() is None
            assert 'selector_diagnostics.json' in z.namelist()
            assert '人工核查200例.html' in z.namelist()
            assert 'all_pair_decisions.csv' in z.namelist()
            assert 'decision_summary.csv' in z.namelist()
            assert 'semantic_parse_coverage.csv' in z.namelist()
        stats=json.loads((run/'selector_diagnostics.json').read_text(encoding='utf-8'))
        assert stats['groups']['holdout']['pairs']>0
        assert len(list(csv.DictReader((run/'all_pair_decisions.csv').open(encoding='utf-8-sig'))))>=50
        # Mock interrupted step; rerun analysis and verify deterministic contents.
        before=sha(run/'all_pair_decisions.csv');build_rows(run)
        assert sha(run/'all_pair_decisions.csv')==before
    log('EXP11 SELF-TEST PASSED: text rules, synthetic analysis, cases, ZIP, deterministic retry.')


if __name__=='__main__':main()
