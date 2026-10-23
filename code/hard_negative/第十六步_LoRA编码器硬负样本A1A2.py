# -*- coding: utf-8 -*-
"""IRRA 硬负样本第十六步：编码器 LoRA 公平对照（Windows 单文件）

A1: 同结构 LoRA + 随机异身份负样本 + 同一排序保护
A2: 同结构 LoRA + H1 错排优先/半困难补齐 + 同一排序保护
B0: 完全原始 Epoch58 IRRA，严格复核历史 Val B0；P1 仅历史对照。

此实验不修改旧权重、不使用 Test、不会下载 CLIP 或其他模型。
仅对 IRRA 原生 CLIP 两路 Transformer 最后两层 MLP 的 c_fc/c_proj 注入 LoRA。
冻结原 IRRA 全部原始参数；每次更新使用微批量1、梯度累积4，适配 6GB 显存。
请先在 PyCharm Parameters 填 --self-test；通过后去掉参数执行。
真实 Windows 6GB 环境需本机验证；CPU 合成自检无法保证真实显存峰值。
"""
from __future__ import annotations
import os,sys,json,csv,hashlib,random,zipfile,tempfile,argparse,gc,datetime,traceback
from pathlib import Path
from collections import defaultdict

if sys.platform=='win32':
    os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
    os.environ['MKL_NUM_THREADS']='1'
    os.environ.setdefault('OMP_NUM_THREADS','1')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(r'/')
BEST_DIR=ROOT/'logs'/'CUHK-PEDES'/'20261009_095215_irra_baseline_b8_resume20'
STEP6_DIR=ROOT/'硬负样本分析'/'第六步_最佳IRRA_Epoch58_硬负样本B0B3_20261021_162009'
STEP9_DIR=''    # 空则自动寻找第九步缓存；没有则只使用Train特征重新挖掘
STEP13_DIR=''   # 空则自动找第十三步 P1 汇总；若缺失不会伪造P1指标
RESUME_DIR=''   # 填入本实验某次输出目录实现断点续跑；留空创建全新目录
DEVICE='cuda'
SEEDS=(42,43,44)
STEPS=1000
GRAD_ACCUM=4       # 每个optimizer step积累4条查询；每查询1正图+1负图+可选保护负图
SAVE_EVERY=100
LR=1e-4
WEIGHT_DECAY=0.01
LORA_RANK=8
LORA_SCALE=16.0
LORA_LAST_BLOCKS=2
TEMPERATURE=0.07
PROTECT_WEIGHT=1.0
DISTILL_WEIGHT=0.05
MAX_GRAD_NORM=1.0
EVAL_IMAGE_BATCH=8
EVAL_TEXT_BATCH=64
EXPECTED_CKPT_SHA='f3ab12028e3406546b836ed6a75c8fc5e02a42146c5da13af423e7f5cf0e7ea6'
EXPECTED_CFG_SHA='ec2fae46ec4e0cce06372789afe1c877a5cebeddf73654781e11b0484fc30fac'
EXPECTED_TRAIN_IMAGES=34054
EXPECTED_TRAIN_TEXTS=68126
EXPECTED_VAL_IMAGES=3078
EXPECTED_VAL_TEXTS=6158
DIM=512
VERSION='step16_lora_last2_mlp_hardneg_control_v1_20261022'
METHODS=('A1_random_lora_protect','A2_H1_lora_protect')
ZIP_NAME='上传给ChatGPT_第十六步_LoRA硬负样本编码器训练.zip'


def log(*x):print(*x,flush=True)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def fp(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def atomic(p,fn):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+p.name+'.',suffix='.partial',dir=p.parent);os.close(fd)
    try:fn(Path(tmp));os.replace(tmp,p)
    finally:
        if Path(tmp).exists():Path(tmp).unlink()
def save_json(p,x):
    def write(tmp):
        with tmp.open('w',encoding='utf-8') as f:
            json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    atomic(p,write)
def save_pt(p,obj):atomic(p,lambda tmp:torch.save(obj,str(tmp)))
def save_csv(p,cols,rows):
    def write(tmp):
        with tmp.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    atomic(p,write)


def safe_yaml(path):
    import yaml
    class Loader(yaml.SafeLoader):pass
    def make_tuple(loader,node):return tuple(loader.construct_sequence(node))
    Loader.add_constructor('tag:yaml.org,2002:python/tuple',make_tuple)
    return yaml.load(Path(path).read_text(encoding='utf-8'),Loader=Loader)


def read_train(root):
    path=Path(root)/'CUHK-PEDES'/'reid_raw.json'
    if not path.is_file():raise FileNotFoundError(path)
    rows=json.loads(path.read_text(encoding='utf-8'))
    img_pids=[];img_paths=[];text_pids=[];texts=[];assigned=[]
    for r in rows:
        if r.get('split')!='train':continue
        pid=int(r['id']);rel=str(r['file_path']).replace('\\','/')
        base=path.parent
        choices=[base/'imgs'/rel,base/rel]
        if rel.startswith('imgs/'):choices.insert(0,base/rel)
        im=next((p for p in choices if p.is_file()),None)
        if im is None:raise FileNotFoundError(f'Train图像缺失: {rel}')
        j=len(img_pids);img_pids.append(pid);img_paths.append(str(im))
        for caption in r['captions']:
            text_pids.append(pid);texts.append(str(caption));assigned.append(j)
    if len(img_pids)!=EXPECTED_TRAIN_IMAGES or len(texts)!=EXPECTED_TRAIN_TEXTS:
        raise RuntimeError(f'Train顺序/规模异常：imgs={len(img_pids)} texts={len(texts)}')
    return dict(ip=np.asarray(img_pids,np.int64),paths=img_paths,tp=np.asarray(text_pids,np.int64),texts=texts,
                assigned=np.asarray(assigned,np.int64),annotation_sha=sha(path))


def load_source():
    if sha(BEST_DIR/'best.pth')!=EXPECTED_CKPT_SHA or sha(BEST_DIR/'configs.yaml')!=EXPECTED_CFG_SHA:
        raise RuntimeError('Epoch58 权重或配置 SHA256 不匹配，拒绝训练')
    cfg=safe_yaml(BEST_DIR/'configs.yaml')
    if tuple(cfg['img_size'])!=(384,128) or cfg.get('dataset_name')!='CUHK-PEDES':
        raise RuntimeError('IRRA输入尺寸/数据集配置错误')
    base=Path(STEP6_DIR)
    ident=read_json(base/'00_最佳模型身份与实验设置.json')
    if (ident.get('checkpoint_sha')!=EXPECTED_CKPT_SHA or ident.get('config_sha')!=EXPECTED_CFG_SHA or
            not ident.get('strict_checkpoint')):raise RuntimeError('第六步模型身份不匹配')
    root=ident['dataset_root']
    ann=read_train(root)
    if ann['annotation_sha']!=ident.get('annotation_sha256'):raise RuntimeError('训练标注与原特征缓存不一致')
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    from datasets.cuhkpedes import CUHKPEDES
    ds=CUHKPEDES(root=str(root),verbose=False)
    val=ds.val;qp=np.asarray(val['caption_pids'],dtype=np.int64);vp=np.asarray(val['image_pids'],dtype=np.int64)
    if (len(vp),len(qp))!=(EXPECTED_VAL_IMAGES,EXPECTED_VAL_TEXTS):raise RuntimeError('Val规模不符')
    if set(qp.tolist()) & set(ann['tp'].tolist()):raise RuntimeError('Train/Val PID 重合')
    identity_fp=hashlib.sha256(json.dumps(ident,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    cache=base/'internal_cache';arrays={};hashes={}
    for name,count in [('train_images',len(ann['ip'])),('train_texts',len(ann['tp'])),
                       ('val_images',len(vp)),('val_texts',len(qp))]:
        mp=cache/(name+'.meta.json');npy=cache/(name+'.npy')
        meta=read_json(mp)
        if (meta.get('fingerprint')!=identity_fp+'|'+name or
                meta.get('shape')!=[count,DIM] or meta.get('done')!=count):
            raise RuntimeError(f'{name} 第六步特征缓存未完成或指纹错误')
        a=np.load(npy,allow_pickle=False)   # 完整复制读取，不留Windows memmap句柄
        if a.shape!=(count,DIM) or a.dtype!=np.float32 or not np.isfinite(a).all():
            raise RuntimeError(f'{name} 特征格式错误')
        if float(np.max(np.abs(np.linalg.norm(a,axis=1)-1)))>0.015:raise RuntimeError(name+' 未归一化')
        arrays[name]=a;hashes[name]=sha(npy)
    b0_info=read_json(base/'01_最佳IRRA_官方Val基线.json')
    if not all(b0_info['checks'].values()):raise RuntimeError('第六步B0未复现')
    return dict(cfg=cfg,train=ann,ds=ds,val=val,qp=qp,vp=vp,arrays=arrays,
                identity=ident,source_fp=identity_fp,source_hashes=hashes,base_dir=base,
                B0_record=b0_info['measured'])


def get_positive_bank_indices(src):
    tp=src['train']['tp'];ip=src['train']['ip'];t=src['arrays']['train_texts'];v=src['arrays']['train_images']
    iq=np.argsort(tp,kind='stable');iv=np.argsort(ip,kind='stable')
    uq,qs,qc=np.unique(tp[iq],return_index=True,return_counts=True)
    ui,vs,vc=np.unique(ip[iv],return_index=True,return_counts=True)
    if not np.array_equal(uq,ui):raise RuntimeError('Train PID 集合不一致')
    pos=np.empty(len(tp),np.int32)
    for j in range(len(uq)):
        qi=iq[qs[j]:qs[j]+qc[j]];vi=iv[vs[j]:vs[j]+vc[j]]
        pos[qi]=vi[np.argmax(t[qi]@v[vi].T,axis=1)]
    if np.any(ip[pos]!=tp):raise RuntimeError('最佳正图出现错PID')
    return pos


def mine_top50(src,device):
    t=src['arrays']['train_texts'];v=src['arrays']['train_images'];tp=src['train']['tp'];ip=src['train']['ip']
    k=50;top=np.empty((len(tp),k),np.int32);wrong=np.empty(len(tp),np.int16)
    pmax=np.empty(len(tp),np.float32);nmax=np.empty(len(tp),np.float32)
    vt=torch.from_numpy(np.ascontiguousarray(v)).to(device);ids=torch.from_numpy(ip).to(device)
    with torch.inference_mode():
        for a in range(0,len(t),48):
            b=min(a+48,len(t));qt=torch.from_numpy(np.ascontiguousarray(t[a:b])).to(device)
            s=qt@vt.T;mask=ids[None,:]==torch.as_tensor(tp[a:b],device=device)[:,None]
            good=s.masked_fill(~mask,-float('inf')).max(1).values
            values,idx=s.masked_fill(mask,-float('inf')).topk(k,dim=1,largest=True,sorted=True)
            top[a:b]=idx.cpu().numpy().astype(np.int32)
            pmax[a:b]=good.cpu().numpy();nmax[a:b]=values[:,0].cpu().numpy()
            wrong[a:b]=(values>good[:,None]+1e-7).sum(1).cpu().numpy().astype(np.int16)
            if (a//48)%180==0 or b==len(t):log('Train挖掘',b,'/',len(t))
    del vt,ids
    return dict(top_idx=top,wrong_count=wrong,best_positive=pmax,best_negative=nmax)


def source_wrong_pool(src,out,device):
    p=out/'internal_cache'/'train_wrong_top50.npz'
    ident=fp(dict(source=src['source_fp'],sha=src['source_hashes'],rule='train_only_top50'))
    if p.is_file():
        with np.load(p,allow_pickle=False) as z:
            if str(z['fingerprint'].item())!=ident:raise RuntimeError('Train错排池缓存指纹错误')
            d={k:np.array(z[k],copy=True) for k in ('top_idx','wrong_count','best_positive','best_negative')}
    else:
        d=None
        base=ROOT/'硬负样本分析'
        folders=([Path(STEP9_DIR)] if STEP9_DIR else sorted(base.glob('第九步_错误排序驱动硬负样本_*'),reverse=True))
        for folder in folders:
            old=folder/'internal_cache'/'train_wrong_ahead_top50.npz'
            if not old.exists():continue
            with np.load(old,allow_pickle=False) as z:
                if not {'top_idx','wrong_count','best_positive','best_negative'}.issubset(set(z.files)):continue
                d={k:np.array(z[k],copy=True) for k in ('top_idx','wrong_count','best_positive','best_negative')}
            log('复用第九步 Train 错排池:',old)
            break
        if d is None:
            log('未找到第九步错排池；仅用 Train 特征重新挖掘一次')
            d=mine_top50(src,device)
        def write(tmp):
            with tmp.open('wb') as f:np.savez_compressed(f,fingerprint=np.array(ident),**d)
        atomic(p,write)
    n=len(src['train']['tp']);ip=src['train']['ip'];tp=src['train']['tp']
    if (d['top_idx'].shape!=(n,50) or d['wrong_count'].shape!=(n,) or
        np.any(d['top_idx']<0) or np.any(d['top_idx']>=len(ip)) or
        np.any(ip[d['top_idx']]==tp[:,None]) or
        np.any(d['wrong_count']<0) or np.any(d['wrong_count']>50)):
        raise RuntimeError('错误领先池无效')
    # 检查旧缓存是否确实对应当前完整 Train 文本索引；不能仅凭路径复用
    vv=src['arrays']['train_images'];tt=src['arrays']['train_texts']
    pick=np.unique(np.linspace(0,n-1,min(n,128),dtype=np.int64))
    s=np.einsum('ij,ij->i',tt[pick],vv[d['top_idx'][pick,0]])
    if np.max(np.abs(s-d['best_negative'][pick]))>0.001:
        raise RuntimeError('错排池 top1 评分与原始 Train 特征不匹配')
    positive=get_positive_bank_indices(src)
    ps=np.einsum('ij,ij->i',tt[pick],vv[positive[pick]])
    if np.max(np.abs(ps-d['best_positive'][pick]))>0.001:
        raise RuntimeError('错排池最佳正图分数与第六步缓存不匹配')
    return d,positive


class LoRALinear(nn.Module):
    """原权重冻结；只训练A/B。CLIP的MHA out_proj不能通过替换Linear来注入，故只注入MLP。"""
    def __init__(self,original,rank=8,scale=16):
        super().__init__()
        if not isinstance(original,nn.Linear):raise TypeError('expected nn.Linear')
        self.base=original
        for p in self.base.parameters():p.requires_grad_(False)
        self.A=nn.Parameter(torch.empty(rank,original.in_features,dtype=torch.float32,device=original.weight.device))
        self.B=nn.Parameter(torch.zeros(original.out_features,rank,dtype=torch.float32,device=original.weight.device))
        nn.init.kaiming_uniform_(self.A,a=5**0.5)
        self.mult=float(scale)/rank
    def forward(self,x):
        y=self.base(x)
        # float32适配器参数可训练；只在出口转换回原层dtype，保持fp16 CLIP一致
        d=F.linear(F.linear(x.float(),self.A),self.B)*self.mult
        return y+d.to(y.dtype)


def inject_lora(model,last=LORA_LAST_BLOCKS,rank=LORA_RANK,scale=LORA_SCALE):
    for p in model.parameters():p.requires_grad_(False)
    count=0
    for transformer in (model.base_model.visual.transformer,model.base_model.transformer):
        blocks=transformer.resblocks
        if len(blocks)<last:raise RuntimeError('Transformer层数不足')
        for block in list(blocks)[-last:]:
            for key in ('c_fc','c_proj'):
                old=getattr(block.mlp,key)
                if isinstance(old,LoRALinear):raise RuntimeError('重复注入LoRA')
                setattr(block.mlp,key,LoRALinear(old,rank,scale));count+=1
    if count!=last*2*2:raise RuntimeError('LoRA模块数不符')
    return count


def lora_tensors(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.endswith('.A') or k.endswith('.B')}
def put_lora(model,state):
    required={k for k in model.state_dict() if k.endswith('.A') or k.endswith('.B')}
    if set(state)!=required:raise RuntimeError('LoRA权重键不完整')
    with torch.no_grad():
        own=dict(model.named_parameters())
        for k,v in state.items():
            if own[k].shape!=v.shape:raise RuntimeError('LoRA形状不匹配: '+k)
            own[k].copy_(v.to(device=own[k].device,dtype=own[k].dtype))


def build_irra(src,device):
    from types import SimpleNamespace
    import importlib
    from model.clip_model import CLIP
    build=importlib.import_module('model.build')
    ck=torch.load(str(BEST_DIR/'best.pth'),map_location='cpu',weights_only=False)
    if not isinstance(ck,dict) or 'model' not in ck:raise RuntimeError('Epoch58 checkpoint结构错误')
    if ck.get('epoch') is not None and int(ck['epoch'])!=58:raise RuntimeError('checkpoint不是58轮')
    state=ck['model']
    if all(k.startswith('module.') for k in state):state={k[7:]:v for k,v in state.items()}
    def shp(x):return state['base_model.'+x].shape
    import re
    vi={int(a) for k in state if (m:=re.match(r'base_model\.visual\.transformer\.resblocks\.(\d+)\.',k)) for a in [m.group(1)]}
    te={int(a) for k in state if (m:=re.match(r'base_model\.transformer\.resblocks\.(\d+)\.',k)) for a in [m.group(1)]}
    if len(vi)!=12 or len(te)!=12:raise RuntimeError('非12层ViT/文本编码器')
    cfg=src['cfg']
    structure=dict(embed_dim=int(shp('text_projection')[1]),image_resolution=tuple(cfg['img_size']),
        vision_layers=len(vi),vision_width=int(shp('visual.conv1.weight')[0]),
        vision_patch_size=int(shp('visual.conv1.weight')[-1]),stride_size=int(cfg['stride_size']),
        context_length=int(shp('positional_embedding')[0]),vocab_size=int(shp('token_embedding.weight')[0]),
        transformer_width=int(shp('ln_final.weight')[0]),
        transformer_heads=int(shp('ln_final.weight')[0])//64,transformer_layers=len(te))
    original=build.build_CLIP_from_openai_pretrained
    build.build_CLIP_from_openai_pretrained=lambda *_:(CLIP(**structure),{'embed_dim':structure['embed_dim']})
    try:net=build.build_model(SimpleNamespace(**cfg),num_classes=len(src['ds'].train_id_container))
    finally:build.build_CLIP_from_openai_pretrained=original
    if device.type=='cpu':net.float()
    net.load_state_dict(state,strict=True)
    del ck,state
    net.to(device).eval()
    for p in net.parameters():p.requires_grad_(False)
    return net


def make_train_io(src):
    from utils.simple_tokenizer import SimpleTokenizer
    from datasets.bases import tokenize
    from utils.iotools import read_image
    from datasets.build import build_transforms
    from functools import lru_cache
    tok=SimpleTokenizer()
    transform=build_transforms(img_size=tuple(src['cfg']['img_size']),is_train=False)
    ann=src['train'];texts=ann['texts'];paths=ann['paths']
    @lru_cache(maxsize=16000)
    def text_at(i):return tokenize(texts[int(i)],tok,text_length=int(src['cfg']['text_length']))
    @lru_cache(maxsize=32)
    def image_at(i):return transform(read_image(paths[int(i)]))
    def get_text(i):return text_at(int(i)).clone()
    def get_image(i):return image_at(int(i)).clone()
    return get_text,get_image


def get_val_io(src):
    from datasets.bases import ImageDataset,TextDataset
    from datasets.build import build_transforms
    from torch.utils.data import DataLoader
    transform=build_transforms(img_size=tuple(src['cfg']['img_size']),is_train=False)
    v=src['val'];n=int(src['cfg']['text_length'])
    iv=ImageDataset(v['image_pids'],v['img_paths'],transform)
    tv=TextDataset(v['caption_pids'],v['captions'],text_length=n)
    return (DataLoader(iv,batch_size=EVAL_IMAGE_BATCH,shuffle=False,num_workers=0),
            DataLoader(tv,batch_size=EVAL_TEXT_BATCH,shuffle=False,num_workers=0))


def score_gallery(text,image,device,qp,vp):
    from utils.metrics import rank
    scores=[];gallery=torch.from_numpy(np.ascontiguousarray(image)).to(device)
    with torch.no_grad():
        for i in range(0,len(text),256):
            tx=torch.from_numpy(np.ascontiguousarray(text[i:i+256])).to(device)
            scores.append((tx@gallery.T).float().cpu().numpy())
    sim=np.concatenate(scores);del gallery
    cmc,ap,inp,_=rank(torch.from_numpy(sim),torch.as_tensor(qp),torch.as_tensor(vp),max_rank=10,get_mAP=True)
    order=np.argsort(-sim,axis=1,kind='stable')
    same=vp[order]==qp[:,None]
    if np.any(same.sum(1)==0):raise RuntimeError('Val查询没有正样本')
    ranks=np.argmax(same,axis=1)+1
    k=np.arange(1,len(vp)+1,dtype=np.float32)
    count=same.sum(1)
    aps=(np.cumsum(same,axis=1)/k*same).sum(1)/count
    inps=count/(len(vp)-np.argmax(same[:,::-1],axis=1))
    metrics=dict(R1=float(cmc[0]),R5=float(cmc[4]),R10=float(cmc[9]),mAP=float(ap),mINP=float(inp))
    independent=dict(R1=float(100*np.mean(ranks<=1)),R5=float(100*np.mean(ranks<=5)),
                     R10=float(100*np.mean(ranks<=10)),mAP=float(100*np.mean(aps)),mINP=float(100*np.mean(inps)))
    for key in metrics:
        if abs(metrics[key]-independent[key])>0.035:raise RuntimeError(f'指标复算不一致 {key} official={metrics[key]} independent={independent[key]}')
    return metrics,dict(rank=ranks,AP=aps,INP=inps)


def eval_model(model,src,device,loaders=None):
    if loaders is None:loaders=get_val_io(src)
    model.eval();ims=[];txs=[]
    with torch.inference_mode():
        for _,images in loaders[0]:
            x=F.normalize(model.encode_image(images.to(device)).float(),dim=1)
            ims.append(x.cpu().numpy())
        for _,tokens in loaders[1]:
            x=F.normalize(model.encode_text(tokens.to(device)).float(),dim=1)
            txs.append(x.cpu().numpy())
    iv=np.concatenate(ims);tv=np.concatenate(txs)
    if iv.shape!=(len(src['vp']),DIM) or tv.shape!=(len(src['qp']),DIM):raise RuntimeError('Val输出维度错误')
    if not np.isfinite(iv).all() or not np.isfinite(tv).all():raise RuntimeError('特征非有限值')
    return score_gallery(tv,iv,device,src['qp'],src['vp'])


def check_initial_features(model,src,device,get_text,get_image):
    model.eval()
    samples=np.unique(np.linspace(0,len(src['train']['tp'])-1,6,dtype=np.int64))
    with torch.inference_mode():
        for j in samples:
            text=F.normalize(model.encode_text(get_text(j).to(device)[None,:]).float(),dim=1)[0]
            old=torch.from_numpy(src['arrays']['train_texts'][j].copy()).to(device)
            cos=F.cosine_similarity(text[None],old[None]).item()
            if cos<0.999:raise RuntimeError(f'初始文本特征与第六步缓存不符: idx={j}, cosine={cos}')
        ids=np.unique(src['train']['assigned'][samples])
        for j in ids:
            im=F.normalize(model.encode_image(get_image(j).to(device)[None,:]).float(),dim=1)[0]
            old=torch.from_numpy(src['arrays']['train_images'][j].copy()).to(device)
            cos=F.cosine_similarity(im[None],old[None]).item()
            if cos<0.999:raise RuntimeError(f'初始图像特征与第六步缓存不符: idx={j}, cosine={cos}')


def choose_query(seed,step,micro,n):
    return int(np.random.default_rng(seed*1000003+step*101+micro*37+53).integers(n))

def choose_negative(method,seed,step,micro,q,src,wrong):
    # 所有策略均在相同查询和正图上训练；只改变这个异身份负样本索引
    rng=np.random.default_rng(seed*1000003+step*97+micro*109+19)
    ip=src['train']['ip'];tp=src['train']['tp'];pid=tp[q]
    if method==METHODS[0]:
        k=int(rng.integers(len(ip)))
        while ip[k]==pid:k=int(rng.integers(len(ip)))
        return k,'random'
    if method!=METHODS[1]:raise ValueError(method)
    count=int(wrong['wrong_count'][q]);take=min(8,count)
    # 与H1一致：8负样本中的前take个来自错误领先，其余由第5~50名半困难补齐
    chosen=[]
    if take:
        chosen.extend(rng.choice(wrong['top_idx'][q,:count],size=take,replace=False).tolist())
    if take<8:
        half=wrong['top_idx'][q,4:50]
        options=half[~np.isin(half,chosen)]
        if len(options)<8-take:raise RuntimeError('半困难池无法补齐')
        chosen.extend(rng.choice(options,size=8-take,replace=False).tolist())
    if len(set(chosen))!=8:raise RuntimeError('负样本重复')
    index=int(rng.integers(8))
    return int(chosen[index]),('wrong_ahead' if index<take else 'semi_hard')


def one_query_loss(model,src,wrong,best_idx,teacher_t,teacher_v,method,seed,step,micro,device,train_io):
    get_text,get_image=train_io
    q=choose_query(seed,step,micro,len(src['train']['tp']))
    pos=int(best_idx[q]);negative,negative_type=choose_negative(method,seed,step,micro,q,src,wrong)
    strongest=int(wrong['top_idx'][q,0]);correct=int(wrong['wrong_count'][q])==0
    ip=src['train']['ip'];pid=src['train']['tp'][q]
    if ip[pos]!=pid or ip[negative]==pid or ip[strongest]==pid:raise RuntimeError('Train身份泄漏/错配')
    # 使用同一已训练编码器计算正图、实际训练负图和保护用最强负图；不借助冻结负图代替梯度
    imgs=[pos,negative]+([strongest] if correct and strongest!=negative else [])
    images=torch.stack([get_image(i) for i in imgs]).to(device)
    token=get_text(q)[None,:].to(device)
    it=F.normalize(model.encode_image(images).float(),dim=1)
    tt=F.normalize(model.encode_text(token).float(),dim=1)[0]
    p=(tt*it[0]).sum();n=(tt*it[1]).sum()
    loss_main=F.softplus((n-p)/TEMPERATURE)
    loss=loss_main
    if correct:
        hardest=it[1] if strongest==negative else it[2]
        student_margin=p-(tt*hardest).sum()
        original_margin=float(wrong['best_positive'][q]-wrong['best_negative'][q])
        # P1型正确间隔保护；A1和A2完全相同
        shortfall=F.relu((torch.as_tensor(original_margin,device=device)-student_margin)/TEMPERATURE)
        loss=loss+PROTECT_WEIGHT*shortfall.square()
    # 防止小数据LoRA快速漂移，A1/A2使用同样强度；教师只提供Train的旧特征
    ref_t=teacher_t[q];ref_i=teacher_v[imgs]
    loss_dist=(tt-ref_t).square().sum()+(it-ref_i).square().sum(dim=1).mean()
    loss=loss+DISTILL_WEIGHT*loss_dist
    if not torch.isfinite(loss):raise RuntimeError('训练损失NaN/Inf')
    return loss,dict(correct_teacher=correct,hard_negative=negative_type=='wrong_ahead',
                     main=float(loss_main.detach()),distill=float(loss_dist.detach()))


def train_fingerprint(src,method,seed):
    return fp(dict(version=VERSION,source_fp=src['source_fp'],source_hashes=src['source_hashes'],
        checkpoint=EXPECTED_CKPT_SHA,method=method,seed=seed,steps=STEPS,accum=GRAD_ACCUM,
        lr=LR,wd=WEIGHT_DECAY,rank=LORA_RANK,scale=LORA_SCALE,last=LORA_LAST_BLOCKS,
        temp=TEMPERATURE,protect=PROTECT_WEIGHT,distill=DISTILL_WEIGHT,maxgrad=MAX_GRAD_NORM))


def optimizer_setup(model):
    trainable=[p for p in model.parameters() if p.requires_grad]
    if not trainable:raise RuntimeError('没有可训练LoRA参数')
    return torch.optim.AdamW(trainable,lr=LR,weight_decay=WEIGHT_DECAY),trainable


def train_model(model,src,wrong,best_idx,method,seed,device,out,train_io,steps_to=None):
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    model.eval()  # 冻结基座没有训练Dropout；LoRA仍会产生梯度
    optimizer,params=optimizer_setup(model)
    path=out/f'{method}_seed{seed}_progress.pt';identity=train_fingerprint(src,method,seed)
    start=0;hist=[];counts={'queries':0,'teacher_correct':0,'wrong_neg_selected':0}
    if path.is_file():
        ck=torch.load(str(path),map_location='cpu',weights_only=False)
        if ck.get('fingerprint')!=identity or ck.get('method')!=method or ck.get('seed')!=seed or not 0<=ck.get('step',-1)<=STEPS:
            raise RuntimeError('训练检查点指纹/步数不一致')
        put_lora(model,ck['lora']);optimizer.load_state_dict(ck['optimizer'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):state[k]=v.to(device)
        start=int(ck['step']);hist=ck['history'];counts=ck['counts'];log('断点恢复',method,seed,start,'/',STEPS)
    # Train教师特征仅用来蒸馏和生成排序保护，不参与Val监督
    teacher_t=torch.from_numpy(np.ascontiguousarray(src['arrays']['train_texts'])).to(device)
    teacher_v=torch.from_numpy(np.ascontiguousarray(src['arrays']['train_images'])).to(device)
    end=STEPS if steps_to is None else min(STEPS,int(steps_to))
    for step in range(start,end):
        optimizer.zero_grad(set_to_none=True)
        observed=[]
        for micro in range(GRAD_ACCUM):
            loss,stats=one_query_loss(model,src,wrong,best_idx,teacher_t,teacher_v,method,seed,step,micro,device,train_io)
            (loss/GRAD_ACCUM).backward()
            observed.append(float(loss.detach()))
            counts['queries']+=1;counts['teacher_correct']+=int(stats['correct_teacher']);counts['wrong_neg_selected']+=int(stats['hard_negative'])
        for p in params:
            if p.grad is not None and not torch.isfinite(p.grad).all():raise RuntimeError('LoRA梯度非有限值')
        norm=nn.utils.clip_grad_norm_(params,MAX_GRAD_NORM)
        optimizer.step()
        if (step+1)%25==0 or step+1==STEPS:log(f'{method} seed={seed} step={step+1}/{STEPS} loss={np.mean(observed):.4f} grad={float(norm):.3f}')
        if (step+1)%SAVE_EVERY==0 or step+1==end:
            hist.append(dict(step=step+1,loss=float(np.mean(observed)),grad=float(norm)))
            ck=dict(version=VERSION,fingerprint=identity,method=method,seed=seed,step=step+1,
                    lora=lora_tensors(model),optimizer=optimizer.state_dict(),history=hist,counts=counts)
            save_pt(path,ck)
    del teacher_t,teacher_v
    return hist,counts,end


def evaluate_case(model,src,device,method,seed,out):
    ckpath=out/f'{method}_seed{seed}_progress.pt';ck=torch.load(str(ckpath),map_location='cpu',weights_only=False)
    if ck['step']!=STEPS or ck['fingerprint']!=train_fingerprint(src,method,seed):raise RuntimeError('无效/未完成的LoRA权重')
    output=out/f'{method}_seed{seed}_metrics.json';csvpath=out/f'{method}_seed{seed}_per_query.csv'
    ksha=sha(ckpath)
    if output.exists() or csvpath.exists():
        if not(output.exists() and csvpath.exists()):raise RuntimeError('已完成结果不完整')
        old=read_json(output)
        if old.get('checkpoint_sha')!=ksha or old.get('fingerprint')!=ck['fingerprint'] or old.get('csv_sha')!=sha(csvpath):
            raise RuntimeError('已有评估缓存权重/结果不一致，拒绝复用')
        with csvpath.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        if len(rows)!=len(src['qp']):raise RuntimeError('评估逐查询记录数量错误')
        return old['summary'],rows
    put_lora(model,ck['lora'])
    m,d=eval_model(model,src,device)
    baseline=src['b0_detail'];b0_hit=baseline['rank']==1;hit=d['rank']==1
    summary=dict(method=method,seed=seed,**m,recovered=int(np.sum(~b0_hit&hit)),
                 regressed=int(np.sum(b0_hit&~hit)),net_correct=int(np.sum(hit)-np.sum(b0_hit)),
                 optimizer_steps=STEPS,query_exposures=STEPS*GRAD_ACCUM)
    rows=[dict(method=method,seed=seed,query_index=i,pid=int(pid),rank=int(d['rank'][i]),
               correct=int(d['rank'][i]==1),AP=float(d['AP'][i]),INP=float(d['INP'][i]))
          for i,pid in enumerate(src['qp'])]
    cols=['method','seed','query_index','pid','rank','correct','AP','INP']
    save_csv(csvpath,cols,rows)
    save_json(output,dict(fingerprint=ck['fingerprint'],checkpoint_sha=ksha,csv_sha=sha(csvpath),summary=summary))
    return summary,rows


def find_p1_historical():
    paths=([Path(STEP13_DIR)] if STEP13_DIR else sorted((ROOT/'硬负样本分析').glob('第十三步_H1已正确排序保护P1_*'),reverse=True))
    for p in paths:
        f=p/'02_所有方案_Val指标.csv'
        if not f.is_file():continue
        with f.open(encoding='utf-8-sig',newline='') as h:rows=list(csv.DictReader(h))
        subset=[r for r in rows if r['method']=='P1_H1_correct_rank_protect' and float(r['alpha'])==1.0]
        if len(subset)==3 and {int(r['seed']) for r in subset}==set(SEEDS):
            return subset,str(f),sha(f)
    return [],None,None


def package(out):
    target=out/ZIP_NAME; temp=out/('.'+ZIP_NAME+'.partial')
    paths=[p for p in sorted(out.iterdir()) if p.is_file() and p.suffix in ('.csv','.json','.txt')]
    if not paths:raise RuntimeError('无报告文件')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths:z.write(p,arcname=p.name)
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:raise RuntimeError('ZIP CRC 错误')
    os.replace(temp,target)
    return target


def run():
    if DEVICE=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA不可用，请确认irra_clean环境和显卡')
    if LORA_LAST_BLOCKS<1 or GRAD_ACCUM<1 or STEPS<1:raise ValueError('超参数不合法')
    device=torch.device(DEVICE)
    torch.backends.cuda.matmul.allow_tf32=False
    src=load_source()
    if RESUME_DIR:
        out=Path(RESUME_DIR).resolve()
        if not out.is_dir():raise FileNotFoundError(out)
    else:
        out=ROOT/'硬负样本分析'/('第十六步_LoRA编码器硬负样本A1A2_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        out.mkdir(parents=True,exist_ok=False)
    (out/'internal_cache').mkdir(exist_ok=True)
    log('输出目录:',out)
    plan=dict(version=VERSION,checkpoint_sha=EXPECTED_CKPT_SHA,source_fp=src['source_fp'],
              source_hashes=src['source_hashes'],methods=list(METHODS),seeds=list(SEEDS),steps=STEPS,
              accum=GRAD_ACCUM,lora_rank=LORA_RANK,scale=LORA_SCALE,last_blocks=LORA_LAST_BLOCKS,
              lr=LR,protect=PROTECT_WEIGHT,distill=DISTILL_WEIGHT,
              val_only=True,test_used=False,baseline='B0 Epoch58',note='P1 only historical comparator')
    manifest=out/'00_实验说明与指纹.json'
    if manifest.exists():
        if read_json(manifest)!=plan:raise RuntimeError('RESUME_DIR 的实验指纹/参数不匹配，拒绝混跑')
    else:save_json(manifest,plan)
    b0,details=score_gallery(src['arrays']['val_texts'],src['arrays']['val_images'],device,src['qp'],src['vp'])
    for k,v in b0.items():
        if abs(v-src['B0_record'][k])>0.035:raise RuntimeError(f'B0历史指标不一致: {k}={v} vs {src["B0_record"][k]}')
    log('B0 原始IRRA Val:',b0)
    src['b0_detail']=details
    wrong,best_idx=source_wrong_pool(src,out,device)
    train_io=make_train_io(src)
    # 一次零增量自检：LoRA注入前后输出必须不变，且与第六步Teacher缓存相符
    torch.manual_seed(SEEDS[0]);model=build_irra(src,device)
    inject_lora(model)
    ntrain=sum(p.numel() for p in model.parameters() if p.requires_grad)
    log('LoRA trainable params:',ntrain)
    check_initial_features(model,src,device,*train_io)
    del model
    if device.type=='cuda':torch.cuda.empty_cache()
    allstats=[dict(method='B0_original',seed='-',**b0,recovered=0,regressed=0,net_correct=0,optimizer_steps=0,query_exposures=0)]
    allrows=[dict(method='B0_original',seed='-',query_index=i,pid=int(pid),rank=int(details['rank'][i]),
                  correct=int(details['rank'][i]==1),AP=float(details['AP'][i]),INP=float(details['INP'][i])) for i,pid in enumerate(src['qp'])]
    prior,p1path,p1sha=find_p1_historical()
    p1rows=[]
    if prior:
        for r in prior:
            p1rows.append(dict(method='P1_historical_alpha1',seed=r['seed'],R1=float(r['R1']),R5=float(r['R5']),
                               R10=float(r['R10']),mAP=float(r['mAP']),mINP=float(r['mINP'])))
        log('复用P1历史指标:',p1path)
    else:log('没有找到第十三步原始P1汇总：本实验仍正常执行，但不会伪造P1对照值')
    save_json(out/'01_原始IRRA基线与历史P1来源.json',dict(B0=b0,P1_source=p1path,P1_source_sha=p1sha,
                  P1_alpha1_per_seed=p1rows,notes='P1权重不参与本轮训练；仅引用历史已评估的Val指标'))
    for method in METHODS:
        for seed in SEEDS:
            torch.manual_seed(seed)   # 保障A1/A2同seed具有完全相同的LoRA初始A/B参数
            model=build_irra(src,device);inject_lora(model)
            hist,counts,last=train_model(model,src,wrong,best_idx,method,seed,device,out,train_io)
            if last!=STEPS:raise RuntimeError('未完成全部训练步')
            save_json(out/f'{method}_seed{seed}_训练统计.json',dict(method=method,seed=seed,history=hist,counts=counts))
            s,r=evaluate_case(model,src,device,method,seed,out)
            allstats.append(s);allrows.extend(r)
            log('结果',method,seed,{k:round(s[k],4) for k in ('R1','R5','R10','mAP','mINP')})
            del model;gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
    save_csv(out/'02_B0_A1_A2_全部指标.csv',list(allstats[0]),allstats)
    save_csv(out/'03_B0_A1_A2_逐查询.csv',list(allrows[0]),allrows)
    if p1rows:save_csv(out/'04_P1历史指标_仅供对照.csv',list(p1rows[0]),p1rows)
    means={}
    for method in METHODS:
        group=[r for r in allstats if r['method']==method]
        means[method]={k:float(np.mean([r[k] for r in group])) for k in ('R1','R5','R10','mAP','mINP')}
    summary=dict(B0=b0,A1=means[METHODS[0]],A2=means[METHODS[1]],
                 A2_minus_A1={k:means[METHODS[1]][k]-means[METHODS[0]][k] for k in means[METHODS[0]]},
                 P1_historical=p1rows,source='Train-only negatives, Val-only evaluation; Test unused')
    save_json(out/'05_三种子均值及负样本增益.json',summary)
    zipfile_path=package(out)
    log('实验完成；上传ZIP:',zipfile_path)


def self_test():
    # 合成小模型的MLP接口与官方CLIP resblocks一致；检验冻结参数、LoRA增量、训练梯度、断点与ZIP。
    from types import SimpleNamespace
    from unittest.mock import patch
    class Block(nn.Module):
        def __init__(self,d=8):
            super().__init__();self.mlp=nn.Sequential()
            self.mlp.add_module('c_fc',nn.Linear(d,16));self.mlp.add_module('act',nn.GELU())
            self.mlp.add_module('c_proj',nn.Linear(16,d))
        def forward(self,x):return x+self.mlp(x)
    class Encoder(nn.Module):
        def __init__(self):super().__init__();self.transformer=SimpleNamespace(resblocks=nn.Sequential(Block(),Block(),Block()))
    # SimpleNamespace不是nn.Module，构造真实module路径以匹配官方结构
    class Branch(nn.Module):
        def __init__(self):
            super().__init__();self.transformer=nn.Module();self.transformer.add_module('resblocks',nn.Sequential(Block(),Block(),Block()))
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__();self.base_model=nn.Module();self.base_model.add_module('visual',Branch())
            self.base_model.add_module('transformer',nn.Module())
            self.base_model.transformer.add_module('resblocks',nn.Sequential(Block(),Block(),Block()))
        def encode_image(self,x):
            for m in self.base_model.visual.transformer.resblocks:x=m(x)
            return x
        def encode_text(self,x):
            for m in self.base_model.transformer.resblocks:x=m(x)
            return x
    torch.manual_seed(7);m=Tiny();x=torch.randn(3,8);old=m.encode_image(x).detach().clone()
    assert inject_lora(m,last=2,rank=2,scale=4)==8
    assert torch.allclose(old,m.encode_image(x),atol=1e-6)
    tr=[p for p in m.parameters() if p.requires_grad]
    assert tr and all(k.endswith(('.A','.B')) for k,p in m.named_parameters() if p.requires_grad)
    assert all(not p.requires_grad for k,p in m.named_parameters() if not k.endswith(('.A','.B')))
    # 梯度确实到LoRA，而非仅修改输出投影的无效包装
    op=torch.optim.AdamW(tr,lr=0.01)
    original=torch.manual_seed(10)
    y=torch.randn(3,8)
    loss=(m.encode_image(x)-y).square().mean()+(m.encode_text(x)-y).square().mean()
    op.zero_grad();loss.backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in tr)
    op.step();assert not torch.allclose(old,m.encode_image(x),atol=1e-6)
    state=lora_tensors(m)
    torch.manual_seed(7);fresh=Tiny();inject_lora(fresh,last=2,rank=2,scale=4)
    put_lora(fresh,state)
    assert torch.allclose(m.encode_image(x),fresh.encode_image(x),atol=1e-5)
    # 目标组比对仅采样规则不同，负样本严格排除同PID，验证随机状态稳定
    pid=np.repeat(np.arange(20),5);n=len(pid)
    fake=dict(train=dict(ip=pid,tp=pid))
    top=np.empty((n,50),np.int32)
    for i in range(n):top[i]=np.flatnonzero(pid!=pid[i])[:50]
    bad=dict(top_idx=top,wrong_count=np.array([3 if i%2 else 0 for i in range(n)],dtype=np.int16))
    for method in METHODS:
        for q in range(n):
            z,_=choose_negative(method,42,1,0,q,fake,bad)
            assert pid[z]!=pid[q]
            z2,_=choose_negative(method,42,1,0,q,fake,bad)
            assert z==z2
    # 实际执行小模型端到端训练，以及【中断恢复 vs 从头训练】逐参数一致性
    import copy
    oldsteps,oldaccum,oldsave=STEPS,GRAD_ACCUM,SAVE_EVERY
    globals()['STEPS']=4;globals()['GRAD_ACCUM']=2;globals()['SAVE_EVERY']=2
    try:
        rng=np.random.default_rng(123)
        a=rng.standard_normal((n,8)).astype(np.float32)
        b=rng.standard_normal((n,8)).astype(np.float32)
        a=a/np.linalg.norm(a,axis=1,keepdims=True)
        b=b/np.linalg.norm(b,axis=1,keepdims=True)
        mock=dict(source_fp='tiny_source',source_hashes={'tiny':'sha'},train=dict(ip=pid,tp=pid),
                  arrays=dict(train_texts=a,train_images=b))
        mock_wrong=dict(top_idx=top,wrong_count=bad['wrong_count'],
                        best_positive=np.ones(n,np.float32),best_negative=np.zeros(n,np.float32))
        def make_tiny(seed):
            torch.manual_seed(7)
            net=Tiny()
            torch.manual_seed(seed)
            inject_lora(net,last=2,rank=LORA_RANK,scale=LORA_SCALE)
            return net
        io=(lambda i:torch.from_numpy(a[int(i)].copy()),lambda i:torch.from_numpy(b[int(i)].copy()))
        best=np.arange(n)
        with tempfile.TemporaryDirectory() as td2:
            aout=Path(td2)/'resume';bout=Path(td2)/'fresh'
            aout.mkdir();bout.mkdir()
            resumed=make_tiny(42)
            _,_,s1=train_model(resumed,mock,mock_wrong,best,METHODS[1],42,torch.device('cpu'),aout,io,steps_to=2)
            assert s1==2
            # 模拟进程重新启动：从新实例恢复状态与optimizer
            resumed=make_tiny(42)
            _,_,s2=train_model(resumed,mock,mock_wrong,best,METHODS[1],42,torch.device('cpu'),aout,io)
            assert s2==4
            fresh=make_tiny(42)
            _,_,s3=train_model(fresh,mock,mock_wrong,best,METHODS[1],42,torch.device('cpu'),bout,io)
            assert s3==4
            for name,weight in lora_tensors(resumed).items():
                assert torch.equal(weight,lora_tensors(fresh)[name]),f'断点恢复参数不一致: {name}'
            # 两组同seed起点必须逐元素相等，排除初始化差异
            st1=lora_tensors(make_tiny(43));st2=lora_tensors(make_tiny(43))
            assert all(torch.equal(st1[k],st2[k]) for k in st1)
    finally:
        globals()['STEPS'],globals()['GRAD_ACCUM'],globals()['SAVE_EVERY']=oldsteps,oldaccum,oldsave
    with tempfile.TemporaryDirectory() as td:
        folder=Path(td);save_pt(folder/'lora.pt',dict(state=state))
        assert (folder/'lora.pt').is_file()
        rows=[dict(method='test',seed=42,R1=66.0,mAP=50.0)]
        save_csv(folder/'test.csv',list(rows[0]),rows)
        save_json(folder/'test.json',dict(pass_test=True))
        z=package(folder)
        with zipfile.ZipFile(z) as f:
            assert f.testzip() is None and 'test.csv' in f.namelist()
    log('[SELF_TEST PASS] LoRA前向及梯度/零初始化/采样排除/训练中断续跑逐参数一致/CSV/ZIP')
    log('[提醒] 合成测试不涉及真实ViT、CUDA峰值；真实数据请先运行 --smoke 验证')


def smoke_real():
    """只运行真实IRRA的一次反向更新，不开启正式800+步训练、不保存权重。"""
    if DEVICE=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA不可用')
    device=torch.device(DEVICE)
    src=load_source()
    with tempfile.TemporaryDirectory(prefix='irra_lora_smoke_') as td:
        folder=Path(td);(folder/'internal_cache').mkdir()
        wrong,best=source_wrong_pool(src,folder,device)
        io=make_train_io(src)
        torch.manual_seed(42)
        model=build_irra(src,device)
        inject_lora(model)
        check_initial_features(model,src,device,*io)
        model.eval()
        teacher_t=torch.from_numpy(np.ascontiguousarray(src['arrays']['train_texts'])).to(device)
        teacher_v=torch.from_numpy(np.ascontiguousarray(src['arrays']['train_images'])).to(device)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        for method in METHODS:
            optimizer,params=optimizer_setup(model)
            optimizer.zero_grad(set_to_none=True)
            loss,stats=one_query_loss(model,src,wrong,best,teacher_t,teacher_v,method,42,0,0,device,io)
            loss.backward()
            norm=nn.utils.clip_grad_norm_(params,MAX_GRAD_NORM)
            optimizer.step()
            log('[真实单步GPU检查]',method,'loss',float(loss.detach()),'grad',float(norm),'info',stats)
            # 每种采样从同一教师初始编码器出发，避免smoke检测本身互相污染
            del loss,stats,optimizer,params,model
            gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
            torch.manual_seed(42);model=build_irra(src,device);inject_lora(model)
            model.eval()
            if device.type=='cuda':torch.cuda.empty_cache()
        if device.type=='cuda':
            log('[CUDA峰值 allocated GiB]',round(torch.cuda.max_memory_allocated()/1024**3,3))
            log('[CUDA device GiB]',round(torch.cuda.get_device_properties(device).total_memory/1024**3,3))
        del model,teacher_t,teacher_v
    log('[SMOKE PASS] 真实Epoch58模型两种负样本完成前向/反向/更新；未训练正式模型')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true',help='CPU合成训练与断点复核')
    p.add_argument('--smoke',action='store_true',help='在真实Epoch58上执行单步前向/反向/显存检查')
    args=p.parse_args()
    if args.self_test:self_test()
    elif args.smoke:smoke_real()
    else:run()

if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();sys.exit(1)
