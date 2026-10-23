# -*- coding: utf-8 -*-
"""
第八步：IRRA Epoch58 + 已训练 B3_mixed (seeds 42/43/44) 固定 alpha=0.5 分支消融。

目的：只判断已训练困难负样本 Adapter 的文本、图像分支分别/共同对检索指标的贡献，
而不是在 Val 或 Test 上继续寻优。此次是推理期分支消融，不是单独重训分支的因果比较。核心四组：
 B0_original：纯 IRRA 冻结特征，不用 Adapter。
 text_only：只变换文本特征，图库图像保持纯 IRRA。
 image_only：只变换图库图像特征，查询文本保持纯 IRRA。
 both：同时变换文本、图像特征，必须复现第六/七步的 B3 alpha=0.5 Val 指标。

三种子独立计算；不重新训练、无需重新提图、绝不读取 Test 数据/报告调参。
也不加载 IRRA 主模型做推理，仅校验其原始 checkpoint/config 文件 SHA。
自动找到第六步原目录（或者填 EXPERIMENT_DIR），结果保存到新的第八步独立目录。
中断后将新目录复制到 RESUME_OUTPUT_DIR；内置每组可验证缓存，支持续跑。
运行：使用 Windows PyCharm irra_clean 环境直接运行；先加 --self-test 可执行合成全链路自检。

输出：原始第六步目录只读；最终 ZIP 内包含关键指标、逐查询详情、按 PID 分组的
配对 Bootstrap、双分支交互效应和原始脚本。该实验只是在 Val 上进行机制消融，
不能视为独立泛化证据；第七步 Test 固定对照不会被重新挑选。
"""
from __future__ import annotations
import os,sys,re,hashlib,json,csv,zipfile,tempfile,traceback,gc,shutil,io
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 注意：必须在 import torch / numpy 之前设置，降低 Windows OMP #15 风险；
# 不设置不安全的 KMP_DUPLICATE_LIB_OK=TRUE。
if sys.platform == 'win32':
    os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ.setdefault('OMP_NUM_THREADS','1')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(r'/')
# 若第六步输出目录有多个，把原实验的完整目录复制到下一行。
EXPERIMENT_DIR = r'G:\PyCharmProjects\IRRA\IRRA-main\硬负样本分析\第六步_最佳IRRA_Epoch58_硬负样本B0B3_20261021_162009'
# 仅在第八步运行中断、需要续跑时填入本脚本上次创建的“第八步_...”完整目录。
RESUME_OUTPUT_DIR = ''
OUTPUT_PREFIX = '第八步_B3固定alpha05_文本图像分支消融'
ZIP_NAME = '上传给ChatGPT_第八步B3文本图像分支消融.zip'

BEST_DIR = ROOT/'logs'/'CUHK-PEDES'/'20261009_095215_irra_baseline_b8_resume20'
EXPECTED_CKPT_SHA = 'f3ab12028e3406546b836ed6a75c8fc5e02a42146c5da13af423e7f5cf0e7ea6'
EXPECTED_CFG_SHA = 'ec2fae46ec4e0cce06372789afe1c877a5cebeddf73654781e11b0484fc30fac'
EXPECTED_EPOCH = 58
EXPECTED_N_IMG = 3078
EXPECTED_N_TXT = 6158
EXPECTED_N_PID = 1000
DIM = 512
SEEDS = (42,43,44)
ALPHA = 0.5                       # 已在前几步冻结，严禁根据 Val 调整
METHOD = 'B3_mixed'
VARIANTS = ('B0_original','text_only','image_only','both')
METRICS = ('R1','R5','R10','mAP','mINP')
VAL_TOLERANCE_PP = 0.06
FEATURE_BATCH = 1024
QUERY_BATCH = 256
BOOT_REPEATS = 3000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VERSION = 'irra_b3_branch_ablation_v1_20261021'

# 冻结的历史参考值：从第六步、第七步 Val 扫描原结果登记，不由本轮指标优化获得。
VAL_B0 = {'R1':66.417668,'R5':85.271192,'R10':90.581358,'mAP':61.865548,'mINP':47.250571}
VAL_BOTH_HALF = {
    42:{'R1':66.856122,'R5':85.189997,'R10':90.662553,'mAP':62.203011,'mINP':47.693317},
    43:{'R1':66.661254,'R5':85.222475,'R10':90.581358,'mAP':62.129231,'mINP':47.556586},
    44:{'R1':66.661254,'R5':85.319909,'R10':90.662553,'mAP':62.114353,'mINP':47.560452},
}

def say(*args): print(*args, flush=True)

def read_json(p):
    with Path(p).open('r',encoding='utf-8') as f:return json.load(f)

def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def atomic_write(p,write):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name+'.',suffix='.partial')
    os.close(fd)
    try:
        write(Path(t))
        os.replace(t,p)
    finally:
        if os.path.exists(t):os.unlink(t)

def atomic_json(p,record):
    def write(t):
        with t.open('w',encoding='utf-8') as f:
            json.dump(record,f,ensure_ascii=False,indent=2,allow_nan=False)
            f.flush();os.fsync(f.fileno())
    atomic_write(p,write)

def write_csv(p,fields,rows):
    def write(t):
        with t.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='raise');w.writeheader();w.writerows(rows)
            f.flush();os.fsync(f.fileno())
    atomic_write(p,write)

def locate_source():
    if EXPERIMENT_DIR:
        p=Path(EXPERIMENT_DIR)
        if not p.is_dir():
            raise FileNotFoundError(f'第六步原目录不存在: {p}\n请检查 EXPERIMENT_DIR（勿把第七步目录填进来）')
        return p.resolve()
    folder=ROOT/'硬负样本分析'
    matches=[x for x in folder.iterdir() if x.is_dir() and
             (x/'00_最佳模型身份与实验设置.json').is_file() and
             all((x/f'B3_mixed_seed{s}_progress.pt').exists() for s in SEEDS)]
    if len(matches)!=1:
        raise RuntimeError(f'发现 {len(matches)} 个第六步候选，须在 EXPERIMENT_DIR 指定目录:\n'+'\n'.join(map(str,matches)))
    return matches[0].resolve()

def check_cache_meta(folder,name,fp,count):
    meta_path=folder/(name+'.meta.json')
    npy_path=folder/(name+'.npy')
    if not meta_path.is_file() or not npy_path.is_file():
        raise FileNotFoundError(f'原始特征缓存不完整（本轮不重新提取）: {npy_path}, {meta_path}')
    meta=read_json(meta_path)
    if (meta.get('fingerprint')!=fp+'|'+name or
        meta.get('shape')!=[count,DIM] or
        type(meta.get('done')) is not int or meta['done']!=count):
        raise RuntimeError(f'原始 {name} 缓存的来源/维度/进度不符')
    # mmap_mode=None 避免 Windows 文件句柄无法释放。
    array=np.load(npy_path,allow_pickle=False,mmap_mode=None)
    if (type(array) is not np.ndarray or array.shape!=(count,DIM) or
        array.dtype!=np.float32 or not np.isfinite(array).all()):
        raise RuntimeError(f'原始 {name} 缓存 shape/dtype/非有限值异常')
    norms=np.linalg.norm(array,axis=1)
    if np.max(abs(norms-1.0))>0.01:
        raise RuntimeError(f'{name} 特征未近似L2归一化，拒绝按余弦检索')
    return array,sha256(npy_path)

def verify_source(base):
    origin_path=base/'00_最佳模型身份与实验设置.json'
    if not origin_path.is_file():raise FileNotFoundError(f'找不到第六步身份文件: {origin_path}')
    origin=read_json(origin_path)
    if (origin.get('epoch')!=EXPECTED_EPOCH or origin.get('checkpoint_sha')!=EXPECTED_CKPT_SHA or
        origin.get('config_sha')!=EXPECTED_CFG_SHA or origin.get('strict_checkpoint') is not True):
        raise RuntimeError('第六步模型身份不符，不允许混用非Epoch58权重')
    ec=origin.get('experiment',{})
    if (ec.get('steps')!=800 or ec.get('bottleneck')!=128 or ec.get('seeds')!=list(SEEDS) or
        ec.get('num_neg')!=8):
        raise RuntimeError('第六步训练实验配置与原方案不一致')
    for p,expected in [(BEST_DIR/'best.pth',EXPECTED_CKPT_SHA),(BEST_DIR/'configs.yaml',EXPECTED_CFG_SHA)]:
        if not p.is_file():raise FileNotFoundError(f'找不到原始文件: {p}')
        if sha256(p)!=expected:raise RuntimeError(f'源文件 SHA256 不匹配: {p}')
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    from datasets.cuhkpedes import CUHKPEDES
    ds=CUHKPEDES(root=str(origin['dataset_root']))
    val=ds.val
    vp=np.asarray(val['image_pids'],dtype=np.int64)
    tp=np.asarray(val['caption_pids'],dtype=np.int64)
    if (len(vp)!=EXPECTED_N_IMG or len(tp)!=EXPECTED_N_TXT or
        len(val['img_paths'])!=EXPECTED_N_IMG or len(val['captions'])!=EXPECTED_N_TXT or
        len(set(vp.tolist()))!=EXPECTED_N_PID or
        len(set(tp.tolist()))!=EXPECTED_N_PID or not set(tp.tolist()).issubset(set(vp.tolist()))):
        raise RuntimeError('官方 Val 查询、图像、PID 数量或匹配关系不符')
    fp=hashlib.sha256(json.dumps(origin,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    cache=base/'internal_cache'
    val_v,hv=check_cache_meta(cache,'val_images',fp,EXPECTED_N_IMG)
    val_t,ht=check_cache_meta(cache,'val_texts',fp,EXPECTED_N_TXT)
    base_record=read_json(base/'01_最佳IRRA_官方Val基线.json')
    if not all(base_record.get('checks',{}).get(k) is True for k in METRICS):
        raise RuntimeError('第六步B0官方指标的5项复现检查不完整或失败')
    history={}
    for seed in SEEDS:
        progress=base/f'B3_mixed_seed{seed}_progress.pt'
        ref_path=base/f'02_B3_mixed_seed{seed}_metrics.json'
        if not progress.is_file() or not ref_path.is_file():
            raise FileNotFoundError(f'第六步 seed{seed} 权重或指标缺失')
        r=read_json(ref_path)
        if (r.get('fingerprint')!=fp or r.get('summary',{}).get('seed')!=seed or
            r.get('summary',{}).get('method')!=METHOD or
            any(k not in r.get('summary',{}) for k in METRICS)):
            raise RuntimeError(f'第六步seed{seed}历史指标来源不符')
        history[seed]=r['summary']
    source_id=fingerprint(dict(version=VERSION,origin=origin,origin_fp=fp,
          val_images_sha=hv,val_texts_sha=ht,
          val_pid=fingerprint(dict(image=vp.tolist(),text=tp.tolist())),
          adapter_shas={str(s):sha256(base/f'B3_mixed_seed{s}_progress.pt') for s in SEEDS},
          alpha=ALPHA,variants=VARIANTS))
    return dict(base=base,origin=origin,fp=fp,tp=tp,vp=vp,t=val_t,v=val_v,
                source_id=source_id,history=history,
                input_sha={'val_images':hv,'val_texts':ht})

def prepare_output(source):
    parent=ROOT/'硬负样本分析'
    parent.mkdir(parents=True,exist_ok=True)
    if RESUME_OUTPUT_DIR:
        out=Path(RESUME_OUTPUT_DIR).resolve()
        if not out.is_dir() or parent.resolve() not in out.parents or not out.name.startswith(OUTPUT_PREFIX+'_'):
            raise RuntimeError(f'仅能复用本实验位于「硬负样本分析」内的第八步结果目录: {out}')
        if out==source['base'] or source['base'] in out.parents:
            raise RuntimeError('禁止修改第六步原实验目录')
        manifest=out/'00_研究设计和来源.json'
        if not manifest.is_file() or read_json(manifest).get('source_id')!=source['source_id']:
            raise RuntimeError('第八步续跑目录不存在对应的来源指纹，不允许复用')
        return out
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    out=parent/(OUTPUT_PREFIX+'_'+stamp)
    seq=2
    while out.exists():
        out=parent/(OUTPUT_PREFIX+f'_{stamp}_{seq}');seq+=1
    out.mkdir()
    return out

class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1=nn.Linear(DIM,128)
        self.l2=nn.Linear(128,DIM)
    def forward(self,x,alpha=ALPHA):
        # 同第六步：l2(GELU(l1(x)))；同第七步：先乘alpha，再归一化。
        return F.normalize(x+float(alpha)*self.l2(F.gelu(self.l1(x))),dim=-1)

class Both(nn.Module):
    def __init__(self):
        super().__init__();self.text=Residual();self.image=Residual()

def load_adapter(source,seed,device):
    p=source['base']/f'B3_mixed_seed{seed}_progress.pt'
    st=torch.load(str(p),map_location='cpu',weights_only=True)
    if (st.get('fingerprint')!=source['fp'] or st.get('method')!=METHOD or
        st.get('seed')!=seed or st.get('step')!=800):
        raise RuntimeError(f'seed {seed}: Adapter身份/训练步数不匹配')
    model=Both().to(device)
    model.load_state_dict(st['model'],strict=True)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model

def apply_branch(features,branch,device,alpha=ALPHA):
    output=[]
    with torch.inference_mode():
        for start in range(0,len(features),FEATURE_BATCH):
            x=torch.as_tensor(np.ascontiguousarray(features[start:start+FEATURE_BATCH]),
                              dtype=torch.float32,device=device)
            r=branch(x,alpha=alpha)
            output.append(r.cpu().numpy())
    arr=np.concatenate(output)
    if arr.shape!=features.shape or not np.isfinite(arr).all():
        raise RuntimeError('Adapter 输出shape或finite检查失败')
    return arr

def evaluate(t,v,tp,vp,device):
    if t.shape!=(len(tp),DIM) or v.shape!=(len(vp),DIM):
        raise RuntimeError(f'特征与Val PID维度不匹配: {t.shape}, {v.shape}')
    ranks=np.empty(len(tp),dtype=np.int32)
    aps=np.empty(len(tp),dtype=np.float64)
    inps=np.empty(len(tp),dtype=np.float64)
    gallery=torch.as_tensor(np.ascontiguousarray(v),dtype=torch.float32,device=device)
    all_positions=np.arange(1,len(vp)+1,dtype=np.float64)[None,:]
    with torch.inference_mode():
        for start in range(0,len(tp),QUERY_BATCH):
            end=min(start+QUERY_BATCH,len(tp))
            query=torch.as_tensor(np.ascontiguousarray(t[start:end]),dtype=torch.float32,device=device)
            score=(query@gallery.T).float().cpu().numpy()  # Torch矩阵乘，不触发NumPy BLAS/OpenMP冲突
            if not np.isfinite(score).all():raise RuntimeError('余弦相似度非有限')
            order=np.argsort(-score,axis=1,kind='stable')
            ok=(vp[order]==tp[start:end,None])
            hits=ok.sum(axis=1)
            if not np.all(hits>0):raise RuntimeError('某些查询在Val图库中没有正样本')
            ranks[start:end]=np.argmax(ok,axis=1)+1
            aps[start:end]=np.sum((np.cumsum(ok,axis=1)/all_positions)*ok,axis=1)/hits
            inps[start:end]=hits/(len(vp)-np.argmax(ok[:,::-1],axis=1))
    return dict(rank=ranks,AP=aps,INP=inps)

def calc_metrics(d):
    r=d['rank'];a=d['AP'];p=d['INP']
    return dict(R1=100*float(np.mean(r==1)),R5=100*float(np.mean(r<=5)),
                R10=100*float(np.mean(r<=10)),mAP=100*float(np.mean(a)),mINP=100*float(np.mean(p)))

def per_query_components(d):
    r=d['rank']
    return np.stack([r==1,r<=5,r<=10,d['AP'],d['INP']],axis=1).astype(np.float64)*100

def verify_detail(d,length,gallery_count):
    if set(d)!={'rank','AP','INP'}:raise RuntimeError('逐查询字段错误')
    r,a,inp=d['rank'],d['AP'],d['INP']
    if (r.shape!=(length,) or a.shape!=(length,) or inp.shape!=(length,) or
        r.dtype.kind not in 'iu' or np.any(r<1) or np.any(r>gallery_count) or
        not np.isfinite(a).all() or not np.isfinite(inp).all() or
        np.any(a<0) or np.any(a>1+1e-6) or np.any(inp<0) or np.any(inp>1+1e-6)):
        raise RuntimeError('逐查询缓存出现坏数据/越界/非有限值')

def case_cache(out,source_id,key,compute,length,gallery_count):
    """单个NPZ内保存完整查询详情+版本指纹，原子替换；安全续跑。"""
    root=out/'internal_cache';root.mkdir(exist_ok=True)
    path=root/(key+'.npz')
    fp=fingerprint(dict(source_id=source_id,case=key,version=VERSION))
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!={'fingerprint','rank','AP','INP'} or str(z['fingerprint'].item())!=fp:
                raise RuntimeError(f'缓存来源/结构不匹配: {path}')
            d={k:np.array(z[k],copy=True) for k in ('rank','AP','INP')}
        verify_detail(d,length,gallery_count)
        say('[复用已完成消融]',key)
        return d
    d=compute()
    verify_detail(d,length,gallery_count)
    def write(temp):
        with temp.open('wb') as f:
            np.savez_compressed(f,fingerprint=np.array(fp),rank=d['rank'],AP=d['AP'],INP=d['INP'])
            f.flush();os.fsync(f.fileno())
    atomic_write(path,write)
    # 复读一次新缓存，确保刚写入的内容可用。
    with np.load(path,allow_pickle=False) as z:
        if str(z['fingerprint'].item())!=fp or not np.array_equal(z['rank'],d['rank']):
            raise RuntimeError(f'新缓存写入核查失败: {path}')
    return d

def check_reference(actual,expected,label):
    error={k:abs(actual[k]-float(expected[k])) for k in METRICS}
    largest=max(error.values())
    say(f'[历史复现] {label} 最大差异={largest:.6f} pp')
    if largest>VAL_TOLERANCE_PP:
        raise RuntimeError(f'{label} 与前次Val报告不一致，拒绝继续: {error}')
    return dict(label=label,max_abs_error_pp=float(largest),errors_pp=error,passed=True)

def changes(base,other):
    def delta_at(n):
        p=base['rank']<=n;q=other['rank']<=n
        return dict(recovered=int(np.sum(~p & q)),regressed=int(np.sum(p & ~q)))
    return {'R1':delta_at(1),'R5':delta_at(5),'R10':delta_at(10)}

def grouped_bootstrap(tp,b0,by_seed,repeats=BOOT_REPEATS,seed=424242):
    """固定三随机种子平均；按Val PID配对重采样。仅为探索性区间。"""
    unique,inv=np.unique(tp,return_inverse=True)
    sizes=np.bincount(inv)
    if not np.all(sizes>0):raise RuntimeError('PID 分组异常')
    value={k:np.mean(np.stack([per_query_components(by_seed[s][k]) for s in SEEDS]),axis=0)
           for k in ('text_only','image_only','both')}
    zero=per_query_components(b0)
    contrasts={'text_minus_B0':value['text_only']-zero,
               'image_minus_B0':value['image_only']-zero,
               'both_minus_B0':value['both']-zero,
               'both_minus_text':value['both']-value['text_only'],
               'both_minus_image':value['both']-value['image_only'],
               # 差分中的差分：仅反映排名指标在两条分支之间的非加性。
               'interaction_both-text-image+B0':value['both']-value['text_only']-value['image_only']+zero}
    rng=np.random.default_rng(seed)
    sampled=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=repeats)
    denom=sampled@sizes
    rows=[]
    for name,d in contrasts.items():
        totals=np.stack([np.bincount(inv,weights=d[:,i],minlength=len(unique)) for i in range(5)],axis=1)
        boots=(sampled@totals)/denom[:,None]
        low,high=np.quantile(boots,[0.025,0.975],axis=0)
        point=totals.sum(axis=0)/sum(sizes)
        for i,m in enumerate(METRICS):
            rows.append(dict(contrast=name,metric=m,mean_pp=float(point[i]),
                             ci_low_pp=float(low[i]),ci_high_pp=float(high[i]),
                             n_pids=int(len(unique)),resamples=int(repeats)))
    return rows

def rank_group_rows(b0,by_seed):
    rank=b0['rank']
    groups=[('rank_1',rank==1),('rank_2_5',(rank>=2)&(rank<=5)),
            ('rank_6_10',(rank>=6)&(rank<=10)),('rank_11_50',(rank>=11)&(rank<=50)),
            ('rank_51_plus',rank>=51)]
    rows=[]
    for name,mask in groups:
        for var in ('text_only','image_only','both'):
            d=np.stack([by_seed[s][var]['AP']-b0['AP'] for s in SEEDS],axis=0)
            changed=np.stack([(by_seed[s][var]['rank']==1).astype(float)-(rank==1).astype(float) for s in SEEDS])
            rows.append(dict(b0_rank_group=name,queries=int(mask.sum()),variant=var,
                             mean_delta_AP_pp=float(100*d[:,mask].mean()) if mask.any() else None,
                             mean_delta_R1_pp=float(100*changed[:,mask].mean()) if mask.any() else None))
    return rows

def export_zip(out):
    dest=out/ZIP_NAME
    partial=out/'.upload_partial.zip'
    try:
        with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for f in sorted(out.iterdir()):
                if f.is_file() and f not in (dest,partial) and f.suffix in ('.csv','.json','.txt'):
                    z.write(f,f.name)
            if Path(__file__).is_file():z.write(Path(__file__).resolve(),'实验脚本/'+Path(__file__).name)
        with zipfile.ZipFile(partial) as z:
            bad=z.testzip()
            if bad:raise RuntimeError(f'ZIP CRC 校验失败: {bad}')
            if '01_单组指标.csv' not in z.namelist() or '02_所有查询逐组结果.csv' not in z.namelist():
                raise RuntimeError('ZIP 缺少核心结果文件')
        os.replace(partial,dest)
    finally:
        if partial.exists():partial.unlink()
    return dest

def execute(source,out,device,strict_references=True,return_data=False):
    """真正的四组*三种子全流程；自检同样走这个函数。"""
    sid=source['source_id'];tp=source['tp'];vp=source['vp'];val_t=source['t'];val_v=source['v']
    origin=out/'00_研究设计和来源.json'
    manifest=dict(version=VERSION,created=datetime.now().isoformat(),source_directory=str(source['base']),
        source_id=sid,checkpoint_sha=EXPECTED_CKPT_SHA,config_sha=EXPECTED_CFG_SHA,
        frozen_alpha=ALPHA,method=METHOD,seeds=list(SEEDS),variants=list(VARIANTS),
        val_text_count=len(tp),val_gallery_count=len(vp),val_pid_count=len(set(tp.tolist())),
        only_val=True,no_training=True,no_test_selection=True,
        notes='仅考察已训练B3 Adapter的模态贡献，不能在本步重新调整alpha或用Test挑选分支。')
    if origin.exists():
        earlier=read_json(origin)
        if earlier.get('source_id')!=sid or earlier.get('frozen_alpha')!=ALPHA or earlier.get('variants')!=list(VARIANTS):
            raise RuntimeError('第八步输出目录已绑定另一套来源或消融设置')
    else:atomic_json(origin,manifest)
    references=[]
    b0=case_cache(out,sid,'B0_original',lambda:evaluate(val_t,val_v,tp,vp,device),len(tp),len(vp))
    base_score=calc_metrics(b0)
    if strict_references:references.append(check_reference(base_score,VAL_B0,'B0 Val'))
    say('[B0]',{k:round(base_score[k],5) for k in METRICS})
    results=[dict(variant='B0_original',seed='baseline',alpha=0.0,
                  text_enabled=0,image_enabled=0,**base_score,
                  **{'delta_'+m+'_pp':0.0 for m in METRICS},R1_recovered=0,R1_regressed=0)]
    cases=[]
    by_seed={}
    for seed in SEEDS:
        ad=load_adapter(source,seed,device)
        try:
            # 首先原强度alpha1两分支复现上一轮，作为权重/公式一致性检查。
            if strict_references:
                full_t=apply_branch(val_t,ad.text,device,alpha=1.0)
                full_v=apply_branch(val_v,ad.image,device,alpha=1.0)
                historical=calc_metrics(evaluate(full_t,full_v,tp,vp,device))
                references.append(check_reference(historical,source['history'][seed],f'B3 seed{seed} alpha=1'))
                del full_t,full_v
            # 四组完全相同的alpha=0.5；原始特征未被原地修改。
            t_half=apply_branch(val_t,ad.text,device)
            v_half=apply_branch(val_v,ad.image,device)
            dtext=case_cache(out,sid,f'B3_seed{seed}_text_only',
                    lambda:evaluate(t_half,val_v,tp,vp,device),len(tp),len(vp))
            dimage=case_cache(out,sid,f'B3_seed{seed}_image_only',
                    lambda:evaluate(val_t,v_half,tp,vp,device),len(tp),len(vp))
            dboth=case_cache(out,sid,f'B3_seed{seed}_both',
                    lambda:evaluate(t_half,v_half,tp,vp,device),len(tp),len(vp))
            if strict_references:
                references.append(check_reference(calc_metrics(dboth),VAL_BOTH_HALF[seed],
                           f'B3 seed{seed} alpha=0.5 (both)'))
            by_seed[seed]=dict(text_only=dtext,image_only=dimage,both=dboth)
            for variant,d in by_seed[seed].items():
                m=calc_metrics(d)
                change=changes(b0,d)['R1']
                record=dict(variant=variant,seed=seed,alpha=ALPHA,
                       text_enabled=int(variant!='image_only'),image_enabled=int(variant!='text_only'),
                       **m,**{'delta_'+x+'_pp':m[x]-base_score[x] for x in METRICS},
                       R1_recovered=change['recovered'],R1_regressed=change['regressed'])
                results.append(record)
                say(f'[Val] seed {seed} {variant}: R1={m["R1"]:.4f} R5={m["R5"]:.4f} '
                    f'R10={m["R10"]:.4f} mAP={m["mAP"]:.4f} mINP={m["mINP"]:.4f}')
            del t_half,v_half
        finally:
            del ad
            if device.type=='cuda':torch.cuda.empty_cache()
    # 形成报告前重新从全部查询数组计算，防止指标文件与查询文件不一致。
    meanrows=[dict(variant='B0_original',n_seeds=0,**base_score,
                   **{'delta_'+m+'_pp':0.0 for m in METRICS})]
    for variant in ('text_only','image_only','both'):
        scores=[calc_metrics(by_seed[s][variant]) for s in SEEDS]
        avg={m:float(np.mean([z[m] for z in scores])) for m in METRICS}
        meanrows.append(dict(variant=variant,n_seeds=len(SEEDS),**avg,
                    **{'delta_'+m+'_pp':avg[m]-base_score[m] for m in METRICS}))
    query_fields=['variant','seed','query_index','pid','rank','correct','R5_hit','R10_hit','AP','INP',
                  'B0_rank','B0_AP','B0_INP','delta_AP_pp','delta_INP_pp']
    def details_rows(variant,seed,d):
        for i in range(len(tp)):
            yield dict(variant=variant,seed=seed,query_index=i,pid=int(tp[i]),
                rank=int(d['rank'][i]),correct=int(d['rank'][i]==1),
                R5_hit=int(d['rank'][i]<=5),R10_hit=int(d['rank'][i]<=10),
                AP=float(d['AP'][i]),INP=float(d['INP'][i]),
                B0_rank=int(b0['rank'][i]),B0_AP=float(b0['AP'][i]),B0_INP=float(b0['INP'][i]),
                delta_AP_pp=float((d['AP'][i]-b0['AP'][i])*100),
                delta_INP_pp=float((d['INP'][i]-b0['INP'][i])*100))
    def iter_rows():
        yield from details_rows('B0_original','baseline',b0)
        for seed in SEEDS:
            for variant in ('text_only','image_only','both'):
                yield from details_rows(variant,seed,by_seed[seed][variant])
    report_fields=['variant','seed','alpha','text_enabled','image_enabled']+list(METRICS)+\
                  ['delta_'+m+'_pp' for m in METRICS]+['R1_recovered','R1_regressed']
    write_csv(out/'01_单组指标.csv',report_fields,results)
    write_csv(out/'02_所有查询逐组结果.csv',query_fields,iter_rows())
    write_csv(out/'03_三种子均值和增量.csv',['variant','n_seeds']+list(METRICS)+
             ['delta_'+m+'_pp' for m in METRICS],meanrows)
    boot=grouped_bootstrap(tp,b0,by_seed)
    write_csv(out/'04_PID配对Bootstrap_含交互项.csv',
              ['contrast','metric','mean_pp','ci_low_pp','ci_high_pp','n_pids','resamples'],boot)
    byrank=rank_group_rows(b0,by_seed)
    write_csv(out/'05_基线排名分组诊断.csv',
              ['b0_rank_group','queries','variant','mean_delta_AP_pp','mean_delta_R1_pp'],byrank)
    atomic_json(out/'06_复现与审计.json',dict(version=VERSION,source_id=sid,reference_checks=references,
            summary_means=meanrows,interactions={x['metric']:x['mean_pp'] for x in boot
              if x['contrast']=='interaction_both-text-image+B0'},
            interpretation='双分支交互是检索指标上的非加性效应，不等于证明学会了跨模态关系；小幅差异不能视为可发表创新。',
            statistical_warning='Val分支消融属探索性后验分析；PID Bootstrap不覆盖独立训练和数据集泛化；Test已使用且不能再调参。',
            no_new_training=True,no_test_reselection=True))
    archive=export_zip(out)
    say('\n[第八步完成] 报告 ZIP：',archive)
    if return_data:return dict(results=results,means=meanrows,bootstrap=boot,by_seed=by_seed,b0=b0,archive=archive)
    return None


def self_test():
    """运行完整合成端到端、缓存续跑、错误来源拒绝、CSV/ZIP复核。"""
    say('[SELF-TEST 1/6] 官方指标独立参考小例子与分支对照')
    rng=np.random.default_rng(219)
    ntext,nimage=18,12
    tp=np.array([i%4 for i in range(ntext)],dtype=np.int64)
    vp=np.array([i%4 for i in range(nimage)],dtype=np.int64)
    t=rng.normal(size=(ntext,DIM)).astype(np.float32)
    v=rng.normal(size=(nimage,DIM)).astype(np.float32)
    t/=np.linalg.norm(t,axis=1,keepdims=True);v/=np.linalg.norm(v,axis=1,keepdims=True)
    dev=torch.device('cpu')
    d=evaluate(t,v,tp,vp,dev)
    # 与完全独立的显式Python逐查询排序、AP/INP公式核对
    scores=(torch.from_numpy(t)@torch.from_numpy(v).T).numpy()
    for i in range(ntext):
        order=np.argsort(-scores[i],kind='stable')
        pos=np.where(vp[order]==tp[i])[0]+1
        independent_ap=np.mean(np.arange(1,len(pos)+1)/pos)
        assert int(d['rank'][i])==int(pos[0])
        assert abs(d['AP'][i]-independent_ap)<1e-10
        assert abs(d['INP'][i]-len(pos)/pos[-1])<1e-10
    say('[SELF-TEST 2/6] 构造合法的B3三种子训练进度文件')
    with tempfile.TemporaryDirectory() as td:
        out=Path(td)/'stage8';out.mkdir()
        origin=Path(td)/'stage6';origin.mkdir()
        fp='synthetic_stage6_fp';sid='synthetic_val_adapter_v1'
        for seed in SEEDS:
            torch.manual_seed(seed)
            ad=Both()
            torch.save(dict(fingerprint=fp,method=METHOD,seed=seed,step=800,model=ad.state_dict()),
                       origin/f'B3_mixed_seed{seed}_progress.pt')
        source=dict(base=origin,fp=fp,source_id=sid,tp=tp,vp=vp,t=t,v=v,history={})
        # 完整执行，参数/缓存路径真实使用；仅跳过真实训练来源的数值参考。
        say('[SELF-TEST 3/6] 合成全链路：B0/文本/图像/双分支 + CSV + PID Bootstrap + ZIP')
        first=execute(source,out,dev,strict_references=False,return_data=True)
        assert len(first['results'])==10 and len(first['means'])==4 and len(first['bootstrap'])==30
        assert all((out/f).is_file() for f in ['01_单组指标.csv','02_所有查询逐组结果.csv',
                '03_三种子均值和增量.csv','04_PID配对Bootstrap_含交互项.csv',
                '05_基线排名分组诊断.csv','06_复现与审计.json',ZIP_NAME])
        with (out/'02_所有查询逐组结果.csv').open('r',encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        assert len(rows)==ntext*10
        for rec in first['results']:
            group=[r for r in rows if r['variant']==rec['variant'] and r['seed']==str(rec['seed'])]
            assert len(group)==ntext
            verify=dict(R1=100*sum(int(r['correct']) for r in group)/ntext,
                        R5=100*sum(int(r['R5_hit']) for r in group)/ntext,
                        R10=100*sum(int(r['R10_hit']) for r in group)/ntext,
                        mAP=100*np.mean([float(r['AP']) for r in group]),
                        mINP=100*np.mean([float(r['INP']) for r in group]))
            assert max(abs(rec[k]-verify[k]) for k in METRICS)<1e-9
        with zipfile.ZipFile(first['archive']) as z:
            assert z.testzip() is None and len(z.namelist())>=7
        say('[SELF-TEST 4/6] 同一结果目录复用全部10组结果，无需重新推理')
        hashes={p.name:sha256(p) for p in (out/'internal_cache').glob('*.npz')}
        second=execute(source,out,dev,strict_references=False,return_data=True)
        assert hashes=={p.name:sha256(p) for p in (out/'internal_cache').glob('*.npz')}
        assert first['means']==second['means']
        say('[SELF-TEST 5/6] 模拟缓存缺失后重算、错误指纹拒绝')
        target=out/'internal_cache'/'B3_seed42_text_only.npz'
        old=target.read_bytes();target.unlink()
        execute(source,out,dev,strict_references=False)
        assert target.exists() and target.read_bytes()==old
        with np.load(target,allow_pickle=False) as z:
            wrong={k:np.array(z[k],copy=True) for k in z.files}
        wrong['fingerprint']=np.array('definitely_wrong')
        with target.open('wb') as f:np.savez_compressed(f,**wrong)
        try:execute(source,out,dev,strict_references=False)
        except RuntimeError as e:assert '缓存来源' in str(e)
        else:raise AssertionError('没有拒绝错误指纹')
        target.write_bytes(old)
        say('[SELF-TEST 6/6] 中断续跑、原目录文件未被修改、PNG/NPY句柄不遗留')
        assert all(p.name.endswith('.pt') for p in origin.iterdir())
        assert all(not p.name.endswith('.partial') for p in out.iterdir())
        shutil.rmtree(out)
        # 再用模拟的官方数据集接口、真实NPY/meta、真实PT验证源目录校验与主入口的目录创建。
        say('[SELF-TEST EXTRA] 官方Val源文件读入/身份校验/独立目录创建与续跑')
        from unittest.mock import patch
        from types import ModuleType
        root=Path(td)/'fake_IRRA-main'; root.mkdir()
        best=root/'fake_best';best.mkdir()
        (best/'best.pth').write_bytes(b'mock epoch58 checkpoint for structural test')
        (best/'configs.yaml').write_text('dataset_name: CUHK-PEDES\n',encoding='utf-8')
        ck_sha=sha256(best/'best.pth');cfg_sha=sha256(best/'configs.yaml')
        info=dict(epoch=58,checkpoint_sha=ck_sha,config_sha=cfg_sha,strict_checkpoint=True,
                  dataset_root=str(root/'datasets'),
                  experiment=dict(steps=800,bottleneck=128,seeds=list(SEEDS),num_neg=8))
        atomic_json(origin/'00_最佳模型身份与实验设置.json',info)
        atomic_json(origin/'01_最佳IRRA_官方Val基线.json',dict(checks={k:True for k in METRICS}))
        fpid=hashlib.sha256(json.dumps(info,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        cache=origin/'internal_cache';cache.mkdir()
        for name,arr in [('val_images',v),('val_texts',t)]:
            np.save(cache/(name+'.npy'),arr)
            atomic_json(cache/(name+'.meta.json'),dict(fingerprint=fpid+'|'+name,
                        shape=list(arr.shape),done=len(arr)))
        for seed in SEEDS:
            progress=origin/f'B3_mixed_seed{seed}_progress.pt'
            st=torch.load(progress,map_location='cpu',weights_only=True)
            st['fingerprint']=fpid
            torch.save(st,progress)
            atomic_json(origin/f'02_B3_mixed_seed{seed}_metrics.json',
                        dict(fingerprint=fpid,summary=dict(seed=seed,method=METHOD,**calc_metrics(d))))
        source_shas={str(s):sha256(origin/f'B3_mixed_seed{s}_progress.pt') for s in SEEDS}
        fakepkg=ModuleType('datasets');fakepkg.__path__=[]
        fakecuhk=ModuleType('datasets.cuhkpedes')
        class FakeCUHK:
            def __init__(self,root):
                self.val=dict(image_pids=vp.tolist(),caption_pids=tp.tolist(),
                    img_paths=['synthetic_'+str(i) for i in range(len(vp))],
                    captions=['mock query '+str(i) for i in range(len(tp))])
        fakecuhk.CUHKPEDES=FakeCUHK
        with patch.dict(sys.modules,{'datasets':fakepkg,'datasets.cuhkpedes':fakecuhk}), \
             patch.dict(globals(),{'ROOT':root,'BEST_DIR':best,'EXPECTED_CKPT_SHA':ck_sha,
                'EXPECTED_CFG_SHA':cfg_sha,'EXPECTED_N_IMG':len(vp),'EXPECTED_N_TXT':len(tp),
                'EXPECTED_N_PID':4,'EXPERIMENT_DIR':str(origin),'RESUME_OUTPUT_DIR':''}):
            imported=verify_source(locate_source())
            assert np.array_equal(imported['t'],t) and np.array_equal(imported['v'],v)
            assert imported['fp']==fpid
            check_shas={str(s):sha256(origin/f'B3_mixed_seed{s}_progress.pt') for s in SEEDS}
            assert check_shas==source_shas
            independent=prepare_output(imported)
            assert independent!=origin and independent.parent==root/'硬负样本分析'
            execute(imported,independent,dev,strict_references=False)
            # 检查第六步文件没有任何副作用
            assert {str(s):sha256(origin/f'B3_mixed_seed{s}_progress.pt') for s in SEEDS}==source_shas
            with patch.dict(globals(),{'RESUME_OUTPUT_DIR':str(independent)}):
                assert prepare_output(imported)==independent
            shutil.rmtree(independent)
        del imported

        assert not out.exists()
    say('[SELF-TEST PASS] 合成端到端、逐查询指标复核、CSV/ZIP、缓存创建/复用/删除后重算、坏指纹拒绝、目录清理全部通过。')
    say('[边界] 没有在真实 Windows/CUDA 环境加载原始 Adapter 与数据运行；真实复现须看你本机输出。')

def main():
    if '--self-test' in sys.argv:
        self_test();return
    if abs(ALPHA-0.5)>1e-12:raise RuntimeError('第八步 alpha 已预先固定为0.5，不允许变更')
    device=torch.device(DEVICE)
    say('[第八步] 固定B3 alpha=0.5，冻结IRRA和Adapter，不重训/不查看Test。设备:',device)
    base=locate_source();say('读取第六步来源：',base)
    source=verify_source(base)
    out=prepare_output(source)
    say('第八步独立结果目录：',out)
    execute(source,out,device,strict_references=True)
    del source
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()

if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();sys.exit(1)
