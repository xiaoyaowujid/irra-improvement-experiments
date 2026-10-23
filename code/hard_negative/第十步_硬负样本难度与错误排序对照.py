# -*- coding: utf-8 -*-
"""
第十步：IRRA Epoch58 硬负样本难度与错误排序定向监督对照

目标：纠正错误身份在检索中压过正确身份的现象，并检验这种训练是否真正提高 Val R1/R5/R10/mAP/mINP。
不会重新训练 B0/B1/B2/B3，不改第六、七、八步源文件，永不使用 Test 调参。

本次仅新增 C1：每条 Train 文本选取原始IRRA相似度最高的8张异身份图像，无论它们是否高于最优正样本。
第九步H1/H2、第六步B0/B1/B2/B3直接读取历史成果及H1/H2权重，禁止重新训练或覆盖。
主检验 alpha=1，预先固定的辅助检验alpha=0.5。仅Val，绝不运行Test。
严格采用第六步同一 Adapter、800步、batch=128、负样本8、SDM-like CE 排序损失、正图、预留项。
主要比较 alpha=1.0；预注册次要评估 alpha=0.5（第六~八步既有发现），绝不继续扫 alpha。

把本文件放到 G:\\PyCharmProjects\\IRRA\\IRRA-main 下，直接运行。
可先在 PyCharm 参数填 --self-test；正式运行不加参数。
中断续跑：将 RESUME_OUTPUT_DIR 设置为本脚本生成的第十步目录。
如本机存在多个第九步运行文件夹，可在 SOURCE_STEP9_DIR 中指定完整路径。
输出路径：项目/硬负样本分析/第十步_难度与错排机制对照_时间戳/
内部缓存+训练进度不打包；上传 ZIP 中含复核所需指标和逐查询 CSV。
"""
from __future__ import annotations
import os, sys, re, json, csv, time, hashlib, random, zipfile, tempfile, traceback, argparse, gc
from pathlib import Path
from datetime import datetime
from collections import Counter

if sys.platform == 'win32':
    os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
    os.environ['MKL_NUM_THREADS']='1'
    os.environ.setdefault('OMP_NUM_THREADS','1')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(r'/')
SOURCE_DIR=Path(r'G:\PyCharmProjects\IRRA\IRRA-main\硬负样本分析\第六步_最佳IRRA_Epoch58_硬负样本B0B3_20261021_162009')
SOURCE_STEP9_DIR=''        # 留空自动查找已完成且指纹一致的第九步目录
RESUME_OUTPUT_DIR=''
OUTPUT_PREFIX='第十步_难度与错排机制对照'
STEP9_PREFIX='第九步_错误排序驱动硬负样本'
STEP10_VERSION='irra_step10_top8_versus_wrong_v1_20261021'
ZIP_NAME='上传给ChatGPT_第十步_负样本难度与错排对照.zip'
VERSION='irra_step9_error_rank_hardneg_v1_20261021'

EXPECTED_CKPT_SHA='f3ab12028e3406546b836ed6a75c8fc5e02a42146c5da13af423e7f5cf0e7ea6'
EXPECTED_CFG_SHA='ec2fae46ec4e0cce06372789afe1c877a5cebeddf73654781e11b0484fc30fac'
EXPECTED_ANN_SHA='fc3d15542f91dd70e23028c2cb40171a8f41734a39afddb9505e352e396cabf0'
BEST_DIR=ROOT/'logs'/'CUHK-PEDES'/'20261009_095215_irra_baseline_b8_resume20'
SEEDS=(42,43,44)
METHODS_OLD=('B1_random','B2_hard','B3_mixed')
METHODS_PRIOR_NEW=('H1_wrong_ahead','H2_wrong_plus_semi')
METHODS_NEW=('C1_top8_unconditional',)
ALPHAS=(1.0,0.5)  # alpha=1 主检验；alpha=0.5 已在前期冻结为辅助检验，禁止临时改动
DIM=512
EXPECTED_TRAIN_IMAGES=34054
EXPECTED_TRAIN_TEXTS=68126
EXPECTED_VAL_IMAGES=3078
EXPECTED_VAL_TEXTS=6158
EXPECTED_VAL_IDS=1000
NUM_NEG=8
RANK_START=5
RANK_END=50
TRAIN_STEPS=800
SAVE_EVERY=100
EMBED_TRAIN_BATCH=128
BOTTLENECK=128
LR=1e-4
TEMPERATURE=0.07
PRESERVE=0.05
TRAIN_QUERY_CHUNK=128
EVAL_QUERY_CHUNK=256
ADAPTER_BATCH=1024
BOOT_REPEATS=2000  # 聚类bootstrap，探索性；不能用来挑Test方案
PROBE_STEPS=tuple(range(0,TRAIN_STEPS,40))
METRICS=('R1','R5','R10','mAP','mINP')
# 第八步曾验证的B3 alpha=0.5 三种子指标：本实验必须能复现，不丢弃历史成果。
HISTORIC_B3_HALF={
 42:dict(R1=66.856122,R5=85.189997,R10=90.662553,mAP=62.203011,mINP=47.693317),
 43:dict(R1=66.661254,R5=85.222475,R10=90.581358,mAP=62.129231,mINP=47.556586),
 44:dict(R1=66.661254,R5=85.319909,R10=90.662553,mAP=62.114353,mINP=47.560452),
}
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'


def say(*args): print(*args,flush=True)

def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()

def read_json(path):
    with Path(path).open(encoding='utf-8') as f:return json.load(f)

def atomic_write(path,fn):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.partial',dir=str(path.parent))
    os.close(fd)
    try:
        fn(Path(temp))
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)

def atomic_json(path,data):
    def save(tmp):
        with tmp.open('w',encoding='utf-8') as f:
            json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False)
            f.flush();os.fsync(f.fileno())
    atomic_write(path,save)

def atomic_torch(path,obj):
    def save(tmp):torch.save(obj,str(tmp))
    atomic_write(path,save)

def atomic_csv(path,headers,rows):
    def save(tmp):
        with tmp.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=headers,extrasaction='raise')
            writer.writeheader();writer.writerows(rows)
            f.flush();os.fsync(f.fileno())
    atomic_write(path,save)

def check_feature(cache,name,fp,num):
    npy=cache/(name+'.npy');mp=cache/(name+'.meta.json')
    if not npy.is_file() or not mp.is_file():raise FileNotFoundError(f'第六步特征缓存缺失: {npy} / {mp}')
    m=read_json(mp)
    if (m.get('fingerprint')!=fp+'|'+name or m.get('shape')!=[num,DIM] or
            type(m.get('done')) is not int or m['done']!=num):
        raise RuntimeError(f'源缓存元信息不匹配：{name}，拒绝使用')
    x=np.load(npy,allow_pickle=False)  # 完整加载RAM，避免WinError32：不使用mmap
    if (type(x) is not np.ndarray or x.shape!=(num,DIM) or x.dtype!=np.float32 or
            not np.isfinite(x).all()):raise RuntimeError(f'{name} 特征形状/类型/数值异常')
    if np.max(abs(np.linalg.norm(x,axis=1)-1))>0.012:
        raise RuntimeError(f'{name} 没有正确L2归一化')
    return x,sha256(npy)

def read_train_order(rootdir):
    ann=Path(rootdir)/'CUHK-PEDES'/'reid_raw.json'
    if not ann.is_file() or sha256(ann)!=EXPECTED_ANN_SHA:
        raise RuntimeError('真实训练标注缺失/哈希与第六步不同，拒绝混用')
    with ann.open(encoding='utf-8') as f:rows=json.load(f)
    img_pids=[];txt_pids=[];positive=[];train_ids=set()
    for r in rows:
        if r.get('split')!='train':continue
        pid=int(r['id']);train_ids.add(pid)
        img_index=len(img_pids);img_pids.append(pid)
        for _ in r['captions']:
            txt_pids.append(pid);positive.append(img_index)
    ip=np.asarray(img_pids,dtype=np.int64);tp=np.asarray(txt_pids,dtype=np.int64)
    pos=np.asarray(positive,dtype=np.int64)
    if (len(ip)!=EXPECTED_TRAIN_IMAGES or len(tp)!=EXPECTED_TRAIN_TEXTS or
        len(train_ids)!=11003 or not np.array_equal(ip[pos],tp)):
        raise RuntimeError('Train划分/文字到图像的索引顺序不匹配')
    return ip,tp,pos

def verify_source():
    base=SOURCE_DIR.resolve();outroot=(ROOT/'硬负样本分析').resolve()
    if not base.is_dir() or outroot not in base.parents:raise FileNotFoundError(f'第六步源实验目录错误: {base}')
    ident=read_json(base/'00_最佳模型身份与实验设置.json')
    if (ident.get('epoch')!=58 or ident.get('checkpoint_sha')!=EXPECTED_CKPT_SHA or
            ident.get('config_sha')!=EXPECTED_CFG_SHA or
            ident.get('annotation_sha256')!=EXPECTED_ANN_SHA or
            ident.get('strict_checkpoint') is not True):
        raise RuntimeError('源Epoch58/原始数据指纹不一致')
    cfg=ident.get('experiment',{})
    expected={'steps':TRAIN_STEPS,'num_neg':NUM_NEG,'bottleneck':BOTTLENECK,
              'embed_train_batch':EMBED_TRAIN_BATCH,'hard_start':RANK_START,'hard_end':RANK_END,
              'lr':LR,'temperature':TEMPERATURE,'preserve':PRESERVE,'seeds':list(SEEDS)}
    for key,value in expected.items():
        if cfg.get(key)!=value:raise RuntimeError(f'第六步关键参数 {key} 不符合原实验: {cfg.get(key)} != {value}')
    if (ident.get('train_images')!=EXPECTED_TRAIN_IMAGES or ident.get('train_captions')!=EXPECTED_TRAIN_TEXTS or
            ident.get('val_images')!=EXPECTED_VAL_IMAGES or ident.get('val_captions')!=EXPECTED_VAL_TEXTS):
        raise RuntimeError('第六步Train/Val规模不匹配')
    # 校验真正的最佳权重/配置，而不是仅依赖源JSON声称已验证
    for path,target in ((BEST_DIR/'best.pth',EXPECTED_CKPT_SHA),(BEST_DIR/'configs.yaml',EXPECTED_CFG_SHA)):
        if not path.is_file() or sha256(path)!=target:raise RuntimeError(f'Epoch58原文件缺失/变更: {path}')
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    from datasets.cuhkpedes import CUHKPEDES
    ds=CUHKPEDES(root=str(ident['dataset_root']))
    val=ds.val
    vp=np.asarray(val['image_pids'],dtype=np.int64)
    qp=np.asarray(val['caption_pids'],dtype=np.int64)
    if (len(vp)!=EXPECTED_VAL_IMAGES or len(qp)!=EXPECTED_VAL_TEXTS or
       len(set(vp.tolist()))!=EXPECTED_VAL_IDS or len(set(qp.tolist()))!=EXPECTED_VAL_IDS or
       not set(qp.tolist()).issubset(set(vp.tolist()))):
        raise RuntimeError('Val数据协议不匹配')
    ip,tp,positive=read_train_order(ident['dataset_root'])
    if set(tp.tolist()) & set(qp.tolist()):raise RuntimeError('Train与Val出现相同PID：数据泄漏')
    fp=hashlib.sha256(json.dumps(ident,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    cache=base/'internal_cache'
    tr_img,hi=check_feature(cache,'train_images',fp,len(ip))
    tr_txt,ht=check_feature(cache,'train_texts',fp,len(tp))
    val_img,hvi=check_feature(cache,'val_images',fp,len(vp))
    val_txt,hvt=check_feature(cache,'val_texts',fp,len(qp))
    bp=cache/'train_hard_pool.npy'; bm=cache/'train_hard_pool.meta.json'
    if not bp.is_file() or not bm.is_file():raise FileNotFoundError('第六步B2第5~50名负样本池缺失')
    meta=read_json(bm)
    if meta.get('fingerprint')!=fp+f'|hard|{RANK_START}|{RANK_END}':
        raise RuntimeError('第六步半困难负样本池指纹不一致')
    half=np.load(bp,allow_pickle=False)
    if (half.shape!=(len(tp),RANK_END-RANK_START+1) or half.dtype.kind not in 'iu' or
        np.any(half<0) or np.any(half>=len(ip)) or np.any(ip[half]==tp[:,None])):
        raise RuntimeError('旧B2负样本池非法')
    baseval=read_json(base/'01_最佳IRRA_官方Val基线.json')
    if not all(baseval.get('checks',{}).get(k) is True for k in METRICS):
        raise RuntimeError('第六步B0原始官方指标核验失败')
    # 训练源权重 + 官方验证缓存的SHA只读绑定本实验；第六步任何文件均不写入。
    history={}
    for method in METHODS_OLD:
        for seed in SEEDS:
            metrics_path=base/f'02_{method}_seed{seed}_metrics.json'
            csv_path=base/f'03_{method}_seed{seed}_per_query.csv'
            weights_path=base/f'{method}_seed{seed}_progress.pt'
            if not all(p.is_file() for p in (metrics_path,csv_path,weights_path)):
                raise FileNotFoundError(f'第六步历史指标/逐查询/权重缺失: {method} seed{seed}')
            item=read_json(metrics_path)
            if (item.get('fingerprint')!=fp or item.get('summary',{}).get('method')!=method or
                item['summary'].get('seed')!=seed or item.get('csv_sha256')!=sha256(csv_path)):
                raise RuntimeError(f'第六步{method} seed{seed}指标/CSV指纹不正确')
            history[(method,seed)]=item['summary']
    sid=fingerprint(dict(version=VERSION,ident=ident,origin_fingerprint=fp,
         cache_shas=dict(train_images=hi,train_texts=ht,val_images=hvi,val_texts=hvt,
                        half_pool=sha256(bp)),
         old_weight_shas={f'{m}_{s}':sha256(base/f'{m}_seed{s}_progress.pt') for m in METHODS_OLD for s in SEEDS},
         old_report_shas={f'{m}_{s}':sha256(base/f'02_{m}_seed{s}_metrics.json') for m in METHODS_OLD for s in SEEDS},
         val_pids=fingerprint(dict(t=qp.tolist(),v=vp.tolist())),params=dict(seeds=SEEDS,
         alphas=ALPHAS,steps=TRAIN_STEPS,num_neg=NUM_NEG,rank_start=RANK_START,rank_end=RANK_END)))
    return dict(base=base,ident=ident,fp=fp,id=sid,ip=ip,tp=tp,positive=positive,qp=qp,vp=vp,
                tri=tr_img,trt=tr_txt,vi=val_img,vt=val_txt,half=half,history=history,
                val_base_reference=baseval['measured'])


def find_step9(src):
    root=ROOT/'硬负样本分析'
    if SOURCE_STEP9_DIR:
        candidates=[Path(SOURCE_STEP9_DIR).resolve()]
    else:
        candidates=sorted((p for p in root.glob(STEP9_PREFIX+'_*') if p.is_dir()),
                          key=lambda p:p.stat().st_mtime,reverse=True)
    valid=[]
    for p in candidates:
        if root.resolve() not in p.parents or not p.name.startswith(STEP9_PREFIX+'_'):
            continue
        manifest=p/'00_方案和来源.json'
        if not manifest.is_file():continue
        x=read_json(manifest)
        if (x.get('source_id')!=src['id'] or x.get('version')!=VERSION or
            x.get('completed') is not True or x.get('test_used') is not False or
            x.get('train_only_mining') is not True or x.get('seeds')!=list(SEEDS) or
            x.get('all_result_rows')!=31 or x.get('query_rows')!=31*len(src['qp'])):
            continue
        required=[p/'01_Train错误排序分布.json',p/'02_所有方案_Val指标.csv',
                  p/'03_全部逐查询_Val.csv',p/'04_三种子汇总及PID_Bootstrap.json',
                  p/'internal_cache'/'train_wrong_ahead_top50.npz']
        required += [p/f'{m}_seed{seed}_progress.pt' for m in METHODS_PRIOR_NEW for seed in SEEDS]
        if not all(r.is_file() for r in required):continue
        valid.append(p)
    if not valid:
        raise FileNotFoundError('找不到带完整训练权重和Val结果的第九步目录。请设置 SOURCE_STEP9_DIR；不能只用第九步上传ZIP代替权重。')
    if len(valid)>1 and not SOURCE_STEP9_DIR:
        say('[提示] 找到多个符合指纹的第九步目录，自动使用最后修改时间最新的：',valid[0])
    return valid[0]

def source_from_step9(src):
    p=find_step9(src)
    say('[第九步只读目录]',p)
    w=p/'internal_cache'/'train_wrong_ahead_top50.npz'
    wanted=fingerprint(dict(source=src['id'],rule='negative_similarity_gt_best_same_pid_similarity',top=RANK_END))
    with np.load(w,allow_pickle=False) as z:
        if set(z.files)!={'fingerprint','top_idx','wrong_count','best_positive','best_negative'}:
            raise RuntimeError('第九步Train挖掘池字段不正确')
        if str(z['fingerprint'].item())!=wanted:
            raise RuntimeError('第九步Train挖掘池身份指纹不匹配')
        wrong={key:np.array(z[key],copy=True) for key in ('top_idx','wrong_count','best_positive','best_negative')}
    n=len(src['tp']);idx=wrong['top_idx']
    if (idx.shape!=(n,RANK_END) or idx.dtype.kind not in 'iu' or
        wrong['wrong_count'].shape!=(n,) or wrong['best_positive'].shape!=(n,) or
        wrong['best_negative'].shape!=(n,) or np.any(idx<0) or np.any(idx>=len(src['ip'])) or
        np.any(src['ip'][idx]==src['tp'][:,None]) or
        np.any(wrong['wrong_count']<0) or np.any(wrong['wrong_count']>RANK_END) or
        not np.isfinite(wrong['best_positive']).all() or not np.isfinite(wrong['best_negative']).all() or
        np.any((wrong['wrong_count']>0)!=(wrong['best_negative']>wrong['best_positive']+1e-7))):
        raise RuntimeError('第九步挖掘池数据不合法')
    # Top8从旧Top50前8个直接复用，严格不使用正负分数间隔做训练抽样
    if RANK_END<NUM_NEG:raise RuntimeError('旧TopK不足8')
    # 抽查原相似度排序，防止已完成缓存只是形状正确但内容被误改
    probe=np.unique(np.linspace(0,n-1,min(41,n),dtype=int))
    for q in probe:
        scores=src['tri'] @ src['trt'][q]
        scores=np.array(scores,copy=True);scores[src['ip']==src['tp'][q]]=-np.inf
        tops=idx[q,:NUM_NEG]
        if np.any(np.diff(scores[tops])>2e-5):
            raise RuntimeError('第九步Top8顺序与原IRRA特征不一致')
        if np.max(scores)>scores[tops[0]]+2e-5:
            raise RuntimeError('第九步Top1不是最相似异身份')
        true_top8=np.sort(scores)[-NUM_NEG:][::-1]
        if not np.allclose(scores[tops],true_top8,rtol=0,atol=2e-5):
            raise RuntimeError('第九步缓存前8个并非真实Top8异身份图像')
    # 预先锁定旧报告+六份权重+挖掘池真实字节身份，旧文件变化将拒绝复用新缓存
    history_paths=[p/'00_方案和来源.json',p/'02_所有方案_Val指标.csv',p/'03_全部逐查询_Val.csv',w]
    history_paths += [p/f'{m}_seed{seed}_progress.pt' for m in METHODS_PRIOR_NEW for seed in SEEDS]
    filehash={str(z.relative_to(p)):sha256(z) for z in history_paths}
    src['step9_dir']=p
    src['step10_source_id']=fingerprint(dict(step10_version=STEP10_VERSION,original_step9_id=src['id'],
        historical_sha=filehash,comparison='unconditional_top1_to_8_all_queries',
        num_neg=NUM_NEG,train_steps=TRAIN_STEPS,seeds=SEEDS,alphas=ALPHAS))
    src['step9_hashes']=filehash
    return wrong

def load_step9_model(src,method,seed,device):
    assert method in METHODS_PRIOR_NEW
    p=src['step9_dir']/f'{method}_seed{seed}_progress.pt'
    state=torch.load(str(p),map_location='cpu',weights_only=False)
    expected=fingerprint(dict(source=src['id'],method=method,seed=seed,version=VERSION))
    if (state.get('fingerprint')!=expected or state.get('source_id')!=src['id'] or
        state.get('method')!=method or state.get('seed')!=seed or state.get('step')!=TRAIN_STEPS):
        raise RuntimeError(f'第九步 {method} seed{seed} 权重身份不一致')
    ad=Both().to(device);ad.load_state_dict(state['model'],strict=True);ad.eval()
    for v in ad.parameters():v.requires_grad_(False)
    return ad

def load_historical_cases(src,device):
    p=src['step9_dir']
    expected={('B0_original',0,0.0)}
    expected.update((m,s,a) for m in METHODS_OLD+METHODS_PRIOR_NEW for s in SEEDS for a in ALPHAS)
    rows={}
    with (p/'02_所有方案_Val指标.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            key=(r['method'],int(r['seed']),float(r['alpha']))
            if key not in expected or key in rows:raise RuntimeError('第九步汇总存在非法/重复组 '+str(key))
            rows[key]=r
    if set(rows)!=expected:raise RuntimeError('第九步31组指标不完整')
    n=len(src['qp']);data={k:dict(rank=np.empty(n,np.int32),AP=np.empty(n),INP=np.empty(n)) for k in expected}
    next_idx={k:0 for k in expected}
    with (p/'03_全部逐查询_Val.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            key=(r['method'],int(r['seed']),float(r['alpha']))
            if key not in data:raise RuntimeError('第九步逐查询结果包含未知组')
            i=int(r['query_index'])
            if i!=next_idx[key] or i>=n or int(r['pid'])!=int(src['qp'][i]):
                raise RuntimeError('第九步逐查询顺序/身份/唯一性异常')
            d=data[key];d['rank'][i]=int(r['rank']);d['AP'][i]=float(r['AP']);d['INP'][i]=float(r['INP'])
            if int(r['correct'])!=int(d['rank'][i]==1):raise RuntimeError('第九步R1正误字段不一致')
            next_idx[key]+=1
    if any(v!=n for v in next_idx.values()):raise RuntimeError('第九步逐查询记录缺失')
    for key,d in data.items():
        validate_detail(d,n,len(src['vp']))
        metrics=get_metrics(d)
        if max(abs(metrics[k]-float(rows[key][k])) for k in METRICS)>1e-5:
            raise RuntimeError('第九步汇总与逐查询指标不一致：'+str(key))
    # B0重算：不能只信旧CSV，同时检查源模型和图库未变
    actual=evaluate(src['vt'],src['vi'],src['qp'],src['vp'],device)
    assert_equal_detail(actual,data[('B0_original',0,0.0)],'B0特征与第九步记录')
    check_close(get_metrics(actual),src['val_base_reference'],'Epoch58 B0')
    # 必须验证第九步H1/H2真实权重与其记录一致，避免只是复制历史数字
    for method in METHODS_PRIOR_NEW:
        for seed in SEEDS:
            ad=load_step9_model(src,method,seed,device)
            for alpha in ALPHAS:
                t=apply_adapter(src['vt'],ad.text,alpha,device)
                v=apply_adapter(src['vi'],ad.image,alpha,device)
                got=evaluate(t,v,src['qp'],src['vp'],device)
                assert_equal_detail(got,data[(method,seed,alpha)],f'第九步{method} seed{seed} alpha{alpha}')
            del ad
    # 重算第六步B2关键对照，确认历史数据和权重绑定
    for seed in SEEDS:
        ad=load_old_adapter(src,'B2_hard',seed,device)
        for alpha in ALPHAS:
            got=evaluate(apply_adapter(src['vt'],ad.text,alpha,device),
                         apply_adapter(src['vi'],ad.image,alpha,device),src['qp'],src['vp'],device)
            assert_equal_detail(got,data[('B2_hard',seed,alpha)],f'第六步B2 seed{seed} alpha{alpha}')
        del ad
    say('[历史结果复用完成] 31组，',31*n,'条查询；B0、H1/H2和B2已由真实权重复现')
    return data

def assert_equal_detail(got,ref,label):
    if (not np.array_equal(got['rank'],ref['rank']) or
        not np.allclose(got['AP'],ref['AP'],rtol=0,atol=2e-6) or
        not np.allclose(got['INP'],ref['INP'],rtol=0,atol=2e-6)):
        raise RuntimeError(label+'逐查询排名/AP/INP无法复现，禁止继续训练')

def prepare_output(src):
    root=ROOT/'硬负样本分析';root.mkdir(parents=True,exist_ok=True)
    if RESUME_OUTPUT_DIR:
        out=Path(RESUME_OUTPUT_DIR).resolve()
        if (not out.is_dir() or root.resolve() not in out.parents or
            not out.name.startswith(OUTPUT_PREFIX+'_') or out==src['base']):
            raise RuntimeError('续跑路径必须是独立的第十步实验目录')
        manifest=out/'00_方案和来源.json'
        if not manifest.is_file() or read_json(manifest).get('source_id')!=src['step10_source_id']:
            raise RuntimeError('第十步源指纹不匹配，拒绝混用已有实验')
        return out
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    out=root/(OUTPUT_PREFIX+'_'+stamp);i=2
    while out.exists():out=root/(OUTPUT_PREFIX+f'_{stamp}_{i}');i+=1
    out.mkdir()
    return out

class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1=nn.Linear(DIM,BOTTLENECK)
        self.l2=nn.Linear(BOTTLENECK,DIM)
        nn.init.zeros_(self.l2.weight);nn.init.zeros_(self.l2.bias)
    def forward(self,x,alpha=1.0):
        return F.normalize(x+float(alpha)*self.l2(F.gelu(self.l1(x))),dim=-1)

class Both(nn.Module):
    def __init__(self):
        super().__init__();self.text=Residual();self.image=Residual()

def load_old_adapter(src,method,seed,device):
    path=src['base']/f'{method}_seed{seed}_progress.pt'
    st=torch.load(str(path),map_location='cpu',weights_only=False)
    if (st.get('fingerprint')!=src['fp'] or st.get('method')!=method or
       st.get('seed')!=seed or st.get('step')!=TRAIN_STEPS):
        raise RuntimeError(f'第六步{method} seed{seed}最后检查点不匹配')
    ad=Both().to(device);ad.load_state_dict(st['model'],strict=True);ad.eval()
    for p in ad.parameters():p.requires_grad_(False)
    return ad

def mine_wrong_ahead(trt,tri,tp,ip,device,rank_end=RANK_END,chunk=TRAIN_QUERY_CHUNK):
    """全Train图库排除同PID，用【最好同身份图】作阈值，避免把另一张正图当错误目标。"""
    n=len(trt);m=len(tri)
    if m-int(np.unique(ip,return_counts=True)[1].max())<rank_end:
        raise RuntimeError('训练图库不同PID负样本数不足指定的TopK，禁止把同ID/-inf混入池')
    k=rank_end
    top_idx=np.empty((n,k),np.int32)
    violating=np.zeros(n,np.int16)
    posmax=np.empty(n,np.float32)
    negmax=np.empty(n,np.float32)
    gallery=torch.as_tensor(np.ascontiguousarray(tri),device=device,dtype=torch.float32)
    ip_t=torch.as_tensor(ip,device=device,dtype=torch.int64)
    with torch.inference_mode():
        for start in range(0,n,chunk):
            end=min(start+chunk,n)
            q=torch.as_tensor(np.ascontiguousarray(trt[start:end]),device=device,dtype=torch.float32)
            qpid=torch.as_tensor(tp[start:end],device=device,dtype=torch.int64)
            scores=q @ gallery.T
            is_pos=(ip_t[None,:]==qpid[:,None])
            ps=scores.masked_fill(~is_pos,-float('inf')).max(dim=1).values
            if not torch.isfinite(ps).all():raise RuntimeError('训练文本缺少同ID图像')
            scores.masked_fill_(is_pos,-float('inf'))
            values,indices=torch.topk(scores,k=k,dim=1,largest=True,sorted=True)
            top_idx[start:end]=indices.cpu().numpy().astype(np.int32)
            v=(values > ps[:,None]+1e-7).sum(dim=1)
            violating[start:end]=v.cpu().numpy().astype(np.int16)
            posmax[start:end]=ps.cpu().numpy().astype(np.float32)
            negmax[start:end]=values[:,0].cpu().numpy().astype(np.float32)
            if start==0 or end%8192<chunk or end==n:
                say('[Train错误排序挖掘]',end,'/',n)
    if np.any(ip[top_idx]==tp[:,None]):raise RuntimeError('错误领先负样本池出现同PID')
    if not np.isfinite(posmax).all() or not np.isfinite(negmax).all():raise RuntimeError('挖掘评分异常')
    return dict(top_idx=top_idx,wrong_count=violating,best_positive=posmax,best_negative=negmax)

def get_wrong_pool(out,src,device):
    p=out/'internal_cache'/'train_wrong_ahead_top50.npz'
    fp=fingerprint(dict(source=src['id'],rule='negative_similarity_gt_best_same_pid_similarity',top=RANK_END))
    if p.is_file():
        with np.load(p,allow_pickle=False) as z:
            if set(z.files)!={'fingerprint','top_idx','wrong_count','best_positive','best_negative'} or str(z['fingerprint'].item())!=fp:
                raise RuntimeError('第九步错误领先池缓存指纹/结构不合法')
            d={k:np.array(z[k],copy=True) for k in ('top_idx','wrong_count','best_positive','best_negative')}
        say('[缓存复用] Train错误领先池')
    else:
        d=mine_wrong_ahead(src['trt'],src['tri'],src['tp'],src['ip'],device,rank_end=RANK_END)
        def write(tmp):
            with tmp.open('wb') as f:
                np.savez_compressed(f,fingerprint=np.array(fp),**d)
                f.flush();os.fsync(f.fileno())
        atomic_write(p,write)
    n=len(src['tp'])
    if (d['top_idx'].shape!=(n,RANK_END) or d['wrong_count'].shape!=(n,) or
        d['best_positive'].shape!=(n,) or d['best_negative'].shape!=(n,) or
        np.any(d['wrong_count']<0) or np.any(d['wrong_count']>RANK_END) or
        np.any(d['top_idx']<0) or np.any(d['top_idx']>=len(src['ip'])) or
        np.any(src['ip'][d['top_idx']]==src['tp'][:,None]) or
        not np.isfinite(d['best_positive']).all() or not np.isfinite(d['best_negative']).all() or
        np.any((d['wrong_count']>0) != (d['best_negative']>d['best_positive']+1e-7))):
        raise RuntimeError('Train错误领先池数据异常')
    summary=dict(total_queries=int(n),wrong_ahead_queries=int(np.sum(d['wrong_count']>0)),
      wrong_ahead_rate=float(np.mean(d['wrong_count']>0)),
      mean_wrong_count_all=float(np.mean(d['wrong_count'])),
      mean_wrong_count_when_exists=float(np.mean(d['wrong_count'][d['wrong_count']>0])) if np.any(d['wrong_count']>0) else 0.0,
      median_top1_margin=float(np.median(d['best_negative']-d['best_positive'])),
      definition='Train原始IRRA余弦分数；异身份score严格高于该查询同身份最好图像score；只存Top50')
    atomic_json(out/'01_Train错误排序分布.json',summary)
    say('[挖掘统计]',summary)
    return d

def choose_negatives(indices,tp,ip,half,wrong,mode,seed,step,num_neg=NUM_NEG):
    """无条件Top8异身份负样本，对所有Train文本一视同仁，不使用正样本阈值。"""
    if mode!='C1_top8_unconditional':raise ValueError('只允许新增C1训练')
    out=np.ascontiguousarray(wrong['top_idx'][indices,:num_neg],dtype=np.int64)
    if out.shape!=(len(indices),num_neg) or np.any(ip[out]==tp[indices,None]):
        raise RuntimeError('Top8负样本包含同PID或形状错误')
    if any(len(set(row.tolist()))!=num_neg for row in out):
        raise RuntimeError('Top8负样本存在重复图像')
    # 字段与第九步训练统计接口兼容，但注意此处没有按错误领先规则挑选
    exceeds=np.arange(num_neg)[None,:]<wrong['wrong_count'][indices,None]
    return out,dict(neg_from_wrong=int(exceeds.sum()),
                    wrong_quota_fallback=0,queries_with_wrong=int(np.count_nonzero(wrong['wrong_count'][indices])))

def train_adapter(src,out,method,seed,wrong,device,stop_at=None):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    ad=Both().to(device)
    opt=torch.optim.AdamW(ad.parameters(),lr=LR,weight_decay=1e-4)
    checkpoint=out/f'{method}_seed{seed}_progress.pt'
    run_fp=fingerprint(dict(source=src['step10_source_id'],method=method,seed=seed,version=STEP10_VERSION))
    start=0;history=[];counter=dict(neg_from_wrong=0,wrong_quota_fallback=0,queries_with_wrong=0)
    if checkpoint.exists():
        st=torch.load(str(checkpoint),map_location='cpu',weights_only=False)
        if (st.get('fingerprint')!=run_fp or st.get('method')!=method or st.get('seed')!=seed or
           st.get('source_id')!=src['step10_source_id'] or not 0<=st.get('step',-1)<=TRAIN_STEPS):
            raise RuntimeError(f'{method} seed{seed}断点指纹/位置不一致')
        ad.load_state_dict(st['model'],strict=True)
        opt.load_state_dict(st['optimizer'])
        # 避免混合GPU/CPU重新保存的optimizer状态
        for state in opt.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):state[k]=v.to(device)
        start=st['step'];history=st['history'];counter=st['counter']
        say('[断点恢复]',method,seed,start,'/',TRAIN_STEPS)
    trt=torch.as_tensor(np.ascontiguousarray(src['trt']),device=device,dtype=torch.float32)
    tri=torch.as_tensor(np.ascontiguousarray(src['tri']),device=device,dtype=torch.float32)
    until=TRAIN_STEPS if stop_at is None else min(int(stop_at),TRAIN_STEPS)
    for step in range(start,until):
        rng=np.random.default_rng(seed*1000003+step*97+3)
        idx=rng.integers(len(src['tp']),size=EMBED_TRAIN_BATCH,dtype=np.int64)
        nidx,stats=choose_negatives(idx,src['tp'],src['ip'],src['half'],wrong,method,seed,step)
        for k,v in stats.items():counter[k]+=v
        it=torch.as_tensor(idx,device=device)
        pos=torch.as_tensor(src['positive'][idx],device=device)
        ni=torch.as_tensor(nidx,device=device)
        x=trt[it];p=tri[pos];n=tri[ni]
        tx=ad.text(x);px=ad.image(p);nx=ad.image(n)
        positive_score=(tx*px).sum(-1,keepdim=True)
        negative_score=torch.einsum('bd,bkd->bk',tx,nx)
        logits=torch.cat([positive_score,negative_score],1)/TEMPERATURE
        loss=F.cross_entropy(logits,torch.zeros(len(idx),device=device,dtype=torch.long))
        preservation=((tx-x).square().sum(-1).mean()+(px-p).square().sum(-1).mean()+
                      (nx-n).square().sum(-1).mean())/3
        obj=loss+PRESERVE*preservation
        if not torch.isfinite(obj):raise RuntimeError(f'{method} seed{seed} step{step}: loss非有限')
        opt.zero_grad(set_to_none=True);obj.backward()
        torch.nn.utils.clip_grad_norm_(ad.parameters(),2.0)
        opt.step()
        if (step+1)%20==0 or step==0:
            history.append(dict(step=step+1,loss=float(loss.detach().cpu()),
               preserve=float(preservation.detach().cpu()),objective=float(obj.detach().cpu())))
        if (step+1)%SAVE_EVERY==0 or step+1==until:
            atomic_torch(checkpoint,dict(fingerprint=run_fp,source_id=src['step10_source_id'],method=method,
                seed=seed,step=step+1,model=ad.state_dict(),optimizer=opt.state_dict(),
                history=history,counter=counter))
            say('[训练]',method,'seed',seed,step+1,'/',TRAIN_STEPS,'loss',round(float(loss.detach().cpu()),5))
    if until==TRAIN_STEPS and start==TRAIN_STEPS:say('[复用权重]',method,seed)
    return ad,history,counter,until

def apply_adapter(x,branch,alpha,device):
    out=[]
    with torch.inference_mode():
        for start in range(0,len(x),ADAPTER_BATCH):
            src=torch.as_tensor(np.ascontiguousarray(x[start:start+ADAPTER_BATCH]),device=device,dtype=torch.float32)
            out.append(branch(src,alpha).cpu().numpy())
    arr=np.concatenate(out,axis=0)
    if not np.isfinite(arr).all() or arr.shape!=x.shape:raise RuntimeError('Adapter输出损坏')
    return arr

def evaluate(t,v,qp,vp,device):
    """与第六/八步相同的全集余弦检索指标；返回逐查询rank/AP/INP。"""
    if t.shape!=(len(qp),DIM) or v.shape!=(len(vp),DIM):raise RuntimeError('评估shape错误')
    ranks=np.empty(len(qp),dtype=np.int32);aps=np.empty(len(qp),np.float64);inps=np.empty(len(qp),np.float64)
    gallery=torch.as_tensor(np.ascontiguousarray(v),device=device,dtype=torch.float32)
    positions=np.arange(1,len(vp)+1,dtype=np.float64)[None,:]
    with torch.inference_mode():
        for start in range(0,len(qp),EVAL_QUERY_CHUNK):
            end=min(start+EVAL_QUERY_CHUNK,len(qp))
            block=torch.as_tensor(np.ascontiguousarray(t[start:end]),device=device,dtype=torch.float32)
            score=(block @ gallery.T).float().cpu().numpy()
            if not np.isfinite(score).all():raise RuntimeError('评估分数含NaN/Inf')
            order=np.argsort(-score,axis=1,kind='stable')
            hits=vp[order]==qp[start:end,None]
            n=hits.sum(1)
            if not np.all(n>0):raise RuntimeError('Val存在无正图的查询')
            ranks[start:end]=np.argmax(hits,axis=1)+1
            aps[start:end]=(((np.cumsum(hits,axis=1)/positions)*hits).sum(1)/n)
            inps[start:end]=n/(len(vp)-np.argmax(hits[:,::-1],axis=1))
    return dict(rank=ranks,AP=aps,INP=inps)

def get_metrics(d):
    return dict(R1=100*float(np.mean(d['rank']==1)),R5=100*float(np.mean(d['rank']<=5)),
       R10=100*float(np.mean(d['rank']<=10)),mAP=100*float(np.mean(d['AP'])),
       mINP=100*float(np.mean(d['INP'])))

def validate_detail(d,n,gallery):
    if set(d)!=set(('rank','AP','INP')):raise RuntimeError('查询指标缺失字段')
    if (d['rank'].shape!=(n,) or d['AP'].shape!=(n,) or d['INP'].shape!=(n,) or
       np.any(d['rank']<1) or np.any(d['rank']>gallery) or
       not np.isfinite(d['AP']).all() or not np.isfinite(d['INP']).all() or
       np.any(d['AP']<0) or np.any(d['AP']>1+1e-6) or
       np.any(d['INP']<0) or np.any(d['INP']>1+1e-6)):
        raise RuntimeError('查询指标不合法')

def cache_eval(src,out,key,compute):
    p=out/'internal_cache'/('eval_'+key+'.npz')
    # 评估缓存除实验指纹外，还绑定最终权重实际SHA256：权重被替换/误写时不能复用旧指标。
    match=re.fullmatch(r'(.+)_seed(\d+)_alpha(-?\d+\.\d+)',key)
    weight_sha=None
    if match:
        method,seed,_=match.groups()
        path=(out if method in METHODS_NEW else src['base'])/f'{method}_seed{seed}_progress.pt'
        if not path.is_file():raise FileNotFoundError(f'缓存评估要求最终Adapter权重存在: {path}')
        weight_sha=sha256(path)
    fp=fingerprint(dict(source=src['step10_source_id'],key=key,version=STEP10_VERSION,adapter_weight_sha256=weight_sha))
    if p.is_file():
        with np.load(p,allow_pickle=False) as z:
            if set(z.files)!={'fingerprint','rank','AP','INP'} or str(z['fingerprint'].item())!=fp:
                raise RuntimeError('逐查询评估缓存指纹不一致: '+str(p))
            d={k:np.array(z[k],copy=True) for k in ('rank','AP','INP')}
        say('[已完成评估复用]',key)
    else:
        d=compute()
        def save(temp):
            with temp.open('wb') as f:
                np.savez_compressed(f,fingerprint=np.array(fp),**d)
                f.flush();os.fsync(f.fileno())
        atomic_write(p,save)
    validate_detail(d,len(src['qp']),len(src['vp']))
    return d

def check_close(got,want,label,tolerance=0.061):
    diffs={k:abs(got[k]-float(want[k])) for k in METRICS}
    say('[历史复现]',label,'最大偏差',f'{max(diffs.values()):.6f} pp')
    if any(delta>tolerance for delta in diffs.values()):
        raise RuntimeError(f'{label} 历史Val无法复现: {diffs}')

def bootstrap_pid(d,ref,pids,n_boot=BOOT_REPEATS):
    # 注意是多个训练seed的均值查询级差，按身份簇重采样；区间仅为探索性，不包含多种子训练方差/多重比较。
    values=np.stack(((d['rank']==1)*100,(d['rank']<=5)*100,(d['rank']<=10)*100,
                     d['AP']*100,d['INP']*100),axis=1)
    baseline=np.stack(((ref['rank']==1)*100,(ref['rank']<=5)*100,(ref['rank']<=10)*100,
                       ref['AP']*100,ref['INP']*100),axis=1)
    delta=values-baseline
    ids,inverse=np.unique(pids,return_inverse=True)
    group_sums=np.zeros((len(ids),5),np.float64)
    group_counts=np.bincount(inverse,minlength=len(ids))
    np.add.at(group_sums,inverse,delta)
    rng=np.random.default_rng(20261021)
    out=np.empty((n_boot,5))
    for i in range(n_boot):
        take=rng.integers(len(ids),size=len(ids))
        out[i]=group_sums[take].sum(0)/group_counts[take].sum()
    return {k:dict(delta_pp=float(delta[:,j].mean()),ci95=[float(x) for x in np.quantile(out[:,j],[0.025,0.975])])
            for j,k in enumerate(METRICS)}

def historical_eval(src,out,method,seed,alpha,device):
    ad=load_old_adapter(src,method,seed,device)
    try:
        t=apply_adapter(src['vt'],ad.text,alpha,device)
        v=apply_adapter(src['vi'],ad.image,alpha,device)
        return evaluate(t,v,src['qp'],src['vp'],device)
    finally:
        del ad

def new_eval(src,out,method,seed,alpha,device):
    # train_adapter 正常已完成；加载自己的最终权重，并验证method/step/source指纹
    p=out/f'{method}_seed{seed}_progress.pt'
    st=torch.load(str(p),map_location='cpu',weights_only=False)
    expected=fingerprint(dict(source=src['step10_source_id'],method=method,seed=seed,version=STEP10_VERSION))
    if (st.get('fingerprint')!=expected or st.get('source_id')!=src['step10_source_id'] or
       st.get('method')!=method or st.get('seed')!=seed or st.get('step')!=TRAIN_STEPS):
        raise RuntimeError(f'{method} seed{seed}最终训练权重不完整')
    ad=Both().to(device);ad.load_state_dict(st['model'],strict=True);ad.eval()
    try:
        t=apply_adapter(src['vt'],ad.text,alpha,device)
        v=apply_adapter(src['vi'],ad.image,alpha,device)
        return evaluate(t,v,src['qp'],src['vp'],device)
    finally:
        del ad

def write_reports(src,out,cases,wrong):
    baseline=cases[('B0_original',0,0.0)]
    rows=[];query_rows=[]
    for method,seed,alpha in sorted(cases.keys(),key=lambda x:(x[0],x[1],x[2])):
        d=cases[(method,seed,alpha)];m=get_metrics(d)
        rec=int(np.sum((baseline['rank']!=1)&(d['rank']==1)))
        reg=int(np.sum((baseline['rank']==1)&(d['rank']!=1)))
        rows.append(dict(method=method,seed=seed,alpha=alpha,**m,
             recovered=rec,regressed=reg,net_correct_delta=rec-reg))
        for i,pid in enumerate(src['qp']):
            query_rows.append(dict(method=method,seed=seed,alpha=alpha,query_index=i,pid=int(pid),
               rank=int(d['rank'][i]),correct=int(d['rank'][i]==1),AP=float(d['AP'][i]),INP=float(d['INP'][i])))
    atomic_csv(out/'02_所有方案_Val指标.csv',
        ['method','seed','alpha',*METRICS,'recovered','regressed','net_correct_delta'],rows)
    atomic_csv(out/'03_全部逐查询_Val.csv',
        ['method','seed','alpha','query_index','pid','rank','correct','AP','INP'],query_rows)
    stats={}
    for alpha in ALPHAS:
        for method in METHODS_OLD+METHODS_PRIOR_NEW+METHODS_NEW:
            group=[cases[(method,s,alpha)] for s in SEEDS]
            means={k:float(np.mean([get_metrics(d)[k] for d in group])) for k in METRICS}
            # 按query先求三个seed的每项指标值，再相对于同一份B0重采样
            pseudo=dict(rank=group[0]['rank'],AP=np.mean([d['AP'] for d in group],axis=0),
                        INP=np.mean([d['INP'] for d in group],axis=0))
            # R1/R5/R10不能用整数rank的平均表示：以三种子query级命中比例聚类抽样
            for key in METRICS:
                if key.startswith('R'):
                    cutoff=int(key[1:])
                    val=np.stack([(d['rank']<=cutoff).astype(np.float64)*100 for d in group]).mean(0)
                    bas=(baseline['rank']<=cutoff).astype(np.float64)*100
                elif key=='mAP':
                    val=pseudo['AP']*100;bas=baseline['AP']*100
                else:
                    val=pseudo['INP']*100;bas=baseline['INP']*100
                ids,inv=np.unique(src['qp'],return_inverse=True)
                counts=np.bincount(inv,minlength=len(ids))
                sums=np.bincount(inv,weights=val-bas,minlength=len(ids))
                rng=np.random.default_rng(20261021)
                boots=np.empty(BOOT_REPEATS)
                for z in range(BOOT_REPEATS):
                    draw=rng.integers(len(ids),size=len(ids))
                    boots[z]=sums[draw].sum()/counts[draw].sum()
                stats.setdefault(method,{})[str(alpha)]=stats.get(method,{}).get(str(alpha),{})|{
                    key:dict(mean=means[key],delta_pp=float(np.mean(val-bas)),
                             ci95=[float(x) for x in np.quantile(boots,[0.025,0.975])])}
    atomic_json(out/'04_三种子汇总及PID_Bootstrap.json',
        dict(note='第十步为Val探索性对照。PID bootstrap仅量化身份采样，不含训练不确定性、多重选择或Test泛化。',
             primary_alpha=1.0,secondary_pre_registered_alpha=0.5,methods=stats))
    # H1/H2 vs B2: 决策以三个种子同种seed严格配对比较，不比较不同α造成的选择偏差
    pair=[]
    for alpha in ALPHAS:
        for method in METHODS_NEW:
            for ref_name in ('B2_hard','H1_wrong_ahead','H2_wrong_plus_semi'):
              for seed in SEEDS:
                x=get_metrics(cases[(method,seed,alpha)])
                b=get_metrics(cases[(ref_name,seed,alpha)])
                pair.append(dict(alpha=alpha,method=method,reference=ref_name,seed=seed,
                                 **{'delta_'+k:float(x[k]-b[k]) for k in METRICS}))
    atomic_csv(out/'05_新方案对旧方法_同种子配对差.csv',
               ['alpha','method','reference','seed',*['delta_'+k for k in METRICS]],pair)
    # 保存旧实验成果连续性：B3的第八步双分支α0.5结果仍在源权重评估中，不废弃。
    manifest=read_json(out/'00_方案和来源.json')
    manifest['all_result_rows']=len(rows)
    manifest['query_rows']=len(query_rows)
    manifest['completed']=True
    atomic_json(out/'00_方案和来源.json',manifest)
    say('[报告完成] 指标',len(rows),'组，逐查询',len(query_rows),'条')
    return rows,stats

def package(out):
    dest=out/ZIP_NAME;temp=out/('.'+ZIP_NAME+'.partial')
    names=[]
    try:
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(out.iterdir()):
                if not p.is_file() or p==dest or p==temp or p.suffix.lower() in ('.pt','.npy','.npz'):
                    continue
                z.write(p,p.name);names.append(p.name)
            z.write(Path(__file__).resolve(),'脚本/'+Path(__file__).name)
        with zipfile.ZipFile(temp) as z:
            bad=z.testzip()
            if bad:raise RuntimeError('ZIP CRC失败 '+bad)
            if not {'02_所有方案_Val指标.csv','03_全部逐查询_Val.csv','04_三种子汇总及PID_Bootstrap.json','05_新方案对旧方法_同种子配对差.csv','07_采样负样本难度_固定Step诊断.csv'}.issubset(set(z.namelist())):
                raise RuntimeError('ZIP遗漏必要结果')
        os.replace(temp,dest)
    finally:
        if temp.exists():temp.unlink()
    say('报告ZIP：',dest)

def probe_difficulty(src,wrong,out):
    """确定性抽样固定训练step，诊断真实负样本分数和错误领先比例。没有反传，无Val参与。"""
    records=[]
    qsteps=tuple(x for x in PROBE_STEPS if x<TRAIN_STEPS)
    for mode in METHODS_PRIOR_NEW+METHODS_NEW:
        for seed in SEEDS:
            total=0;score_sum=0.;gap_sum=0.;wrong_exposure=0;top8_exposure=0
            for step in qsteps:
                rng=np.random.default_rng(seed*1000003+step*97+3)
                idx=rng.integers(len(src['tp']),size=EMBED_TRAIN_BATCH,dtype=np.int64)
                if mode in METHODS_PRIOR_NEW:
                    # 仅诊断时按第九步原抽样精确重放；训练新C1不调用此方法
                    neg=choose_old_negatives(idx,src['tp'],src['ip'],src['half'],wrong,mode,seed,step)
                else:
                    neg=choose_negatives(idx,src['tp'],src['ip'],src['half'],wrong,mode,seed,step)[0]
                text=src['trt'][idx];img=src['tri'][neg]
                scores=np.einsum('bd,bkd->bk',text,img,optimize=True)
                margin=scores-wrong['best_positive'][idx,None]
                wrong_exposure+=int(np.count_nonzero(margin>1e-7))
                top8_exposure+=int(np.count_nonzero(np.any(neg[:,:,None]==wrong['top_idx'][idx,None,:8],axis=2)))
                total+=int(scores.size)
                score_sum+=float(scores.sum(dtype=np.float64))
                gap_sum+=float(margin.sum(dtype=np.float64))
            records.append(dict(method=mode,seed=seed,probed_steps=len(qsteps),
                negative_selections=total,mean_cosine=score_sum/total,
                mean_minus_best_positive=gap_sum/total,
                share_wrong_ahead=wrong_exposure/total,
                share_top8=top8_exposure/total))
    atomic_csv(out/'07_采样负样本难度_固定Step诊断.csv',
               ['method','seed','probed_steps','negative_selections','mean_cosine',
                'mean_minus_best_positive','share_wrong_ahead','share_top8'],records)
    return records

def choose_old_negatives(indices,tp,ip,half,wrong,mode,seed,step):
    """仅历史H1/H2训练曝光统计使用，重演第九步已定抽样，不训练旧模型。"""
    if mode not in METHODS_PRIOR_NEW:raise ValueError('unknown prior mode')
    rng=np.random.default_rng(seed*1000003+step*97+19)
    res=np.empty((len(indices),NUM_NEG),dtype=np.int64)
    desired=NUM_NEG if mode=='H1_wrong_ahead' else NUM_NEG//2
    for j,q in enumerate(indices):
        used=min(desired,int(wrong['wrong_count'][q]))
        if used:res[j,:used]=rng.choice(wrong['top_idx'][q,:int(wrong['wrong_count'][q])],size=used,replace=False)
        if used<NUM_NEG:
            available=half[q][~np.isin(half[q],res[j,:used])]
            res[j,used:]=rng.choice(available,size=NUM_NEG-used,replace=False)
    if np.any(ip[res]==tp[indices,None]):raise RuntimeError('旧负样本重演出现同PID')
    return res

def run(src,out,device,wrong=None):
    if wrong is None:wrong=source_from_step9(src)
    manifest=out/'00_方案和来源.json'
    if manifest.exists():
        if read_json(manifest).get('source_id')!=src['step10_source_id']:
            raise RuntimeError('第十步目录包含不兼容实验，拒绝复用')
    else:
        atomic_json(manifest,dict(version=STEP10_VERSION,source_id=src['step10_source_id'],
            epoch58_source_id=src['id'],step6_folder=str(src['base']),step9_folder=str(src['step9_dir']),
            previous_weight_and_report_hashes=src['step9_hashes'],
            purpose='检验H1相对B2的增益是否仅由高相似度异身份Top8难度造成',
            independent_group='C1_top8_unconditional',prior_groups=['B0',*METHODS_OLD,*METHODS_PRIOR_NEW],
            fixed_alpha_primary=1.0,fixed_alpha_secondary=0.5,seeds=list(SEEDS),
            train_steps=TRAIN_STEPS,loss_and_adapter='完全同第九步',
            only_new_variable='negative_sample_selection',
            caveat='Top8固定集合与H1/H2可变难度+回退池不完全匹配；本实验不能单独证明因果机制',
            test_used=False,train_only_mining=True))
    cases=load_historical_cases(src,device)
    stats_probe=probe_difficulty(src,wrong,out)
    say('[已复用第六至九步] 31组；接下来只训练3个C1模型')
    for seed in SEEDS:
        ad,history,counter,last=train_adapter(src,out,'C1_top8_unconditional',seed,wrong,device)
        if last!=TRAIN_STEPS:raise RuntimeError('C1训练未结束')
        del ad
        if device.type=='cuda':torch.cuda.empty_cache()
        atomic_json(out/f'06_C1_top8_unconditional_seed{seed}_训练统计.json',
                    dict(method='C1_top8_unconditional',seed=seed,steps=last,
                         counter=counter,history=history))
        for alpha in ALPHAS:
            key=f'C1_top8_unconditional_seed{seed}_alpha{alpha:.2f}'
            d=cache_eval(src,out,key,lambda s=seed,a=alpha:new_eval(src,out,'C1_top8_unconditional',s,a,device))
            cases[('C1_top8_unconditional',seed,alpha)]=d
    rows,stats=write_reports(src,out,cases,wrong)
    if len(rows)!=37:raise RuntimeError('结果应该有31个历史组+6个新组=37')
    package(out)
    say('[主要比较 α=1；α=0.5为历史固定辅助分析，不能再搜索]')
    for method in ('B2_hard','H1_wrong_ahead','H2_wrong_plus_semi','C1_top8_unconditional'):
        for a in ALPHAS:
            mm=stats[method][str(a)]
            say(method,'alpha',a,'R1',round(mm['R1']['mean'],4),
                'mAP',round(mm['mAP']['mean'],4),'R5',round(mm['R5']['mean'],4))
    say('[难度对照]',[(x['method'],x['seed'],round(x['share_wrong_ahead'],3)) for x in stats_probe])
    return rows

def self_test():
    """CPU合成全链路：历史31组只读复用、C1训练+续跑、指标、报告、ZIP和缓存篡改拦截。"""
    prev={k:globals()[k] for k in ('ROOT','SOURCE_STEP9_DIR','TRAIN_STEPS','SAVE_EVERY',
                                   'EMBED_TRAIN_BATCH','RANK_END','PROBE_STEPS','BOOT_REPEATS')}
    try:
        globals()['TRAIN_STEPS']=4
        globals()['SAVE_EVERY']=2
        globals()['EMBED_TRAIN_BATCH']=12
        globals()['RANK_END']=12
        globals()['PROBE_STEPS']=(0,1,2,3)
        globals()['BOOT_REPEATS']=40
        device=torch.device('cpu')
        rng=np.random.default_rng(20261021)
        ip=np.repeat(np.arange(6),3).astype(np.int64)
        tp=ip.copy();pos=np.arange(len(ip),dtype=np.int64)
        vi=rng.normal(size=(18,DIM)).astype(np.float32)
        vi/=np.linalg.norm(vi,axis=1,keepdims=True)
        vt=vi.copy();vt[0]=vi[4].copy()
        half=np.stack([np.flatnonzero(ip!=pid)[:10] for pid in tp]).astype(np.int32)
        wrong=mine_wrong_ahead(vt,vi,tp,ip,device,rank_end=12,chunk=5)
        assert wrong['wrong_count'][0]>0 and wrong['wrong_count'][1]==0
        q=np.arange(18)
        for seed in SEEDS:
            chosen,ctr=choose_negatives(q,tp,ip,half,wrong,'C1_top8_unconditional',seed,0)
            assert np.array_equal(chosen,wrong['top_idx'][:,:NUM_NEG])
            assert np.all(ip[chosen]!=tp[:,None])
        base=evaluate(vt,vi,tp,ip,device)
        say('[自检1/5] Top8选择、异PID约束、B0指标完成')
        with tempfile.TemporaryDirectory() as folder:
            temp=Path(folder)
            globals()['ROOT']=temp/'IRRA-main'
            root=ROOT/'硬负样本分析'
            root.mkdir(parents=True)
            old6=root/'第六步_模拟历史成果';old6.mkdir()
            old9=root/(STEP9_PREFIX+'_模拟历史成果');old9.mkdir()
            globals()['SOURCE_STEP9_DIR']=str(old9)
            src=dict(id='synthetic_step9_validated_id',base=old6,fp='synthetic_original_cache',
                     ip=ip,tp=tp,positive=pos,trt=vt,tri=vi,half=half,
                     vt=vt,vi=vi,qp=tp,vp=ip,val_base_reference=get_metrics(base),
                     history={},synthetic=True)
            for method in METHODS_OLD+METHODS_PRIOR_NEW:
                for seed in SEEDS:
                    torch.manual_seed(seed)
                    ad=Both().to(device)
                    if method in METHODS_PRIOR_NEW:
                        p=old9/f'{method}_seed{seed}_progress.pt'
                        checksum=fingerprint(dict(source=src['id'],method=method,seed=seed,version=VERSION))
                        payload=dict(fingerprint=checksum,source_id=src['id'],method=method,
                                     seed=seed,step=TRAIN_STEPS,model=ad.state_dict())
                    else:
                        p=old6/f'{method}_seed{seed}_progress.pt'
                        payload=dict(fingerprint=src['fp'],method=method,seed=seed,
                                     step=TRAIN_STEPS,model=ad.state_dict())
                    atomic_torch(p,payload)
            wf=fingerprint(dict(source=src['id'],rule='negative_similarity_gt_best_same_pid_similarity',top=RANK_END))
            cache=old9/'internal_cache';cache.mkdir()
            atomic_write(cache/'train_wrong_ahead_top50.npz',lambda dest:
                _save_wrong_npz(dest,wf,wrong))
            atomic_json(old9/'00_方案和来源.json',dict(source_id=src['id'],version=VERSION,
                completed=True,test_used=False,train_only_mining=True,
                seeds=list(SEEDS),all_result_rows=31,query_rows=31*len(tp)))
            atomic_json(old9/'01_Train错误排序分布.json',dict(total_queries=len(tp)))
            atomic_json(old9/'04_三种子汇总及PID_Bootstrap.json',dict(synthetic=True))
            historical={('B0_original',0,0.0):base}
            for method in METHODS_OLD+METHODS_PRIOR_NEW:
                for seed in SEEDS:
                    ad=load_old_adapter(src,method,seed,device) if method in METHODS_OLD else _load_fake9_model(old9,src,method,seed,device)
                    for alpha in ALPHAS:
                        t=apply_adapter(vt,ad.text,alpha,device)
                        v=apply_adapter(vi,ad.image,alpha,device)
                        historical[(method,seed,alpha)]=evaluate(t,v,tp,ip,device)
                    del ad
            metric_rows=[];query_rows=[]
            for k,d in sorted(historical.items()):
                method,seed,alpha=k
                metric_rows.append(dict(method=method,seed=seed,alpha=alpha,**get_metrics(d)))
                for i,pid in enumerate(tp):
                    query_rows.append(dict(method=method,seed=seed,alpha=alpha,query_index=i,
                        pid=int(pid),rank=int(d['rank'][i]),correct=int(d['rank'][i]==1),
                        AP=float(d['AP'][i]),INP=float(d['INP'][i])))
            assert len(metric_rows)==31
            atomic_csv(old9/'02_所有方案_Val指标.csv',['method','seed','alpha',*METRICS],metric_rows)
            atomic_csv(old9/'03_全部逐查询_Val.csv',
                       ['method','seed','alpha','query_index','pid','rank','correct','AP','INP'],query_rows)
            verified=source_from_step9(src)
            cases=load_historical_cases(src,device)
            assert len(cases)==31 and all(np.array_equal(verified[k],wrong[k]) for k in ('top_idx','wrong_count'))
            say('[自检2/5] 31组历史文件读取、真实权重复现与文件SHA绑定完成')
            split=root/'single_split';fresh=root/'single_full';split.mkdir();fresh.mkdir()
            partial,_,_,step=train_adapter(src,split,'C1_top8_unconditional',42,verified,device,stop_at=2)
            assert step==2
            resumed,_,_,step=train_adapter(src,split,'C1_top8_unconditional',42,verified,device)
            unbroken,_,_,step=train_adapter(src,fresh,'C1_top8_unconditional',42,verified,device)
            assert step==4
            for name,weight in resumed.state_dict().items():
                assert torch.equal(weight,unbroken.state_dict()[name]),name
            del partial,resumed,unbroken
            say('[自检3/5] C1中断续跑权重逐参数与不中断训练一致')
            out=prepare_output(src)
            report=run(src,out,device,verified)
            assert len(report)==37 and (out/'07_采样负样本难度_固定Step诊断.csv').exists()
            assert (out/ZIP_NAME).is_file()
            with zipfile.ZipFile(out/ZIP_NAME) as z:
                assert z.testzip() is None
                assert '03_全部逐查询_Val.csv' in z.namelist()
            with (out/'03_全部逐查询_Val.csv').open(encoding='utf-8-sig',newline='') as f:
                assert sum(1 for _ in csv.DictReader(f))==37*len(tp)
            before=sha256(out/'03_全部逐查询_Val.csv')
            report2=run(src,out,device,verified)
            assert len(report2)==37 and sha256(out/'03_全部逐查询_Val.csv')==before
            say('[自检4/5] 37组报告、完整ZIP、再次运行缓存/权重复用完成')
            wp=out/'C1_top8_unconditional_seed42_progress.pt'
            st=torch.load(str(wp),map_location='cpu',weights_only=False)
            st['model']['text.l1.weight']=st['model']['text.l1.weight']+0.02
            atomic_torch(wp,st)
            try:
                cache_eval(src,out,'C1_top8_unconditional_seed42_alpha1.00',lambda:base)
            except RuntimeError as e:
                assert '指纹' in str(e)
            else:raise AssertionError('篡改训练权重后未拒绝缓存')
            assert (old9/'02_所有方案_Val指标.csv').is_file()
            # 历史报告一旦被任何方式改写，第十步源哈希随之改变：旧运行目录不得续跑。
            old_reports=old9/'02_所有方案_Val指标.csv'
            with old_reports.open('a',encoding='utf-8') as f:f.write('\n')
            altered=dict(src)
            source_from_step9(altered)
            assert altered['step10_source_id']!=src['step10_source_id']
            former_resume=globals()['RESUME_OUTPUT_DIR']
            globals()['RESUME_OUTPUT_DIR']=str(out)
            try:
                try:prepare_output(altered)
                except RuntimeError as e:assert '指纹' in str(e)
                else:raise AssertionError('历史文件修改后仍被错误复用')
            finally:
                globals()['RESUME_OUTPUT_DIR']=former_resume
            say('[自检5/5] 篡改训练权重、修改历史文件均触发缓存/续跑拦截')
        say('[SELF_TEST PASS] 完整CPU合成全链路通过；不代表真实Windows/CUDA实验已验证')
    finally:
        globals().update(prev)


def _save_wrong_npz(path,fp,d):
    with Path(path).open('wb') as f:
        np.savez_compressed(f,fingerprint=np.array(fp),**d)
        f.flush();os.fsync(f.fileno())


def _load_fake9_model(path,src,method,seed,device):
    st=torch.load(str(path/f'{method}_seed{seed}_progress.pt'),map_location='cpu',weights_only=False)
    ad=Both().to(device);ad.load_state_dict(st['model']);ad.eval()
    return ad


def main():
    parser=argparse.ArgumentParser(description='IRRA第十步：Top8最高相似度负样本 vs 错误领先定向监督，旧成果严格只读')
    parser.add_argument('--self-test',action='store_true',help='合成CPU完整端到端测试，不读取本机数据')
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    say('第十步：比较无条件Top8与错误领先样本；GPU/CPU：',DEVICE)
    src=verify_source()               # 验证第六步Epoch58和数据指纹
    wrong=source_from_step9(src)      # 第九步必须存在真实完整权重和完整Val结果
    out=prepare_output(src)
    say('第六步只读目录:',src['base'])
    say('第九步只读目录:',src['step9_dir'])
    say('第十步独立输出:',out)
    run(src,out,torch.device(DEVICE),wrong)


if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc()
        raise
