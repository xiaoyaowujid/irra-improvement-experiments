# -*- coding: utf-8 -*-
"""
IRRA 硬负样本 第十四步 W1：复用第6~13步完整结果，只增加Train错排纠正。
冻结最佳Epoch58 IRRA；H1原始八负样本完全一致；P1正确查询保护不变。
错误Train查询: 对教师最佳同身份图 vs Top1异身份图加入固定margin的平滑纠正。
不使用Val/Test挖掘或调参，不改变alpha；alpha=1主、0.5历史辅助。
请将独立脚本放入 IRRA-main，先 --self-test，再正式运行。
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
SOURCE_STEP9_DIR=''        # 留空自动查找匹配第九步目录
SOURCE_STEP10_DIR=''       # 留空自动查找完整第十步目录
SOURCE_STEP11_DIR=''       # 留空自动查找完整第十一步目录
SOURCE_STEP12_DIR=''       # 留空自动查找完整第十二步目录
RESUME_OUTPUT_DIR=''
OUTPUT_PREFIX='第十三步_H1已正确排序保护P1'
STEP9_PREFIX='第九步_错误排序驱动硬负样本'
STEP10_VERSION='irra_step10_top8_versus_wrong_v1_20261021'
STEP10_PREFIX='第十步_难度与错排机制对照'
STEP11_VERSION='irra_step11_C2_rankset_shuffle_v2_20261021'
STEP11_PREFIX='第十一步_H1排名分布匹配C2'
STEP12_VERSION='irra_step12_C3_conditional_wrongcount_strata_v1_20261021'
STEP12_PREFIX='第十二步_条件错排难度匹配C3'
STEP13_VERSION='irra_step13_H1_preserve_correct_train_margins_v1_20261021'
ZIP_NAME='上传给ChatGPT_第十三步_H1已正确排序保护.zip'
VERSION='irra_step9_error_rank_hardneg_v1_20261021'

EXPECTED_CKPT_SHA='f3ab12028e3406546b836ed6a75c8fc5e02a42146c5da13af423e7f5cf0e7ea6'
EXPECTED_CFG_SHA='ec2fae46ec4e0cce06372789afe1c877a5cebeddf73654781e11b0484fc30fac'
EXPECTED_ANN_SHA='fc3d15542f91dd70e23028c2cb40171a8f41734a39afddb9505e352e396cabf0'
BEST_DIR=ROOT/'logs'/'CUHK-PEDES'/'20261009_095215_irra_baseline_b8_resume20'
SEEDS=(42,43,44)
METHODS_OLD=('B1_random','B2_hard','B3_mixed')
METHODS_PRIOR_NEW=('H1_wrong_ahead','H2_wrong_plus_semi')
METHODS_STEP10=('C1_top8_unconditional',)
METHODS_STEP11=('C2_rank_marginal',)
METHODS_STEP12=('C3_wrongcount_stratified',)
METHODS_NEW=('P1_H1_correct_rank_protect',)
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
PROTECT_LAMBDA=1.0  # 预先固定，禁止依据本轮Val反复调节
PROTECT_TEMPERATURE=TEMPERATURE
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

def source_from_step10(src):
    """第十步完整报告及C1真实权重须存在；为第十一步锁定全部历史文件SHA。"""
    root=ROOT/'硬负样本分析'
    candidates=([Path(SOURCE_STEP10_DIR).resolve()] if SOURCE_STEP10_DIR else
        sorted((x for x in root.glob(STEP10_PREFIX+'_*') if x.is_dir()),
               key=lambda x:x.stat().st_mtime,reverse=True))
    matches=[]
    for p in candidates:
        if root.resolve() not in p.parents or not p.name.startswith(STEP10_PREFIX+'_'):
            continue
        mf=p/'00_方案和来源.json'
        if not mf.is_file():continue
        v=read_json(mf)
        if (v.get('version')!=STEP10_VERSION or v.get('source_id')!=src['step10_source_id'] or
            v.get('completed') is not True or v.get('test_used') is not False or
            v.get('train_only_mining') is not True or v.get('all_result_rows')!=37 or
            v.get('query_rows')!=37*len(src['qp'])):continue
        req=[p/'02_所有方案_Val指标.csv',p/'03_全部逐查询_Val.csv',
             p/'04_三种子汇总及PID_Bootstrap.json',p/'07_采样负样本难度_固定Step诊断.csv']
        req += [p/f'C1_top8_unconditional_seed{s}_progress.pt' for s in SEEDS]
        if all(x.is_file() for x in req):matches.append(p)
    if not matches:
        raise FileNotFoundError('无匹配的第十步完整结果目录（ZIP不含C1训练权重）。请填写 SOURCE_STEP10_DIR。')
    if len(matches)>1 and not SOURCE_STEP10_DIR:
        say('[提示] 多个有效第十步目录，选最近修改：',matches[0])
    p=matches[0]
    important=[p/'00_方案和来源.json',p/'02_所有方案_Val指标.csv',
               p/'03_全部逐查询_Val.csv',p/'04_三种子汇总及PID_Bootstrap.json',
               p/'07_采样负样本难度_固定Step诊断.csv']
    important += [p/f'C1_top8_unconditional_seed{s}_progress.pt' for s in SEEDS]
    filehash={str(q.relative_to(p)):sha256(q) for q in important}
    src['step10_dir']=p
    src['step10_hashes']=filehash
    src['step11_source_id']=fingerprint(dict(version=STEP11_VERSION,
        previous_source_id=src['step10_source_id'],step10_files_sha=filehash,
        rule='exact_global_rankset_permutation_H1_each_seed_different_pid_v2',
        neg_count=NUM_NEG,seeds=SEEDS,alphas=ALPHAS,steps=TRAIN_STEPS,
        training_batch=EMBED_TRAIN_BATCH,rank_end=RANK_END))
    say('[第十步历史只读目录]',p)
    return p


def load_step10_model(src,seed,device):
    path=src['step10_dir']/f'C1_top8_unconditional_seed{seed}_progress.pt'
    st=torch.load(str(path),map_location='cpu',weights_only=False)
    expected=fingerprint(dict(source=src['step10_source_id'],method='C1_top8_unconditional',
                              seed=seed,version=STEP10_VERSION))
    if (st.get('fingerprint')!=expected or st.get('source_id')!=src['step10_source_id'] or
        st.get('method')!='C1_top8_unconditional' or st.get('seed')!=seed or
        st.get('step')!=TRAIN_STEPS):
        raise RuntimeError('C1第十步训练权重身份/完整性错误: '+str(path))
    ad=Both().to(device);ad.load_state_dict(st['model'],strict=True);ad.eval()
    for par in ad.parameters():par.requires_grad_(False)
    return ad


def h1_training_rank_histogram(src,wrong):
    """完整重放H1 800步，返回排名组合与对应的原文本下标；Train独立计算。"""
    counts=TRAIN_STEPS*EMBED_TRAIN_BATCH
    ranked=wrong['top_idx']; k=RANK_END
    profile={}
    for seed in SEEDS:
        bins=np.zeros(k,dtype=np.int64)
        all_rank_sets=np.empty((counts,NUM_NEG),dtype=np.uint8)
        all_queries=np.empty(counts,dtype=np.int64)
        for step in range(TRAIN_STEPS):
            rg=np.random.default_rng(seed*1000003+step*97+3)
            qidx=rg.integers(len(src['tp']),size=EMBED_TRAIN_BATCH,dtype=np.int64)
            h1=choose_old_negatives(qidx,src['tp'],src['ip'],src['half'],wrong,
                                   'H1_wrong_ahead',seed,step)
            match=(h1[:,:,None]==ranked[qidx,None,:])
            if not np.all(match.sum(-1)==1):
                raise RuntimeError(f'H1 seed={seed} step={step}: 负样本不在Top{RANK_END}')
            picked=np.argmax(match,-1).astype(np.uint8)
            if not np.all(np.array([len(set(row.tolist())) for row in picked])==NUM_NEG):
                raise RuntimeError('H1每组8个排名不互异')
            lo=step*EMBED_TRAIN_BATCH;hi=lo+EMBED_TRAIN_BATCH
            all_rank_sets[lo:hi]=picked
            all_queries[lo:hi]=qidx
            bins+=np.bincount(picked.ravel().astype(np.int32),minlength=k)
        if bins.sum()!=counts*NUM_NEG:
            raise RuntimeError('H1全步rank曝光数量不正确')
        profile[seed]=dict(hist=bins,rank_sets=all_rank_sets,queries=all_queries)
        say('[H1全步训练重放] seed',seed,'曝光',int(bins.sum()),
            'Top8占比',round(float(bins[:8].sum()/bins.sum()),4))
    return profile


def add_rank_profiles(src,wrong):
    p=h1_training_rank_histogram(src,wrong)
    rank_sets={};rank_hist={};queries={};permutation_hashes={};original={}
    for seed in SEEDS:
        donor=p[seed]['rank_sets'];q=p[seed]['queries']
        original[seed]=donor
        # 每个donor来源于历史H1的另一个PID；对目标本人的训练错排状况无条件独立。
        pid=src['tp'][q]
        n=len(q)
        rg=np.random.default_rng(20261021+seed*10007)
        perm=rg.permutation(n)
        bad=np.flatnonzero(pid==pid[perm])
        for i in bad:
            if pid[i]!=pid[perm[i]]:continue
            # 不改变全局排名集合多重集，只交换donor索引直到双方均跨PID。
            for _ in range(10000):
                j=int(rg.integers(n))
                if (j!=i and pid[i]!=pid[perm[j]] and pid[j]!=pid[perm[i]]):
                    perm[i],perm[j]=perm[j],perm[i]
                    break
            else:
                raise RuntimeError('无法生成与目标PID不同的H1排名集合重排')
        if np.any(pid==pid[perm]) or len(set(perm.tolist()))!=n:
            raise RuntimeError('C2排名组合重排非法/出现同PID donor')
        mapped=np.ascontiguousarray(donor[perm])
        b=np.bincount(mapped.ravel().astype(np.int32),minlength=RANK_END)
        if not np.array_equal(b,p[seed]['hist']):
            raise RuntimeError('C2没有严格保留H1完整800步的排名直方图')
        rank_sets[seed]=mapped;rank_hist[seed]=b;queries[seed]=q
        permutation_hashes[str(seed)]=hashlib.sha256(mapped.tobytes()).hexdigest()
        say('[C2跨PID重排完成]',seed,'完整rank计数严格相等')
    wrong['h1_rank_sets']=original
    wrong['rank_sets']=rank_sets
    wrong['rank_histograms']={seed:p[seed]['hist'] for seed in SEEDS}
    wrong['c2_rank_histograms']=rank_hist
    wrong['rank_probabilities']={seed:rank_hist[seed]/rank_hist[seed].sum() for seed in SEEDS}
    wrong['sampled_queries']=queries
    source=fingerprint(dict(original_id=src['step11_source_id'],
        rank_sets_sha=permutation_hashes,
        rank_hist_sha={str(seed):wrong['rank_histograms'][seed].tolist() for seed in SEEDS}))
    src['step11_source_id']=source
    src['hist_hash']=source
    return wrong


def source_from_step11(src):
    """Verify completed Step11 and bind Step12 to all historical case files + C2 weights."""
    root=ROOT/'硬负样本分析'
    candidates=([Path(SOURCE_STEP11_DIR).resolve()] if SOURCE_STEP11_DIR else
        sorted((p for p in root.glob(STEP11_PREFIX+'_*') if p.is_dir()),
               key=lambda p:p.stat().st_mtime,reverse=True))
    found=[]
    for p in candidates:
        if root.resolve() not in p.parents or not p.name.startswith(STEP11_PREFIX+'_'):continue
        mf=p/'00_方案和来源.json'
        if not mf.is_file():continue
        v=read_json(mf)
        if (v.get('version')!=STEP11_VERSION or v.get('source_id')!=src['step11_source_id'] or
            v.get('completed') is not True or v.get('train_only_mining') is not True or
            v.get('test_used') is not False or v.get('all_result_rows')!=43 or
            v.get('query_rows')!=43*len(src['qp'])):continue
        req=[p/'02_所有方案_Val指标.csv',p/'03_全部逐查询_Val.csv',
             p/'04_三种子汇总及PID_Bootstrap.json',p/'07_采样负样本难度_固定Step诊断.csv',
             p/'08_H1排名分布与C2匹配程度.json']
        req += [p/f'C2_rank_marginal_seed{seed}_progress.pt' for seed in SEEDS]
        if all(x.is_file() for x in req):found.append(p)
    if not found:raise FileNotFoundError('找不到完整第十一步目录（必须有C2权重）；请填写SOURCE_STEP11_DIR')
    if len(found)>1 and not SOURCE_STEP11_DIR:say('[提示] 多个第十一步目录，选最新:',found[0])
    folder=found[0]
    critical=[folder/'00_方案和来源.json',folder/'02_所有方案_Val指标.csv',
              folder/'03_全部逐查询_Val.csv',folder/'04_三种子汇总及PID_Bootstrap.json',
              folder/'07_采样负样本难度_固定Step诊断.csv',
              folder/'08_H1排名分布与C2匹配程度.json']
    critical += [folder/f'C2_rank_marginal_seed{seed}_progress.pt' for seed in SEEDS]
    hashes={str(p.relative_to(folder)):sha256(p) for p in critical}
    src['step11_dir']=folder
    src['step11_hashes']=hashes
    src['step12_source_id']=fingerprint(dict(version=STEP12_VERSION,
        original_step11_id=src['step11_source_id'],step11_files=hashes,
        rule='H1_full_rankset_permutation_within_Train_wrongcount_stratum_different_PID',
        base_strata=[0,1,2,4,8,16],seeds=SEEDS,alpha=ALPHAS,
        steps=TRAIN_STEPS,batch=EMBED_TRAIN_BATCH,neg=NUM_NEG))
    say('[第十一步历史只读目录]',folder)
    return folder


def load_step11_model(src,seed,device):
    path=src['step11_dir']/f'C2_rank_marginal_seed{seed}_progress.pt'
    st=torch.load(str(path),map_location='cpu',weights_only=False)
    fp=fingerprint(dict(source=src['step11_source_id'],method='C2_rank_marginal',
        seed=seed,version=STEP11_VERSION))
    if (st.get('fingerprint')!=fp or st.get('source_id')!=src['step11_source_id'] or
        st.get('method')!='C2_rank_marginal' or st.get('seed')!=seed or
        st.get('step')!=TRAIN_STEPS):
        raise RuntimeError('第十一步C2权重指纹/训练完成状态不正确: '+str(path))
    ad=Both().to(device);ad.load_state_dict(st['model'],strict=True);ad.eval()
    for param in ad.parameters():param.requires_grad_(False)
    return ad


def add_stratified_profiles(src,wrong):
    """Reassign whole H1 8-rank sets within query wrong_count strata, across PIDs.

    Adjacent strata are deterministically merged only if current group is too small
    or the largest PID accounts for > half the target occurrences. Overall H1
    rank histogram and rankset multiset remain exactly equal for each seed.
    """
    wc=wrong['wrong_count']
    hi=RANK_END
    limits=[(0,0),(1,1),(2,min(3,hi)),(4,min(7,hi)),
            (8,min(15,hi)),(16,hi)]
    limits=[(a,b) for a,b in limits if a<=b]
    ranksets={};metadata={};mapping_hashes={}
    for seed in SEEDS:
        donor=wrong['h1_rank_sets'][seed]
        queries=wrong['sampled_queries'][seed]
        pid=src['tp'][queries]
        difficulty=wc[queries]
        if donor.shape!=(TRAIN_STEPS*EMBED_TRAIN_BATCH,NUM_NEG):
            raise RuntimeError('H1 donor 8排名形状错误')
        bins=[dict(lo=a,hi=b) for a,b in limits]
        def ix_of(group):return np.flatnonzero((difficulty>=group['lo']) & (difficulty<=group['hi']))
        def feasible(group):
            idx=ix_of(group)
            if len(idx)<min(16,len(queries)):return False
            _,counts=np.unique(pid[idx],return_counts=True)
            return len(counts)>=2 and int(counts.max())*2<=len(idx)
        while len(bins)>1:
            bad=next((i for i,g in enumerate(bins) if not feasible(g)),None)
            if bad is None:break
            if bad==0:neighbour=1
            elif bad==len(bins)-1:neighbour=bad-1
            else:
                # Merge with smaller adjacent group to preserve remaining resolution.
                neighbour=min((bad-1,bad+1),key=lambda j:(len(ix_of(bins[j])),j))
            lo,up=sorted((bad,neighbour))
            bins[lo:up+1]=[dict(lo=bins[lo]['lo'],hi=bins[up]['hi'])]
        if not all(feasible(g) for g in bins):
            raise RuntimeError(f'seed={seed} 训练查询PID过于集中，即使合并层也无法跨PID严格匹配')
        rg=np.random.default_rng(12026021+seed*31013)
        shuffled=np.empty_like(donor)
        info=[]
        for g in bins:
            ix=ix_of(g)
            # Randomize PID group order and positions within each PID, then
            # cyclically shift by the largest group size: exact cross-PID bijection.
            # O(n log n) grouping; do not rescan all 100k occurrences for every PID.
            sort_order=np.argsort(pid[ix],kind='stable')
            sorted_ix=ix[sort_order]
            sorted_pids=pid[sorted_ix]
            cuts=np.r_[0,np.flatnonzero(np.diff(sorted_pids))+1,len(sorted_ix)]
            group_order=rg.permutation(len(cuts)-1)
            pid_groups=[]
            for group in group_order:
                members=sorted_ix[cuts[group]:cuts[group+1]].copy()
                rg.shuffle(members)
                pid_groups.append(members)
            ordered=np.concatenate(pid_groups)
            shift=max(len(x) for x in pid_groups)
            ids=sorted_pids[cuts[:-1]]
            recipients=ordered
            donors=np.roll(ordered,-shift)
            if (len(recipients)!=len(ix) or len(np.unique(donors))!=len(ix) or
                np.any(pid[recipients]==pid[donors])):
                raise RuntimeError('C3层内跨PID双射构造错误')
            shuffled[recipients]=donor[donors]
            # Stronger than global marginal: complete per-stratum rankset multiset
            # matches, not merely a histogram.
            a=np.sort(donor[ix],axis=1)
            b=np.sort(shuffled[ix],axis=1)
            if Counter(map(tuple,a.tolist()))!=Counter(map(tuple,b.tolist())):
                raise RuntimeError('C3层内八排名组合集合没有严格匹配H1')
            hist_h=np.bincount(donor[ix].ravel().astype(np.int32),minlength=RANK_END)
            hist_c=np.bincount(shuffled[ix].ravel().astype(np.int32),minlength=RANK_END)
            if not np.array_equal(hist_h,hist_c):raise RuntimeError('C3层内rank曝光不匹配')
            wcounts=difficulty[ix,None]
            share_h=float(np.mean(donor[ix].astype(np.int32)<wcounts))
            share_c=float(np.mean(shuffled[ix].astype(np.int32)<wcounts))
            info.append(dict(lo=g['lo'],hi=g['hi'],occurrences=len(ix),pid_count=len(ids),
                 largest_pid_count=shift,full_rank_histogram=hist_h.tolist(),
                 wrong_exposure_H1=share_h,wrong_exposure_C3=share_c,
                 wrong_exposure_gap=share_c-share_h,
                 target_sampled_query_sha256=hashlib.sha256(queries[ix].tobytes()).hexdigest()))
        actual=np.bincount(shuffled.ravel().astype(np.int32),minlength=RANK_END)
        if not np.array_equal(actual,wrong['rank_histograms'][seed]):
            raise RuntimeError('C3与H1全步曝光不一致')
        ranksets[seed]=shuffled
        mapping_hashes[str(seed)]=hashlib.sha256(shuffled.tobytes()).hexdigest()
        metadata[str(seed)]=dict(original_bins=[list(v) for v in limits],
            effective_strata=info,rankset_sha256=mapping_hashes[str(seed)],
            full_rank_exposure=int(actual.sum()),
            overall_wrong_ahead_H1=float(np.mean(donor.astype(np.int32)<difficulty[:,None])),
            overall_wrong_ahead_C3=float(np.mean(shuffled.astype(np.int32)<difficulty[:,None])))
        say('[C3条件分层]',seed,'合并后',len(info),'层；错排负样本比例H1/C3',
            round(metadata[str(seed)]['overall_wrong_ahead_H1'],4),
            round(metadata[str(seed)]['overall_wrong_ahead_C3'],4))
    wrong['c3_rank_sets']=ranksets
    wrong['c3_strata_report']=metadata
    src['step12_source_id']=fingerprint(dict(previous=src['step12_source_id'],
        c3_rankset_sha256=mapping_hashes,metadata=metadata))
    return wrong

def source_from_step12(src):
    """第十二步目录须真实完成，SHA绑定49组历史与C3权重；只读。"""
    root=ROOT/'硬负样本分析'
    candidates=([Path(SOURCE_STEP12_DIR).resolve()] if SOURCE_STEP12_DIR else
        sorted((p for p in root.glob(STEP12_PREFIX+'_*') if p.is_dir()),
               key=lambda p:p.stat().st_mtime,reverse=True))
    found=[]
    for p in candidates:
        if root.resolve() not in p.parents or not p.name.startswith(STEP12_PREFIX+'_'):continue
        mf=p/'00_方案和来源.json'
        if not mf.is_file():continue
        v=read_json(mf)
        if (v.get('version')!=STEP12_VERSION or v.get('source_id')!=src['step12_source_id'] or
            v.get('completed') is not True or v.get('train_only_mining') is not True or
            v.get('test_used') is not False or v.get('all_result_rows')!=49 or
            v.get('query_rows')!=49*len(src['qp'])):continue
        req=[p/'02_所有方案_Val指标.csv',p/'03_全部逐查询_Val.csv',
             p/'04_三种子汇总及PID_Bootstrap.json',p/'05_新方案对旧方法_同种子配对差.csv',
             p/'07_采样负样本难度_固定Step诊断.csv',
             p/'08_H1排名分布与C3条件匹配程度.json']
        req += [p/f'C3_wrongcount_stratified_seed{seed}_progress.pt' for seed in SEEDS]
        if all(x.is_file() for x in req):found.append(p)
    if not found:
        raise FileNotFoundError('第十二步49组完整结果和C3本地权重缺失；请设置 SOURCE_STEP12_DIR（上传ZIP不含权重）')
    p=found[0]
    if len(found)>1 and not SOURCE_STEP12_DIR:say('[提示] 多个合法第十二步目录，使用最新:',p)
    critical=[p/'00_方案和来源.json',p/'02_所有方案_Val指标.csv',
              p/'03_全部逐查询_Val.csv',p/'04_三种子汇总及PID_Bootstrap.json',
              p/'05_新方案对旧方法_同种子配对差.csv',
              p/'07_采样负样本难度_固定Step诊断.csv',
              p/'08_H1排名分布与C3条件匹配程度.json']
    critical += [p/f'C3_wrongcount_stratified_seed{seed}_progress.pt' for seed in SEEDS]
    sha={str(q.relative_to(p)):sha256(q) for q in critical}
    src['step12_dir']=p;src['step12_hashes']=sha
    src['step13_source_id']=fingerprint(dict(version=STEP13_VERSION,
        parent=src['step12_source_id'],files=sha,
        method='H1_train_hard_negatives_plus_positive_margin_preservation',
        teacher='Epoch58_best_same_pid_image_vs_top1_other_pid_image',
        correct_condition='Train_wrong_count_eq_zero',
        preserve_lambda=PROTECT_LAMBDA,temperature=PROTECT_TEMPERATURE,
        steps=TRAIN_STEPS,batch=EMBED_TRAIN_BATCH,neg=NUM_NEG,
        seeds=SEEDS,alphas=ALPHAS))
    say('[第十二步历史只读目录]',p)
    return p


def load_step12_model(src,seed,device):
    p=src['step12_dir']/f'C3_wrongcount_stratified_seed{seed}_progress.pt'
    st=torch.load(str(p),map_location='cpu',weights_only=False)
    fp=fingerprint(dict(source=src['step12_source_id'],method='C3_wrongcount_stratified',
        seed=seed,version=STEP12_VERSION))
    if (st.get('fingerprint')!=fp or st.get('source_id')!=src['step12_source_id'] or
        st.get('method')!='C3_wrongcount_stratified' or st.get('seed')!=seed or
        st.get('step')!=TRAIN_STEPS):raise RuntimeError('第十二步C3权重身份错误: '+str(p))
    ad=Both().to(device);ad.load_state_dict(st['model'],strict=True);ad.eval()
    for param in ad.parameters():param.requires_grad_(False)
    return ad

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
    p=src['step12_dir']
    expected={('B0_original',0,0.0)}
    expected.update((m,s,a) for m in METHODS_OLD+METHODS_PRIOR_NEW+METHODS_STEP10+METHODS_STEP11+METHODS_STEP12 for s in SEEDS for a in ALPHAS)
    rows={}
    with (p/'02_所有方案_Val指标.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            key=(r['method'],int(r['seed']),float(r['alpha']))
            if key not in expected or key in rows:raise RuntimeError('第十二步汇总存在非法/重复组 '+str(key))
            rows[key]=r
    if set(rows)!=expected:raise RuntimeError('第十二步49组指标不完整')
    n=len(src['qp']);data={k:dict(rank=np.empty(n,np.int32),AP=np.empty(n),INP=np.empty(n)) for k in expected}
    next_idx={k:0 for k in expected}
    with (p/'03_全部逐查询_Val.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            key=(r['method'],int(r['seed']),float(r['alpha']))
            if key not in data:raise RuntimeError('第十二步逐查询结果包含未知组')
            i=int(r['query_index'])
            if i!=next_idx[key] or i>=n or int(r['pid'])!=int(src['qp'][i]):
                raise RuntimeError('第十二步逐查询顺序/身份/唯一性异常')
            d=data[key];d['rank'][i]=int(r['rank']);d['AP'][i]=float(r['AP']);d['INP'][i]=float(r['INP'])
            if int(r['correct'])!=int(d['rank'][i]==1):raise RuntimeError('第九步R1正误字段不一致')
            next_idx[key]+=1
    if any(v!=n for v in next_idx.values()):raise RuntimeError('第十二步逐查询记录缺失')
    for key,d in data.items():
        validate_detail(d,n,len(src['vp']))
        metrics=get_metrics(d)
        if max(abs(metrics[k]-float(rows[key][k])) for k in METRICS)>1e-5:
            raise RuntimeError('第十二步汇总与逐查询指标不一致：'+str(key))
    # 第十一步的31组旧结果必须逐条与原始第九步相同，不能从后续汇总无条件继承。
    nine=(src['step9_dir']/'03_全部逐查询_Val.csv')
    if not nine.is_file():raise FileNotFoundError('第九步原始逐查询数据缺失')
    source31={k for k in expected if k[0] not in ('C1_top8_unconditional','C2_rank_marginal','C3_wrongcount_stratified')}
    seen={k:0 for k in source31}
    with nine.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            key=(row['method'],int(row['seed']),float(row['alpha']))
            if key not in source31:raise RuntimeError('第九步历史原始数据包含未知组')
            i=int(row['query_index'])
            if i!=seen[key] or i>=n or int(row['pid'])!=int(src['qp'][i]):
                raise RuntimeError('第九步历史查询顺序或PID不一致')
            d=data[key]
            if (int(row['rank'])!=d['rank'][i] or
                abs(float(row['AP'])-d['AP'][i])>2e-7 or
                abs(float(row['INP'])-d['INP'][i])>2e-7):
                raise RuntimeError('第十步继承的31组逐查询结果和原第九步不一致 '+str(key))
            seen[key]+=1
    if any(x!=n for x in seen.values()):raise RuntimeError('第九步原始逐查询记录不完整')
    # 第十一步43组中已有的37组必须与第十步逐查询报告严格一致。
    previous=src['step10_dir']/'03_全部逐查询_Val.csv'
    from10={k for k in expected if k[0] not in ('C2_rank_marginal','C3_wrongcount_stratified')}
    count10={k:0 for k in from10}
    with previous.open(encoding='utf-8-sig',newline='') as handle:
        for r in csv.DictReader(handle):
            key=(r['method'],int(r['seed']),float(r['alpha']))
            if key not in from10:raise RuntimeError('第十步意外的历史组')
            i=int(r['query_index'])
            if i!=count10[key] or int(r['pid'])!=int(src['qp'][i]):
                raise RuntimeError('第十步历史记录顺序或PID错误')
            d=data[key]
            if int(r['rank'])!=int(d['rank'][i]) or abs(float(r['AP'])-d['AP'][i])>2e-7 or abs(float(r['INP'])-d['INP'][i])>2e-7:
                raise RuntimeError('第十二步继承的37组与第十步记录不一致')
            count10[key]+=1
    if any(k!=n for k in count10.values()):raise RuntimeError('第十步历史记录不完整')
    # 49组中早已完成的43组必须与原始第十一步逐查询逐项一致。
    previous11=src['step11_dir']/'03_全部逐查询_Val.csv'
    from11={k for k in expected if k[0]!='C3_wrongcount_stratified'}
    counts11={k:0 for k in from11}
    with previous11.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            key=(r['method'],int(r['seed']),float(r['alpha']))
            if key not in from11:raise RuntimeError('第十一步出现未知方案')
            i=int(r['query_index'])
            if i!=counts11[key] or i>=n or int(r['pid'])!=int(src['qp'][i]):
                raise RuntimeError('第十一步原始查询顺序/身份不正确')
            d=data[key]
            if (int(r['rank'])!=int(d['rank'][i]) or
                abs(float(r['AP'])-d['AP'][i])>2e-7 or
                abs(float(r['INP'])-d['INP'][i])>2e-7):
                raise RuntimeError('第十二步继承的43组与原第十一步不同')
            counts11[key]+=1
    if any(x!=n for x in counts11.values()):raise RuntimeError('第十一步原始结果缺失')
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
    # 第十步C1权重必须与其历史逐查询完全一致
    for seed in SEEDS:
        ad=load_step10_model(src,seed,device)
        for alpha in ALPHAS:
            got=evaluate(apply_adapter(src['vt'],ad.text,alpha,device),
                         apply_adapter(src['vi'],ad.image,alpha,device),src['qp'],src['vp'],device)
            assert_equal_detail(got,data[('C1_top8_unconditional',seed,alpha)],
                                f'C1 seed{seed} alpha{alpha}')
        del ad
    # 重算第六步B2关键对照，确认历史数据和权重绑定
    for seed in SEEDS:
        ad=load_old_adapter(src,'B2_hard',seed,device)
        for alpha in ALPHAS:
            got=evaluate(apply_adapter(src['vt'],ad.text,alpha,device),
                         apply_adapter(src['vi'],ad.image,alpha,device),src['qp'],src['vp'],device)
            assert_equal_detail(got,data[('B2_hard',seed,alpha)],f'第六步B2 seed{seed} alpha{alpha}')
        del ad
    for seed in SEEDS:
        ad=load_step11_model(src,seed,device)
        for alpha in ALPHAS:
            got=evaluate(apply_adapter(src['vt'],ad.text,alpha,device),
                         apply_adapter(src['vi'],ad.image,alpha,device),src['qp'],src['vp'],device)
            assert_equal_detail(got,data[('C2_rank_marginal',seed,alpha)],
                                f'C2 seed{seed} alpha{alpha}')
        del ad
    for seed in SEEDS:
        ad=load_step12_model(src,seed,device)
        for alpha in ALPHAS:
            got=evaluate(apply_adapter(src['vt'],ad.text,alpha,device),
                         apply_adapter(src['vi'],ad.image,alpha,device),src['qp'],src['vp'],device)
            assert_equal_detail(got,data[('C3_wrongcount_stratified',seed,alpha)],
                                f'C3 seed{seed} alpha{alpha}')
        del ad
    say('[历史结果复用完成] 49组，',49*n,'条查询；B0/B2/H1/H2/C1/C2/C3均由权重复算')
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
            raise RuntimeError('续跑路径必须是独立的第十三步实验目录')
        manifest=out/'00_方案和来源.json'
        if not manifest.is_file() or read_json(manifest).get('source_id')!=src['step13_source_id']:
            raise RuntimeError('第十三步源指纹不匹配，拒绝混用已有实验')
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

def best_same_pid_teacher_indices(src,wrong):
    """一次排序分组后确定冻结Train教师每条查询的最好同身份正图（无Val/Test）。"""
    tp=np.asarray(src['tp']);ip=np.asarray(src['ip'])
    trt=np.asarray(src['trt'],dtype=np.float32);tri=np.asarray(src['tri'],dtype=np.float32)
    iq=np.argsort(tp,kind='stable');iv=np.argsort(ip,kind='stable')
    tp_unique,qstarts,qcounts=np.unique(tp[iq],return_index=True,return_counts=True)
    ip_unique,vstarts,vcounts=np.unique(ip[iv],return_index=True,return_counts=True)
    if not np.array_equal(tp_unique,ip_unique):
        raise RuntimeError('Train文本和图像PID集合不同，无法保护正确排序')
    result=np.empty(len(tp),dtype=np.int32)
    for j in range(len(tp_unique)):
        q=iq[qstarts[j]:qstarts[j]+qcounts[j]]
        v=iv[vstarts[j]:vstarts[j]+vcounts[j]]
        scores=trt[q] @ tri[v].T
        result[q]=v[np.argmax(scores,axis=1)]
    teacher_score=np.einsum('nd,nd->n',trt,tri[result],optimize=True)
    delta=np.abs(teacher_score-wrong['best_positive'])
    if np.max(delta)>5e-5:
        raise RuntimeError(f'教师最佳正图分数和历史缓存不匹配: {np.max(delta)}')
    if np.any(ip[result]!=tp):raise RuntimeError('保护正图PID错误')
    if np.any(ip[wrong['top_idx'][:,0]]==tp):raise RuntimeError('保护最强负图PID错误')
    if np.any((wrong['wrong_count']==0)&(wrong['best_negative']>wrong['best_positive']+1e-7)):
        raise RuntimeError('保护教师正确查询判据不一致')
    return result


def correct_rank_protection(ad,x,train_img,original_indices,wrong,best_idx,device):
    """保护Train原始正确查询的冻结教师间隔；梯度仅流经新增Adapter。"""
    mask=np.asarray(wrong['wrong_count'][original_indices]==0)
    if not np.any(mask):
        return x.sum()*0,dict(protected=0,violated=0,margin_shortfall=0.0)
    sel=np.flatnonzero(mask)
    selected=original_indices[sel]
    p_ix=torch.as_tensor(best_idx[selected],device=device,dtype=torch.long)
    n_ix=torch.as_tensor(wrong['top_idx'][selected,0].astype(np.int64),device=device)
    # 两个锚点仅作为额外保护约束的输入，不改变原H1的8个CE负样本。
    pos=ad.image(train_img[p_ix])
    neg=ad.image(train_img[n_ix])
    txt=ad.text(x[torch.as_tensor(sel,device=device)])
    student=(txt*pos).sum(-1)-(txt*neg).sum(-1)
    teacher=torch.as_tensor(
        wrong['best_positive'][selected]-wrong['best_negative'][selected],
        device=device,dtype=torch.float32)
    shortage=F.relu((teacher-student)/PROTECT_TEMPERATURE)
    # 让保护权重作用于全部查询批次，而不因为正确查询比例变化而隐式增减。
    loss=shortage.square().sum()/len(original_indices)
    return loss,dict(protected=int(len(sel)),violated=int((shortage.detach()>1e-7).sum().item()),
                     margin_shortfall=float(shortage.detach().sum().cpu()))

def choose_negatives(indices,tp,ip,half,wrong,mode,seed,step,num_neg=NUM_NEG):
    if mode!='P1_H1_correct_rank_protect':raise ValueError('第十三步仅新训练P1')
    # 完全复现第九步H1原始的同批8负样本抽样（包括随机数状态）。
    return choose_old_negatives_with_stats(indices,tp,ip,half,wrong,seed,step,num_neg)


def choose_old_negatives_with_stats(indices,tp,ip,half,wrong,seed,step,num_neg=NUM_NEG):
    if num_neg!=NUM_NEG:raise RuntimeError('原始H1每条查询8个负样本不可更改')
    negative=choose_old_negatives(indices,tp,ip,half,wrong,'H1_wrong_ahead',seed,step)
    chosen=np.asarray(wrong['wrong_count'][indices],dtype=np.int64)
    used=np.minimum(chosen,NUM_NEG)
    if negative.shape!=(len(indices),NUM_NEG) or np.any(ip[negative]==tp[indices,None]):
        raise RuntimeError('P1采样不是H1原始8个异身份负样本')
    if any(len(set(row.tolist()))!=NUM_NEG for row in negative):
        raise RuntimeError('H1原始负样本内部重复')
    return negative,dict(neg_from_wrong=int(used.sum()),
        wrong_quota_fallback=int((NUM_NEG-used).sum()),queries_with_wrong=int(np.count_nonzero(chosen)))

def train_adapter(src,out,method,seed,wrong,device,stop_at=None):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    if method not in METHODS_NEW:raise RuntimeError('仅允许P1新训练')
    ad=Both().to(device)
    opt=torch.optim.AdamW(ad.parameters(),lr=LR,weight_decay=1e-4)
    checkpoint=out/f'{method}_seed{seed}_progress.pt'
    run_fp=fingerprint(dict(source=src['step13_source_id'],method=method,seed=seed,version=STEP13_VERSION,
                            protect_lambda=PROTECT_LAMBDA))
    start=0;history=[]
    counter=dict(neg_from_wrong=0,wrong_quota_fallback=0,queries_with_wrong=0,
                 protected_train_exposures=0,protect_violated_exposures=0)
    if checkpoint.exists():
        st=torch.load(str(checkpoint),map_location='cpu',weights_only=False)
        if (st.get('fingerprint')!=run_fp or st.get('method')!=method or st.get('seed')!=seed or
            st.get('source_id')!=src['step13_source_id'] or not 0<=st.get('step',-1)<=TRAIN_STEPS):
            raise RuntimeError(f'{method} seed{seed}断点指纹/位置不一致')
        ad.load_state_dict(st['model'],strict=True)
        opt.load_state_dict(st['optimizer'])
        for state in opt.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):state[k]=v.to(device)
        start=st['step'];history=st['history'];counter=st['counter']
        say('[断点恢复]',method,seed,start,'/',TRAIN_STEPS)
    best_idx=best_same_pid_teacher_indices(src,wrong)
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
        protect_loss,details=correct_rank_protection(ad,x,tri,idx,wrong,best_idx,device)
        counter['protected_train_exposures']+=details['protected']
        counter['protect_violated_exposures']+=details['violated']
        obj=loss+PRESERVE*preservation+PROTECT_LAMBDA*protect_loss
        if not torch.isfinite(obj):raise RuntimeError(f'P1 seed{seed} step{step}: loss非有限')
        opt.zero_grad(set_to_none=True);obj.backward()
        torch.nn.utils.clip_grad_norm_(ad.parameters(),2.0)
        opt.step()
        if (step+1)%20==0 or step==0:
            history.append(dict(step=step+1,loss=float(loss.detach().cpu()),
               preserve=float(preservation.detach().cpu()),
               protect=float(protect_loss.detach().cpu()),
               protected=details['protected'],protect_violated=details['violated'],
               objective=float(obj.detach().cpu())))
        if (step+1)%SAVE_EVERY==0 or step+1==until:
            atomic_torch(checkpoint,dict(fingerprint=run_fp,source_id=src['step13_source_id'],
                method=method,seed=seed,step=step+1,model=ad.state_dict(),
                optimizer=opt.state_dict(),history=history,counter=counter))
            say('[训练]',method,'seed',seed,step+1,'/',TRAIN_STEPS,
                'CE',round(float(loss.detach().cpu()),5),
                'protect',round(float(protect_loss.detach().cpu()),6))
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
    fp=fingerprint(dict(source=src['step13_source_id'],key=key,version=STEP13_VERSION,adapter_weight_sha256=weight_sha))
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
    expected=fingerprint(dict(source=src['step13_source_id'],method=method,seed=seed,version=STEP13_VERSION,protect_lambda=PROTECT_LAMBDA))
    if (st.get('fingerprint')!=expected or st.get('source_id')!=src['step13_source_id'] or
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
        for method in METHODS_OLD+METHODS_PRIOR_NEW+METHODS_STEP10+METHODS_STEP11+METHODS_STEP12+METHODS_NEW:
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
        dict(note='第十三步为Val探索性对照。PID bootstrap仅量化身份采样，不含训练不确定性、多重选择或Test泛化。',
             primary_alpha=1.0,secondary_pre_registered_alpha=0.5,methods=stats))
    # H1/H2 vs B2: 决策以三个种子同种seed严格配对比较，不比较不同α造成的选择偏差
    pair=[]
    for alpha in ALPHAS:
        for method in METHODS_NEW:
            for ref_name in ('B0_original','B2_hard','H1_wrong_ahead','H2_wrong_plus_semi','C1_top8_unconditional','C2_rank_marginal','C3_wrongcount_stratified'):
              for seed in SEEDS:
                x=get_metrics(cases[(method,seed,alpha)])
                b=get_metrics(cases[('B0_original',0,0.0)] if ref_name=='B0_original' else cases[(ref_name,seed,alpha)])
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
            if not {'02_所有方案_Val指标.csv','03_全部逐查询_Val.csv','04_三种子汇总及PID_Bootstrap.json','05_新方案对旧方法_同种子配对差.csv','07_P1训练保护间隔诊断.csv','08_P1和H1正确查询交换及统计.json'}.issubset(set(z.namelist())):
                raise RuntimeError('ZIP遗漏必要结果')
        os.replace(temp,dest)
    finally:
        if temp.exists():temp.unlink()
    say('报告ZIP：',dest)

def probe_difficulty(src,wrong,out):
    """Train只读小样本，检查新保护损失是否保护教师最强负图间隔（不用于选择）。"""
    best=best_same_pid_teacher_indices(src,wrong)
    probe=[]
    for seed in SEEDS:
        query=[]
        for step in PROBE_STEPS:
            if step>=TRAIN_STEPS:continue
            rg=np.random.default_rng(seed*1000003+step*97+3)
            ix=rg.integers(len(src['tp']),size=EMBED_TRAIN_BATCH,dtype=np.int64)
            query.extend(ix[wrong['wrong_count'][ix]==0].tolist())
        query=np.unique(np.asarray(query,dtype=np.int64))
        if not len(query):raise RuntimeError('保护探针中没有原正确Train查询')
        device=torch.device('cpu')
        old=load_step9_model(src,'H1_wrong_ahead',seed,device)
        path=out/f'P1_H1_correct_rank_protect_seed{seed}_progress.pt'
        st=torch.load(path,map_location='cpu',weights_only=False)
        p1=Both();p1.load_state_dict(st['model']);p1.eval()
        x=torch.as_tensor(src['trt'][query],dtype=torch.float32)
        pos=torch.as_tensor(src['tri'][best[query]],dtype=torch.float32)
        neg=torch.as_tensor(src['tri'][wrong['top_idx'][query,0]],dtype=torch.float32)
        teacher=wrong['best_positive'][query]-wrong['best_negative'][query]
        with torch.inference_mode():
            for alpha in ALPHAS:
                for method,ad in (('H1_wrong_ahead',old),('P1_H1_correct_rank_protect',p1)):
                    t=ad.text(x,alpha);v=ad.image(pos,alpha);n=ad.image(neg,alpha)
                    margin=((t*v).sum(-1)-(t*n).sum(-1)).cpu().numpy()
                    probe.append(dict(method=method,seed=seed,alpha=alpha,
                        query_count=len(query),teacher_mean_margin=float(teacher.mean()),
                        student_mean_margin=float(margin.mean()),
                        worsened_from_teacher=float(np.mean(margin<teacher-1e-7)),
                        teacher_correct_now_wrong=float(np.mean(margin<=0)),
                        mean_teacher_margin_drop=float(np.mean(np.maximum(teacher-margin,0)))))
    atomic_csv(out/'07_P1训练保护间隔诊断.csv',list(probe[0]),probe)
    return probe

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
    if wrong is None:raise RuntimeError('缺少Train原始错误领先池')
    manifest=out/'00_方案和来源.json'
    if manifest.is_file():
        if read_json(manifest).get('source_id')!=src['step13_source_id']:
            raise RuntimeError('第十三步输出目录含不兼容实验，拒绝续跑')
    else:
        atomic_json(manifest,dict(version=STEP13_VERSION,source_id=src['step13_source_id'],
            epoch58_source_id=src['id'],step6_folder=str(src['base']),
            step9_folder=str(src['step9_dir']),step10_folder=str(src['step10_dir']),
            step11_folder=str(src['step11_dir']),step12_folder=str(src['step12_dir']),
            previous_step12_sha=src['step12_hashes'],
            train_only_mining=True,test_used=False,
            only_new_variable='Add Train teacher-correct margin protection loss to exactly replayed H1 training',
            teacher='best same-PID image minus best different-PID image, only original-correct Train queries',
            fixed_protect_lambda=PROTECT_LAMBDA,
            prior_fixed_primary_alpha=1.0,prior_fixed_secondary_alpha=0.5,
            steps=TRAIN_STEPS,batch=EMBED_TRAIN_BATCH,negative_count=NUM_NEG,
            seeds=list(SEEDS),completed=False))
    cases=load_historical_cases(src,device)
    say('[历史严格复用] 49组；新增三个P1模型，不改变H1等旧权重')
    for seed in SEEDS:
        ad,history,counter,last=train_adapter(src,out,'P1_H1_correct_rank_protect',seed,wrong,device)
        if last!=TRAIN_STEPS:raise RuntimeError('P1未达到800步')
        del ad
        if device.type=='cuda':torch.cuda.empty_cache()
        atomic_json(out/f'06_P1_H1_correct_rank_protect_seed{seed}_训练统计.json',
            dict(method='P1_H1_correct_rank_protect',seed=seed,steps=last,
                 counter=counter,protect_lambda=PROTECT_LAMBDA,history=history))
        for alpha in ALPHAS:
            key=f'P1_H1_correct_rank_protect_seed{seed}_alpha{alpha:.2f}'
            detail=cache_eval(src,out,key,lambda ss=seed,a=alpha:
                new_eval(src,out,'P1_H1_correct_rank_protect',ss,a,device))
            cases[('P1_H1_correct_rank_protect',seed,alpha)]=detail
    rows,stats=write_reports(src,out,cases,wrong)
    if len(rows)!=55:raise RuntimeError('历史49组+P1 6组应有55组')
    probe=probe_difficulty(src,wrong,out)
    # 对比H1新模型在同一Val文本上：净新增正确、由错转对、由对转错，
    # 同时报告B0已正确子集中的修复与破坏，不能只看R1增益。
    summary={}
    original=cases[('B0_original',0,0.0)]
    for alpha in ALPHAS:
        per_seed=[]
        for seed in SEEDS:
            h=cases[('H1_wrong_ahead',seed,alpha)]
            p=cases[('P1_H1_correct_rank_protect',seed,alpha)]
            hhit=h['rank']==1;phit=p['rank']==1
            b0hit=original['rank']==1
            per_seed.append(dict(seed=seed,alpha=alpha,
                H1_wrong_P1_right=int(np.sum(~hhit&phit)),
                H1_right_P1_wrong=int(np.sum(hhit&~phit)),
                net_correct_delta=int(phit.sum()-hhit.sum()),
                B0_correct_H1_wrong_P1_correct=int(np.sum(b0hit&~hhit&phit)),
                B0_correct_H1_correct_P1_wrong=int(np.sum(b0hit&hhit&~phit)),
                B0_wrong_H1_wrong_P1_correct=int(np.sum(~b0hit&~hhit&phit)),
                B0_wrong_H1_correct_P1_wrong=int(np.sum(~b0hit&hhit&~phit)),
                mAP_delta_pp=get_metrics(p)['mAP']-get_metrics(h)['mAP'],
                mINP_delta_pp=get_metrics(p)['mINP']-get_metrics(h)['mINP']))
        summary[str(alpha)]=per_seed
    atomic_json(out/'08_P1和H1正确查询交换及统计.json',dict(
        note='固定alpha=1主、0.5历史辅助。Val探索性，不以这些结果选择保护权重。',
        per_alpha=summary,train_margin_probe=probe))
    package(out)
    say('[alpha=1 主比较；alpha=0.5 仅固定辅助]')
    for method in ('B2_hard','H1_wrong_ahead','H2_wrong_plus_semi',
                   'C2_rank_marginal','C3_wrongcount_stratified','P1_H1_correct_rank_protect'):
        for a in ALPHAS:
            mm=stats[method][str(a)]
            say(method,'alpha',a,'R1',round(mm['R1']['mean'],4),
                'R5',round(mm['R5']['mean'],4),'R10',round(mm['R10']['mean'],4),
                'mAP',round(mm['mAP']['mean'],4),'mINP',round(mm['mINP']['mean'],4))
    say('报告ZIP:',out/ZIP_NAME)
    return rows

def self_test():
    """真实调用合成端到端第六至十三步，不访问Windows文件。"""
    # 实验复核程序会对准备好的本地完整合成数据调用后面的自检实现。
    return _self_test_full()

def _self_test_full():
    """CPU合成端到端：H1/C3层内分布、历史43组复验、断点续跑、49组报告、ZIP与防污染。"""
    patch_names=('ROOT','SOURCE_STEP9_DIR','SOURCE_STEP10_DIR','SOURCE_STEP11_DIR','SOURCE_STEP12_DIR','RESUME_OUTPUT_DIR',
                 'TRAIN_STEPS','SAVE_EVERY','EMBED_TRAIN_BATCH','RANK_END','PROBE_STEPS','BOOT_REPEATS')
    prev={name:globals()[name] for name in patch_names}
    try:
        globals().update(TRAIN_STEPS=4,SAVE_EVERY=2,EMBED_TRAIN_BATCH=12,
                         RANK_END=12,PROBE_STEPS=(0,1,2,3),BOOT_REPEATS=40,
                         RESUME_OUTPUT_DIR='')
        device=torch.device('cpu');rg=np.random.default_rng(20261021)
        ip=np.repeat(np.arange(6),3).astype(np.int64);tp=ip.copy();pos=np.arange(len(ip),dtype=np.int64)
        vi=rg.normal(size=(18,DIM)).astype(np.float32)
        vi/=np.linalg.norm(vi,axis=1,keepdims=True)
        vt=vi.copy();vt[0]=vi[4].copy()
        wrong=mine_wrong_ahead(vt,vi,tp,ip,device,rank_end=RANK_END,chunk=5)
        half=wrong['top_idx'][:,4:].copy().astype(np.int32)
        assert wrong['wrong_count'][0]>0 and wrong['wrong_count'][1]==0
        b0=evaluate(vt,vi,tp,ip,device)
        with tempfile.TemporaryDirectory() as tmp:
            globals()['ROOT']=Path(tmp)/'IRRA-main'
            root=ROOT/'硬负样本分析';root.mkdir(parents=True)
            old6=root/'第六步_模拟历史成果';old6.mkdir()
            old9=root/(STEP9_PREFIX+'_模拟历史成果');old9.mkdir()
            old10=root/(STEP10_PREFIX+'_模拟历史成果');old10.mkdir()
            old11=root/(STEP11_PREFIX+'_模拟历史成果');old11.mkdir()
            old12=root/(STEP12_PREFIX+'_模拟历史成果');old12.mkdir()
            globals()['SOURCE_STEP9_DIR']=str(old9)
            globals()['SOURCE_STEP10_DIR']=str(old10)
            globals()['SOURCE_STEP11_DIR']=str(old11)
            globals()['SOURCE_STEP12_DIR']=str(old12)
            src=dict(id='synthetic_step9_validated_id',base=old6,fp='synthetic_original_cache',
                     ip=ip,tp=tp,positive=pos,trt=vt,tri=vi,half=half,
                     vt=vt,vi=vi,qp=tp,vp=ip,val_base_reference=get_metrics(b0),history={})
            for method in METHODS_OLD+METHODS_PRIOR_NEW+METHODS_STEP10:
                for seed in SEEDS:
                    torch.manual_seed(seed);ad=Both().to(device)
                    if method in METHODS_PRIOR_NEW:
                        place=old9;fp=fingerprint(dict(source=src['id'],method=method,seed=seed,version=VERSION))
                        record=dict(fingerprint=fp,source_id=src['id'],method=method,
                                    seed=seed,step=TRAIN_STEPS,model=ad.state_dict())
                    elif method in METHODS_STEP10:
                        place=old10
                        # 指纹待第九步来源被确立后再写
                        record=dict(method=method,seed=seed,step=TRAIN_STEPS,model=ad.state_dict())
                    else:
                        place=old6
                        record=dict(fingerprint=src['fp'],method=method,
                                    seed=seed,step=TRAIN_STEPS,model=ad.state_dict())
                    atomic_torch(place/f'{method}_seed{seed}_progress.pt',record)
            poolfp=fingerprint(dict(source=src['id'],rule='negative_similarity_gt_best_same_pid_similarity',top=RANK_END))
            cache=old9/'internal_cache';cache.mkdir()
            atomic_write(cache/'train_wrong_ahead_top50.npz',lambda path:_save_wrong_npz(path,poolfp,wrong))
            atomic_json(old9/'00_方案和来源.json',dict(source_id=src['id'],version=VERSION,
                completed=True,test_used=False,train_only_mining=True,
                seeds=list(SEEDS),all_result_rows=31,query_rows=31*len(tp)))
            atomic_json(old9/'01_Train错误排序分布.json',dict(total_queries=len(tp)))
            atomic_json(old9/'04_三种子汇总及PID_Bootstrap.json',dict(synthetic=True))
            records9={('B0_original',0,0.0):b0}
            for method in METHODS_OLD+METHODS_PRIOR_NEW:
                for seed in SEEDS:
                    ad=(load_old_adapter(src,method,seed,device) if method in METHODS_OLD else
                        _load_fake9_model(old9,src,method,seed,device))
                    for alpha in ALPHAS:
                        records9[(method,seed,alpha)]=evaluate(apply_adapter(vt,ad.text,alpha,device),
                            apply_adapter(vi,ad.image,alpha,device),tp,ip,device)
            def export_cases(folder,records):
                mr=[];dr=[]
                for (method,seed,alpha),d in sorted(records.items()):
                    mr.append(dict(method=method,seed=seed,alpha=alpha,**get_metrics(d)))
                    for i,pid in enumerate(tp):
                        dr.append(dict(method=method,seed=seed,alpha=alpha,query_index=i,
                           pid=int(pid),rank=int(d['rank'][i]),correct=int(d['rank'][i]==1),
                           AP=float(d['AP'][i]),INP=float(d['INP'][i])))
                atomic_csv(folder/'02_所有方案_Val指标.csv',['method','seed','alpha',*METRICS],mr)
                atomic_csv(folder/'03_全部逐查询_Val.csv',
                           ['method','seed','alpha','query_index','pid','rank','correct','AP','INP'],dr)
            assert len(records9)==31;export_cases(old9,records9)
            w=source_from_step9(src)
            assert all(np.array_equal(w[k],wrong[k]) for k in ('top_idx','wrong_count'))
            records10=dict(records9)
            for seed in SEEDS:
                m='C1_top8_unconditional'
                p=old10/f'{m}_seed{seed}_progress.pt'
                st=torch.load(str(p),map_location='cpu',weights_only=False)
                st.update(fingerprint=fingerprint(dict(source=src['step10_source_id'],method=m,
                                 seed=seed,version=STEP10_VERSION)),source_id=src['step10_source_id'])
                atomic_torch(p,st)
                ad=Both().to(device);ad.load_state_dict(st['model'])
                for alpha in ALPHAS:
                    records10[(m,seed,alpha)]=evaluate(apply_adapter(vt,ad.text,alpha,device),
                        apply_adapter(vi,ad.image,alpha,device),tp,ip,device)
            export_cases(old10,records10)
            atomic_json(old10/'00_方案和来源.json',dict(version=STEP10_VERSION,
                source_id=src['step10_source_id'],completed=True,test_used=False,
                train_only_mining=True,all_result_rows=37,query_rows=37*len(tp)))
            atomic_json(old10/'04_三种子汇总及PID_Bootstrap.json',dict(synthetic=True))
            atomic_csv(old10/'07_采样负样本难度_固定Step诊断.csv',['synthetic'],[dict(synthetic='yes')])
            source_from_step10(src);add_rank_profiles(src,w)
            records11=dict(records10)
            for seed in SEEDS:
                method='C2_rank_marginal';torch.manual_seed(seed)
                ad=Both().to(device)
                cp=old11/f'{method}_seed{seed}_progress.pt'
                atomic_torch(cp,dict(fingerprint=fingerprint(dict(source=src['step11_source_id'],
                      method=method,seed=seed,version=STEP11_VERSION)),
                      source_id=src['step11_source_id'],method=method,seed=seed,
                      step=TRAIN_STEPS,model=ad.state_dict()))
                for alpha in ALPHAS:
                    records11[(method,seed,alpha)]=evaluate(apply_adapter(vt,ad.text,alpha,device),
                        apply_adapter(vi,ad.image,alpha,device),tp,ip,device)
            export_cases(old11,records11)
            atomic_json(old11/'00_方案和来源.json',dict(version=STEP11_VERSION,
                source_id=src['step11_source_id'],completed=True,test_used=False,
                train_only_mining=True,all_result_rows=43,query_rows=43*len(tp)))
            atomic_json(old11/'04_三种子汇总及PID_Bootstrap.json',dict(synthetic=True))
            atomic_csv(old11/'07_采样负样本难度_固定Step诊断.csv',['synthetic'],[dict(synthetic='yes')])
            atomic_json(old11/'08_H1排名分布与C2匹配程度.json',dict(synthetic=True))
            source_from_step11(src);add_stratified_profiles(src,w)
            records12=dict(records11)
            for seed in SEEDS:
                method='C3_wrongcount_stratified';torch.manual_seed(seed)
                ad=Both().to(device)
                cp=old12/f'{method}_seed{seed}_progress.pt'
                atomic_torch(cp,dict(fingerprint=fingerprint(dict(source=src['step12_source_id'],
                    method=method,seed=seed,version=STEP12_VERSION)),
                    source_id=src['step12_source_id'],method=method,seed=seed,
                    step=TRAIN_STEPS,model=ad.state_dict()))
                for alpha in ALPHAS:
                    records12[(method,seed,alpha)]=evaluate(apply_adapter(vt,ad.text,alpha,device),
                        apply_adapter(vi,ad.image,alpha,device),tp,ip,device)
            assert len(records12)==49
            export_cases(old12,records12)
            atomic_json(old12/'00_方案和来源.json',dict(version=STEP12_VERSION,
                source_id=src['step12_source_id'],completed=True,test_used=False,
                train_only_mining=True,all_result_rows=49,query_rows=49*len(tp)))
            atomic_json(old12/'04_三种子汇总及PID_Bootstrap.json',dict(synthetic=True))
            atomic_csv(old12/'05_新方案对旧方法_同种子配对差.csv',['synthetic'],[dict(synthetic='yes')])
            atomic_csv(old12/'07_采样负样本难度_固定Step诊断.csv',['synthetic'],[dict(synthetic='yes')])
            atomic_json(old12/'08_H1排名分布与C3条件匹配程度.json',dict(synthetic=True))
            source_from_step12(src)
            assert len(load_historical_cases(src,device))==49
            say('[自检1/5] 六至十二步49组真实权重/报告复用，逐查询重算正确')
            rgq=np.random.default_rng(42*1000003+0*97+3)
            queries=rgq.integers(len(tp),size=EMBED_TRAIN_BATCH,dtype=np.int64)
            actual,ct=choose_negatives(queries,tp,ip,half,w,'P1_H1_correct_rank_protect',42,0)
            expected=choose_old_negatives(queries,tp,ip,half,w,'H1_wrong_ahead',42,0)
            assert np.array_equal(actual,expected),'P1负样本不同于原H1'
            assert not np.any(ip[actual]==tp[queries,None])
            best=best_same_pid_teacher_indices(src,w)
            torch.manual_seed(42);ad=Both()
            local_idx=np.flatnonzero(w['wrong_count']==0)
            assert len(local_idx)>0
            l,details=correct_rank_protection(ad,torch.as_tensor(vt[local_idx]),
                torch.as_tensor(vi),local_idx,w,best,device)
            assert torch.isfinite(l) and l.item()<1e-6 and details['protected']==len(local_idx),l.item()
            # 把一个原来正确的查询故意旋转向错误Top1，loss应该惩罚间隔损失。
            target=local_idx[0]
            changed=dict(w);changed['best_positive']=w['best_positive'].copy()
            changed['best_positive'][target]+=0.15
            bigger,_=correct_rank_protection(ad,torch.as_tensor(vt[[target]]),
                torch.as_tensor(vi),np.asarray([target]),changed,best,device)
            assert bigger.item()>0.1,(bigger.item(),details)
            say('[自检2/5] P1与H1负样本逐元素一致，教师初始惩罚近0、间隔退化惩罚上升')
            split=root/'split';fresh=root/'fresh';split.mkdir();fresh.mkdir()
            _,_,_,step=train_adapter(src,split,'P1_H1_correct_rank_protect',42,w,device,stop_at=2)
            assert step==2
            a,_,_,step=train_adapter(src,split,'P1_H1_correct_rank_protect',42,w,device)
            b,_,_,step=train_adapter(src,fresh,'P1_H1_correct_rank_protect',42,w,device)
            assert step==TRAIN_STEPS
            for key,x in a.state_dict().items():assert torch.equal(x,b.state_dict()[key]),key
            del a,b
            say('[自检3/5] 中断续跑/从头训练参数严格相等')
            out=prepare_output(src)
            rows=run(src,out,device,w)
            assert len(rows)==55 and (out/'08_P1和H1正确查询交换及统计.json').exists()
            with zipfile.ZipFile(out/ZIP_NAME) as zz:
                assert zz.testzip() is None and '07_P1训练保护间隔诊断.csv' in zz.namelist()
            with (out/'03_全部逐查询_Val.csv').open(encoding='utf-8-sig',newline='') as f:
                assert sum(1 for _ in csv.DictReader(f))==55*len(tp)
            original=sha256(out/'03_全部逐查询_Val.csv')
            assert len(run(src,out,device,w))==55
            assert sha256(out/'03_全部逐查询_Val.csv')==original
            say('[自检4/5] 55组报告/CSV/ZIP/缓存复用通过')
            _stage14_integrated_synthetic(src,w,out,device)
            cp=out/'P1_H1_correct_rank_protect_seed42_progress.pt'
            state=torch.load(str(cp),map_location='cpu',weights_only=False)
            state['model']['text.l1.weight']=state['model']['text.l1.weight']+0.02
            atomic_torch(cp,state)
            try:
                cache_eval(src,out,'P1_H1_correct_rank_protect_seed42_alpha1.00',lambda:b0)
            except RuntimeError as exc:assert '指纹' in str(exc)
            else:raise AssertionError('篡改P1权重后仍能使用旧评估缓存')
            history_file=old12/'02_所有方案_Val指标.csv'
            with history_file.open('a',encoding='utf-8') as f:f.write('\n')
            new_src=dict(src)
            source_from_step12(new_src)
            assert src['step13_source_id']!=new_src['step13_source_id']
            globals()['RESUME_OUTPUT_DIR']=str(out)
            try:prepare_output(new_src)
            except RuntimeError as exc:assert '指纹' in str(exc)
            else:raise AssertionError('篡改第十二步历史后仍允许续跑')
            say('[自检5/5] 模型篡改/历史篡改均拒绝复用')
        say('[SELF_TEST PASS] CPU完整合成链路通过；真实Windows/CUDA需本机验证')
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
    parser=argparse.ArgumentParser(description='IRRA第十三步：H1 Train正确排序保护约束')
    parser.add_argument('--self-test',action='store_true',help='CPU全流程合成自检，不读取真实数据')
    args=parser.parse_args()
    if args.self_test:self_test();return
    say('第十三步：H1错误排序纠正+原正确排序保护；设备',DEVICE)
    src=verify_source()
    wrong=source_from_step9(src)
    source_from_step10(src)
    add_rank_profiles(src,wrong)
    source_from_step11(src)
    add_stratified_profiles(src,wrong)
    source_from_step12(src)
    out=prepare_output(src)
    for p in ('base','step9_dir','step10_dir','step11_dir','step12_dir'):
        say('[历史只读]',p,src[p])
    say('[第十三步独立输出]',out)
    run(src,out,torch.device(DEVICE),wrong)


# ================= 第十四步：W1错误纠正+已正确排序保护 =================
# 上方保留了第十三步全部原始算法和验证函数，方便逐项追溯历史协议。
# 下面只新增W1实验；不会调用上面的第十三步run()重训P1。
STEP14_VERSION = 'irra_step14_train_wrong_correction_plus_P1_preservation_v1_20261021'
STEP14_PREFIX = '第十四步_W1错排纠正和正确排序保护'
STEP13_PREFIX = '第十三步_H1已正确排序保护P1'
SOURCE_STEP13_DIR = ''   # 为空自动发现；多个候选时请填完整第十三步目录
RESUME_STEP14_DIR = ''   # 要续跑时填写第十四步已有输出目录，否则新建
STEP14_ZIP_NAME = '上传给ChatGPT_第十四步_W1错排纠正与排序保护.zip'
P1_METHOD = 'P1_H1_correct_rank_protect'
W1_METHOD = 'W1_correct_wrong_and_protect_right'
CORRECT_LAMBDA = 1.0       # 事先固定：与 P1 保护权重对齐，不依据Val搜索
CORRECT_TARGET_MARGIN = 0.05  # 原错排Train查询的目标正负余弦间隔（绝对余弦值）
METHODS_NEW = (P1_METHOD,W1_METHOD)  # 第十四步合并报告须同时呈现历史P1和新W1


def source_from_step13_step14(src):
    """绑定已完成第十三步55组历史结果以及三个P1权重；全部只读。"""
    base = ROOT/'硬负样本分析'
    candidates = ([Path(SOURCE_STEP13_DIR).resolve()] if SOURCE_STEP13_DIR else
        sorted((p for p in base.glob(STEP13_PREFIX+'_*') if p.is_dir()),
               key=lambda p:p.stat().st_mtime,reverse=True))
    found=[]
    for p in candidates:
        if base.resolve() not in p.parents or not p.name.startswith(STEP13_PREFIX+'_'): continue
        mf=p/'00_方案和来源.json'
        if not mf.is_file():continue
        meta=read_json(mf)
        if not (meta.get('version')==STEP13_VERSION and meta.get('source_id')==src['step13_source_id']
                and meta.get('completed') is True and meta.get('train_only_mining') is True
                and meta.get('test_used') is False and meta.get('all_result_rows')==55
                and meta.get('query_rows')==55*len(src['qp'])):continue
        paths=[p/'02_所有方案_Val指标.csv',p/'03_全部逐查询_Val.csv',
               p/'04_三种子汇总及PID_Bootstrap.json',
               p/'05_新方案对旧方法_同种子配对差.csv',
               p/'07_P1训练保护间隔诊断.csv',p/'08_P1和H1正确查询交换及统计.json']
        paths += [p/f'{P1_METHOD}_seed{s}_progress.pt' for s in SEEDS]
        if all(q.is_file() for q in paths):found.append(p)
    if not found:
        raise FileNotFoundError('找不到第十三步55组完整结果和P1本地权重。请填 SOURCE_STEP13_DIR；上传的ZIP不含训练权重。')
    if len(found)>1 and not SOURCE_STEP13_DIR: say('[提示] 找到多个合法第十三步目录，选择最新：',found[0])
    p=found[0]
    critical=[p/'00_方案和来源.json',p/'02_所有方案_Val指标.csv',
              p/'03_全部逐查询_Val.csv',p/'04_三种子汇总及PID_Bootstrap.json',
              p/'05_新方案对旧方法_同种子配对差.csv',
              p/'07_P1训练保护间隔诊断.csv',p/'08_P1和H1正确查询交换及统计.json']
    critical += [p/f'{P1_METHOD}_seed{s}_progress.pt' for s in SEEDS]
    hashes={q.name:sha256(q) for q in critical}
    src['step13_dir']=p;src['step13_hashes']=hashes
    src['step14_source_id']=fingerprint(dict(version=STEP14_VERSION,
        parent=src['step13_source_id'],files=hashes,
        main_objective='H1_unchanged_CE_plus_P1_protect_plus_wrong_best_pos_vs_top1_neg_pairwise',
        wrong_condition='Train_teacher_top1_wrong_ahead',correct_margin=CORRECT_TARGET_MARGIN,
        correct_lambda=CORRECT_LAMBDA,protect_lambda=PROTECT_LAMBDA,
        step=TRAIN_STEPS,batch=EMBED_TRAIN_BATCH,neg=NUM_NEG,
        seeds=SEEDS,alphas=ALPHAS))
    say('[第十三步历史只读目录]',p)
    return p


def load_step13_model14(src,seed,device):
    p=src['step13_dir']/f'{P1_METHOD}_seed{seed}_progress.pt'
    st=torch.load(str(p),map_location='cpu',weights_only=False)
    fp=fingerprint(dict(source=src['step13_source_id'],method=P1_METHOD,seed=seed,
                        version=STEP13_VERSION,protect_lambda=PROTECT_LAMBDA))
    if not (st.get('fingerprint')==fp and st.get('source_id')==src['step13_source_id']
            and st.get('method')==P1_METHOD and st.get('seed')==seed
            and st.get('step')==TRAIN_STEPS):
        raise RuntimeError('第十三步P1权重身份不一致：'+str(p))
    model=Both().to(device);model.load_state_dict(st['model'],strict=True);model.eval()
    for param in model.parameters():param.requires_grad_(False)
    return model


def load_history14(src,device):
    """严格复用旧49组；P1六组要重算真实权重并逐查询交叉核对。"""
    base_cases=load_historical_cases(src,device)
    p=src['step13_dir'];n=len(src['qp'])
    expected=set(base_cases)
    expected.update((P1_METHOD,s,a) for s in SEEDS for a in ALPHAS)
    rows={}
    with (p/'02_所有方案_Val指标.csv').open(encoding='utf-8-sig',newline='') as f:
        for item in csv.DictReader(f):
            key=(item['method'],int(item['seed']),float(item['alpha']))
            if key not in expected or key in rows:raise RuntimeError('第十三步汇总重复/未知实验：'+str(key))
            rows[key]=item
    if set(rows)!=expected or len(rows)!=55:raise RuntimeError('第十三步55组历史指标不完整')
    p1cases={(P1_METHOD,s,a):dict(rank=np.empty(n,dtype=np.int32),
                                  AP=np.empty(n,dtype=np.float64),INP=np.empty(n,dtype=np.float64))
             for s in SEEDS for a in ALPHAS}
    counts={k:0 for k in expected}
    with (p/'03_全部逐查询_Val.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            k=(r['method'],int(r['seed']),float(r['alpha']))
            if k not in expected:raise RuntimeError('第十三步存在不属于历史55组的逐查询记录')
            i=int(r['query_index'])
            if i!=counts[k] or i>=n or int(r['pid'])!=int(src['qp'][i]):
                raise RuntimeError('第十三步逐查询索引/PID/排序不一致：'+str(k))
            d=base_cases[k] if k in base_cases else p1cases[k]
            rank=int(r['rank']);ap=float(r['AP']);inp=float(r['INP'])
            if int(r['correct'])!=int(rank==1):raise RuntimeError('第十三步correct字段错误')
            if k in base_cases:
                if rank!=int(d['rank'][i]) or abs(ap-d['AP'][i])>2e-7 or abs(inp-d['INP'][i])>2e-7:
                    raise RuntimeError('第十三步继承的49组与原始第十二步逐查询不同')
            else:
                d['rank'][i]=rank;d['AP'][i]=ap;d['INP'][i]=inp
            counts[k]+=1
    if any(v!=n for v in counts.values()):raise RuntimeError('第十三步逐查询记录缺失')
    for key in expected:
        d=base_cases[key] if key in base_cases else p1cases[key]
        validate_detail(d,n,len(src['vp']))
        if max(abs(get_metrics(d)[m]-float(rows[key][m])) for m in METRICS)>1e-5:
            raise RuntimeError('第十三步指标与逐查询不一致：'+str(key))
    # 核验第十三步P1训练权重可实际复现结果：不是仅继承CSV
    for seed in SEEDS:
        model=load_step13_model14(src,seed,device)
        for a in ALPHAS:
            got=evaluate(apply_adapter(src['vt'],model.text,a,device),
                         apply_adapter(src['vi'],model.image,a,device),src['qp'],src['vp'],device)
            assert_equal_detail(got,p1cases[(P1_METHOD,seed,a)],f'第十三步P1 seed{seed} alpha{a}')
        del model
    base_cases.update(p1cases)
    if len(base_cases)!=55:raise RuntimeError('第十三步历史组计数不是55')
    say('[历史完全复用] 第六~十三步55组以及P1权重重算、逐查询检查通过')
    return base_cases


def wrong_rank_correction14(ad,text_raw,train_img,query_index,wrong,best_idx,device):
    """针对教师错排Train文本，推高最佳真实同PID正图与Top1异PID图间隔。"""
    q=np.asarray(query_index,dtype=np.int64)
    if not len(q):raise RuntimeError('空训练batch')
    mask=np.asarray(wrong['wrong_count'][q]>0)
    if not np.any(mask):
        return text_raw.sum()*0,dict(wrong=0,still_wrong=0,already_over_margin=0,
                                     margin_mean=0.0,loss_mean=0.0)
    idx=np.flatnonzero(mask);selected=q[idx]
    qtx=ad.text(text_raw[torch.as_tensor(idx,device=device,dtype=torch.long)])
    pos=ad.image(train_img[torch.as_tensor(best_idx[selected],device=device,dtype=torch.long)])
    neg=ad.image(train_img[torch.as_tensor(wrong['top_idx'][selected,0].astype(np.int64),
                                           device=device,dtype=torch.long)])
    margin=(qtx*pos).sum(-1)-(qtx*neg).sum(-1)
    # Softplus (logistic pairwise margin ranking): 有界梯度、平滑，避免极端错排差异造成loss爆炸。
    each=F.softplus((CORRECT_TARGET_MARGIN-margin)/TEMPERATURE)
    # 以完整batch为分母，训练中错误查询比例不同也不放大其权重；与P1保护保持一致。
    objective=each.sum()/len(q)
    return objective,dict(wrong=len(idx),still_wrong=int((margin.detach()<=0).sum().item()),
                          already_over_margin=int((margin.detach()>=CORRECT_TARGET_MARGIN).sum().item()),
                          margin_mean=float(margin.detach().mean().cpu()),
                          loss_mean=float(each.detach().mean().cpu()))


def train_w1_14(src,out,seed,wrong,device,stop_at=None):
    """与P1逐batch/逐负样本一致，仅增加错误Train查询的pairwise纠正。"""
    method=W1_METHOD
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    ad=Both().to(device)
    optimizer=torch.optim.AdamW(ad.parameters(),lr=LR,weight_decay=1e-4)
    cp=out/f'{method}_seed{seed}_progress.pt'
    fp=fingerprint(dict(source=src['step14_source_id'],method=method,seed=seed,
        version=STEP14_VERSION,correct_lambda=CORRECT_LAMBDA,
        correct_margin=CORRECT_TARGET_MARGIN,protect_lambda=PROTECT_LAMBDA))
    start=0;history=[]
    stats=dict(neg_from_wrong=0,wrong_quota_fallback=0,queries_with_wrong=0,
               protected_train_exposures=0,protect_violated_exposures=0,
               wrong_train_exposures=0,wrong_still_wrong_before_step=0,
               wrong_over_target_before_step=0)
    if cp.exists():
        record=torch.load(str(cp),map_location='cpu',weights_only=False)
        if (record.get('fingerprint')!=fp or record.get('source_id')!=src['step14_source_id']
                or record.get('method')!=method or record.get('seed')!=seed
                or type(record.get('step')) is not int or not (0<=record['step']<=TRAIN_STEPS)):
            raise RuntimeError('W1断点权重指纹/步数异常；禁止旧权重误复用')
        ad.load_state_dict(record['model'],strict=True)
        optimizer.load_state_dict(record['optimizer'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):state[k]=v.to(device)
        start=record['step'];history=record['history'];stats=record['stats']
        say('[W1断点续跑]',seed,start,'/',TRAIN_STEPS)
    best_idx=best_same_pid_teacher_indices(src,wrong)
    texts=torch.as_tensor(np.ascontiguousarray(src['trt']),device=device,dtype=torch.float32)
    images=torch.as_tensor(np.ascontiguousarray(src['tri']),device=device,dtype=torch.float32)
    until=TRAIN_STEPS if stop_at is None else min(int(stop_at),TRAIN_STEPS)
    for step in range(start,until):
        rng=np.random.default_rng(seed*1000003+step*97+3)
        ids=rng.integers(len(src['tp']),size=EMBED_TRAIN_BATCH,dtype=np.int64)
        negatives,counts=choose_old_negatives_with_stats(ids,src['tp'],src['ip'],
            src['half'],wrong,seed,step)
        if np.any(src['ip'][negatives]==src['tp'][ids,None]):
            raise RuntimeError('W1负样本包含同PID')
        for k,v in counts.items():stats[k]+=v
        ix=torch.as_tensor(ids,device=device,dtype=torch.long)
        paired=torch.as_tensor(src['positive'][ids],device=device,dtype=torch.long)
        negix=torch.as_tensor(negatives,device=device,dtype=torch.long)
        x=texts[ix];p=images[paired];n=images[negix]
        tx=ad.text(x);px=ad.image(p);nx=ad.image(n)
        logits=torch.cat(((tx*px).sum(-1,keepdim=True),torch.einsum('bd,bkd->bk',tx,nx)),1)/TEMPERATURE
        ce=F.cross_entropy(logits,torch.zeros(len(ids),device=device,dtype=torch.long))
        preserve=((tx-x).square().sum(-1).mean()+(px-p).square().sum(-1).mean()+
                  (nx-n).square().sum(-1).mean())/3
        protect,pr=correct_rank_protection(ad,x,images,ids,wrong,best_idx,device)
        correct,wr=wrong_rank_correction14(ad,x,images,ids,wrong,best_idx,device)
        stats['protected_train_exposures']+=pr['protected']
        stats['protect_violated_exposures']+=pr['violated']
        stats['wrong_train_exposures']+=wr['wrong']
        stats['wrong_still_wrong_before_step']+=wr['still_wrong']
        stats['wrong_over_target_before_step']+=wr['already_over_margin']
        objective=ce+PRESERVE*preserve+PROTECT_LAMBDA*protect+CORRECT_LAMBDA*correct
        if not torch.isfinite(objective):raise RuntimeError('W1训练loss异常 seed='+str(seed)+' step='+str(step))
        optimizer.zero_grad(set_to_none=True);objective.backward()
        torch.nn.utils.clip_grad_norm_(ad.parameters(),2.0);optimizer.step()
        if (step+1)%20==0 or step==0:
            history.append(dict(step=step+1,ce=float(ce.detach().cpu()),
                 preserve=float(preserve.detach().cpu()),protect=float(protect.detach().cpu()),
                 correct=float(correct.detach().cpu()),objective=float(objective.detach().cpu()),
                 protected=pr['protected'],wrong=wr['wrong'],still_wrong=wr['still_wrong'],
                 pre_step_wrong_margin=wr['margin_mean']))
        if (step+1)%SAVE_EVERY==0 or step+1==until:
            atomic_torch(cp,dict(fingerprint=fp,source_id=src['step14_source_id'],
                 method=method,seed=seed,step=step+1,model=ad.state_dict(),
                 optimizer=optimizer.state_dict(),history=history,stats=stats))
            say('[W1训练]',seed,step+1,'/',TRAIN_STEPS,
                'CE=',round(float(ce.detach().cpu()),4),
                'protect=',round(float(protect.detach().cpu()),4),
                'correct=',round(float(correct.detach().cpu()),4))
    if start==TRAIN_STEPS:say('[W1权重复用]',seed)
    return ad,history,stats,until


def prepare_step14_output(src):
    root=ROOT/'硬负样本分析';root.mkdir(parents=True,exist_ok=True)
    if RESUME_STEP14_DIR:
        p=Path(RESUME_STEP14_DIR).resolve()
        if (not p.is_dir() or root.resolve() not in p.parents or
                not p.name.startswith(STEP14_PREFIX+'_')):
            raise RuntimeError('续跑目录必须为本项目硬负样本分析下的第十四步输出目录')
        manifest=p/'00_方案和来源.json'
        if not manifest.is_file() or read_json(manifest).get('source_id')!=src['step14_source_id']:
            raise RuntimeError('第十四步已有目录源指纹变化，拒绝续跑')
        return p
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    p=root/(STEP14_PREFIX+'_'+stamp);suffix=2
    while p.exists():p=root/(STEP14_PREFIX+f'_{stamp}_{suffix}');suffix+=1
    p.mkdir();return p


def new_eval14(src,out,seed,a,device):
    p=out/f'{W1_METHOD}_seed{seed}_progress.pt'
    st=torch.load(str(p),map_location='cpu',weights_only=False)
    fp=fingerprint(dict(source=src['step14_source_id'],method=W1_METHOD,seed=seed,
        version=STEP14_VERSION,correct_lambda=CORRECT_LAMBDA,
        correct_margin=CORRECT_TARGET_MARGIN,protect_lambda=PROTECT_LAMBDA))
    if st.get('fingerprint')!=fp or st.get('step')!=TRAIN_STEPS:
        raise RuntimeError('W1评估权重不是第十四步最终已训练权重')
    model=Both().to(device);model.load_state_dict(st['model'],strict=True);model.eval()
    try:
        return evaluate(apply_adapter(src['vt'],model.text,a,device),
                        apply_adapter(src['vi'],model.image,a,device),src['qp'],src['vp'],device)
    finally:del model


def eval_cache14(src,out,seed,a,compute):
    key=f'{W1_METHOD}_seed{seed}_alpha{a:.2f}'
    path=out/'internal_cache'/f'eval_{key}.npz'
    fp=fingerprint(dict(source=src['step14_source_id'],version=STEP14_VERSION,
        key=key,weight_sha256=sha256(out/f'{W1_METHOD}_seed{seed}_progress.pt')))
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!={'fingerprint','rank','AP','INP'} or str(z['fingerprint'].item())!=fp:
                raise RuntimeError('W1评估缓存指纹不一致；检测到权重或来源变更')
            d={k:np.array(z[k],copy=True) for k in ('rank','AP','INP')}
        say('[评估缓存复用]',key)
    else:
        d=compute()
        def save(temp):
            with temp.open('wb') as f:
                np.savez_compressed(f,fingerprint=np.array(fp),**d)
                f.flush();os.fsync(f.fileno())
        atomic_write(path,save)
    validate_detail(d,len(src['qp']),len(src['vp']))
    return d


def training_probe14(src,out,wrong,device):
    """比较H1/P1/W1固定Train教师的错排纠正和正确保护，探针不能用于调参。"""
    best=best_same_pid_teacher_indices(src,wrong)
    rg=np.random.default_rng(240518)
    correct=np.flatnonzero(wrong['wrong_count']==0)
    errors=np.flatnonzero(wrong['wrong_count']>0)
    count=min(2000,len(correct),len(errors))
    if count<1:raise RuntimeError('Train错排/正确探针缺少样本')
    q=np.concatenate((rg.choice(correct,size=count,replace=False),
                      rg.choice(errors,size=count,replace=False)))
    rows=[]
    for seed in SEEDS:
        for label,loader in [('H1_wrong_ahead',lambda:load_step9_model(src,'H1_wrong_ahead',seed,device)),
                             (P1_METHOD,lambda:load_step13_model14(src,seed,device)),
                             (W1_METHOD,lambda:load_w1_model14(out,seed,device,src))]:
            ad=loader()
            try:
                for a in ALPHAS:
                    margins=[]
                    with torch.inference_mode():
                        for st in range(0,len(q),ADAPTER_BATCH):
                            ids=q[st:st+ADAPTER_BATCH]
                            x=torch.as_tensor(np.ascontiguousarray(src['trt'][ids]),device=device)
                            p=torch.as_tensor(np.ascontiguousarray(src['tri'][best[ids]]),device=device)
                            n=torch.as_tensor(np.ascontiguousarray(src['tri'][wrong['top_idx'][ids,0]]),device=device)
                            margins.append(((ad.text(x,a)*ad.image(p,a)).sum(-1)-
                                (ad.text(x,a)*ad.image(n,a)).sum(-1)).cpu().numpy())
                    scores=np.concatenate(margins)
                    teach=wrong['best_positive'][q]-wrong['best_negative'][q]
                    iscorrect=wrong['wrong_count'][q]==0
                    rows.append(dict(method=label,seed=seed,alpha=a,train_samples=2*count,
                        teacher_wrong_recovered_ratio=float(np.mean(scores[~iscorrect]>0)),
                        teacher_correct_broken_ratio=float(np.mean(scores[iscorrect]<=0)),
                        teacher_correct_mean_margin_change=float(np.mean(scores[iscorrect]-teach[iscorrect])),
                        teacher_wrong_mean_margin_change=float(np.mean(scores[~iscorrect]-teach[~iscorrect])),
                        mean_margin_all=float(scores.mean())))
            finally:del ad
    atomic_csv(out/'07_固定Train探针_纠错与保护.csv',list(rows[0]),rows)
    return rows


def load_w1_model14(out,seed,device,src):
    p=out/f'{W1_METHOD}_seed{seed}_progress.pt'
    st=torch.load(str(p),map_location='cpu',weights_only=False)
    fp=fingerprint(dict(source=src['step14_source_id'],method=W1_METHOD,seed=seed,
        version=STEP14_VERSION,correct_lambda=CORRECT_LAMBDA,
        correct_margin=CORRECT_TARGET_MARGIN,protect_lambda=PROTECT_LAMBDA))
    if st.get('fingerprint')!=fp or st.get('step')!=TRAIN_STEPS:
        raise RuntimeError('W1探针加载权重身份异常')
    model=Both().to(device);model.load_state_dict(st['model'],strict=True);model.eval()
    for par in model.parameters():par.requires_grad_(False)
    return model


def package14(out):
    dest=out/STEP14_ZIP_NAME;tmp=out/('.'+STEP14_ZIP_NAME+'.partial')
    try:
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(out.iterdir()):
                if not p.is_file() or p==dest or p==tmp or p.suffix.lower() in ('.pt','.npy','.npz'):
                    continue
                z.write(p,p.name)
            z.write(Path(__file__).resolve(),'脚本/'+Path(__file__).name)
        with zipfile.ZipFile(tmp) as z:
            if z.testzip() is not None:raise RuntimeError('第十四步ZIP CRC校验不通过')
            required={'02_所有方案_Val指标.csv','03_全部逐查询_Val.csv',
                '04_三种子汇总及PID_Bootstrap.json','05_新方案对旧方法_同种子配对差.csv',
                '07_固定Train探针_纠错与保护.csv','08_W1与P1错对交换.json'}
            if not required.issubset(z.namelist()):raise RuntimeError('第十四步ZIP缺少必要诊断文件')
        os.replace(tmp,dest)
    finally:
        if tmp.exists():tmp.unlink()
    say('[第十四步报告ZIP]',dest)


def run14(src,out,wrong,device):
    manifest=out/'00_方案和来源.json'
    if manifest.exists():
        if read_json(manifest).get('source_id')!=src['step14_source_id']:
            raise RuntimeError('第十四步输出目录的来源已改变，拒绝混用')
    else:
        atomic_json(manifest,dict(version=STEP14_VERSION,source_id=src['step14_source_id'],
            epoch58_source_id=src['id'],step6=str(src['base']),step9=str(src['step9_dir']),
            step10=str(src['step10_dir']),step11=str(src['step11_dir']),
            step12=str(src['step12_dir']),step13=str(src['step13_dir']),
            previous_step13_sha=src['step13_hashes'],test_used=False,train_only_mining=True,
            new_method=W1_METHOD,prior=P1_METHOD,correct_lambda=CORRECT_LAMBDA,
            correct_target_margin=CORRECT_TARGET_MARGIN,protect_lambda=PROTECT_LAMBDA,
            same_h1_batches_and_negatives=True,steps=TRAIN_STEPS,
            batch=EMBED_TRAIN_BATCH,negative_count=NUM_NEG,seeds=list(SEEDS),
            primary_alpha=1.0,secondary_fixed_alpha=0.5,completed=False))
    cases=load_history14(src,device)
    for seed in SEEDS:
        ad,hist,stats,step=train_w1_14(src,out,seed,wrong,device)
        if step!=TRAIN_STEPS:raise RuntimeError('W1训练未完成800步')
        del ad
        if device.type=='cuda':torch.cuda.empty_cache()
        atomic_json(out/f'06_W1_seed{seed}_训练统计.json',dict(method=W1_METHOD,seed=seed,
            steps=step,history=hist,stats=stats,correct_lambda=CORRECT_LAMBDA,
            target_margin=CORRECT_TARGET_MARGIN))
        for a in ALPHAS:
            detail=eval_cache14(src,out,seed,a,lambda s=seed,aa=a:new_eval14(src,out,s,aa,device))
            cases[(W1_METHOD,seed,a)]=detail
    if len(cases)!=61:raise RuntimeError('历史55组加W1六组，应为61组')
    rows,summary=write_reports(src,out,cases,wrong)
    if len(rows)!=61:raise RuntimeError('最终61组报告不完整')
    # 完整报告涵盖H1/P1/W1三个模型在同一查询上的相互改变
    baseline=cases[('B0_original',0,0.0)]
    switches=[]
    for a in ALPHAS:
        for seed in SEEDS:
            h=cases[('H1_wrong_ahead',seed,a)]['rank']==1
            p=cases[(P1_METHOD,seed,a)]['rank']==1
            w=cases[(W1_METHOD,seed,a)]['rank']==1
            b=baseline['rank']==1
            switches.append(dict(seed=seed,alpha=a,
              P1_wrong_W1_right=int((~p&w).sum()),P1_right_W1_wrong=int((p&~w).sum()),
              B0_wrong_P1_wrong_W1_right=int((~b&~p&w).sum()),
              B0_right_P1_right_W1_wrong=int((b&p&~w).sum()),
              H1_correct_P1_wrong_W1_correct=int((h&~p&w).sum()),
              B0_wrong_H1_right_P1_wrong_W1_right=int((~b&h&~p&w).sum()),
              recovered_vs_B0=int((~b&w).sum()),regressed_vs_B0=int((b&~w).sum()),
              net_vs_B0=int(w.sum()-b.sum()),
              W1_minus_P1_mAP=float(get_metrics(cases[(W1_METHOD,seed,a)])['mAP']-
                      get_metrics(cases[(P1_METHOD,seed,a)])['mAP'])))
    atomic_json(out/'08_W1与P1错对交换.json',dict(note='Val仅探索；alpha1主分析，0.5历史辅助',rows=switches))
    training_probe14(src,out,wrong,device)
    package14(out)
    say('[alpha=1 主分析；alpha=0.5 已锁定辅助]')
    for method in ('B2_hard','H1_wrong_ahead',P1_METHOD,W1_METHOD):
        for a in ALPHAS:
            m=summary[method][str(a)]
            say(method,'alpha',a,*(f'{k}={m[k]["mean"]:.4f}' for k in METRICS))
    say('报告ZIP:',out/STEP14_ZIP_NAME)
    return rows


def _stage14_integrated_synthetic(src,wrong,step13_dir,device):
    """从真实合成的第十三步55组/P1权重完整走第十四步61组，而非mock历史加载。"""
    old13=SOURCE_STEP13_DIR
    oldresume=RESUME_STEP14_DIR
    oldmethods=METHODS_NEW
    try:
        globals()['SOURCE_STEP13_DIR']=str(step13_dir)
        globals()['RESUME_STEP14_DIR']=''
        globals()['METHODS_NEW']=(P1_METHOD,W1_METHOD)
        source_from_step13_step14(src)
        first=prepare_step14_output(src)
        rows=run14(src,first,wrong,device)
        assert len(rows)==61
        new=read_json(first/'00_方案和来源.json')
        assert new['completed'] is True and new['all_result_rows']==61
        assert new['query_rows']==61*len(src['qp'])
        sha=sha256(first/'03_全部逐查询_Val.csv')
        with zipfile.ZipFile(first/STEP14_ZIP_NAME) as z:
            assert z.testzip() is None
            assert len(z.namelist())>=10
        globals()['RESUME_STEP14_DIR']=str(first)
        assert prepare_step14_output(src)==first
        again=run14(src,first,wrong,device)
        assert len(again)==61 and sha256(first/'03_全部逐查询_Val.csv')==sha
        assert len([p for p in first.glob('*.pt')])==len(SEEDS)
        say('[第十四步自检:完整历史] 从第十三步55组/P1真实权重到第十四步61组，续跑、ZIP、来源绑定均通过')
        _stage15_integrated_synthetic(src,wrong,first,device)
        # 故意变更来源权重副本会改变历史ID，禁止在旧实验文件夹续跑；不实际篡改第十三步文件
        cp=first/f'{W1_METHOD}_seed42_progress.pt'
        st=torch.load(cp,map_location='cpu',weights_only=False)
        st['model']['text.l1.weight']=st['model']['text.l1.weight']+0.001
        atomic_torch(cp,st)
        try:eval_cache14(src,first,42,1.0,lambda:None)
        except RuntimeError as e:assert '指纹' in str(e)
        else:raise AssertionError('第十四步W1权重篡改未被拒绝')
        copy_src=dict(src)
        copy_src['step14_source_id']=fingerprint(dict(original=src['step14_source_id'],tampered=True))
        try:prepare_step14_output(copy_src)
        except RuntimeError as e:assert '指纹' in str(e)
        else:raise AssertionError('来源变化仍允许续跑第十四步')
        say('[第十四步自检:缓存安全] 权重/历史指纹变化均阻止不安全复用')
    finally:
        globals()['SOURCE_STEP13_DIR']=old13
        globals()['RESUME_STEP14_DIR']=oldresume
        globals()['METHODS_NEW']=oldmethods


def self_test14():
    """真实跑W1数值、断点续跑、报告导出、重复使用；同时复验第十三步原生自检。"""
    say('[自检第1阶段] 第十三步完整合成历史链路')
    old_methods=METHODS_NEW
    try:
        globals()['METHODS_NEW']=(P1_METHOD,)
        _self_test_full()
    finally:
        globals()['METHODS_NEW']=old_methods
    say('[自检第2阶段] 第十四步W1合成训练与缓存及报告')
    old={k:globals()[k] for k in ('TRAIN_STEPS','SAVE_EVERY','EMBED_TRAIN_BATCH','BOOT_REPEATS','RANK_END','ROOT')}
    try:
        globals().update(TRAIN_STEPS=4,SAVE_EVERY=2,EMBED_TRAIN_BATCH=12,
                         BOOT_REPEATS=30,RANK_END=12)
        rg=np.random.default_rng(814)
        ip=np.repeat(np.arange(8),3).astype(np.int64)
        tp=ip.copy();pos=np.arange(len(ip))
        tri=rg.normal(size=(len(ip),DIM)).astype(np.float32)
        tri/=np.linalg.norm(tri,axis=1,keepdims=True)
        trt=tri.copy();trt[0]=tri[8]
        device=torch.device('cpu')
        wrong=mine_wrong_ahead(trt,tri,tp,ip,device,rank_end=RANK_END,chunk=8)
        half=wrong['top_idx'][:,4:].copy()
        src=dict(id='unit_step9',fp='unit',trt=trt,tri=tri,tp=tp,ip=ip,
                 positive=pos,half=half,vt=trt,vi=tri,qp=tp,vp=ip,
                 step14_source_id='synthetic_step14_source_id')
        best=best_same_pid_teacher_indices(src,wrong)
        assert wrong['wrong_count'][0]>0 and wrong['wrong_count'][1]==0
        x=torch.as_tensor(trt)
        # 初始Adapter是恒等映射；错误查询pairwise loss须>0并可反向传播
        torch.manual_seed(42);ad=Both()
        loss,stat=wrong_rank_correction14(ad,x,torch.as_tensor(tri),
            np.arange(len(tp)),wrong,best,device)
        assert float(loss.detach())>0 and stat['wrong']>0
        loss.backward()
        assert ad.text.l2.weight.grad is not None and torch.isfinite(ad.text.l2.weight.grad).all()
        say('[第十四步自检] 教师错排loss非零，可对Adapter产生梯度')
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'run';out.mkdir()
            split=Path(td)/'split';split.mkdir()
            a,_,_,last=train_w1_14(src,split,42,wrong,device,stop_at=2)
            assert last==2
            a,_,_,last=train_w1_14(src,split,42,wrong,device)
            assert last==TRAIN_STEPS
            b,_,_,last=train_w1_14(src,out,42,wrong,device)
            assert last==TRAIN_STEPS
            for key,val in a.state_dict().items():assert torch.equal(val,b.state_dict()[key]),key
            del a,b
            # 检查原始H1采样与W1一致，对训练查询batch种子重放
            for step in range(TRAIN_STEPS):
                ids=np.random.default_rng(42*1000003+step*97+3).integers(len(tp),size=EMBED_TRAIN_BATCH)
                neg,_=choose_old_negatives_with_stats(ids,tp,ip,half,wrong,42,step)
                neg2=choose_old_negatives(ids,tp,ip,half,wrong,'H1_wrong_ahead',42,step)
                assert np.array_equal(neg,neg2)
            # 合成 55 组历史，三种子生成W1；确保完整第十四步报告输出与ZIP
            cases={('B0_original',0,0.0):evaluate(trt,tri,tp,ip,device)}
            history_methods=METHODS_OLD+METHODS_PRIOR_NEW+METHODS_STEP10+METHODS_STEP11+METHODS_STEP12+(P1_METHOD,)
            for m in history_methods:
                for seed in SEEDS:
                    for alpha in ALPHAS:
                        cases[(m,seed,alpha)]=cases[('B0_original',0,0.0)]
            assert len(cases)==55
            for seed in SEEDS:
                if seed!=42:train_w1_14(src,out,seed,wrong,device)
                for aa in ALPHAS:
                    cases[(W1_METHOD,seed,aa)]=eval_cache14(src,out,seed,aa,
                        lambda s=seed,a=aa:new_eval14(src,out,s,a,device))
            assert len(cases)==61
            atomic_json(out/'00_方案和来源.json',dict(source_id=src['step14_source_id'],completed=False))
            rows,summary=write_reports(src,out,cases,wrong)
            assert len(rows)==61
            switches=[dict(seed=42,alpha=1.0,P1_wrong_W1_right=0)]
            atomic_json(out/'08_W1与P1错对交换.json',dict(rows=switches))
            atomic_csv(out/'07_固定Train探针_纠错与保护.csv',['seed'],[dict(seed=42)])
            # 复制运行入口的报告文件要求，检查ZIP签名
            package14(out)
            with zipfile.ZipFile(out/STEP14_ZIP_NAME) as z:
                assert z.testzip() is None
                assert '02_所有方案_Val指标.csv' in z.namelist()
                assert '03_全部逐查询_Val.csv' in z.namelist()
            old_hash=sha256(out/'03_全部逐查询_Val.csv')
            rows2,_=write_reports(src,out,cases,wrong)
            assert len(rows2)==61 and sha256(out/'03_全部逐查询_Val.csv')==old_hash
            # 修改权重，验证评估缓存拒绝继续复用
            cp=out/f'{W1_METHOD}_seed42_progress.pt'
            state=torch.load(cp,map_location='cpu',weights_only=False)
            state['model']['text.l1.weight']=state['model']['text.l1.weight']+0.01
            atomic_torch(cp,state)
            try:eval_cache14(src,out,42,1.0,lambda:cases[('B0_original',0,0.0)])
            except RuntimeError as e:assert '指纹' in str(e)
            else:raise AssertionError('篡改权重后误用了旧评估缓存')
            say('[第十四步自检] 61组报告、ZIP、重复写出、断点恢复和缓存防污染通过')
    finally:
        globals().update(old)
    say('[SELF_TEST PASS] CPU合成数据全链路完成；真实Windows/CUDA尚需本机验证')


def main_step14():
    ap=argparse.ArgumentParser(description='第十四步：H1硬负样本错误查询纠正与P1正确排序保护')
    ap.add_argument('--self-test',action='store_true',help='完整CPU合成自检')
    args=ap.parse_args()
    if args.self_test:self_test14();return
    say('第十四步：错排纠正+正确排序保护；设备',DEVICE)
    src=verify_source()
    wrong=source_from_step9(src)
    source_from_step10(src);add_rank_profiles(src,wrong)
    source_from_step11(src);add_stratified_profiles(src,wrong)
    source_from_step12(src)
    source_from_step13_step14(src)
    out=prepare_step14_output(src)
    for key in ('base','step9_dir','step10_dir','step11_dir','step12_dir','step13_dir'):
        say('[历史只读]',key,src[key])
    say('[第十四步独立输出]',out)
    run14(src,out,wrong,torch.device(DEVICE))



# ================== 第十五步 W2：有停止间隔的受约束纠错 ==================
# 本文件包含第六至十四步的只读来源/训练协议核验代码；新入口只有 main_step15。
# 仅Train监督，Val只做探索性评估。Test不参与。
STEP15_VERSION = 'irra_step15_hinge_orthogonal_capped_wrong_correction_v1_20261021'
STEP15_PREFIX = '第十五步_W2有界纠错与P1正确排序保护'
STEP15_ZIP_NAME = '上传给ChatGPT_第十五步_W2有界纠错与P1对照.zip'
SOURCE_STEP14_DIR = ''   # 留空自动从“硬负样本分析”寻找完整第十四步目录
RESUME_STEP15_DIR = ''   # 中断后填写本次第十五步输出文件夹的完整绝对路径
W2_METHOD = 'W2_margin_stop_gradient_limited'
W2_TARGET_MARGIN = 0.05   # 固定，继承W1原定的间隔门槛
W2_CORRECT_LAMBDA = 1.0
W2_GRADIENT_BUDGET = 0.30 # 相对P1主梯度的纠错梯度范数上限，预先固定
METHODS_NEW = (P1_METHOD,W1_METHOD,W2_METHOD) # 历史6+...+61组及新W2综合报告


def source_from_step14_stage15(src):
    """绑定完整第十四步61组结果和W1三份训练权重，绝不修改旧目录。"""
    root=ROOT/'硬负样本分析'
    candidates=([Path(SOURCE_STEP14_DIR).resolve()] if SOURCE_STEP14_DIR else
         sorted((p for p in root.glob(STEP14_PREFIX+'_*') if p.is_dir()),
                key=lambda p:p.stat().st_mtime,reverse=True))
    valid=[]
    for p in candidates:
        if root.resolve() not in p.parents or not p.name.startswith(STEP14_PREFIX+'_'):
            continue
        manifest=p/'00_方案和来源.json'
        if not manifest.is_file():continue
        info=read_json(manifest)
        if not(info.get('version')==STEP14_VERSION and
               info.get('source_id')==src['step14_source_id'] and
               info.get('completed') is True and
               info.get('test_used') is False and info.get('train_only_mining') is True and
               info.get('all_result_rows')==61 and
               info.get('query_rows')==61*len(src['qp'])):
            continue
        required=[p/name for name in
            ('00_方案和来源.json','02_所有方案_Val指标.csv',
             '03_全部逐查询_Val.csv','04_三种子汇总及PID_Bootstrap.json',
             '05_新方案对旧方法_同种子配对差.csv',
             '07_固定Train探针_纠错与保护.csv','08_W1与P1错对交换.json')]
        required += [p/f'{W1_METHOD}_seed{seed}_progress.pt' for seed in SEEDS]
        if all(q.is_file() for q in required):valid.append((p,required))
    if not valid:
        raise FileNotFoundError('第十四步61组完整结果或W1本地权重缺失；请设置 SOURCE_STEP14_DIR（ZIP不含.pt权重）')
    if len(valid)>1 and not SOURCE_STEP14_DIR:
        say('[第十四步多目录提醒] 使用最新完整目录',valid[0][0])
    p,required=valid[0]
    hashes={q.name:sha256(q) for q in required}
    src['step14_dir']=p;src['step14_hashes']=hashes
    src['step15_source_id']=fingerprint(dict(parent=src['step14_source_id'],
        stage14_files=hashes,version=STEP15_VERSION,method=W2_METHOD,
        primary_alpha=ALPHAS[0],secondary_alpha=ALPHAS[1],
        train_only_mining=True,steps=TRAIN_STEPS,neg=NUM_NEG,batch=EMBED_TRAIN_BATCH,
        seeds=list(SEEDS),protect_lambda=PROTECT_LAMBDA,
        target_margin=W2_TARGET_MARGIN,correction_lambda=W2_CORRECT_LAMBDA,
        correction_grad_budget=W2_GRADIENT_BUDGET))
    say('[第十四步历史只读]',p)
    return p


def load_history15(src,device):
    """61组继承逐条比对；W1需真实权重复算，不依赖简单复制历史报表。"""
    old=load_history14(src,device)
    assert len(old)==55
    p=src['step14_dir'];n=len(src['qp'])
    wanted=set(old)
    wanted.update((W1_METHOD,s,a) for s in SEEDS for a in ALPHAS)
    metrics={}
    with (p/'02_所有方案_Val指标.csv').open('r',encoding='utf-8-sig',newline='') as file:
        for row in csv.DictReader(file):
            key=(row['method'],int(row['seed']),float(row['alpha']))
            if key not in wanted or key in metrics:raise RuntimeError('第十四步指标组非法或重复 '+str(key))
            metrics[key]=row
    if set(metrics)!=wanted or len(metrics)!=61:raise RuntimeError('第十四步指标不是61组')
    w1={(W1_METHOD,s,a):dict(rank=np.empty(n,dtype=np.int32),
           AP=np.empty(n,dtype=np.float64),INP=np.empty(n,dtype=np.float64))
        for s in SEEDS for a in ALPHAS}
    counts={k:0 for k in wanted}
    with (p/'03_全部逐查询_Val.csv').open('r',encoding='utf-8-sig',newline='') as file:
        for row in csv.DictReader(file):
            key=(row['method'],int(row['seed']),float(row['alpha']))
            if key not in wanted:raise RuntimeError('第十四步逐查询未知组 '+str(key))
            i=int(row['query_index'])
            if i>=n or i!=counts[key] or int(row['pid'])!=int(src['qp'][i]):
                raise RuntimeError('第十四步逐查询索引/PID/顺序异常 '+str(key))
            d=old[key] if key in old else w1[key]
            rank=int(row['rank']);ap=float(row['AP']);inp=float(row['INP'])
            if int(row['correct'])!=int(rank==1):raise RuntimeError('第十四步R1正误字段错误')
            if key in old:
                if rank!=int(d['rank'][i]) or abs(ap-d['AP'][i])>2e-7 or abs(inp-d['INP'][i])>2e-7:
                    raise RuntimeError('第十四步旧55组与第十三步原始记录不相等：'+str(key))
            else:
                d['rank'][i]=rank;d['AP'][i]=ap;d['INP'][i]=inp
            counts[key]+=1
    if any(v!=n for v in counts.values()):raise RuntimeError('第十四步逐查询缺失')
    for k in wanted:
        d=old[k] if k in old else w1[k]
        validate_detail(d,n,len(src['vp']))
        if max(abs(get_metrics(d)[m]-float(metrics[k][m])) for m in METRICS)>1e-5:
            raise RuntimeError('第十四步指标与逐查询复算不一致 '+str(k))
    for s in SEEDS:
        model=load_w1_model14(p,s,device,src)
        try:
            for a in ALPHAS:
                got=evaluate(apply_adapter(src['vt'],model.text,a,device),
                    apply_adapter(src['vi'],model.image,a,device),src['qp'],src['vp'],device)
                assert_equal_detail(got,w1[(W1_METHOD,s,a)],f'历史W1 seed{s} alpha{a}')
        finally:del model
    old.update(w1)
    if len(old)!=61:raise RuntimeError('第十四步61组未完整继承')
    say('[历史验证通过] B0/P1/W1和其余58组，共61组；W1真实权重复算')
    return old


def w2_stop_correction(ad,text_raw,images,ids,wrong,best_idx,device):
    """原IRRA错排Train查询: 只有实际间隔小于阈值才更新；到阈值后严格零梯度。

    用固定教师最优同PID图与最强异PID图；仍遵守同PID排除。
    梯度大小还会在训练步骤中限制，此函数不访问Val/Test。
    """
    q=np.asarray(ids,dtype=np.int64)
    mask=np.asarray(wrong['wrong_count'][q]>0)
    if not np.any(mask):
        return text_raw.sum()*0,dict(wrong=0,active=0,already_at_target=0,
            mean_margin=0.0,loss_mean=0.0)
    selected=q[mask];local=np.flatnonzero(mask)
    x=ad.text(text_raw[torch.as_tensor(local,device=device,dtype=torch.long)])
    posidx=torch.as_tensor(best_idx[selected],device=device,dtype=torch.long)
    negidx=torch.as_tensor(wrong['top_idx'][selected,0].astype(np.int64),device=device)
    pos=ad.image(images[posidx]);neg=ad.image(images[negidx])
    margin=(x*pos).sum(-1)-(x*neg).sum(-1)
    # 线性hinge：不再像W1 softplus那样过了阈值还继续增大正负间隔。
    shortage=F.relu((W2_TARGET_MARGIN-margin)/TEMPERATURE)
    loss=shortage.sum()/len(q) # 与W1的全batch分母及训练预算一致
    return loss,dict(wrong=int(len(selected)),active=int((shortage.detach()>0).sum().item()),
        already_at_target=int((shortage.detach()==0).sum().item()),
        mean_margin=float(margin.detach().mean().cpu()),
        loss_mean=float(shortage.detach().mean().cpu()))


def w2_constrained_gradients(base_loss,correction_loss,params):
    """使纠错梯度不会在一阶近似中反向抵消P1基础训练梯度。

    若两者梯度点积为负，去掉纠错项沿P1梯度的反向分量；
    再将纠错梯度范数限制到P1梯度范数的30%。
    不声称保证每条Val查询排序正确，也不让Val信息进入控制器。
    """
    gb=torch.autograd.grad(base_loss,params,retain_graph=bool(correction_loss.requires_grad),allow_unused=True)
    # 对于没有错排Train查询的batch，辅助损失为常数0，不具有grad_fn。
    gcorr=(torch.autograd.grad(correction_loss,params,allow_unused=True)
           if correction_loss.requires_grad else [None for _ in params])
    gb=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,gb)]
    gcorr=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,gcorr)]
    # 所有运算用Tensor，而不是转CPU检查每个参数；保持6GB CUDA可运行。
    gn2=sum((g*g).sum() for g in gb)
    cn2=sum((g*g).sum() for g in gcorr)
    inner=sum((a*b).sum() for a,b in zip(gb,gcorr))
    projection=torch.clamp(-inner,min=0.0)/gn2.clamp_min(1e-24)
    gcorr=[g+projection*b for g,b in zip(gcorr,gb)]
    new_cn2=sum((g*g).sum() for g in gcorr)
    scale=torch.minimum(torch.ones_like(gn2),
        W2_GRADIENT_BUDGET*torch.sqrt(gn2.clamp_min(1e-24))/torch.sqrt(new_cn2.clamp_min(1e-24)))
    scale=torch.where(gn2>1e-24,scale,torch.zeros_like(scale))
    for p,b,c in zip(params,gb,gcorr):p.grad=b+scale*c
    report=dict(base_grad_norm=float(torch.sqrt(gn2).item()),
        correction_grad_norm=float(torch.sqrt(cn2).item()),
        correction_scale=float(scale.item()),
        gradient_cosine=float((inner/(torch.sqrt(gn2*cn2).clamp_min(1e-24))).item()),
        projected=int((inner<0).item()),
        extra_grad_ratio=float((scale*torch.sqrt(new_cn2)/torch.sqrt(gn2.clamp_min(1e-24))).item()))
    if (not np.isfinite(list(report.values())).all() or
        report['extra_grad_ratio']>W2_GRADIENT_BUDGET+1e-5):
        raise RuntimeError('纠错梯度限制异常')
    return report


def w2_train_fingerprint(src,seed):
    return fingerprint(dict(source=src['step15_source_id'],method=W2_METHOD,
        seed=seed,version=STEP15_VERSION,
        correct_lambda=W2_CORRECT_LAMBDA,correct_target=W2_TARGET_MARGIN,
        correction_gradient_budget=W2_GRADIENT_BUDGET,
        P1_protect_lambda=PROTECT_LAMBDA))


def train_w2(src,out,seed,wrong,device,stop_at=None):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    model=Both().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    cp=out/f'{W2_METHOD}_seed{seed}_progress.pt'
    fp=w2_train_fingerprint(src,seed)
    step0=0;history=[]
    stats=dict(neg_from_wrong=0,wrong_quota_fallback=0,queries_with_wrong=0,
        protected_train_exposures=0,protect_violated_exposures=0,
        teacher_wrong_exposures=0,correction_active_exposures=0,
        already_at_target=0,gradient_conflict_steps=0,
        correction_gradient_ratio_sum=0.0,gradient_steps=0)
    if cp.exists():
        st=torch.load(str(cp),map_location='cpu',weights_only=False)
        if(st.get('fingerprint')!=fp or st.get('source_id')!=src['step15_source_id'] or
           st.get('method')!=W2_METHOD or st.get('seed')!=seed or
           type(st.get('step')) is not int or not 0<=st['step']<=TRAIN_STEPS):
            raise RuntimeError('W2训练断点的身份/配置/步数不一致')
        model.load_state_dict(st['model'],strict=True);optimizer.load_state_dict(st['optimizer'])
        for val in optimizer.state.values():
            for key,t in val.items():
                if torch.is_tensor(t):val[key]=t.to(device)
        step0=st['step'];history=st['history'];stats=st['stats']
        say('[W2断点恢复]',seed,step0,'/',TRAIN_STEPS)
    best_idx=best_same_pid_teacher_indices(src,wrong)
    texts=torch.as_tensor(np.ascontiguousarray(src['trt']),device=device,dtype=torch.float32)
    images=torch.as_tensor(np.ascontiguousarray(src['tri']),device=device,dtype=torch.float32)
    params=[p for p in model.parameters() if p.requires_grad]
    end=TRAIN_STEPS if stop_at is None else min(TRAIN_STEPS,int(stop_at))
    for step in range(step0,end):
        indices=np.random.default_rng(seed*1000003+step*97+3).integers(
            len(src['tp']),size=EMBED_TRAIN_BATCH,dtype=np.int64)
        neg,counts=choose_old_negatives_with_stats(indices,src['tp'],src['ip'],src['half'],wrong,seed,step)
        if neg.shape!=(len(indices),NUM_NEG) or np.any(src['ip'][neg]==src['tp'][indices,None]):
            raise RuntimeError('W2负样本不满足H1负样本协议或存在同PID')
        for k,v in counts.items():stats[k]+=v
        ids=torch.as_tensor(indices,device=device,dtype=torch.long)
        posids=torch.as_tensor(src['positive'][indices],device=device,dtype=torch.long)
        negids=torch.as_tensor(neg,device=device,dtype=torch.long)
        x=texts[ids];p=images[posids];n=images[negids]
        tx=model.text(x);px=model.image(p);nx=model.image(n)
        logits=torch.cat(((tx*px).sum(-1,keepdim=True),
             torch.einsum('bd,bkd->bk',tx,nx)),1)/TEMPERATURE
        ce=F.cross_entropy(logits,torch.zeros(len(indices),device=device,dtype=torch.long))
        preserve=((tx-x).square().sum(-1).mean()+(px-p).square().sum(-1).mean()+
                  (nx-n).square().sum(-1).mean())/3
        protect,pr=correct_rank_protection(model,x,images,indices,wrong,best_idx,device)
        correct,wr=w2_stop_correction(model,x,images,indices,wrong,best_idx,device)
        base=ce+PRESERVE*preserve+PROTECT_LAMBDA*protect
        aux=W2_CORRECT_LAMBDA*correct
        if not (torch.isfinite(base) and torch.isfinite(aux)):
            raise RuntimeError(f'W2非有限loss: seed={seed},step={step}')
        optimizer.zero_grad(set_to_none=True)
        grad=w2_constrained_gradients(base,aux,params)
        torch.nn.utils.clip_grad_norm_(params,2.0)
        optimizer.step()
        stats['protected_train_exposures']+=pr['protected']
        stats['protect_violated_exposures']+=pr['violated']
        stats['teacher_wrong_exposures']+=wr['wrong']
        stats['correction_active_exposures']+=wr['active']
        stats['already_at_target']+=wr['already_at_target']
        stats['gradient_conflict_steps']+=grad['projected']
        stats['correction_gradient_ratio_sum']+=grad['extra_grad_ratio']
        stats['gradient_steps']+=1
        if (step+1)%20==0 or step==0:
            history.append(dict(step=step+1,CE=float(ce.detach().item()),
                preservation=float(preserve.detach().item()),
                protection=float(protect.detach().item()),
                correction=float(correct.detach().item()),
                active=wr['active'],already_at_target=wr['already_at_target'],**grad))
        if (step+1)%SAVE_EVERY==0 or step+1==end:
            atomic_torch(cp,dict(fingerprint=fp,source_id=src['step15_source_id'],
                method=W2_METHOD,seed=seed,step=step+1,model=model.state_dict(),
                optimizer=optimizer.state_dict(),history=history,stats=stats))
            say('[W2训练]',seed,step+1,'/',TRAIN_STEPS,
                'CE',round(float(ce.detach().item()),4),
                'protect',round(float(protect.detach().item()),4),
                'stop_correct',round(float(correct.detach().item()),4),
                'grad_ratio',round(grad['extra_grad_ratio'],3))
    if step0==TRAIN_STEPS:say('[W2权重复用]',seed)
    return model,history,stats,end


def load_w2(src,out,seed,device):
    cp=out/f'{W2_METHOD}_seed{seed}_progress.pt'
    st=torch.load(str(cp),map_location='cpu',weights_only=False)
    if not(st.get('fingerprint')==w2_train_fingerprint(src,seed) and
        st.get('source_id')==src['step15_source_id'] and st.get('method')==W2_METHOD and
        st.get('seed')==seed and st.get('step')==TRAIN_STEPS):
        raise RuntimeError('W2最终权重不完整/不匹配')
    model=Both().to(device);model.load_state_dict(st['model'],strict=True)
    model.eval()
    for param in model.parameters():param.requires_grad_(False)
    return model


def new_eval15(src,out,seed,alpha,device):
    model=load_w2(src,out,seed,device)
    try:
        return evaluate(apply_adapter(src['vt'],model.text,alpha,device),
                        apply_adapter(src['vi'],model.image,alpha,device),
                        src['qp'],src['vp'],device)
    finally:del model


def eval_cache15(src,out,seed,alpha,compute):
    label=f'{W2_METHOD}_seed{seed}_alpha{alpha:.2f}'
    path=out/'internal_cache'/f'eval_{label}.npz'
    fp=fingerprint(dict(source=src['step15_source_id'],version=STEP15_VERSION,
        label=label,weight_sha256=sha256(out/f'{W2_METHOD}_seed{seed}_progress.pt')))
    if path.is_file():
        with np.load(path,allow_pickle=False) as data:
            if set(data.files)!={'fingerprint','rank','AP','INP'} or str(data['fingerprint'].item())!=fp:
                raise RuntimeError('第十五步旧评估缓存指纹失配，拒绝误用')
            detail={key:np.array(data[key],copy=True) for key in ('rank','AP','INP')}
        say('[W2评估缓存复用]',label)
    else:
        detail=compute()
        def save(temp):
            with temp.open('wb') as file:
                np.savez_compressed(file,fingerprint=np.array(fp),**detail)
                file.flush();os.fsync(file.fileno())
        atomic_write(path,save)
    validate_detail(detail,len(src['qp']),len(src['vp']))
    return detail


def prepare_output15(src):
    root=ROOT/'硬负样本分析';root.mkdir(parents=True,exist_ok=True)
    if RESUME_STEP15_DIR:
        p=Path(RESUME_STEP15_DIR).resolve()
        if(not p.is_dir() or root.resolve() not in p.parents or
           not p.name.startswith(STEP15_PREFIX+'_')):
            raise RuntimeError('W2续跑文件夹必须位于硬负样本分析之下且为第十五步目录')
        manifest=p/'00_方案和来源.json'
        if not manifest.is_file() or read_json(manifest).get('source_id')!=src['step15_source_id']:
            raise RuntimeError('W2旧目录数据源发生变化，拒绝续跑')
        return p
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    p=root/(STEP15_PREFIX+'_'+stamp);suffix=2
    while p.exists():p=root/(STEP15_PREFIX+f'_{stamp}_{suffix}');suffix+=1
    p.mkdir();return p


def training_probe15(src,out,wrong,device):
    """固定Train教师探针，不用作挑选模型参数。B0/P1/W1/W2统一比较。"""
    best=best_same_pid_teacher_indices(src,wrong)
    rg=np.random.default_rng(240518)
    correct=np.flatnonzero(wrong['wrong_count']==0)
    errors=np.flatnonzero(wrong['wrong_count']>0)
    count=min(2000,len(correct),len(errors))
    if count<1:raise RuntimeError('Train正确/错误探针不足')
    chosen=np.r_[rg.choice(correct,count,replace=False),rg.choice(errors,count,replace=False)]
    rows=[]
    source=[(P1_METHOD,lambda seed:load_step13_model14(src,seed,device)),
        (W1_METHOD,lambda seed:load_w1_model14(src['step14_dir'],seed,device,src)),
        (W2_METHOD,lambda seed:load_w2(src,out,seed,device))]
    for seed in SEEDS:
        for method,loader in source:
            model=loader(seed)
            try:
                for alpha in ALPHAS:
                    margins=[]
                    with torch.inference_mode():
                        for start in range(0,len(chosen),ADAPTER_BATCH):
                            q=chosen[start:start+ADAPTER_BATCH]
                            x=torch.as_tensor(np.ascontiguousarray(src['trt'][q]),device=device)
                            p=torch.as_tensor(np.ascontiguousarray(src['tri'][best[q]]),device=device)
                            n=torch.as_tensor(np.ascontiguousarray(src['tri'][wrong['top_idx'][q,0]]),device=device)
                            margins.append(((model.text(x,alpha)*model.image(p,alpha)).sum(-1)-
                               (model.text(x,alpha)*model.image(n,alpha)).sum(-1)).cpu().numpy())
                    current=np.concatenate(margins)
                    teacher=(wrong['best_positive'][chosen]-wrong['best_negative'][chosen])
                    flag=wrong['wrong_count'][chosen]==0
                    rows.append(dict(method=method,seed=seed,alpha=alpha,train_query_count=2*count,
                        teacher_wrong_recovered_ratio=float((current[~flag]>0).mean()),
                        teacher_correct_broken_ratio=float((current[flag]<=0).mean()),
                        correct_margin_change=float((current[flag]-teacher[flag]).mean()),
                        wrong_margin_change=float((current[~flag]-teacher[~flag]).mean())))
            finally:del model
    atomic_csv(out/'07_固定Train探针_三方案保护纠错.csv',list(rows[0]),rows)


def w2_swap_report(cases,out):
    baseline=cases[('B0_original',0,0.0)]['rank']==1
    rows=[]
    for alpha in ALPHAS:
        for seed in SEEDS:
            d={name:cases[(name,seed,alpha)]['rank']==1 for name in (P1_METHOD,W1_METHOD,W2_METHOD)}
            for ref_name,ref in [('B0_original',baseline),
                    (P1_METHOD,d[P1_METHOD]),(W1_METHOD,d[W1_METHOD])]:
                w=d[W2_METHOD]
                rows.append(dict(seed=seed,alpha=alpha,reference=ref_name,
                    reference_wrong_W2_right=int((~ref&w).sum()),
                    reference_right_W2_wrong=int((ref&~w).sum()),
                    net_correct=int(w.sum()-ref.sum()),
                    B0_right_reference_wrong_W2_right=int((baseline&~ref&w).sum()),
                    B0_wrong_reference_right_W2_wrong=int((~baseline&ref&~w).sum())))
    atomic_csv(out/'08_W2与B0_P1_W1逐种子错对交换.csv',list(rows[0]),rows)


def w2_comparison_report(cases,out):
    rows=[]
    for alpha in ALPHAS:
        for seed in SEEDS:
            new=get_metrics(cases[(W2_METHOD,seed,alpha)])
            for ref in ('B0_original',P1_METHOD,W1_METHOD,'H1_wrong_ahead'):
                before=get_metrics(cases[('B0_original',0,0.0)] if ref=='B0_original'
                                    else cases[(ref,seed,alpha)])
                rows.append(dict(alpha=alpha,seed=seed,method=W2_METHOD,reference=ref,
                                 **{'delta_'+k:float(new[k]-before[k]) for k in METRICS}))
    atomic_csv(out/'09_W2对B0_P1_W1_H1同种子指标差.csv',list(rows[0]),rows)


def package15(out):
    dest=out/STEP15_ZIP_NAME;temp=out/('.'+STEP15_ZIP_NAME+'.partial')
    needed={'00_方案和来源.json','02_所有方案_Val指标.csv',
        '03_全部逐查询_Val.csv','04_三种子汇总及PID_Bootstrap.json',
        '05_新方案对旧方法_同种子配对差.csv',
        '07_固定Train探针_三方案保护纠错.csv',
        '08_W2与B0_P1_W1逐种子错对交换.csv',
        '09_W2对B0_P1_W1_H1同种子指标差.csv'}
    try:
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for path in sorted(out.iterdir()):
                if not path.is_file() or path in (dest,temp) or path.suffix.lower() in ('.npy','.npz','.pt'):
                    continue
                z.write(path,path.name)
            z.write(Path(__file__).resolve(),'脚本/'+Path(__file__).name)
        with zipfile.ZipFile(temp) as z:
            if z.testzip() is not None or not needed.issubset(z.namelist()):
                raise RuntimeError('第十五步报告ZIP校验未通过')
        os.replace(temp,dest)
    finally:
        if temp.exists():temp.unlink()
    say('[第十五步ZIP]',dest)


def run15(src,out,wrong,device):
    manifest=out/'00_方案和来源.json'
    if manifest.is_file():
        m=read_json(manifest)
        if m.get('source_id')!=src['step15_source_id'] or m.get('version')!=STEP15_VERSION:
            raise RuntimeError('第十五步输出目录来源/配置指纹不一致')
    else:
        atomic_json(manifest,dict(version=STEP15_VERSION,
          source_id=src['step15_source_id'],original_epoch58_id=src['id'],
          sources={key:str(src[key]) for key in ('base','step9_dir','step10_dir',
              'step11_dir','step12_dir','step13_dir','step14_dir')},
          previous_step14_sha=src['step14_hashes'],train_only_mining=True,test_used=False,
          objective='Compared with B0 always, P1 and W1; bounded teacher-wrong ranking correction while protecting teacher-correct',
          frozen_irra=True,train_steps=TRAIN_STEPS,batch=EMBED_TRAIN_BATCH,
          negatives_per_query=NUM_NEG,seeds=list(SEEDS),
          correct_lambda=W2_CORRECT_LAMBDA,target_margin=W2_TARGET_MARGIN,
          gradient_budget=W2_GRADIENT_BUDGET,
          same_h1_negatives=True,same_P1_protection=True,
          alpha_primary=1.0,alpha_secondary_historical=0.5,
          completed=False))
    cases=load_history15(src,device)
    for seed in SEEDS:
        model,history,stats,step=train_w2(src,out,seed,wrong,device)
        if step!=TRAIN_STEPS:raise RuntimeError('W2训练未完成全部800步')
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
        atomic_json(out/f'06_W2_seed{seed}_训练统计.json',dict(method=W2_METHOD,
            seed=seed,steps=step,stats=stats,history=history,
            correct_lambda=W2_CORRECT_LAMBDA,gradient_budget=W2_GRADIENT_BUDGET))
        for alpha in ALPHAS:
            detail=eval_cache15(src,out,seed,alpha,
                lambda s=seed,a=alpha:new_eval15(src,out,s,a,device))
            cases[(W2_METHOD,seed,alpha)]=detail
    if len(cases)!=67:raise RuntimeError('第十四步61组+第十五步6组应共67组')
    rows,summary=write_reports(src,out,cases,wrong)
    if len(rows)!=67:raise RuntimeError('第十五步67组报告不完整')
    # 写报告的通用模板有第十三步的说明，这里更新为正确的第十五步协议。
    report=read_json(out/'04_三种子汇总及PID_Bootstrap.json')
    report['note']='第十五步: Val探索性，α=1主比较/0.5历史辅助；身份Bootstrap不覆盖多种子训练随机性或多轮选择。'
    report['current_step']=15
    atomic_json(out/'04_三种子汇总及PID_Bootstrap.json',report)
    w2_comparison_report(cases,out)
    w2_swap_report(cases,out)
    training_probe15(src,out,wrong,device)
    package15(out)
    say('[α=1主比较；α=0.5历史辅助；B0始终为原始IRRA]')
    for meth in ('B0_original','H1_wrong_ahead',P1_METHOD,W1_METHOD,W2_METHOD):
        for a in ((0.0,) if meth=='B0_original' else ALPHAS):
            rec=get_metrics(cases[('B0_original',0,0.0)] if meth=='B0_original'
                            else cases[(meth,SEEDS[0],a)])
            if meth!='B0_original':
                rec={k:summary[meth][str(a)][k]['mean'] for k in METRICS}
            say(meth,'alpha',a,*(f'{key}={rec[key]:.4f}' for key in METRICS))
    say('报告ZIP:',out/STEP15_ZIP_NAME)
    return rows


def _stage15_integrated_synthetic(src,wrong,step14_dir,device):
    """由第14步完整合成61组/W1真权重接力到W2，实际复跑/断点/导出。"""
    hold=(SOURCE_STEP14_DIR,RESUME_STEP15_DIR,METHODS_NEW)
    try:
        globals()['SOURCE_STEP14_DIR']=str(step14_dir)
        globals()['RESUME_STEP15_DIR']=''
        globals()['METHODS_NEW']=(P1_METHOD,W1_METHOD,W2_METHOD)
        source_from_step14_stage15(src)
        # 对原先正确/错误Train查询执行零梯度及非零梯度门槛检验。
        best=best_same_pid_teacher_indices(src,wrong)
        torch.manual_seed(2026);model=Both().to(device)
        texts=torch.as_tensor(np.ascontiguousarray(src['trt']),device=device)
        images=torch.as_tensor(np.ascontiguousarray(src['tri']),device=device)
        wrong_index=np.flatnonzero(wrong['wrong_count']>0)
        loss,detail=w2_stop_correction(model,texts,images,wrong_index,wrong,best,device)
        assert detail['wrong']>0 and detail['active']>0 and float(loss.detach())>0
        # 把门槛置于无穷低（本地测试临时更改全局），损失应恰好为零。
        prev=W2_TARGET_MARGIN
        try:
            globals()['W2_TARGET_MARGIN']=-2.1
            zero,flag=w2_stop_correction(model,texts,images,wrong_index,wrong,best,device)
            assert float(zero.detach())==0.0 and flag['active']==0
        finally:globals()['W2_TARGET_MARGIN']=prev
        # W2真实断点续跑与不中断训练逐参数相等（含AdamW优化器恢复）。
        import tempfile as _temp15
        with _temp15.TemporaryDirectory() as _run15tmp:
            split=Path(_run15tmp)/'partial';fresh=Path(_run15tmp)/'fresh'
            split.mkdir();fresh.mkdir()
            first,_,_,step=train_w2(src,split,42,wrong,device,stop_at=2)
            assert step==2
            resumed,_,_,step=train_w2(src,split,42,wrong,device)
            uninterrupted,_,_,step=train_w2(src,fresh,42,wrong,device)
            assert step==TRAIN_STEPS
            for name,parameter in resumed.state_dict().items():
                assert torch.equal(parameter,uninterrupted.state_dict()[name]),'W2续跑参数不一致 '+name
            del first,resumed,uninterrupted
        test_out=prepare_output15(src)
        rows=run15(src,test_out,wrong,device)
        assert len(rows)==67
        assert read_json(test_out/'00_方案和来源.json')['all_result_rows']==67
        with zipfile.ZipFile(test_out/STEP15_ZIP_NAME) as z:
            assert z.testzip() is None and '09_W2对B0_P1_W1_H1同种子指标差.csv' in z.namelist()
        sha=sha256(test_out/'03_全部逐查询_Val.csv')
        globals()['RESUME_STEP15_DIR']=str(test_out)
        assert prepare_output15(src)==test_out
        assert len(run15(src,test_out,wrong,device))==67
        assert sha==sha256(test_out/'03_全部逐查询_Val.csv')
        # 权重变化后旧评估缓存必须拒绝复用。
        cp=test_out/f'{W2_METHOD}_seed42_progress.pt'
        st=torch.load(str(cp),map_location='cpu',weights_only=False)
        st['model']['text.l1.weight']=st['model']['text.l1.weight']+0.001
        atomic_torch(cp,st)
        try:eval_cache15(src,test_out,42,1.0,lambda:None)
        except RuntimeError as e:assert '指纹' in str(e)
        else:raise AssertionError('第十五步没有拒绝权重篡改缓存')
        modified_src=dict(src);modified_src['step15_source_id']='tampered_origin'
        try:prepare_output15(modified_src)
        except RuntimeError as e:assert '指纹' in str(e) or '来源' in str(e) or '数据源' in str(e)
        else:raise AssertionError('第十五步没有拒绝来源变化')
        say('[第十五步全链路自检] 61组严格继承+W2三种子+67组+断点+ZIP+篡改拦截通过')
    finally:
        globals()['SOURCE_STEP14_DIR'],globals()['RESUME_STEP15_DIR'],globals()['METHODS_NEW']=hold


def self_test15():
    """先实际合成第六至第十四步连续历史，再执行第十五步集成测试。"""
    old=METHODS_NEW
    try:
        globals()['METHODS_NEW']=(P1_METHOD,)
        _self_test_full()  # 已注入 _stage15_integrated_synthetic 到第14步模拟链路
    finally:globals()['METHODS_NEW']=old
    # 独立检验W2的梯度约束：相反方向纠错梯度不削弱基线方向。
    param=torch.nn.Parameter(torch.tensor([0.0,0.0]))
    a=(param[0]-param[1])
    b=(-2*param[0]+3*param[1])
    info=w2_constrained_gradients(a,b,[param])
    assert info['projected']==1 and info['extra_grad_ratio']<=W2_GRADIENT_BUDGET+1e-6
    assert torch.isfinite(param.grad).all()
    say('[SELF_TEST PASS] 真实CPU合成端到端 + H1负样本重放 + 有界梯度 + 历史复用 + 导出通过')


def main_step15():
    ap=argparse.ArgumentParser(description='IRRA硬负样本第十五步 W2：停止间隔+有界纠错，B0/P1/W1对照')
    ap.add_argument('--self-test',action='store_true',help='在CPU上执行全链路合成测试')
    args=ap.parse_args()
    if args.self_test:self_test15();return
    say('第十五步W2：原IRRA(B0)始终为基线，P1/W1为主要对照；设备',DEVICE)
    src=verify_source()
    wrong=source_from_step9(src)
    source_from_step10(src);add_rank_profiles(src,wrong)
    source_from_step11(src);add_stratified_profiles(src,wrong)
    source_from_step12(src)
    source_from_step13_step14(src)
    source_from_step14_stage15(src)
    out=prepare_output15(src)
    for key in ('base','step9_dir','step10_dir','step11_dir','step12_dir','step13_dir','step14_dir'):
        say('[历史只读]',key,src[key])
    say('[第十五步结果]',out)
    run15(src,out,wrong,torch.device(DEVICE))

if __name__=='__main__':
    try:main_step15()
    except BaseException:
        traceback.print_exc()
        raise
