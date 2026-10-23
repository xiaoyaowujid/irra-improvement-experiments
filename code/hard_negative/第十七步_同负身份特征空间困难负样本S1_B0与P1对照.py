# -*- coding: utf-8 -*-
"""
第十三步：冻结Epoch58 IRRA + H1负样本，新增P1「已正确查询排序保护」损失。
严格只在Train定义原始正确：同PID最佳图像余弦 > 最强异PID图像余弦。
保持H1完全相同batch查询、8负样本、随机数、Adapter、800步及优化器。
P1新增固定lambda=1的保序惩罚，仅在原始正确的Train查询上：
  [max(0,(teacher(best-positive - top1-negative)-student_margin)/T)]^2
best-positive和top1-negative都来自同一冻结Epoch58 Train图库，不用Val/Test标签；
正/负图像锚点保持教师索引固定，不额外加入主CE八负样本。
第6/9/10/11/12步已保存权重全部只读，历史49组严格核验，新增P1三种子、
两档已预注册alpha=1主/0.5辅助，合计55组Val结果。禁止Test调参。
--self-test: CPU合成端到端检查，Windows/CUDA真实性仍需用户本机实测。
RESUME_OUTPUT_DIR 可手动粘贴第十三步输出文件夹，原实验结果绝不覆盖。
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



# ====================================================================
# 第十七步 S1：同错误身份内特征混合困难负样本，保留 P1 所有监督。
# 所有第6~13步历史成果仅仅读取；不依赖未实际完成的第15步输出目录。
# S0 = 旧P1（完全真实H1负样本），S1 = P1 + 固定比例的同负PID特征混合。
# ====================================================================
STEP17_VERSION = 'irra_step17_same_negative_pid_feature_mix_v1_20261022'
STEP17_PREFIX = '第十七步_同身份特征混合困难负样本S1'
STEP17_METHOD = 'S1_same_negative_pid_feature_mix'
STEP17_ZIP = '上传给ChatGPT_第十七步_硬负样本特征空间混合.zip'
SOURCE_STEP13_DIR = ''    # 自动查找第十三步完整实验；多目录时填写绝对路径
RESUME_STEP17_DIR = ''    # 中断后续跑填写本次第十七步文件夹绝对路径
MIX_SLOTS = 4             # 原H1的8个异PID负样本中最多替换4个
MIX_COEFFS = (0.25, 0.5, 0.75)  # 固定，训练文本对同PID混合候选选最大余弦
MIX_MIN_GAIN = 0.00001    # 合成候选比原负图稍难才替换，否则保留原图


def source_from_step13_stage17(src):
    """严格绑定已完成第十三步P1源文件；不要求第14~16步曾运行。"""
    root=(ROOT/'硬负样本分析').resolve()
    candidates=([Path(SOURCE_STEP13_DIR).resolve()] if SOURCE_STEP13_DIR else
        sorted((p.resolve() for p in root.glob('第十三步_H1已正确排序保护P1_*') if p.is_dir()),
               key=lambda p:p.stat().st_mtime,reverse=True))
    choices=[]
    for p in candidates:
        if root not in p.parents or (not SOURCE_STEP13_DIR and not p.name.startswith('第十三步_H1已正确排序保护P1_')):
            continue
        manifest=p/'00_方案和来源.json'
        if not manifest.is_file():continue
        m=read_json(manifest)
        if not (m.get('version')==STEP13_VERSION and m.get('source_id')==src['step13_source_id']
                and m.get('completed') is True and m.get('test_used') is False
                and m.get('train_only_mining') is True and m.get('all_result_rows')==55
                and m.get('query_rows')==55*len(src['qp'])):
            continue
        required=[p/name for name in ('00_方案和来源.json','02_所有方案_Val指标.csv',
                '03_全部逐查询_Val.csv','04_三种子汇总及PID_Bootstrap.json',
                '08_P1和H1正确查询交换及统计.json')]
        required += [p/f'P1_H1_correct_rank_protect_seed{s}_progress.pt' for s in SEEDS]
        if all(x.is_file() for x in required):choices.append((p,required))
    if not choices:
        raise FileNotFoundError('缺少第十三步已完成P1实验目录/本地三种子.pt。\n'
            '请填写脚本顶部 SOURCE_STEP13_DIR；上传ZIP不含.pt，不能代替本地权重。')
    p,files=choices[0]
    hashes={x.name:sha256(x) for x in files}
    src['step13_dir']=p
    src['step17_source_id']=fingerprint(dict(parent=src['step13_source_id'],
        version=STEP17_VERSION,history_files_sha=hashes,training=TRAIN_STEPS,
        batch=EMBED_TRAIN_BATCH,num_neg=NUM_NEG,seed=list(SEEDS),
        lmbda_protect=PROTECT_LAMBDA,alpha=ALPHAS,mix_slots=MIX_SLOTS,
        mix_coefficients=MIX_COEFFS,mix_min_gain=MIX_MIN_GAIN))
    src['step17_history_hashes']=hashes
    say('[P1历史严格只读]',p)
    return p


def load_p1_17(src,seed,device):
    p=src['step13_dir']/f'P1_H1_correct_rank_protect_seed{seed}_progress.pt'
    st=torch.load(str(p),map_location='cpu',weights_only=False)
    expected=fingerprint(dict(source=src['step13_source_id'],
        method='P1_H1_correct_rank_protect',seed=seed,version=STEP13_VERSION,
        protect_lambda=PROTECT_LAMBDA))
    if not(st.get('fingerprint')==expected and st.get('source_id')==src['step13_source_id']
       and st.get('method')=='P1_H1_correct_rank_protect' and st.get('seed')==seed
       and st.get('step')==TRAIN_STEPS):
        raise RuntimeError(f'历史P1 seed{seed}权重不匹配')
    model=Both().to(device);model.load_state_dict(st['model'],strict=True)
    model.eval();return model


def load_history17(src,device):
    """读取49组历史，并逐项比对第十三步55组汇总与原P1权重的真正预测。"""
    old=load_historical_cases(src,device)
    assert len(old)==49
    p=src['step13_dir'];n=len(src['qp'])
    wanted=set(old)|{('P1_H1_correct_rank_protect',s,a) for s in SEEDS for a in ALPHAS}
    got_metrics={}
    with (p/'02_所有方案_Val指标.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            key=(row['method'],int(row['seed']),float(row['alpha']))
            if key in got_metrics or key not in wanted:
                raise RuntimeError(f'第十三步历史指标存在未知/重复键: {key}')
            got_metrics[key]=row
    if set(got_metrics)!=wanted or len(got_metrics)!=55:
        raise RuntimeError('第十三步历史指标必须恰为55组')
    p1={('P1_H1_correct_rank_protect',s,a):dict(
        rank=np.empty(n,dtype=np.int32),AP=np.empty(n,dtype=np.float64),
        INP=np.empty(n,dtype=np.float64)) for s in SEEDS for a in ALPHAS}
    counts={key:0 for key in wanted}
    with (p/'03_全部逐查询_Val.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            key=(row['method'],int(row['seed']),float(row['alpha']))
            if key not in wanted:raise RuntimeError(f'第十三步未知逐查询组 {key}')
            i=int(row['query_index'])
            if i!=counts[key] or i>=n or int(row['pid'])!=int(src['qp'][i]):
                raise RuntimeError('第十三步查询顺序/PID错误')
            rk=int(row['rank']);ap=float(row['AP']);inp=float(row['INP'])
            if int(row['correct'])!=int(rk==1):raise RuntimeError('历史正误字段不一致')
            d=old[key] if key in old else p1[key]
            if key in old:
                if rk!=int(d['rank'][i]) or abs(ap-d['AP'][i])>2e-7 or abs(inp-d['INP'][i])>2e-7:
                    raise RuntimeError('历史49组逐查询与第十二步不一致')
            else:
                d['rank'][i]=rk;d['AP'][i]=ap;d['INP'][i]=inp
            counts[key]+=1
    if any(v!=n for v in counts.values()):raise RuntimeError('历史逐查询不完整')
    for key in wanted:
        d=old[key] if key in old else p1[key]
        validate_detail(d,n,len(src['vp']))
        if max(abs(get_metrics(d)[m]-float(got_metrics[key][m])) for m in METRICS)>1e-5:
            raise RuntimeError('历史指标与逐查询复算不一致 '+str(key))
    for seed in SEEDS:
        model=load_p1_17(src,seed,device)
        try:
            with torch.inference_mode():
                for a in ALPHAS:
                    check=evaluate(apply_adapter(src['vt'],model.text,a,device),
                        apply_adapter(src['vi'],model.image,a,device),src['qp'],src['vp'],device)
                    assert_equal_detail(check,p1[('P1_H1_correct_rank_protect',seed,a)],
                                        f'第十三步P1重跑 seed{seed} alpha{a}')
        finally:del model
    old.update(p1)
    say('[历史全部复核]',len(old),'组；P1三种子真实权重复算通过')
    return old


def make_pid_partner_lookup(ip):
    """为每个真实负图找同PID不同图的等概率候选，索引只来自Train。"""
    ip=np.asarray(ip,dtype=np.int64)
    order=np.argsort(ip,kind='stable')
    sorted_pids=ip[order]
    starts=np.flatnonzero(np.r_[True,sorted_pids[1:]!=sorted_pids[:-1]])
    ends=np.r_[starts[1:],len(ip)]
    location=np.empty(len(ip),dtype=np.int64)
    group_start=np.empty(len(ip),dtype=np.int64)
    group_count=np.empty(len(ip),dtype=np.int64)
    for lo,hi in zip(starts,ends):
        indexes=order[lo:hi]
        location[indexes]=np.arange(hi-lo,dtype=np.int64)
        group_start[indexes]=lo
        group_count[indexes]=hi-lo
    return dict(order=order,location=location,group_start=group_start,
                group_count=group_count)


def replace_h1_with_same_pid_mixes(text_raw,images,queries,neg_ids,src,partner,seed,step):
    """在H1原负样本中尝试替换前4个；混合负图必须与锚负图PID相同且不同于查询PID。"""
    if neg_ids.shape!=(len(queries),NUM_NEG):raise RuntimeError('H1负样本维度不合法')
    if np.any(src['ip'][neg_ids]==src['tp'][queries,None]):
        raise RuntimeError('合成前H1负样本出现正身份')
    device=images.device
    raw=images[torch.as_tensor(neg_ids,device=device,dtype=torch.long)].clone()
    b=len(queries)
    # 从H1本身的前4槽进行混合，其他4槽始终保持真实图像特征。
    qrow=np.repeat(np.arange(b),MIX_SLOTS)
    slot=np.tile(np.arange(MIX_SLOTS),b)
    anchors=neg_ids[qrow,slot].astype(np.int64)
    qty=partner['group_count'][anchors]
    can=qty>=2
    stats=dict(attempted=int(len(anchors)),eligible=int(can.sum()),
        replaced=0,skipped_single_image=int((~can).sum()),
        skipped_no_gain=0,sim_gain_sum=0.0)
    if not np.any(can):return raw,stats
    rng=np.random.default_rng(int(seed)*1000003+int(step)*97+17017)
    original_rows=qrow[can];original_slots=slot[can]
    anchor=anchors[can];counts=qty[can]
    offsets=rng.integers(0,counts-1)
    otherpos=(partner['location'][anchor]+1+offsets)%counts
    other=partner['order'][partner['group_start'][anchor]+otherpos]
    if np.any(other==anchor) or np.any(src['ip'][other]!=src['ip'][anchor]) or np.any(
            src['ip'][other]==src['tp'][queries[original_rows]]):
        raise RuntimeError('S1合成负样本违反同负身份或异查询身份条件')
    ta=text_raw[torch.as_tensor(original_rows,device=device)]
    va=images[torch.as_tensor(anchor,device=device)]
    vb=images[torch.as_tensor(other,device=device)]
    coeff=torch.as_tensor(MIX_COEFFS,device=device,dtype=torch.float32)[None,:,None]
    mixture=F.normalize(coeff*va[:,None,:]+(1-coeff)*vb[:,None,:],dim=-1)
    teacher_scores=(mixture*ta[:,None,:]).sum(-1)
    top_score,top_choice=teacher_scores.max(1)
    score_real=(va*ta).sum(-1)
    use=(top_score>score_real+MIX_MIN_GAIN)
    stats['replaced']=int(use.sum().item())
    stats['skipped_no_gain']=int((~use).sum().item())
    stats['sim_gain_sum']=float((top_score-score_real)[use].sum().item())
    if bool(use.any()):
        chosen=mixture[torch.arange(len(top_choice),device=device),top_choice]
        rq=torch.as_tensor(original_rows,device=device)[use]
        rs=torch.as_tensor(original_slots,device=device)[use]
        raw[rq,rs]=chosen[use]
        # 最佳同PID教师得分可在外部统计；此处只用Train身份与文本余弦选择混合候选。
    if not torch.isfinite(raw).all():raise RuntimeError('合成负样本含NaN/Inf')
    if (raw.norm(dim=-1)-1).abs().max()>0.002:raise RuntimeError('合成负样本未归一化')
    return raw,stats


def train_s1(src,out,wrong,seed,device,stop_at=None):
    """严格继承P1的主损失+正确排序保护；仅替换部分H1真实负图。"""
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    ad=Both().to(device)
    optimizer=torch.optim.AdamW(ad.parameters(),lr=LR,weight_decay=1e-4)
    runid=fingerprint(dict(source=src['step17_source_id'],version=STEP17_VERSION,
        seed=seed,method=STEP17_METHOD,mix_slots=MIX_SLOTS,coeffs=MIX_COEFFS,
        mix_min_gain=MIX_MIN_GAIN,training=TRAIN_STEPS))
    cp=out/f'{STEP17_METHOD}_seed{seed}_progress.pt'
    begin=0;history=[];stats=dict(attempted=0,eligible=0,replaced=0,
        skipped_single_image=0,skipped_no_gain=0,sim_gain_sum=0.,
        protected_train_exposures=0,protect_violated_exposures=0,
        neg_from_wrong=0,wrong_quota_fallback=0,queries_with_wrong=0)
    if cp.exists():
        record=torch.load(str(cp),map_location='cpu',weights_only=False)
        if not(record.get('fingerprint')==runid and record.get('source_id')==src['step17_source_id']
          and record.get('method')==STEP17_METHOD and record.get('seed')==seed and
          type(record.get('step')) is int and 0<=record['step']<=TRAIN_STEPS):
            raise RuntimeError('S1已有权重/训练断点指纹不匹配，拒绝覆盖')
        ad.load_state_dict(record['model'],strict=True)
        optimizer.load_state_dict(record['optimizer'])
        for st in optimizer.state.values():
            for key,val in st.items():
                if torch.is_tensor(val):st[key]=val.to(device)
        begin=record['step'];history=record['history'];stats=record['stats']
        say('[S1断点复用]',seed,begin,'/',TRAIN_STEPS)
    partner=make_pid_partner_lookup(src['ip'])
    best_idx=best_same_pid_teacher_indices(src,wrong)
    texts=torch.as_tensor(np.ascontiguousarray(src['trt']),device=device,dtype=torch.float32)
    images=torch.as_tensor(np.ascontiguousarray(src['tri']),device=device,dtype=torch.float32)
    end=TRAIN_STEPS if stop_at is None else min(int(stop_at),TRAIN_STEPS)
    for step in range(begin,end):
        idx=np.random.default_rng(seed*1000003+step*97+3).integers(
            len(src['tp']),size=EMBED_TRAIN_BATCH,dtype=np.int64)
        negatives,original_stats=choose_old_negatives_with_stats(
            idx,src['tp'],src['ip'],src['half'],wrong,seed,step,NUM_NEG)
        for k,v in original_stats.items():stats[k]+=v
        ids=torch.as_tensor(idx,device=device,dtype=torch.long)
        pos=torch.as_tensor(src['positive'][idx],device=device,dtype=torch.long)
        x=texts[ids];p=images[pos]
        n,mix_stats=replace_h1_with_same_pid_mixes(x,images,idx,negatives,
                                                     src,partner,seed,step)
        for k,v in mix_stats.items():
            if k in stats:stats[k]+=v
        tx=ad.text(x);px=ad.image(p);nx=ad.image(n)
        positive_score=(tx*px).sum(-1,keepdim=True)
        negative_scores=torch.einsum('bd,bkd->bk',tx,nx)
        logits=torch.cat([positive_score,negative_scores],dim=1)/TEMPERATURE
        ce=F.cross_entropy(logits,torch.zeros(len(idx),device=device,dtype=torch.long))
        preservation=((tx-x).square().sum(-1).mean()+(px-p).square().sum(-1).mean()+
                      (nx-n).square().sum(-1).mean())/3
        protect,detail=correct_rank_protection(ad,x,images,idx,wrong,best_idx,device)
        stats['protected_train_exposures']+=detail['protected']
        stats['protect_violated_exposures']+=detail['violated']
        loss=ce+PRESERVE*preservation+PROTECT_LAMBDA*protect
        if not torch.isfinite(loss):raise RuntimeError(f'S1 seed{seed} step{step} loss无效')
        optimizer.zero_grad(set_to_none=True)
        loss.backward();torch.nn.utils.clip_grad_norm_(ad.parameters(),2.0)
        optimizer.step()
        if (step+1)%20==0 or step==0:
            history.append(dict(step=step+1,CE=float(ce.detach().item()),
                preservation=float(preservation.detach().item()),
                protection=float(protect.detach().item()),objective=float(loss.detach().item()),
                batch_mix_replaced=mix_stats['replaced'],
                cumulative_mix_replaced=stats['replaced']))
        if (step+1)%SAVE_EVERY==0 or step+1==end:
            atomic_torch(cp,dict(fingerprint=runid,source_id=src['step17_source_id'],
                version=STEP17_VERSION,method=STEP17_METHOD,seed=seed,step=step+1,
                model=ad.state_dict(),optimizer=optimizer.state_dict(),
                stats=stats,history=history))
            say('[S1训练]',seed,step+1,'/',TRAIN_STEPS,'CE',round(float(ce.detach().item()),4),
                '保护',round(float(protect.detach().item()),4),
                '累计合成',stats['replaced'],'/',stats['attempted'])
    return ad,history,stats,end


def load_s1(src,out,seed,device):
    cp=out/f'{STEP17_METHOD}_seed{seed}_progress.pt'
    st=torch.load(str(cp),map_location='cpu',weights_only=False)
    runid=fingerprint(dict(source=src['step17_source_id'],version=STEP17_VERSION,
        seed=seed,method=STEP17_METHOD,mix_slots=MIX_SLOTS,coeffs=MIX_COEFFS,
        mix_min_gain=MIX_MIN_GAIN,training=TRAIN_STEPS))
    if not(st.get('fingerprint')==runid and st.get('source_id')==src['step17_source_id']
       and st.get('method')==STEP17_METHOD and st.get('step')==TRAIN_STEPS and st.get('seed')==seed):
        raise RuntimeError('S1最终权重缺失/指纹不匹配')
    ad=Both().to(device);ad.load_state_dict(st['model'],strict=True);ad.eval()
    for par in ad.parameters():par.requires_grad_(False)
    return ad


def prepare_output17(src):
    root=(ROOT/'硬负样本分析').resolve();root.mkdir(parents=True,exist_ok=True)
    if RESUME_STEP17_DIR:
        p=Path(RESUME_STEP17_DIR).resolve()
        if not(p.is_dir() and root in p.parents and p.name.startswith(STEP17_PREFIX+'_')):
            raise RuntimeError('第十七步续跑目录非法')
        mf=p/'00_方案和来源.json'
        if not mf.is_file() or read_json(mf).get('source_id')!=src['step17_source_id']:
            raise RuntimeError('第十七步来源指纹变化，拒绝续跑')
        return p
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    p=root/(STEP17_PREFIX+'_'+stamp)
    i=2
    while p.exists():p=root/(STEP17_PREFIX+f'_{stamp}_{i}');i+=1
    p.mkdir();return p


def eval_cache17(src,out,seed,alpha,device):
    label=f'{STEP17_METHOD}_seed{seed}_alpha{alpha:.2f}'
    cp=out/f'{STEP17_METHOD}_seed{seed}_progress.pt'
    key=fingerprint(dict(source=src['step17_source_id'],label=label,
        version=STEP17_VERSION,weight_sha256=sha256(cp)))
    cached=out/'internal_cache'/f'eval_{label}.npz'
    cached.parent.mkdir(parents=True,exist_ok=True)
    if cached.exists():
        with np.load(cached,allow_pickle=False) as z:
            if set(z.files)!={'fingerprint','rank','AP','INP'} or str(z['fingerprint'].item())!=key:
                raise RuntimeError('S1权重改变但评估缓存未更新，拒绝复用')
            detail={k:np.array(z[k],copy=True) for k in ('rank','AP','INP')}
        say('[复用S1评估]',label)
    else:
        model=load_s1(src,out,seed,device)
        try:
            detail=evaluate(apply_adapter(src['vt'],model.text,alpha,device),
                            apply_adapter(src['vi'],model.image,alpha,device),
                            src['qp'],src['vp'],device)
        finally:del model
        def save(temp):
            with temp.open('wb') as f:
                np.savez_compressed(f,fingerprint=np.array(key),**detail)
                f.flush();os.fsync(f.fileno())
        atomic_write(cached,save)
    validate_detail(detail,len(src['qp']),len(src['vp']))
    return detail


def comparisons17(src,out,cases):
    metrics=[];swaps=[]
    b0=cases[('B0_original',0,0.0)]
    for alpha in ALPHAS:
        for seed in SEEDS:
            now=cases[(STEP17_METHOD,seed,alpha)]
            for refer in ('B0_original','P1_H1_correct_rank_protect','H1_wrong_ahead'):
                before=b0 if refer=='B0_original' else cases[(refer,seed,alpha)]
                m=get_metrics(now);refm=get_metrics(before)
                metrics.append(dict(seed=seed,alpha=alpha,reference=refer,
                    **{'delta_'+k:m[k]-refm[k] for k in METRICS}))
                a=before['rank']==1;z=now['rank']==1
                bb=b0['rank']==1
                swaps.append(dict(seed=seed,alpha=alpha,reference=refer,
                    reference_wrong_S1_right=int((~a&z).sum()),
                    reference_right_S1_wrong=int((a&~z).sum()),
                    net_correct=int(z.sum()-a.sum()),
                    B0_right_reference_wrong_S1_right=int((bb&~a&z).sum()),
                    B0_wrong_reference_right_S1_wrong=int((~bb&a&~z).sum())))
    atomic_csv(out/'08_S1与B0_P1_H1_同种子指标差.csv',list(metrics[0]),metrics)
    atomic_csv(out/'09_S1与B0_P1_H1_查询错对交换.csv',list(swaps[0]),swaps)


def package17(out):
    dest=out/STEP17_ZIP;tmp=out/('.'+STEP17_ZIP+'.partial')
    require={'00_方案和来源.json','02_所有方案_Val指标.csv',
        '03_全部逐查询_Val.csv','04_三种子汇总及PID_Bootstrap.json',
        '07_S1三个种子合成统计.json','08_S1与B0_P1_H1_同种子指标差.csv',
        '09_S1与B0_P1_H1_查询错对交换.csv'}
    try:
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(out.iterdir()):
                if not p.is_file() or p in (dest,tmp) or p.suffix.lower() in ('.pt','.npy','.npz'):
                    continue
                z.write(p,p.name)
            z.write(Path(__file__).resolve(),'脚本/'+Path(__file__).name)
        with zipfile.ZipFile(tmp) as z:
            if z.testzip() is not None or not require.issubset(set(z.namelist())):
                raise RuntimeError('S1报告ZIP不完整')
        os.replace(tmp,dest)
    finally:
        if tmp.exists():tmp.unlink()
    say('[第十七步报告ZIP]',dest)


def run17(src,out,wrong,device):
    meta=out/'00_方案和来源.json'
    if meta.is_file():
        m=read_json(meta)
        if m.get('source_id')!=src['step17_source_id'] or m.get('version')!=STEP17_VERSION:
            raise RuntimeError('第十七步实验目录指纹不匹配')
    else:
        atomic_json(meta,dict(version=STEP17_VERSION,source_id=src['step17_source_id'],
            epoch58_sha=EXPECTED_CKPT_SHA,step6=str(src['base']),
            step13=str(src['step13_dir']),history_shas=src['step17_history_hashes'],
            objective='same negative PID feature-space hard mixing against P1',
            primary_reference='B0_original',historical_reference='P1_H1_correct_rank_protect',
            unchanged_H1_pool=True,unchanged_P1_protection=True,train_only=True,
            test_used=False,train_only_mining=True,original_irra_frozen=True,
            negative_count=NUM_NEG,mix_slots=MIX_SLOTS,mix_coefficients=MIX_COEFFS,
            mix_min_gain=MIX_MIN_GAIN,steps=TRAIN_STEPS,batch=EMBED_TRAIN_BATCH,
            seeds=list(SEEDS),primary_alpha=1.,historical_secondary_alpha=0.5,
            completed=False))
    cases=load_history17(src,device)
    per_seed={}
    for seed in SEEDS:
        ad,hist,stats,last=train_s1(src,out,wrong,seed,device)
        if last!=TRAIN_STEPS:raise RuntimeError('S1训练步数不完整')
        del ad
        if device.type=='cuda':torch.cuda.empty_cache()
        per_seed[str(seed)]=dict(stats=stats,history=hist,steps=last,
            eligible_rate=stats['eligible']/max(1,stats['attempted']),
            replacement_rate=stats['replaced']/max(1,stats['attempted']),
            mean_similarity_gain=stats['sim_gain_sum']/max(1,stats['replaced']))
        for a in ALPHAS:
            cases[(STEP17_METHOD,seed,a)]=eval_cache17(src,out,seed,a,device)
    atomic_json(out/'07_S1三个种子合成统计.json',dict(method=STEP17_METHOD,
        note='仅Train异身份负图的同PID特征混合；无跨PID、无额外GPU编码器',
        per_seed=per_seed))
    if len(cases)!=61:raise RuntimeError(f'历史55组+S1六组应61组，实际{len(cases)}')
    old_methods=METHODS_NEW
    try:
        globals()['METHODS_NEW']=('P1_H1_correct_rank_protect',STEP17_METHOD)
        rows,summary=write_reports(src,out,cases,wrong)
    finally:globals()['METHODS_NEW']=old_methods
    if len(rows)!=61:raise RuntimeError('第十七步报告61组数量错误')
    report=read_json(out/'04_三种子汇总及PID_Bootstrap.json')
    report['note']='第十七步Train合成硬负样本。Val探索性，不含多轮调参校正；alpha=1主对照，0.5历史辅助。'
    report['current_step']=17
    atomic_json(out/'04_三种子汇总及PID_Bootstrap.json',report)
    comparisons17(src,out,cases)
    package17(out)
    say('[主结果: alpha=1；历史辅助 alpha=0.5]')
    origin=get_metrics(cases[('B0_original',0,0.0)])
    say('B0_original',origin)
    for name in ('H1_wrong_ahead','P1_H1_correct_rank_protect',STEP17_METHOD):
        for a in ALPHAS:
            values={k:summary[name][str(a)][k]['mean'] for k in METRICS}
            say(name,'alpha',a,values)
    say('[完成报告]',out/STEP17_ZIP)
    return rows


def self_test17():
    """完整合成第6~13步并钩入第17步；测试独立训练、中断与报告。"""
    original=globals()['run']
    injected={'done':False}
    def intercept(src,out,device,wrong=None):
        result=original(src,out,device,wrong)
        if not injected['done']:
            injected['done']=True
            old13=globals()['SOURCE_STEP13_DIR']
            try:
                globals()['SOURCE_STEP13_DIR']=str(out)
                source_from_step13_stage17(src)
            finally:globals()['SOURCE_STEP13_DIR']=old13
            root=ROOT/'硬负样本分析'
            fresh=root/(STEP17_PREFIX+'_模拟运行')
            fresh.mkdir(exist_ok=True)
            # 先检查同PID混合与H1负样本不跨身份。
            ids=np.random.default_rng(42*1000003+3).integers(
                len(src['tp']),size=EMBED_TRAIN_BATCH,dtype=np.int64)
            n,_=choose_old_negatives_with_stats(ids,src['tp'],src['ip'],src['half'],wrong,42,0)
            t=torch.as_tensor(src['trt'][ids]);v=torch.as_tensor(src['tri'])
            neg,diag=replace_h1_with_same_pid_mixes(t,v,ids,n,src,
                make_pid_partner_lookup(src['ip']),42,0)
            assert neg.shape==(len(ids),NUM_NEG,DIM) and torch.isfinite(neg).all()
            assert diag['attempted']==len(ids)*MIX_SLOTS
            # 更强的公平性测试：完全关闭合成时，S1训练须逐参数等于历史P1。
            prior_slots=globals()['MIX_SLOTS']
            try:
                globals()['MIX_SLOTS']=0
                nop=root/'S1_no_mixing';p1copy=root/'P1_comparison'
                nop.mkdir(exist_ok=True);p1copy.mkdir(exist_ok=True)
                nomix,_,nomix_stat,_=train_s1(src,nop,wrong,42,device)
                oldp1,_,_,_=train_adapter(src,p1copy,'P1_H1_correct_rank_protect',42,wrong,device)
                assert nomix_stat['replaced']==0
                for key,weight in nomix.state_dict().items():
                    assert torch.equal(weight,oldp1.state_dict()[key]),'零混合与P1权重不相同 '+key
                del nomix,oldp1
            finally:
                globals()['MIX_SLOTS']=prior_slots
            # 测试中断/恢复与完整训练权重逐元素一致。
            partial=root/'S1_partial';single=root/'S1_full'
            partial.mkdir(exist_ok=True);single.mkdir(exist_ok=True)
            _,_,_,s=train_s1(src,partial,wrong,42,device,stop_at=2)
            assert s==2
            resumed,_,_,s=train_s1(src,partial,wrong,42,device)
            uninterrupted,_,_,s=train_s1(src,single,wrong,42,device)
            assert s==TRAIN_STEPS
            for key,item in resumed.state_dict().items():
                assert torch.equal(item,uninterrupted.state_dict()[key]),'S1恢复参数错误 '+key
            del resumed,uninterrupted
            rows=run17(src,fresh,wrong,device)
            assert len(rows)==61
            zipfile_path=fresh/STEP17_ZIP
            with zipfile.ZipFile(zipfile_path) as z:
                assert z.testzip() is None
                assert '09_S1与B0_P1_H1_查询错对交换.csv' in z.namelist()
            oldhash=sha256(fresh/'03_全部逐查询_Val.csv')
            assert len(run17(src,fresh,wrong,device))==61
            assert sha256(fresh/'03_全部逐查询_Val.csv')==oldhash
            cp=fresh/f'{STEP17_METHOD}_seed42_progress.pt'
            info=torch.load(str(cp),map_location='cpu',weights_only=False)
            info['model']['text.l1.weight']=info['model']['text.l1.weight']+0.002
            atomic_torch(cp,info)
            try:eval_cache17(src,fresh,42,1.0,device)
            except RuntimeError:pass
            else:raise AssertionError('S1修改权重后旧评估缓存没有被拒绝')
            say('[第十七步全链路合成自检通过] P1/B0、混合、断点、61组、CSV/ZIP和缓存拒绝')
        return result
    globals()['run']=intercept
    try:
        _self_test_full()
        assert injected['done']
    finally:
        globals()['run']=original
    say('[SELF_TEST PASS] 第十七步完整CPU合成链路通过')


def main17():
    parser=argparse.ArgumentParser(description='IRRA第十七步：轻量同负PID特征空间困难负样本增强；B0与P1对照')
    parser.add_argument('--self-test',action='store_true',help='完整CPU合成测试，不依赖真实数据')
    parser.add_argument('--device',choices=['cuda','cpu'],default=DEVICE)
    args=parser.parse_args()
    if args.self_test:self_test17();return
    if not (1<=MIX_SLOTS<=NUM_NEG):raise RuntimeError('正式训练混合槽位必须介于1~8')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA不可用，请改 --device cpu')
    say('第十七步S1：原Epoch58 IRRA始终为B0，P1为严格主对照；设备',device)
    src=verify_source()
    wrong=source_from_step9(src)
    source_from_step10(src);add_rank_profiles(src,wrong)
    source_from_step11(src);add_stratified_profiles(src,wrong)
    source_from_step12(src)
    source_from_step13_stage17(src)
    out=prepare_output17(src)
    for k in ('base','step9_dir','step10_dir','step11_dir','step12_dir','step13_dir'):
        say('[历史文件仅只读]',k,src[k])
    say('[本轮独立文件夹]',out)
    run17(src,out,wrong,device)


if __name__=='__main__':
    try:main17()
    except BaseException:
        traceback.print_exc();raise
