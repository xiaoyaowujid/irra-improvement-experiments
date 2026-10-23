# -*- coding: utf-8 -*-
"""
第十二步（只研究硬负样本）：H1错误领先 vs C3同查询难度层的条件排名匹配。
复用第六、九、十、十一步Epoch58/Train/Val/权重/逐查询结果，只新训练C3三个种子。
每个种子完全重放H1的800步文本batch及8负样本排名组合，按Train文本的wrong_count
(原IRRA异身份Top50中高于最佳同身份图像的数量)分层；在每一层内部跨PID重分配
完整8负样本排名组合，必要时确定性合并相邻层以满足可行性。
精确保留每个seed每个实际难度层的H1排名组合多重集，且禁止来源和目标为同PID。
C3抽样不使用目标查询的具体错排名单，只使用已冻结Train层标签。它并非完全
因果识别：层内margin、余弦、重复曝光仍可能不同，报告必须明确展示。
训练保持同一Adapter/损失/optimizer/800步/128文本batch/8负样本/seed42-44。
alpha=1主比较；历史固定alpha=0.5辅助对照，不在Val或Test搜索alpha。
严格只在Train挖掘，Val完整图库评估；不读取Test。旧目录全部只读。
先 --self-test 再正常运行；RESUME_OUTPUT_DIR填写第十二步目录可安全续跑。
上传ZIP不含.pt权重，运行需你本机第六/九/十/十一步完整目录。
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
RESUME_OUTPUT_DIR=''
OUTPUT_PREFIX='第十二步_条件错排难度匹配C3'
STEP9_PREFIX='第九步_错误排序驱动硬负样本'
STEP10_VERSION='irra_step10_top8_versus_wrong_v1_20261021'
STEP10_PREFIX='第十步_难度与错排机制对照'
STEP11_VERSION='irra_step11_C2_rankset_shuffle_v2_20261021'
STEP11_PREFIX='第十一步_H1排名分布匹配C2'
STEP12_VERSION='irra_step12_C3_conditional_wrongcount_strata_v1_20261021'
ZIP_NAME='上传给ChatGPT_第十二步_C3条件难度匹配对照.zip'
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
METHODS_NEW=('C3_wrongcount_stratified',)
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
    p=src['step11_dir']
    expected={('B0_original',0,0.0)}
    expected.update((m,s,a) for m in METHODS_OLD+METHODS_PRIOR_NEW+METHODS_STEP10+METHODS_STEP11 for s in SEEDS for a in ALPHAS)
    rows={}
    with (p/'02_所有方案_Val指标.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            key=(r['method'],int(r['seed']),float(r['alpha']))
            if key not in expected or key in rows:raise RuntimeError('第十一步汇总存在非法/重复组 '+str(key))
            rows[key]=r
    if set(rows)!=expected:raise RuntimeError('第十一步43组指标不完整')
    n=len(src['qp']);data={k:dict(rank=np.empty(n,np.int32),AP=np.empty(n),INP=np.empty(n)) for k in expected}
    next_idx={k:0 for k in expected}
    with (p/'03_全部逐查询_Val.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            key=(r['method'],int(r['seed']),float(r['alpha']))
            if key not in data:raise RuntimeError('第十一步逐查询结果包含未知组')
            i=int(r['query_index'])
            if i!=next_idx[key] or i>=n or int(r['pid'])!=int(src['qp'][i]):
                raise RuntimeError('第十一步逐查询顺序/身份/唯一性异常')
            d=data[key];d['rank'][i]=int(r['rank']);d['AP'][i]=float(r['AP']);d['INP'][i]=float(r['INP'])
            if int(r['correct'])!=int(d['rank'][i]==1):raise RuntimeError('第九步R1正误字段不一致')
            next_idx[key]+=1
    if any(v!=n for v in next_idx.values()):raise RuntimeError('第十一步逐查询记录缺失')
    for key,d in data.items():
        validate_detail(d,n,len(src['vp']))
        metrics=get_metrics(d)
        if max(abs(metrics[k]-float(rows[key][k])) for k in METRICS)>1e-5:
            raise RuntimeError('第十一步汇总与逐查询指标不一致：'+str(key))
    # 第十一步的31组旧结果必须逐条与原始第九步相同，不能从后续汇总无条件继承。
    nine=(src['step9_dir']/'03_全部逐查询_Val.csv')
    if not nine.is_file():raise FileNotFoundError('第九步原始逐查询数据缺失')
    source31={k for k in expected if k[0] not in ('C1_top8_unconditional','C2_rank_marginal')}
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
    from10={k for k in expected if k[0]!='C2_rank_marginal'}
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
                raise RuntimeError('第十一步继承的37组与第十步记录不一致')
            count10[key]+=1
    if any(k!=n for k in count10.values()):raise RuntimeError('第十步历史记录不完整')
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
    say('[历史结果复用完成] 43组，',43*n,'条查询；B0/B2/H1/H2/C1/C2由真实权重重算')
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
            raise RuntimeError('续跑路径必须是独立的第十二步实验目录')
        manifest=out/'00_方案和来源.json'
        if not manifest.is_file() or read_json(manifest).get('source_id')!=src['step12_source_id']:
            raise RuntimeError('第十二步源指纹不匹配，拒绝混用已有实验')
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
    """C3 frozen ranksets from different-PID Train queries within wrong_count strata."""
    if mode!='C3_wrongcount_stratified':raise ValueError('第十二步只新增C3')
    left=step*EMBED_TRAIN_BATCH;right=left+EMBED_TRAIN_BATCH
    if len(indices)!=EMBED_TRAIN_BATCH or right>len(wrong['c3_rank_sets'][seed]):
        raise RuntimeError('C3训练batch形状/step不符合预先约定')
    if not np.array_equal(indices,wrong['sampled_queries'][seed][left:right]):
        raise RuntimeError('C3训练query顺序不等于H1历史重放')
    ranks=wrong['c3_rank_sets'][seed][left:right].astype(np.int64)
    negative=np.take_along_axis(wrong['top_idx'][indices],ranks,axis=1).astype(np.int64)
    if negative.shape!=(len(indices),num_neg) or np.any(ip[negative]==tp[indices,None]):
        raise RuntimeError('C3负样本出现同身份或维度错误')
    if any(len(set(row.tolist()))!=num_neg for row in negative):
        raise RuntimeError('C3负样本内存在重复图像')
    count=wrong['wrong_count'][indices]
    return negative,dict(neg_from_wrong=int((ranks<count[:,None]).sum()),
        wrong_quota_fallback=0,queries_with_wrong=int(np.count_nonzero(count)))

def train_adapter(src,out,method,seed,wrong,device,stop_at=None):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    ad=Both().to(device)
    opt=torch.optim.AdamW(ad.parameters(),lr=LR,weight_decay=1e-4)
    checkpoint=out/f'{method}_seed{seed}_progress.pt'
    run_fp=fingerprint(dict(source=src['step12_source_id'],method=method,seed=seed,version=STEP12_VERSION))
    start=0;history=[];counter=dict(neg_from_wrong=0,wrong_quota_fallback=0,queries_with_wrong=0)
    if checkpoint.exists():
        st=torch.load(str(checkpoint),map_location='cpu',weights_only=False)
        if (st.get('fingerprint')!=run_fp or st.get('method')!=method or st.get('seed')!=seed or
           st.get('source_id')!=src['step12_source_id'] or not 0<=st.get('step',-1)<=TRAIN_STEPS):
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
            atomic_torch(checkpoint,dict(fingerprint=run_fp,source_id=src['step12_source_id'],method=method,
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
    fp=fingerprint(dict(source=src['step12_source_id'],key=key,version=STEP12_VERSION,adapter_weight_sha256=weight_sha))
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
    expected=fingerprint(dict(source=src['step12_source_id'],method=method,seed=seed,version=STEP12_VERSION))
    if (st.get('fingerprint')!=expected or st.get('source_id')!=src['step12_source_id'] or
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
        for method in METHODS_OLD+METHODS_PRIOR_NEW+METHODS_STEP10+METHODS_STEP11+METHODS_NEW:
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
        dict(note='第十二步为Val探索性对照。PID bootstrap仅量化身份采样，不含训练不确定性、多重选择或Test泛化。',
             primary_alpha=1.0,secondary_pre_registered_alpha=0.5,methods=stats))
    # H1/H2 vs B2: 决策以三个种子同种seed严格配对比较，不比较不同α造成的选择偏差
    pair=[]
    for alpha in ALPHAS:
        for method in METHODS_NEW:
            for ref_name in ('B2_hard','H1_wrong_ahead','H2_wrong_plus_semi','C1_top8_unconditional','C2_rank_marginal'):
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
            if not {'02_所有方案_Val指标.csv','03_全部逐查询_Val.csv','04_三种子汇总及PID_Bootstrap.json','05_新方案对旧方法_同种子配对差.csv','07_采样负样本难度_固定Step诊断.csv','08_H1排名分布与C3条件匹配程度.json'}.issubset(set(z.namelist())):
                raise RuntimeError('ZIP遗漏必要结果')
        os.replace(temp,dest)
    finally:
        if temp.exists():temp.unlink()
    say('报告ZIP：',dest)

def probe_difficulty(src,wrong,out):
    """固定20步训练曝光对照，记录rank/余弦/间隔；禁止据此宣称完全匹配。"""
    records=[];summ={};steps=tuple(x for x in PROBE_STEPS if x<TRAIN_STEPS)
    for mode in METHODS_PRIOR_NEW+METHODS_STEP10+METHODS_STEP11+METHODS_NEW:
        summ[mode]={}
        for seed in SEEDS:
            bins=np.zeros(RANK_END,np.int64)
            total=0;score_sum=0.;gap_sum=0.;wrong_exp=0;top8_exp=0
            per_query=Counter()
            for step in steps:
                rg=np.random.default_rng(seed*1000003+step*97+3)
                idx=rg.integers(len(src['tp']),size=EMBED_TRAIN_BATCH,dtype=np.int64)
                if mode in METHODS_PRIOR_NEW:
                    neg=choose_old_negatives(idx,src['tp'],src['ip'],src['half'],wrong,mode,seed,step)
                elif mode in METHODS_STEP10:
                    neg=wrong['top_idx'][idx,:NUM_NEG]
                elif mode in METHODS_STEP11:
                    ranks=wrong['rank_sets'][seed][step*EMBED_TRAIN_BATCH:(step+1)*EMBED_TRAIN_BATCH].astype(np.int64)
                    if not np.array_equal(idx,wrong['sampled_queries'][seed][step*EMBED_TRAIN_BATCH:(step+1)*EMBED_TRAIN_BATCH]):
                        raise RuntimeError('历史C2探针query顺序不正确')
                    neg=np.take_along_axis(wrong['top_idx'][idx],ranks,axis=1)
                else:
                    neg=choose_negatives(idx,src['tp'],src['ip'],src['half'],wrong,mode,seed,step)[0]
                isrank=(neg[:,:,None]==wrong['top_idx'][idx,None,:])
                if not np.all(isrank.sum(2)==1):raise RuntimeError('探针负样本不属于Top50')
                ranks=np.argmax(isrank,2)
                bins+=np.bincount(ranks.ravel(),minlength=RANK_END)
                cosine=np.einsum('bd,bkd->bk',src['trt'][idx],src['tri'][neg],optimize=True)
                gaps=cosine-wrong['best_positive'][idx,None]
                wrong_exp+=int((gaps>1e-7).sum())
                top8_exp+=int((ranks<8).sum())
                total+=int(cosine.size)
                score_sum+=float(cosine.sum(dtype=np.float64))
                gap_sum+=float(gaps.sum(dtype=np.float64))
                for j,q in enumerate(idx):
                    for n in neg[j]:per_query[(int(q),int(n))]+=1
            # 近似重复曝光比例：同一个query和负图在固定探针内首次以后再次出现的频次。
            dup_count=sum(max(0,x-1) for x in per_query.values())
            stat=dict(method=mode,seed=seed,probed_steps=len(steps),negative_selections=total,
                mean_cosine=score_sum/total,mean_minus_best_positive=gap_sum/total,
                share_wrong_ahead=wrong_exp/total,share_top8=top8_exp/total,
                same_query_negative_repeat_exposure=dup_count/total)
            records.append(stat)
            summ[mode][str(seed)]={'histogram':bins.tolist(),**stat}
    atomic_csv(out/'07_采样负样本难度_固定Step诊断.csv',
        ['method','seed','probed_steps','negative_selections','mean_cosine',
         'mean_minus_best_positive','share_wrong_ahead','share_top8',
         'same_query_negative_repeat_exposure'],records)
    conditional={}
    for seed in SEEDS:
        h=summ['H1_wrong_ahead'][str(seed)]
        c=summ['C3_wrongcount_stratified'][str(seed)]
        hist_h=np.asarray(h['histogram'],np.float64)
        hist_c=np.asarray(c['histogram'],np.float64)
        hist_h/=hist_h.sum();hist_c/=hist_c.sum()
        conditional[str(seed)]=dict(probe_rank_TV=float(np.sum(abs(hist_h-hist_c))/2),
            probe_cosine_gap_C3_minus_H1=float(c['mean_cosine']-h['mean_cosine']),
            probe_wrong_exposure_gap_C3_minus_H1=float(c['share_wrong_ahead']-h['share_wrong_ahead']),
            complete_stratification=wrong['c3_strata_report'][str(seed)])
    atomic_json(out/'08_H1排名分布与C3条件匹配程度.json',dict(
        note='Train wrong_count条件分层、同层不同PID全训练步排列；H1和C3每层八排名组合多重集及全局曝光严格相等。余弦、margin、重复曝光并非严格匹配。',
        per_seed=conditional,probe_steps=list(steps),all_method_probe=summ))
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
    if wrong is None:raise RuntimeError('须先验证至第十一步的完整历史权重，并生成C3条件分层排名组合')
    manifest=out/'00_方案和来源.json'
    if manifest.is_file():
        if read_json(manifest).get('source_id')!=src['step12_source_id']:
            raise RuntimeError('第十二步目录包含不兼容实验，拒绝复用')
    else:
        atomic_json(manifest,dict(version=STEP12_VERSION,source_id=src['step12_source_id'],
            epoch58_source_id=src['id'],step6_folder=str(src['base']),
            step9_folder=str(src['step9_dir']),step10_folder=str(src['step10_dir']),
            step11_folder=str(src['step11_dir']),
            previous_step9_sha=src['step9_hashes'],previous_step10_sha=src['step10_hashes'],
            previous_step11_sha=src['step11_hashes'],
            h1_hist_sha=src['hist_hash'],train_only_mining=True,test_used=False,
            c3_stratification=wrong['c3_strata_report'],
            purpose='同Train查询wrong_count难度层匹配H1负样本排名组合，检验精细错排选择价值',
            caveat='在每个合并后wrong_count层内完整800步rankset多重集完全匹配，严格跨PID；余弦、逐query margin、重复曝光不能保证匹配',
            only_new_variable='C3 conditional cross-PID rankset permutation within Train wrong_count strata',
            fixed_alpha_primary=1.0,fixed_alpha_secondary=0.5,train_steps=TRAIN_STEPS,
            seeds=list(SEEDS),completed=False))
    cases=load_historical_cases(src,device)
    probe=probe_difficulty(src,wrong,out)
    say('[已复用历史] 43组；仅新增三个C3模型')
    for seed in SEEDS:
        ad,history,counter,last=train_adapter(src,out,'C3_wrongcount_stratified',seed,wrong,device)
        if last!=TRAIN_STEPS:raise RuntimeError('C3未训练完成')
        del ad
        if device.type=='cuda':torch.cuda.empty_cache()
        atomic_json(out/f'06_C3_wrongcount_stratified_seed{seed}_训练统计.json',
            dict(method='C3_wrongcount_stratified',seed=seed,steps=last,
                 counter=counter,history=history))
        for alpha in ALPHAS:
            key=f'C3_wrongcount_stratified_seed{seed}_alpha{alpha:.2f}'
            detail=cache_eval(src,out,key,
               lambda ss=seed,a=alpha:new_eval(src,out,'C3_wrongcount_stratified',ss,a,device))
            cases[('C3_wrongcount_stratified',seed,alpha)]=detail
    rows,stats=write_reports(src,out,cases,wrong)
    if len(rows)!=49:raise RuntimeError('期望43组历史+6组C3=49组')
    package(out)
    say('[alpha=1 主比较；alpha=0.5 已固定仅辅助]')
    for method in ('B2_hard','H1_wrong_ahead','H2_wrong_plus_semi','C1_top8_unconditional','C2_rank_marginal','C3_wrongcount_stratified'):
        for a in ALPHAS:
            mm=stats[method][str(a)]
            say(method,'alpha',a,'R1',round(mm['R1']['mean'],4),
                'R5',round(mm['R5']['mean'],4),'mAP',round(mm['mAP']['mean'],4),
                'mINP',round(mm['mINP']['mean'],4))
    say('[难度曝光]',[(v['method'],v['seed'],round(v['share_wrong_ahead'],3)) for v in probe])
    return rows


def self_test():
    """CPU合成端到端：H1/C3层内分布、历史43组复验、断点续跑、49组报告、ZIP与防污染。"""
    patch_names=('ROOT','SOURCE_STEP9_DIR','SOURCE_STEP10_DIR','SOURCE_STEP11_DIR','RESUME_OUTPUT_DIR',
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
            globals()['SOURCE_STEP9_DIR']=str(old9)
            globals()['SOURCE_STEP10_DIR']=str(old10)
            globals()['SOURCE_STEP11_DIR']=str(old11)
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
            assert len(load_historical_cases(src,device))==43
            say('[自检1/5] 第六/九/十/十一步43组历史复用及权重复算通过')
            samples=w['sampled_queries'][42][:EMBED_TRAIN_BATCH]
            choice,stat=choose_negatives(samples,tp,ip,half,w,'C3_wrongcount_stratified',42,0)
            assert choice.shape==(len(samples),NUM_NEG) and not np.any(ip[choice]==tp[samples,None])
            changed=dict(w);changed['wrong_count']=np.zeros(len(tp),np.int16)
            same,_=choose_negatives(samples,tp,ip,half,changed,'C3_wrongcount_stratified',42,0)
            assert np.array_equal(choice,same),'C3运行时抽样意外依赖正负间隔阈值'
            assert all(x.sum()==TRAIN_STEPS*EMBED_TRAIN_BATCH*NUM_NEG for x in w['rank_histograms'].values())
            assert all(np.array_equal(np.bincount(w['c3_rank_sets'][seed].ravel().astype(np.int32),minlength=RANK_END),
                w['rank_histograms'][seed]) for seed in SEEDS)
            say('[自检2/5] 全步H1条件层内rankset精确匹配、跨PID重排通过')
            split=root/'split';fresh=root/'fresh';split.mkdir();fresh.mkdir()
            _,_,_,step=train_adapter(src,split,'C3_wrongcount_stratified',42,w,device,stop_at=2)
            assert step==2
            a,_,_,step=train_adapter(src,split,'C3_wrongcount_stratified',42,w,device)
            b,_,_,step=train_adapter(src,fresh,'C3_wrongcount_stratified',42,w,device)
            assert step==TRAIN_STEPS
            for key,x in a.state_dict().items():assert torch.equal(x,b.state_dict()[key]),key
            del a,b
            say('[自检3/5] 中断续跑权重与完整训练严格一致')
            out=prepare_output(src)
            rows=run(src,out,device,w)
            assert len(rows)==49 and (out/'08_H1排名分布与C3条件匹配程度.json').exists()
            with zipfile.ZipFile(out/ZIP_NAME) as zz:
                assert zz.testzip() is None and '03_全部逐查询_Val.csv' in zz.namelist()
            with (out/'03_全部逐查询_Val.csv').open(encoding='utf-8-sig',newline='') as f:
                assert sum(1 for _ in csv.DictReader(f))==49*len(tp)
            original=sha256(out/'03_全部逐查询_Val.csv')
            assert len(run(src,out,device,w))==49
            assert sha256(out/'03_全部逐查询_Val.csv')==original
            say('[自检4/5] 49组报告和ZIP、重复运行缓存复用通过')
            cp=out/'C3_wrongcount_stratified_seed42_progress.pt'
            state=torch.load(str(cp),map_location='cpu',weights_only=False)
            state['model']['text.l1.weight']=state['model']['text.l1.weight']+0.02
            atomic_torch(cp,state)
            try:
                cache_eval(src,out,'C3_wrongcount_stratified_seed42_alpha1.00',lambda:b0)
            except RuntimeError as exc:assert '指纹' in str(exc)
            else:raise AssertionError('篡改权重后评估缓存仍被接受')
            history_file=old11/'02_所有方案_Val指标.csv'
            with history_file.open('a',encoding='utf-8') as f:f.write('\n')
            new_src=dict(src)
            source_from_step11(new_src);add_stratified_profiles(new_src,dict(w))
            assert src['step12_source_id']!=new_src['step12_source_id']
            globals()['RESUME_OUTPUT_DIR']=str(out)
            try:
                prepare_output(new_src)
            except RuntimeError as exc:assert '指纹' in str(exc)
            else:raise AssertionError('篡改第十一步报告后仍允许续跑')
            say('[自检5/5] 篡改模型及历史文件均触发缓存拒绝')
        say('[SELF_TEST PASS] CPU合成端到端通过；真实Windows/CUDA需本机验证')
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
    parser=argparse.ArgumentParser(description='IRRA第十二步：C3 Train查询错排难度分层匹配H1排名组合')
    parser.add_argument('--self-test',action='store_true',help='合成CPU端到端自检，不访问本机IRRA数据')
    args=parser.parse_args()
    if args.self_test:self_test();return
    say('第十二步：条件难度分层对照C3；设备',DEVICE)
    src=verify_source()
    wrong=source_from_step9(src)
    source_from_step10(src)
    add_rank_profiles(src,wrong)
    source_from_step11(src)
    add_stratified_profiles(src,wrong)
    out=prepare_output(src)
    say('第六步只读目录:',src['base'])
    say('第九步只读目录:',src['step9_dir'])
    say('第十步只读目录:',src['step10_dir'])
    say('第十一步只读目录:',src['step11_dir'])
    say('第十二步独立输出:',out)
    run(src,out,torch.device(DEVICE),wrong)


if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc()
        raise
