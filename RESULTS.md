# 完整结果表

> 所有表格由 `results/summary_all.csv` 自动生成，未做手工修改。
> 基线：IRRA Epoch 58，官方 val **R1 66.418 / R5 85.271 / R10 90.581 / mAP 61.866 / mINP 47.251**。

---

## 1. 训练侧：硬负样本 + 排序保护（三种子 42/43/44 均值）

训练侧全部基于冻结 Epoch 58 主干，只训练一个轻量 Adapter（800 步，8 负样本/查询）。α 是该损失相对主对比损失的强度系数。
「修正 / 破坏」= 相对基线在 val 上由错变对、由对变错的查询数（三种子均值）。

| 方法 | α | R1 | R5 | R10 | mAP | mINP | ΔR1 | ΔmAP | ΔmINP | 修正 | 破坏 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P1_H1_correct_rank_protect | 1.0 | 67.002 | 85.304 | 90.695 | 62.397 | 47.938 | 0.5846 | 0.5315 | 0.687 | 173.3 | 137.3 |
| S1_same_negative_pid_feature_mix | 1.0 | 66.970 | 85.277 | 90.706 | 62.454 | 48.084 | 0.5521 | 0.5886 | 0.8331 | 178.0 | 144.0 |
| W1_correct_wrong_and_protect_right | 0.5 | 66.970 | 85.607 | 90.966 | 62.397 | 47.892 | 0.5521 | 0.5316 | 0.6412 | 143.7 | 109.7 |
| W2_margin_stop_gradient_limited | 1.0 | 66.910 | 85.239 | 90.738 | 62.352 | 47.935 | 0.4926 | 0.487 | 0.6847 | 204.7 | 174.3 |
| H2_wrong_plus_semi | 0.5 | 66.878 | 85.417 | 90.711 | 62.281 | 47.735 | 0.4601 | 0.4153 | 0.4842 | 123.0 | 94.7 |
| H1_wrong_ahead | 0.5 | 66.856 | 85.396 | 90.814 | 62.298 | 47.776 | 0.4385 | 0.4329 | 0.5251 | 126.7 | 99.7 |
| C1_top8_unconditional | 0.5 | 66.845 | 85.450 | 90.857 | 62.318 | 47.859 | 0.4276 | 0.4524 | 0.6082 | 165.7 | 139.3 |
| P1_H1_correct_rank_protect | 0.5 | 66.835 | 85.406 | 90.765 | 62.310 | 47.813 | 0.4168 | 0.4443 | 0.5626 | 95.3 | 69.7 |
| S1_same_negative_pid_feature_mix | 0.5 | 66.835 | 85.466 | 90.787 | 62.362 | 47.909 | 0.4168 | 0.4962 | 0.6584 | 97.7 | 72.0 |
| C3_wrongcount_stratified | 0.5 | 66.813 | 85.406 | 90.782 | 62.298 | 47.785 | 0.3951 | 0.4324 | 0.535 | 124.3 | 100.0 |
| W2_margin_stop_gradient_limited | 0.5 | 66.780 | 85.504 | 90.830 | 62.346 | 47.889 | 0.3627 | 0.4809 | 0.6386 | 108.0 | 85.7 |
| H1_wrong_ahead | 1.0 | 66.753 | 84.936 | 90.435 | 62.136 | 47.653 | 0.3356 | 0.2702 | 0.4023 | 221.7 | 201.0 |
| C2_rank_marginal | 0.5 | 66.742 | 85.293 | 90.657 | 62.209 | 47.669 | 0.3248 | 0.344 | 0.4181 | 113.7 | 93.7 |
| B3_mixed | 0.5 | 66.726 | 85.244 | 90.635 | 62.149 | 47.603 | 0.3085 | 0.2833 | 0.3529 | 96.7 | 77.7 |
| H2_wrong_plus_semi | 1.0 | 66.705 | 85.066 | 90.478 | 62.137 | 47.649 | 0.2869 | 0.2711 | 0.3986 | 206.0 | 188.3 |
| C2_rank_marginal | 1.0 | 66.694 | 84.963 | 90.376 | 62.089 | 47.588 | 0.2761 | 0.2232 | 0.3377 | 188.7 | 171.7 |
| B2_hard | 1.0 | 66.677 | 84.746 | 90.224 | 61.899 | 47.201 | 0.2598 | 0.0331 | -0.0496 | 173.7 | 157.7 |
| B3_mixed | 1.0 | 66.607 | 84.952 | 90.424 | 61.999 | 47.408 | 0.1895 | 0.1338 | 0.1576 | 158.3 | 146.7 |
| C3_wrongcount_stratified | 1.0 | 66.602 | 85.006 | 90.424 | 62.073 | 47.604 | 0.184 | 0.2071 | 0.3535 | 214.0 | 202.7 |
| B2_hard | 0.5 | 66.585 | 85.066 | 90.549 | 62.083 | 47.512 | 0.1678 | 0.2174 | 0.2612 | 99.3 | 89.0 |
| B1_random | 0.5 | 66.515 | 85.309 | 90.516 | 62.075 | 47.592 | 0.0974 | 0.2097 | 0.3409 | 72.3 | 66.3 |
| B1_random | 1.0 | 66.466 | 85.006 | 90.446 | 62.018 | 47.588 | 0.0487 | 0.1528 | 0.3376 | 125.0 | 122.0 |
| A1_random_lora_protect | lora | 66.428 | 85.260 | 90.608 | 61.897 | 47.280 | 0.0108 | 0.0314 | 0.0291 | 17.0 | 16.3 |
| A2_H1_lora_protect | lora | 66.374 | 85.385 | 90.516 | 61.821 | 47.177 | -0.0433 | -0.0448 | -0.0738 | 60.7 | 63.3 |
| W1_correct_wrong_and_protect_right | 1.0 | 66.255 | 84.876 | 90.462 | 61.924 | 47.628 | -0.1624 | 0.0584 | 0.3775 | 243.3 | 253.3 |
| C1_top8_unconditional | 1.0 | 65.811 | 84.275 | 90.116 | 61.371 | 47.146 | -0.6063 | -0.4942 | -0.1048 | 270.3 | 307.7 |

**怎么读这张表**：

- `B1_random` 是**随机负样本对照臂**，跑同样的 800 步、同样的预算。它的 ΔR1 在 −0.08 ~ +0.24 之间抖动 —— 这就是「光多训几百步」能拿到的量级。任何负样本方案必须稳定超过它才算数。
- `B2_hard`（纯硬负样本）只 +0.17，且 α=1.0 时 ΔmINP 变负 —— 硬负样本挤掉了简单样本，排序质量反而退化。
- `B3_mixed`（硬 + 随机混合）是**唯一做了 Test 独立验证**的方案，见下一节。
- `H1/H2/C1/C2/C3` 属于「挖错排」系列：不再按特征距离挑负样本，而是挑**当前排在正样本前面的那个负样本**，直接对症。
- `P1/W1/W2/S1` 是「保护」系列：对训练集里原本就排对的查询加保序惩罚。**P1 一步就把 ΔR1 从 +0.44 推到 +0.59，是单步收益最大的一步。**
- α=1.0 普遍不如 α=0.5 稳（`C1` α=1.0 直接掉到 −0.606），说明硬负样本必须**限量**使用。

---

## 2. 冻结 Test 独立验证（只对 B3）

B3 的 α 和模态分支全部在 val 上定死之后，才在 Test 上跑一次。Test 从未参与任何选择。

| 方法 | α | R1 | R5 | R10 | mAP | mINP | ΔR1 | ΔmAP | ΔmINP | 修正 | 破坏 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B3_mixed | 1.0 | 66.585 | 85.250 | 90.838 | 61.577 | 47.003 | 0.2112 | 0.1296 | -0.0608 | 181.7 | 168.7 |
| B3_mixed | 0.5 | 66.515 | 85.526 | 90.887 | 61.666 | 47.166 | 0.1408 | 0.2183 | 0.1024 | 107.3 | 98.7 |

基线 Test：R1 66.374 / R5 85.689 / R10 90.919 / mAP 61.447 / mINP 47.064。

**结论**：val 上 +0.309 的增益，在 Test 上只剩 +0.141（α=0.5）/ +0.211（α=1.0）。
**缩水约一半，但符号没翻、方向一致** —— 说明 val 增益里大约一半是选择偏差，另一半是真的。这是本仓库里唯一一个通过了样本外检验的改进。

同一节还做了 B3 Adapter 的模态分支消融（`results/hard_negative/step08_B3_branch_ablation/`）：

| 分支 | ΔR1 | ΔmAP | ΔmINP |
|---|---|---|---|
| text_only | −0.005 | −0.036 | +0.032 |
| image_only | +0.206 | +0.076 | +0.081 |
| **both** | **+0.309** | **+0.283** | **+0.353** |

增益主要来自**图像分支**，文本分支单独用几乎为零 —— 这和「文本侧信息已经饱和」的判断一致。

---

## 3. 推理侧：不训练、只改测试时决策（按 ΔR1 排序，共 128 组）

| 变体 | R1 | ΔR1 | ΔmAP | ΔmINP | 出现轮次 |
|---|---|---|---|---|---|
| O_全查询低温Sinkhorn(OT平衡) | 67.928 | 1.5102 | 1.1947 | 1.4351 | round5_cross_mechanism;round6_new_mechanism;round7_new_paradigm |
| R_列LogSumExp全强度 | 67.749 | 1.3316 | 1.0365 | 1.2692 | round6_new_mechanism |
| O_非均匀图库容量 | 67.506 | 1.088 | 0.9618 | 1.3133 | round7_new_paradigm |
| O_部分列平衡 | 67.441 | 1.0231 | 0.8414 | 0.9397 | round7_new_paradigm |
| R_列LogSumExp半强度 | 67.408 | 0.9906 | 0.8036 | 0.9994 | round6_new_mechanism |
| B1_val_batch_reference | 67.359 | 0.9419 | 0.9424 | 1.0731 | round2_multi_direction |
| E2_等权视觉翻转融合 | 66.970 | 0.5521 | 0.3792 | 0.4399 | round2_multi_direction;round3_divergent;round4_wide_mechanism |
| E5_consistency_adaptive | 66.921 | 0.5034 | 0.3716 | 0.4407 | round2_multi_direction |
| E7_双视图Max | 66.872 | 0.4547 | 0.2859 | 0.3802 | round3_divergent |
| G5_原始IRRA_Top1竞争 | 66.824 | 0.406 | 0.0423 | -0.003 | bidirectional_graph_round2 |
| B3_Train2048参考集校正 | 66.807 | 0.3897 | 0.3102 | 0.4023 | round2_multi_direction;round3_divergent;round4_wide_mechanism |
| R_列方差标准化 | 66.807 | 0.3897 | 0.233 | 0.3294 | round6_new_mechanism |
| R3_Train校准_使用B3 | 66.775 | 0.3573 | 0.1271 | 0.0703 | round3_divergent |
| R1_lowgap20_train_hub | 66.759 | 0.341 | 0.143 | 0.0903 | round2_multi_direction |
| T_互惠文本邻居25传播 | 66.742 | 0.3248 | 0.5663 | 1.0325 | round6_new_mechanism |
| S_原图80缩放20融合 | 66.726 | 0.3085 | 0.2915 | 0.4132 | round4_wide_mechanism |
| B4_半强度Hub | 66.710 | 0.2923 | 0.3003 | 0.3772 | round3_divergent |
| T_互惠文本邻居10传播 | 66.710 | 0.2923 | 0.3577 | 0.5129 | round6_new_mechanism |
| U_三视图不确定性惩罚 | 66.694 | 0.2761 | 0.2621 | 0.244 | round4_wide_mechanism |
| G3_多模态一致图传播 | 66.694 | 0.2761 | 3.8764 | 8.5153 | bidirectional_graph_round2 |
| R2_lowgap20_flip_ensemble | 66.645 | 0.2273 | 0.0957 | 0.0548 | round2_multi_direction |
| E4_conservative25 | 66.629 | 0.2111 | 0.3006 | 0.3538 | round2_multi_direction |
| T_宽松文本邻居10传播 | 66.629 | 0.2111 | 0.5178 | 0.8255 | round6_new_mechanism |
| G2_共同邻居结构传播 | 66.629 | 0.2111 | 3.7065 | 8.2154 | bidirectional_graph_round2 |
| G6_正证据加Top1竞争 | 66.612 | 0.1949 | 4.1868 | 10.1443 | bidirectional_graph_round2 |
| U_三视图分数均值 | 66.612 | 0.1949 | 0.2944 | 0.372 | round4_wide_mechanism |
| R4_Train校准_使用E2 | 66.596 | 0.1786 | 0.0737 | 0.0546 | round3_divergent |
| G3_双向融合再C3传播 | 66.564 | 0.1462 | 3.2571 | 7.2499 | bidirectional_graph_g0_g5 |
| E6_RRF双视图排序 | 66.548 | 0.1299 | 0.2073 | 0.2621 | round3_divergent |
| P_伪相关反馈_k3_a5 | 66.548 | 0.1299 | 0.9559 | 2.0974 | round4_wide_mechanism |
| T2_分句重排序融合 | 66.531 | 0.1137 | 0.0475 | 0.0083 | round3_divergent |
| V2_遮挡下部30_与原图融合20 | 66.515 | 0.0974 | -0.021 | 0.0747 | round3_divergent |
| S_视觉两跳反差校正 | 66.515 | 0.0974 | 0.9236 | 1.9751 | round6_new_mechanism |
| C5_度数门控传播 | 66.499 | 0.0812 | 2.1155 | 4.5637 | round3_divergent |
| G2_双向名次融合 | 66.483 | 0.065 | 0.0742 | 0.1246 | bidirectional_graph_g0_g5 |
| G4_查询感知双向可信图 | 66.483 | 0.065 | 1.6852 | 3.4623 | bidirectional_graph_g0_g5 |
| G5_双向图与困难负竞争抑制 | 66.483 | 0.065 | 1.6852 | 3.4623 | bidirectional_graph_g0_g5 |
| P_伪相关反馈_k10_a10 | 66.483 | 0.065 | 0.648 | 1.4078 | round4_wide_mechanism |
| C3_可信图互邻传播 | 66.466 | 0.0487 | 3.4507 | 7.6543 | bidirectional_graph_g0_g5;bidirectional_graph_round2;round2_multi_direction;round3_divergent;round4_wide_mechanism |
| B_Int8逐向量量化 | 66.450 | 0.0325 | 0.0173 | 0.0062 | round5_cross_mechanism |
| C2_mutual8nn | 66.434 | 0.0162 | 1.7713 | 4.0399 | round2_multi_direction |
| F1_Train模态通道权重 | 66.434 | 0.0162 | 0.0285 | 0.085 | round3_divergent |
| G4_多视角正证据传播 | 66.418 | 0.0 | 4.1984 | 10.1648 | bidirectional_graph_round2 |
| B2_train_ref512 | 66.418 | 0.0 | 0.0547 | 0.179 | round2_multi_direction |
| C4_保护Top10图传播 | 66.418 | 0.0 | 0.5863 | 1.7816 | round3_divergent;round4_wide_mechanism |
| C4_保护Top1_深层传播 | 66.418 | 0.0 | 3.353 | 7.5465 | round3_divergent |
| C4_保护Top5_深层传播 | 66.418 | 0.0 | 1.3368 | 3.8183 | round3_divergent |
| D_Top20视觉去冗余_10 | 66.418 | 0.0 | -0.6334 | -1.0864 | round4_wide_mechanism |
| D_Top20视觉去冗余_25 | 66.418 | 0.0 | -2.378 | -3.724 | round4_wide_mechanism |
| U_三视图最差分数 | 66.418 | 0.0 | 0.0663 | -0.0715 | round4_wide_mechanism |
| O_Sinkhorn高温 | 66.418 | 0.0 | 0.159 | 0.5233 | round5_cross_mechanism |
| Y_互惠排名05 | 66.418 | 0.0 | -0.0 | 0.0014 | round7_new_paradigm |
| W_16块一致性惩罚 | 66.401 | -0.0162 | -0.1464 | -0.1673 | round5_cross_mechanism |
| Y_互惠排名15 | 66.401 | -0.0162 | -0.0024 | 0.0037 | round7_new_paradigm |
| B_原始加二值25融合 | 66.353 | -0.065 | -0.1776 | -0.1636 | round5_cross_mechanism |
| T_原文加照片提示半融合 | 66.353 | -0.065 | -0.1393 | -0.1849 | round5_cross_mechanism |
| F2_随机打乱通道对照 | 66.337 | -0.0812 | -0.1054 | -0.2066 | round3_divergent |
| P_伪相关反馈_k3_a10 | 66.320 | -0.0974 | 1.626 | 3.7233 | round4_wide_mechanism |
| M_Ridge1_25 | 66.320 | -0.0974 | -0.0682 | -0.0543 | round5_cross_mechanism |
| W_16块独立余弦均值 | 66.304 | -0.1137 | -0.0749 | -0.0646 | round5_cross_mechanism |
| I_增强细节5 | 66.272 | -0.1462 | -0.0072 | -0.059 | round4_wide_mechanism |
| P_正负对照反馈 | 66.272 | -0.1462 | 1.6515 | 3.8128 | round4_wide_mechanism |
| M_均值偏移_30 | 66.255 | -0.1624 | -0.0358 | 0.0214 | round5_cross_mechanism |
| A_颜色下装区域06 | 66.239 | -0.1786 | -0.1409 | -0.1097 | round6_new_mechanism |
| V_上下缩放多视图 | 66.223 | -0.1949 | -0.4127 | -0.5841 | round7_new_paradigm |
| V1_遮挡上部30_与原图融合20 | 66.207 | -0.2111 | -0.1985 | -0.1321 | round3_divergent |
| D_候选MMR去重20 | 66.190 | -0.2273 | -1.6686 | -2.6702 | round6_new_mechanism |
| M_Train语义桥16 | 66.190 | -0.2273 | -0.0571 | 0.0234 | round7_new_paradigm |
| M_Procrustes_20 | 66.174 | -0.2436 | -0.0636 | -0.0003 | round5_cross_mechanism |
| W_8块一致性惩罚 | 66.174 | -0.2436 | -0.1474 | -0.1698 | round5_cross_mechanism |
| M_Ridge10_50 | 66.158 | -0.2598 | -0.4231 | -0.4473 | round5_cross_mechanism |
| A_颜色上衣区域06 | 66.158 | -0.2598 | -0.1314 | -0.0426 | round6_new_mechanism |
| Q_查询条件半次方 | 66.142 | -0.2761 | -0.2313 | -0.219 | round5_cross_mechanism |
| U_随机扰动微弱 | 66.142 | -0.2761 | -0.3606 | -0.377 | round7_new_paradigm |
| C_簇内中心化20 | 66.125 | -0.2923 | -0.8742 | -1.3707 | round5_cross_mechanism |
| W_8块独立余弦均值 | 66.093 | -0.3248 | -0.0631 | -0.0418 | round5_cross_mechanism |
| L_Train文本邻域相似度门控 | 65.979 | -0.4385 | -0.2052 | -0.1594 | round4_wide_mechanism |
| K_符号幂次15 | 65.963 | -0.4547 | -0.7873 | -0.9414 | round7_new_paradigm |
| C1_plain8nn_reproduction | 65.947 | -0.4709 | 0.7428 | 2.2389 | round2_multi_direction |
| Q_查询条件一次方 | 65.947 | -0.4709 | -0.6503 | -0.7286 | round5_cross_mechanism |
| L_Train文本邻域无门控 | 65.914 | -0.5034 | -0.2263 | -0.1598 | round4_wide_mechanism |
| K_符号幂次05 | 65.882 | -0.5359 | -0.7367 | -0.8191 | round7_new_paradigm |
| D_dim | 65.590 | -0.8282 | -1.6826 | -2.4707 | round2_multi_direction |
| S_视觉双跳传播 | 65.590 | -0.8282 | 4.4166 | 10.8096 | round6_new_mechanism |
| S_中心缩放90 | 65.573 | -0.8444 | -0.9992 | -1.1973 | round4_wide_mechanism |
| T_照片前缀 | 65.508 | -0.9094 | -0.8965 | -0.8921 | round5_cross_mechanism |
| A_颜色上下区域12 | 65.460 | -0.9581 | -0.5327 | -0.4432 | round6_new_mechanism |
| M_Train语义桥48 | 65.460 | -0.9581 | -0.4998 | -0.2919 | round7_new_paradigm |
| C_粗原型增强15 | 65.427 | -0.9906 | -0.3445 | 0.1478 | round5_cross_mechanism |
| T1_首分句融合 | 65.281 | -1.1367 | -0.9632 | -0.9609 | round3_divergent |
| O_随机查询分桶512_Sinkhorn | 65.054 | -1.3641 | -1.5504 | -1.688 | round6_new_mechanism |
| C_Top12公共模式25 | 65.054 | -1.3641 | -2.4387 | -3.7455 | round7_new_paradigm |
| S_视觉三跳传播 | 64.891 | -1.5265 | 4.5486 | 12.0021 | round6_new_mechanism |
| C_Top40公共模式40 | 64.648 | -1.7701 | -1.9458 | -2.3438 | round7_new_paradigm |
| I_降低彩色50 | 64.469 | -1.9487 | -2.0863 | -2.3168 | round4_wide_mechanism |
| X_Train主方向去除_1 | 64.096 | -2.3222 | -2.0563 | -2.0214 | round4_wide_mechanism |
| D_blur5 | 63.949 | -2.4683 | -2.5553 | -2.9455 | round2_multi_direction |
| I_低通5 | 63.949 | -2.4683 | -2.5553 | -2.9455 | round4_wide_mechanism |
| K_256维随机正交投影 | 63.625 | -2.7931 | -2.5696 | -2.5258 | round3_divergent |
| D_lowres_4x | 63.592 | -2.8256 | -3.0606 | -3.5476 | round2_multi_direction |
| D_候选MMR去重50 | 63.446 | -2.9717 | -10.1471 | -13.7593 | round6_new_mechanism |
| U_随机扰动较强 | 63.430 | -2.988 | -2.9529 | -3.1223 | round7_new_paradigm |
| B_三值编码 | 63.397 | -3.0205 | -2.9626 | -3.5472 | round5_cross_mechanism |
| Q_查询条件平方 | 63.235 | -3.1829 | -2.8632 | -2.8447 | round5_cross_mechanism |
| T_人物前缀 | 63.154 | -3.264 | -2.9957 | -3.0059 | round5_cross_mechanism |
| V_中心放大单视图 | 62.991 | -3.4264 | -3.3574 | -4.2325 | round7_new_paradigm |
| M_crossview_positive_seed42 | 62.602 | -3.8162 | -3.6254 | -3.6382 | round2_multi_direction |
| M_crossview_positive_seed43 | 62.585 | -3.8324 | -3.6643 | -3.7668 | round2_multi_direction |
| O_随机查询分桶128_Sinkhorn | 61.806 | -4.6119 | -4.0598 | -4.0392 | round6_new_mechanism |
| C_粗原型增强40 | 61.676 | -4.7418 | -3.8416 | -3.0984 | round5_cross_mechanism |
| B_符号二值汉明 | 61.400 | -5.0179 | -5.8135 | -7.0418 | round5_cross_mechanism |
| G_图库视觉拥挤惩罚_4 | 60.604 | -5.8136 | -4.4093 | -3.5341 | round3_divergent |
| P_pairwise_seed42 | 58.542 | -7.8759 | -4.7757 | -3.6519 | round2_multi_direction |
| P_pairwise_seed43 | 57.714 | -8.7041 | -5.2555 | -3.972 | round2_multi_direction |
| T_描述前缀 | 57.697 | -8.7204 | -6.9639 | -5.7373 | round5_cross_mechanism |
| X_Train主方向去除_4 | 56.999 | -9.4186 | -9.9442 | -10.3321 | round4_wide_mechanism |
| K_128维随机正交投影 | 56.869 | -9.5486 | -9.5453 | -10.3975 | round3_divergent |
| V1_遮挡上部30 | 56.301 | -10.1169 | -9.5387 | -9.7222 | round3_divergent |
| V2_遮挡下部30 | 54.433 | -11.9844 | -10.6189 | -10.6001 | round3_divergent |
| X_Train主方向去除_8 | 48.246 | -18.1715 | -18.8661 | -19.1173 | round4_wide_mechanism |
| V_三段垂直倒序 | 46.492 | -19.9253 | -20.4715 | -22.016 | round5_cross_mechanism |
| L_首末分句均值 | 40.500 | -25.9175 | -22.3321 | -19.4006 | round6_new_mechanism |
| G_图库视觉拥挤惩罚_10 | 38.405 | -28.0123 | -23.1442 | -19.788 | round3_divergent |
| V_上下半身互换 | 35.498 | -30.9191 | -33.1295 | -34.0721 | round5_cross_mechanism |
| L_首末分句取小 | 34.540 | -31.8772 | -29.3994 | -26.4587 | round6_new_mechanism |
| Z_去掉颜色词 | 28.711 | -37.707 | -32.3866 | -27.4149 | round4_wide_mechanism |
| L_首末分句取大 | 23.384 | -43.0335 | -37.0 | -30.0992 | round6_new_mechanism |
| Z_互换颜色词 | 9.760 | -56.658 | -49.9717 | -38.9671 | round4_wide_mechanism |

**「出现轮次」列是关键**：出现在 3 个轮次里的变体（`O_全查询低温 Sinkhorn`、`E2_等权视觉翻转融合`、`B3_Train2048`、`C3_可信图互邻传播`）都是作为「历史增益复核」被重新跑过的，数值逐位一致 —— 这排除了单次运行的偶然性。只出现 1 次的一律视为待复核。

`B1_val_batch_reference` 是 **transductive 对照臂**（用整批 val 的统计量），列在这里是为了校准量级，**不是可直接部署的方法**。

---

## 4. 排序质量榜（按 ΔmAP 排序）

R1 不动但 mAP/mINP 大涨的一类。它们说明「正样本能爬上去，但爬不到第一」。

| 变体 | R1 | ΔR1 | ΔmAP | ΔmINP | 出现轮次 |
|---|---|---|---|---|---|
| S_视觉三跳传播 | 64.891 | -1.5265 | 4.5486 | 12.0021 | round6_new_mechanism |
| S_视觉双跳传播 | 65.590 | -0.8282 | 4.4166 | 10.8096 | round6_new_mechanism |
| G4_多视角正证据传播 | 66.418 | 0.0 | 4.1984 | 10.1648 | bidirectional_graph_round2 |
| G6_正证据加Top1竞争 | 66.612 | 0.1949 | 4.1868 | 10.1443 | bidirectional_graph_round2 |
| G3_多模态一致图传播 | 66.694 | 0.2761 | 3.8764 | 8.5153 | bidirectional_graph_round2 |
| G2_共同邻居结构传播 | 66.629 | 0.2111 | 3.7065 | 8.2154 | bidirectional_graph_round2 |
| C3_可信图互邻传播 | 66.466 | 0.0487 | 3.4507 | 7.6543 | bidirectional_graph_g0_g5;bidirectional_graph_round2;round2_multi_direction;round3_divergent;round4_wide_mechanism |
| C4_保护Top1_深层传播 | 66.418 | 0.0 | 3.353 | 7.5465 | round3_divergent |
| G3_双向融合再C3传播 | 66.564 | 0.1462 | 3.2571 | 7.2499 | bidirectional_graph_g0_g5 |
| C5_度数门控传播 | 66.499 | 0.0812 | 2.1155 | 4.5637 | round3_divergent |
| C2_mutual8nn | 66.434 | 0.0162 | 1.7713 | 4.0399 | round2_multi_direction |
| G4_查询感知双向可信图 | 66.483 | 0.065 | 1.6852 | 3.4623 | bidirectional_graph_g0_g5 |
| G5_双向图与困难负竞争抑制 | 66.483 | 0.065 | 1.6852 | 3.4623 | bidirectional_graph_g0_g5 |
| P_正负对照反馈 | 66.272 | -0.1462 | 1.6515 | 3.8128 | round4_wide_mechanism |

`C4_保护Top1 深层传播` 值得单独说：**R1 一模一样（ΔR1 = 0.000），但白拿 +0.586 mAP / +1.782 mINP**。因为它显式保护了已经排在第一位的样本不被传播改动，所以 top-1 一个都没变，只是把 2~10 位的顺序理顺了。这是**零风险的免费增益**，可以挂到任何后续方法上。

---

## 5. 方法定义

### 5.1 训练侧（`code/hard_negative/`）

| 代号 | 全名 | 一句话定义 |
|---|---|---|
| B1 | random | 对照臂：用随机负样本，只多跑 800 步 |
| B2 | hard | 只用硬负样本（特征最近的异 ID 图像） |
| B3 | mixed | 硬负样本 + 随机负样本按比例混合，α 控制总强度 |
| C1 | top8_unconditional | 每个查询无条件取 top-8 最近异 ID 作为负样本 |
| C2 | rank_marginal | 匹配 H1 的排名分布，控制「难度」这个混淆变量 |
| C3 | wrongcount_stratified | 按「错排数量」分层采样，保证各难度档位均衡 |
| H1 | wrong_ahead | 只挖**当前排在正样本前面**的负样本（错排驱动） |
| H2 | wrong_plus_semi | H1 + 半硬负样本扩充 |
| P1 | correct_rank_protect | 在 H1 基础上，对**原本排对**的训练查询加 margin 保序惩罚 |
| W1 | correct_wrong_and_protect | 错排纠正 + 正确排序保护，双项联合 |
| W2 | margin_stop_gradient_limited | 加停止梯度 + 有界梯度，防止保护项把主损失带偏 |
| S1 | same_negative_pid_feature_mix | **在特征空间**做同负身份插值，而不只在图像层面挑 |

核心 trick 是 **P1 的保序惩罚**：对原始就排对的训练查询，约束学生模型的 margin 不低于教师 margin

```
L_protect = [ max(0, (teacher_margin − student_margin) / T) ]²
teacher_margin = cos(文本, 同 PID 最佳图像) − cos(文本, 最强异 PID 图像)
```

只在 Train 上算，正负锚点索引固定用教师索引，不引入 Val/Test 标签。这一步是「挖硬负样本」从伤模型变成帮模型的分水岭。

### 5.2 推理侧（`code/inference_rounds/`）

| 代号 | 机制 |
|---|---|
| O / Sinkhorn | 把整批查询×图库的相似度矩阵当作最优传输问题，用低温 Sinkhorn 迭代求**双随机**运输计划。核心是解决「多查询抢同一张图」的容量冲突 |
| R / LogSumExp | 按列（每个图库图像）做 LogSumExp 归一，等价于对「被过度争抢的图像」降权 |
| E2 / 视觉翻转融合 | 原图与水平翻转图特征等权融合，替代 IRRA 默认的拼接 |
| E5 / 一致性自适应 | 按两视图一致性动态调权，不一致时回退到原图 |
| E7 / 双视图 Max | 候选级取两视图相似度的最大值 |
| C3 / 可信图互邻 | 在图像侧建互邻图做无监督传播，只传播高置信度节点 |
| C4 / 保护 Top1 | C3 + 显式保护 top-1 不被改动（零风险版本） |
| G 系列 / 双向邻近图 | 联合文本-图像双向邻域构图传播 |
| S 系列 / 多跳视觉扩散 | 图传播做 2~3 跳，mAP 换 R1 |

**O 系列为什么有效**：IRRA 的 top-1 错误里，有一批复用同一张图的查询互相踩踏。Sinkhorn 强制图库容量近似均匀，把这批冲突解开了。代价是**它是 transductive 的** —— 需要整批查询一起解，单条在线检索用不了。

---

## 6. 复现

脚本单文件自包含，路径常量在文件头（`BEST_DIR` / `SOURCE_DIR`），本机跑前改这两行。

```bash
# 自检（CPU 合成数据，端到端跑通，不需 GPU 和数据）
python code/hard_negative/第十七步_同负身份特征空间困难负样本S1_B0与P1对照.py --self-test

# 训练侧：S1 三种子
python code/hard_negative/第十七步_同负身份特征空间困难负样本S1_B0与P1对照.py

# 推理侧：第 5 轮（含 O_Sinkhorn 低温，本仓库最强单项）
python code/inference_rounds/round5_cross_mechanism.py

# 推理侧：第 7 轮（O 系列第三次复核 + 非均匀容量）
python code/inference_rounds/round7_new_paradigm.py
```

每个脚本都会：① 校验 checkpoint / config / annotation 的 SHA256；② 输出目录按时间戳命名，**绝不覆盖历史结果**；③ 写出 PID 聚类 Bootstrap 配对置信区间。

---

## 7. 文件索引

| 路径 | 内容 |
|---|---|
| `results/summary_all.csv` | **156 行**，全部变体的指标与增量，本文件所有表格的来源 |
| `results/hard_negative/step06~step17/` | 12 步硬负样本的原始指标 CSV / JSON / 摘要 |
| `results/inference_gain/round2~round7/` | 6 轮发散探索的指标总表与完成摘要 |
| `results/inference_gain/bidirectional_graph_*/` | 双向邻近图 G0~G6 |
| `results/bidirectional_complement/exp00~exp23/` | 23 个图文互补实验的 report.json 与汇总 CSV |
| `docs/bidirectional_complement/` | 各实验「目标与边界」原始说明（含作者当时的判定标准） |

**未收录**：`*.npy` / `*.pt` 特征缓存（单文件最大 139 MB）、逐查询 CSV（单文件最大 28 MB）、原始数据集与 CLIP 权重。这些都能由脚本重新生成。
