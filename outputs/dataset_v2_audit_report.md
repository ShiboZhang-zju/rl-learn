# kk-v2 数据审计报告

## 完整性

- Generator：`kk-v2-statement-first`；seed：`20260829`。
- 总样本：`6500`；唯一 puzzle：`6500`。
- duplicate puzzle：`0`，重复多出的行数：`0`。
- 生成方式：先随机生成 statements，再调用 exact solver，仅保留唯一解。

## Split 与标签分布

| Split | Count | KKK | KKN | KNK | KNN | NKK | NKN | NNK | NNN | Unique solution rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 5000 | 591 | 621 | 674 | 603 | 597 | 633 | 628 | 653 | 1.000 |
| val | 500 | 60 | 73 | 52 | 63 | 60 | 63 | 65 | 64 | 1.000 |
| test | 1000 | 126 | 120 | 128 | 140 | 98 | 133 | 129 | 126 | 1.000 |

## 难度与长度

| Split | Prompt chars mean | Answer chars mean | Expr nodes mean | Expr depth mean |
|---|---:|---:|---:|---:|
| train | 351.0 | 56.5 | 6.56 | 2.39 |
| val | 357.9 | 56.5 | 6.92 | 2.48 |
| test | 354.7 | 56.5 | 6.71 | 2.46 |

## Shortcut audit

互信息越低表示该结构特征对答案标签的直接关联越弱；它不是独立性证明。

| Split | same | different | and | or | not | top-op signature |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.0927 | 0.0977 | 0.0225 | 0.0218 | 0.0032 | 0.7807 |
| val | 0.1575 | 0.0885 | 0.0665 | 0.0761 | 0.0410 | 1.8798 |
| test | 0.1236 | 0.1184 | 0.0345 | 0.0512 | 0.0195 | 1.3984 |

## 人工抽查

固定随机抽查 `30` 条 train 样本；每条同时保存 raw statements、solver answer、SFT prompt、SFT completion，并检查 solver answer 与记录一致。

明细保存在本报告对应的 JSON 文件 `dataset_v2_feature_audit.json`。

