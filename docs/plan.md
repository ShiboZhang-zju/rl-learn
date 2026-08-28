# 实施计划

## 已冻结的 baseline

| 项目 | 选择 |
|---|---|
| Task | 3-person Knights & Knaves |
| Model | Qwen2.5-0.5B-Instruct |
| Local framework | Transformers + Datasets + PEFT + TRL |
| First training | LoRA SFT on Mac |
| Future RL | GRPO with exact verifier reward |
| Reward | correct answer `+1.0` + valid format `+0.1` |

## Phase 0–5：当前实现

- [x] 项目结构和配置
- [x] 表达式与 exact solver
- [x] 固定 seed 的数据生成
- [x] train/validation/test 独立 split
- [x] 程序化 reasoning trace
- [x] 数据审计
- [x] tokenizer debug
- [x] 统一 evaluator/reward
- [x] GRPO 数学独立 debug
- [x] Mac 友好的 LoRA SFT 入口
- [x] balanced 8-sample overfit 数据
- [x] prediction pattern distribution 统计
- [x] balanced 20-step 实验与同条件评测
- [x] 下载 tokenizer 并检查真实 token mask
- [ ] 32 条样本过拟合
- [ ] 1000 条正式 SFT
- [ ] Base vs SFT 错误分析

## 当前实验结论

balanced 20-step 实验：

```text
Format Accuracy: 8/8
Exact Accuracy: 1/8
Prediction: KKK × 7, KKN × 1
```

因此：

```text
G0 Data / verifier              PASS
G1 Tokenization / masking       PASS
G2 LoRA optimization            PASS
G3 Checkpoint + generation      PASS
G4 Format learning              PASS
G5 True overfit                 NOT PASS
G6 Generalization               NOT STARTED
G7 GRPO                         NOT STARTED
```

这说明当前 loss/token accuracy 的下降主要反映格式和公共 reasoning token 学习，尚不能证明模型能够根据 puzzle 选择不同答案模式。

## Phase 2：后续 RL

1. 用同一套 test/evaluator 验证 SFT checkpoint。
2. 对每个 prompt 生成多个 completion，记录每条 completion 与 reward。
3. 先实现一个不训练模型的 rollout/reward/advantage dump。
4. 用 TRL `GRPOTrainer` 做 64 prompts 的 smoke test。
5. 观察 reward mean/std、zero-variance group、response length、KL、entropy 和 grad norm。
6. 最后才跑正式 GRPO，并固定 B0/Base、B1/SFT、B2/GRPO、B3/SFT→GRPO。
