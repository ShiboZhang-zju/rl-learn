# Preregistered design — H5 causal intervention: K=8 vs K=16

Created **before** the K16 run, before any fresh-holdout result was observed.
Date: 2026-08-29

---

## Hypothesis

```text
Increasing K from 8 to 16 increases the probability that
low-support gold answers are sampled during on-policy rollout,
thereby reducing lower-tail collapse.
```

Mechanistically: with `P(miss) = (1-p)^K`, doubling K multiplies the miss
probability by `(1-p)^8`. For p=0.10 that is 43.0% → 18.5%; for p=0.05,
66.3% → 44.0%. If the rich-get-richer loop is driven by finite-K sampling
coverage, more coverage should preferentially help the low-support region.

## Reference

```text
historical GRPO-V2 K=8   (outputs/grpo_v2_kl001/, best = checkpoint-600)
```

Validity of that reference is gated by §2 of the task spec
(`HISTORICAL_K8_CONTROL_VALID` must be established first).

## Intervention

```text
K = 16   (num_generations: 8 -> 16)
```

Everything else is frozen: SFT init (checkpoint-1252), frozen reference
(checkpoint-1252), beta = 0.01, exact reward, LR = 1e-5, weight decay 0,
max_grad_norm 1.0, prompt_batch_size 8, temperature 0.8, top_p 0.95,
max_completion_length 64, 1 epoch, 625 optimizer steps, seed 20260830,
train file `data/processed/grpo_v1_train.jsonl`.

## Primary mechanism outcomes

```text
1. p10 gold_q
2. fraction gold_q < 0.05
3. median Δgold_q in initial gold_q < 0.20
4. common-eval K=8 all-wrong rate
5. common-eval K=8 Pass@8
```

## Secondary outcomes

```text
exact greedy accuracy
Top1 / Top3 coverage
entropy
effective support
```

## Selection and inference rules (committed in advance)

```text
Do not select conclusions using only final exact accuracy.
```

- Best checkpoint is chosen by **V2 Val exact accuracy only**, ties → earlier.
  The fresh holdout is never used for selection and is not inspected before
  the best checkpoint is fixed.
- Primary endpoint is **lower-tail probability collapse**, not exact accuracy.
  A non-significant exact-accuracy delta does **not** by itself refute H5.
- Training-time group statistics (all-wrong, Pass@K) are **process
  diagnostics only**: with K=16, `P(all-wrong)` falls mechanically and
  `Pass@16` is not comparable to `Pass@8`. They are never used as evidence.
- All behavioural comparisons use a **common evaluation protocol**
  `K_eval = 8, temperature 0.8, top_p 0.95, max_new_tokens 64`, so that
  "training K may differ, evaluation K must not".
- Both comparisons are reported:
  - **Comparison A** (equal optimizer steps / prompt exposure)
  - **Comparison B** (equal total rollout budget)
  Neither alone supports a K-only attribution.
- Statistics: 10000 paired bootstrap, fixed seed, for probability-landscape
  endpoints; McNemar + paired bootstrap for exact accuracy.

## Decision rules (from the task spec)

```text
H5_CAUSAL_SUPPORT
  lower-tail collapse in initial q<0.20 significantly mitigated
  AND fraction gold_q<0.05 significantly down
  AND p10 gold_q clearly up
  AND common K_eval=8 all-wrong down / Pass@8 up
  AND equal-step comparison supports it
  AND equal-rollout-budget comparison does not reverse it

H5_CAUSAL_NOT_SUPPORTED
  coverage clearly increased but lower tail / p10 /
  common-eval all-wrong / Pass@8 all unchanged

H5_COMPUTE_CONFOUNDED
  equal-step K16 clearly better,
  but the advantage disappears or reverses under equal rollout budget

H5_CAUSAL_INCONCLUSIVE
  sampler misalignment, inconsistent historical K8 training semantics,
  or OOM forcing changes to other variables
```

## Blocking gates that must pass before training

```text
1. HISTORICAL_K8_CONTROL_VALID   (§2)
2. same_unique_prompt_order      (§3, sampler alignment, first 100 steps)
3. H2_SMOKE_PASS on K16 at 128 completions / step, with no change to
   prompt_batch_size / gradient accumulation / gradient checkpointing /
   max_completion_length  (§6)
```

If gate 3 fails because of memory: `K16_MEMORY_BLOCKED`, stop.

## Out of scope this round

```text
K = 32            (forbidden, regardless of outcome)
new reward        (mode stays exact)
new beta          (0.01)
new LR
new generator / new training data
```
