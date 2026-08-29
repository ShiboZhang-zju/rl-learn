# Phase A — reward-rescue audit (zero training)

Question: for a group whose 8 rollouts are all **exact-wrong**, does the
3-person **partial** reward restore non-zero variance (hence non-zero advantage)?

Rewards: `exact = 1[parsed == gt]`; `partial = correct_person_count / 3`;
parse failure / invalid -> 0.0 in both modes.
Advantages use TRL 0.23 semantics: `(r - mean) / (sample_std_ddof1 + 1e-4)`.

## Summary

| Source | groups | exact all-wrong | partial gains variance | still zero-var | rescue rate | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| grpo_v2_200prompt | 200 | 32 | 10 | 22 | 31.2% | [0.180, 0.486] |
| sft_epoch4_200prompt | 200 | 13 | 7 | 6 | 53.8% | [0.291, 0.768] |
| grpo_v1_200prompt | 200 | 27 | 7 | 20 | 25.9% | [0.132, 0.447] |
| grpo_v2_probe_20prompt | 100 | 14 | 5 | 9 | 35.7% | [0.163, 0.612] |

## Primary source detail (GRPO-V2, 200-prompt subset)

- Exact all-wrong groups: **32**
- Among them, partial reward still zero-variance: **22**
- Among them, partial reward gains variance: **10**
- **reward rescue rate = 0.3125**

Zero-variance groups overall (all 8 rollouts identical reward):

- exact mode:   150 / 200
- partial mode: 140 / 200

Partial reward distribution over all rollouts:

- mean = 0.8592, std = 0.2660
- unique values = [0.0, 0.333333, 0.666667, 1.0]
- counts = {0.0: 31, 0.333333: 221, 0.666667: 141, 1.0: 1207}

Among the exact all-wrong groups, how many *distinct* partial reward values:

- {1: 22, 2: 10}

- rescued groups: mean max|advantage| = 1.502233201265335

## Examples

### rescued

- `kk_grpo_v1_final_holdout_000009` gt=KNK
  - predictions: ['NNN', 'NNK', 'NNN', 'NNK', 'NNN', 'NNN', 'NNK', 'NNK']
  - exact:   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  - partial: [0.3333, 0.6667, 0.3333, 0.6667, 0.3333, 0.3333, 0.6667, 0.6667]
  - partial_std = 0.178174
  - max|advantage| = 0.9349
- `kk_grpo_v1_final_holdout_000027` gt=NKN
  - predictions: ['KKN', 'KNN', 'KKN', 'KKN', 'KNN', 'KNN', 'KNN', 'KKN']
  - exact:   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  - partial: [0.6667, 0.3333, 0.6667, 0.6667, 0.3333, 0.3333, 0.3333, 0.6667]
  - partial_std = 0.178174
  - max|advantage| = 0.9349
- `kk_grpo_v1_final_holdout_000038` gt=NKK
  - predictions: ['KKN', 'KKN', 'KKN', 'KNN', 'KKN', 'KKN', 'KKN', 'KKN']
  - exact:   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  - partial: [0.3333, 0.3333, 0.3333, 0.0, 0.3333, 0.3333, 0.3333, 0.3333]
  - partial_std = 0.117851
  - max|advantage| = 2.4728

### not_rescued

- `kk_grpo_v1_final_holdout_000014` gt=KNK
  - predictions: ['KKK', 'KKK', 'KKK', 'KKK', 'KKK', 'KKK', 'KKK', 'KKK']
  - exact:   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  - partial: [0.6667, 0.6667, 0.6667, 0.6667, 0.6667, 0.6667, 0.6667, 0.6667]
  - partial_std = 0.000000
- `kk_grpo_v1_final_holdout_000019` gt=KNN
  - predictions: ['NKN', 'NNK', 'NNK', 'NKN', 'NNK', 'NNK', 'NNK', 'NNK']
  - exact:   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  - partial: [0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333]
  - partial_std = 0.000000
- `kk_grpo_v1_final_holdout_000031` gt=NKN
  - predictions: ['NNK', 'KNN', 'NNK', 'NNK', 'NNK', 'NNK', 'NNK', 'NNK']
  - exact:   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  - partial: [0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333]
  - partial_std = 0.000000

## Gate

- primary rescue rate = **0.3125**
- STOP if < 0.10, continue to Phase B if > 0.25 (engineering gate, not a significance test)
- decision = **CONTINUE_TO_PHASE_B**

