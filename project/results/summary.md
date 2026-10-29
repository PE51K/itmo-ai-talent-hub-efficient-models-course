| Config | Run | Accuracy, % | F1, % | Seeds |
|---|---|---|---|---|
| Teacher, 10% labels | `teacher_p10` | 93.7 | 93.8 | 1 |
| Teacher, 100% labels | `teacher_p100` | 95.9 | 96.0 | 1 |
| Student, CE, 10% labels | `hard_p10` | 84.3 ± 1.3 | 84.7 ± 1.2 | 3 |
| Student, KD on the 10% | `kd_p10` | 84.1 ± 0.6 | 84.7 ± 0.5 | 3 |
| Student, KD on all texts, 10% labels | `kdu_p10` | 88.1 ± 0.2 | 88.4 ± 0.1 | 3 |
| Student, CE, k-center 10% | `hard_kc10` | 79.1 ± 1.0 | 79.2 ± 1.2 | 3 |
| Student, KD on all texts, k-center 10% | `kdu_kc10` | 86.7 ± 0.3 | 87.0 ± 0.3 | 3 |
| Student, CE, 100% labels | `hard_p100` | 89.5 ± 0.6 | 89.8 ± 0.5 | 3 |
| Student, KD, 100% labels | `kd_p100` | 89.0 ± 0.3 | 89.4 ± 0.3 | 3 |

| Student | `hard_p10`, % | `kdu_p10`, % | `hard_p100`, % |
|---|---|---|---|
| Ettin-encoder-17m | 84.3 ± 1.3 | 88.1 ± 0.2 | 89.5 ± 0.6 |

| Model | Params, M | Non-embedding params, M | CPU, batch of 32, ms | T4, sentences/s |
|---|---|---|---|---|
| `jhu-clsp/ettin-encoder-400m` | 395.8 | 344.3 | 6042 | 312 |
| `jhu-clsp/ettin-encoder-17m` | 16.9 | 4.0 | 93 | 3350 |

CPU latency with 2 PyTorch threads, GPU: Tesla T4.
