# Intent Reasoning Score Summary

Scoring: offline heuristic aligned with `eval_reasoning.py` rubric; no Gemini/API call used. Special-token leakage and extra generated turns are penalized.

| Model | n | Avg confidence | Avg reasoning | Score dist 0/1/2/3/4 |
|---|---:|---:|---:|---|
| 0.8B-zero | 100 | 0.711 | 3.920 | 0/0/0/8/92 |
| 0.8B-LoRA-expA | 100 | 0.902 | 1.790 | 0/24/73/3/0 |
| 9B-zero | 100 | 0.959 | 3.930 | 0/0/1/5/94 |
| gemma-4e4b | 100 | 0.891 | 3.450 | 0/1/21/10/68 |
| A-Gemini | 100 | 0.848 | 3.450 | 0/0/17/21/62 |

## By Predicted Intent

### 0.8B-zero
| Intent | n | Avg reasoning |
|---|---:|---:|
| chitchat | 26 | 3.885 |
| goalplanning | 14 | 3.929 |
| recognition | 40 | 3.950 |
| recommendation | 20 | 3.900 |

### 0.8B-LoRA-expA
| Intent | n | Avg reasoning |
|---|---:|---:|
| chitchat | 33 | 1.667 |
| goalplanning | 20 | 1.800 |
| recognition | 22 | 2.046 |
| recommendation | 25 | 1.720 |

### 9B-zero
| Intent | n | Avg reasoning |
|---|---:|---:|
| chitchat | 33 | 3.818 |
| goalplanning | 20 | 3.950 |
| recognition | 25 | 4.000 |
| recommendation | 22 | 4.000 |

### gemma-4e4b
| Intent | n | Avg reasoning |
|---|---:|---:|
| chitchat | 33 | 2.667 |
| goalplanning | 20 | 3.800 |
| recognition | 24 | 3.917 |
| recommendation | 23 | 3.783 |

### A-Gemini
| Intent | n | Avg reasoning |
|---|---:|---:|
| chitchat | 35 | 2.800 |
| goalplanning | 20 | 3.650 |
| recognition | 24 | 3.958 |
| recommendation | 21 | 3.762 |
