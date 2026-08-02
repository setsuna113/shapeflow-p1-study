# BrowseComp-Plus: P1 against P0

## What this is

- Baseline arm: **P0**
- Cells: 520 committed of 560
- Tasks: 80
- Run: `campaign1` layer `b1_select` lanes `['0', '1']`
- Execution binding: `3e73c63194e7a7fd`
- Answer key: `0374946363a5c59f`
- Bootstrap: 10000 resamples, seed 20260731

## Arms

| arm | cells | committed | accuracy | evidence recall | GPU-busy s | prompt tok | completion tok | e2e s |
|---|---|---|---|---|---|---|---|---|
| `C_CPU_CONTROL` | 80 | 62 | 0.032 | 0.089 | 152.6 | 151,267 | 16,341 | 156.2 |
| `C_ID` | 80 | 76 | 0.092 | 0.129 | 230.3 | 180,069 | 23,714 | 234.2 |
| `H_CPU_CONTROL` | 80 | 80 | 0.1 | 0.116 | 87.1 | 58,470 | 5,893 | 94.3 |
| `H_MARKDOWN_ID` | 80 | 77 | 0.104 | 0.131 | 280 | 270,438 | 26,713 | 284.7 |
| `H_PLUS_C` | 80 | 74 | 0.108 | 0.105 | 262.3 | 288,331 | 24,499 | 266.9 |
| `H_PROSE_CONTROL` | 80 | 77 | 0.104 | 0.117 | 112 | 139,448 | 8,958 | 115.5 |
| `P0` | 80 | 74 | 0.095 | 0.134 | 265.4 | 174,302 | 27,579 | 269.7 |

## Did the treatment fire?

| arm | H published / offered | C published / offered | fallbacks | close failures | overall rate |
|---|---|---|---|---|---|
| `C_CPU_CONTROL` | -- | 50 / 62 (81%) | 0 | 12 | 0.806 |
| `C_ID` | -- | 5 / 76 (7%) | 0 | 71 | 0.066 |
| `H_CPU_CONTROL` | 206 / 206 (100%) | -- | 0 | 0 | 1 |
| `H_MARKDOWN_ID` | 2 / 188 (1%) | -- | 186 | 0 | 0.011 |
| `H_PLUS_C` | 4 / 181 (2%) | 3 / 73 (4%) | 177 | 70 | 0.028 |
| `H_PROSE_CONTROL` | 129 / 141 (91%) | -- | 12 | 0 | 0.915 |

## `C_CPU_CONTROL` vs `P0`

Paired on 61 tasks where both arms committed.
**Survivorship warning:** only 81% of the smaller arm's committed tasks survived the pairing, so this contrast is computed on a task set that the arms' own failures selected.

| metric | baseline | treatment | paired diff | 95% CI | wins/losses | verdict |
|---|---|---|---|---|---|---|
| Accuracy (official grader) | 0.049 | 0.033 | -0.016 (-33.3%) | [-0.049, 0] | 0/1 | no detectable difference |
| Evidence recall (agent-level) | 0.096 | 0.09 | -0.006 (-6%) | [-0.019, 0.007] | 1/3 | no detectable difference |
| GPU-busy seconds (primary work) | 211.919 | 144.707 | -67.212 (-31.7%) | [-99.424, -41.561] | 52/9 | better |
| Prompt tokens (co-primary) | 168,594 | 152,030 | -16,564 (-9.8%) | [-27,068, -5,040] | 47/13 | better |
| Completion tokens (co-primary) | 21,125 | 15,694 | -5,431 (-25.7%) | [-9,051, -2,690] | 48/12 | better |
| End-to-end seconds | 215.584 | 148.298 | -67.286 (-31.2%) | [-100.124, -41.268] | 52/9 | better |
| Cached prompt tokens | 0 | 0 | 0 (0%) | [0, 0] | 0/0 | no detectable difference |
| Gold recall (agent-level) | 0.08 | 0.091 | 0.011 (13.7%) | [-0.016, 0.049] | 1/1 | no detectable difference |
| Search queries issued | 5.295 | 5.311 | 0.016 (0.3%) | [-0.393, 0.459] | 10/14 | no detectable difference |

## `C_ID` vs `P0`

Paired on 74 tasks where both arms committed.

| metric | baseline | treatment | paired diff | 95% CI | wins/losses | verdict |
|---|---|---|---|---|---|---|
| Accuracy (official grader) | 0.095 | 0.081 | -0.014 (-14.3%) | [-0.041, 0] | 0/1 | no detectable difference |
| Evidence recall (agent-level) | 0.134 | 0.125 | -0.009 (-6.6%) | [-0.033, 0.014] | 4/7 | no detectable difference |
| GPU-busy seconds (primary work) | 265.417 | 229.875 | -35.543 (-13.4%) | [-77.109, 0.594] | 38/36 | no detectable difference |
| Prompt tokens (co-primary) | 174,302 | 182,442 | 8,140 (4.7%) | [-3,808, 20,675] | 28/45 | no detectable difference |
| Completion tokens (co-primary) | 27,579 | 23,770 | -3,810 (-13.8%) | [-8,413, 187.419] | 41/32 | no detectable difference |
| End-to-end seconds | 269.684 | 233.844 | -35.84 (-13.3%) | [-77.8, 0.721] | 37/37 | no detectable difference |
| Cached prompt tokens | 0 | 0 | 0 (0%) | [0, 0] | 0/0 | no detectable difference |
| Gold recall (agent-level) | 0.114 | 0.107 | -0.007 (-5.9%) | [-0.027, 0.009] | 1/2 | no detectable difference |
| Search queries issued | 5.297 | 5.324 | 0.027 (0.5%) | [-0.351, 0.405] | 16/18 | no detectable difference |

## `H_CPU_CONTROL` vs `P0`

Paired on 74 tasks where both arms committed.
**Survivorship warning:** only 92% of the smaller arm's committed tasks survived the pairing, so this contrast is computed on a task set that the arms' own failures selected.

| metric | baseline | treatment | paired diff | 95% CI | wins/losses | verdict |
|---|---|---|---|---|---|---|
| Accuracy (official grader) | 0.095 | 0.081 | -0.014 (-14.3%) | [-0.041, 0] | 0/1 | no detectable difference |
| Evidence recall (agent-level) | 0.134 | 0.107 | -0.027 (-20.1%) | [-0.054, -0.004] | 4/10 | WORSE |
| GPU-busy seconds (primary work) | 265.417 | 87.292 | -178.126 (-67.1%) | [-234.426, -124.513] | 67/7 | better |
| Prompt tokens (co-primary) | 174,302 | 59,186 | -115,116 (-66%) | [-133,193, -98,012] | 72/2 | better |
| Completion tokens (co-primary) | 27,579 | 5,918 | -21,662 (-78.5%) | [-28,414, -15,445] | 73/1 | better |
| End-to-end seconds | 269.684 | 94.578 | -175.106 (-64.9%) | [-231.821, -121.082] | 66/8 | better |
| Cached prompt tokens | 0 | 0 | 0 (0%) | [0, 0] | 0/0 | no detectable difference |
| Gold recall (agent-level) | 0.114 | 0.106 | -0.008 (-6.9%) | [-0.038, 0.019] | 3/3 | no detectable difference |
| Search queries issued | 5.297 | 5.649 | 0.351 (6.6%) | [-0.135, 0.838] | 25/16 | no detectable difference |

## `H_MARKDOWN_ID` vs `P0`

Paired on 73 tasks where both arms committed.
**Survivorship warning:** only 94% of the smaller arm's committed tasks survived the pairing, so this contrast is computed on a task set that the arms' own failures selected.

| metric | baseline | treatment | paired diff | 95% CI | wins/losses | verdict |
|---|---|---|---|---|---|---|
| Accuracy (official grader) | 0.096 | 0.096 | 0 (0%) | [0, 0] | 0/0 | no detectable difference |
| Evidence recall (agent-level) | 0.136 | 0.12 | -0.016 (-11.7%) | [-0.039, 0.005] | 3/7 | no detectable difference |
| GPU-busy seconds (primary work) | 255.114 | 272.453 | 17.339 (6.8%) | [-23.187, 57.288] | 22/51 | no detectable difference |
| Prompt tokens (co-primary) | 173,132 | 275,322 | 102,189 (59%) | [81,835, 122,382] | 4/69 | WORSE |
| Completion tokens (co-primary) | 26,359 | 26,211 | -148.753 (-0.6%) | [-4,303, 3,764] | 26/47 | no detectable difference |
| End-to-end seconds | 259.267 | 277.067 | 17.8 (6.9%) | [-23.216, 58.117] | 21/52 | no detectable difference |
| Cached prompt tokens | 0 | 0 | 0 (0%) | [0, 0] | 0/0 | no detectable difference |
| Gold recall (agent-level) | 0.116 | 0.118 | 0.002 (2%) | [-0.025, 0.037] | 1/2 | no detectable difference |
| Search queries issued | 5.26 | 5.274 | 0.014 (0.3%) | [-0.384, 0.397] | 13/17 | no detectable difference |

## `H_PLUS_C` vs `P0`

Paired on 72 tasks where both arms committed.
**Survivorship warning:** only 95% of the smaller arm's committed tasks survived the pairing, so this contrast is computed on a task set that the arms' own failures selected.

| metric | baseline | treatment | paired diff | 95% CI | wins/losses | verdict |
|---|---|---|---|---|---|---|
| Accuracy (official grader) | 0.097 | 0.111 | 0.014 (14.3%) | [0, 0.042] | 1/0 | no detectable difference |
| Evidence recall (agent-level) | 0.121 | 0.108 | -0.014 (-11.4%) | [-0.032, 0.002] | 4/8 | no detectable difference |
| GPU-busy seconds (primary work) | 257.345 | 266.72 | 9.375 (3.6%) | [-26.22, 42.457] | 22/50 | no detectable difference |
| Prompt tokens (co-primary) | 175,153 | 293,636 | 118,483 (67.6%) | [94,303, 145,357] | 4/68 | WORSE |
| Completion tokens (co-primary) | 26,140 | 24,945 | -1,194 (-4.6%) | [-4,767, 2,070] | 26/46 | no detectable difference |
| End-to-end seconds | 261.469 | 271.451 | 9.982 (3.8%) | [-26.134, 43.574] | 22/50 | no detectable difference |
| Cached prompt tokens | 0 | 0 | 0 (0%) | [0, 0] | 0/0 | no detectable difference |
| Gold recall (agent-level) | 0.097 | 0.109 | 0.013 (13.2%) | [-0.009, 0.047] | 2/1 | no detectable difference |
| Search queries issued | 5.25 | 5.264 | 0.014 (0.3%) | [-0.361, 0.403] | 12/18 | no detectable difference |

## `H_PROSE_CONTROL` vs `P0`

Paired on 71 tasks where both arms committed.
**Survivorship warning:** only 89% of the smaller arm's committed tasks survived the pairing, so this contrast is computed on a task set that the arms' own failures selected.

| metric | baseline | treatment | paired diff | 95% CI | wins/losses | verdict |
|---|---|---|---|---|---|---|
| Accuracy (official grader) | 0.085 | 0.07 | -0.014 (-16.7%) | [-0.07, 0.042] | 2/3 | no detectable difference |
| Evidence recall (agent-level) | 0.128 | 0.116 | -0.012 (-9.1%) | [-0.045, 0.02] | 7/10 | no detectable difference |
| GPU-busy seconds (primary work) | 265.884 | 114.216 | -151.668 (-57%) | [-208.672, -100.059] | 62/9 | better |
| Prompt tokens (co-primary) | 174,860 | 144,083 | -30,777 (-17.6%) | [-54,653, -5,376] | 51/20 | better |
| Completion tokens (co-primary) | 27,825 | 9,209 | -18,616 (-66.9%) | [-25,435, -12,733] | 66/5 | better |
| End-to-end seconds | 270.181 | 117.728 | -152.453 (-56.4%) | [-210.38, -100.1] | 61/10 | better |
| Cached prompt tokens | 0 | 0 | 0 (0%) | [0, 0] | 0/0 | no detectable difference |
| Gold recall (agent-level) | 0.105 | 0.11 | 0.005 (4.5%) | [-0.031, 0.045] | 3/3 | no detectable difference |
| Search queries issued | 5.366 | 4.352 | -1.014 (-18.9%) | [-1.634, -0.394] | 16/38 | WORSE |

## Reading

**`C_CPU_CONTROL`** -- 50 P1 publications.
- Improves: GPU-busy seconds (primary work), Prompt tokens (co-primary), Completion tokens (co-primary), End-to-end seconds
- Costs: nothing measurably
- Unchanged within the interval: Accuracy (official grader), Evidence recall (agent-level), Gold recall (agent-level), Cached prompt tokens, Search queries issued

**`C_ID`** -- 5 P1 publications.
- Improves: nothing measurably
- Costs: nothing measurably
- Unchanged within the interval: Accuracy (official grader), Evidence recall (agent-level), Gold recall (agent-level), GPU-busy seconds (primary work), Prompt tokens (co-primary), Completion tokens (co-primary), Cached prompt tokens, End-to-end seconds, Search queries issued

**`H_CPU_CONTROL`** -- 206 P1 publications.
- Improves: GPU-busy seconds (primary work), Prompt tokens (co-primary), Completion tokens (co-primary), End-to-end seconds
- Costs: Evidence recall (agent-level)
- Unchanged within the interval: Accuracy (official grader), Gold recall (agent-level), Cached prompt tokens, Search queries issued

**`H_MARKDOWN_ID`** -- 2 P1 publications.
- Improves: nothing measurably
- Costs: Prompt tokens (co-primary)
- Unchanged within the interval: Accuracy (official grader), Evidence recall (agent-level), Gold recall (agent-level), GPU-busy seconds (primary work), Completion tokens (co-primary), Cached prompt tokens, End-to-end seconds, Search queries issued

**`H_PLUS_C`** -- 7 P1 publications.
- Improves: nothing measurably
- Costs: Prompt tokens (co-primary)
- Unchanged within the interval: Accuracy (official grader), Evidence recall (agent-level), Gold recall (agent-level), GPU-busy seconds (primary work), Completion tokens (co-primary), Cached prompt tokens, End-to-end seconds, Search queries issued

**`H_PROSE_CONTROL`** -- 129 P1 publications.
- Improves: GPU-busy seconds (primary work), Prompt tokens (co-primary), Completion tokens (co-primary), End-to-end seconds
- Costs: Search queries issued
- Unchanged within the interval: Accuracy (official grader), Evidence recall (agent-level), Gold recall (agent-level), Cached prompt tokens
