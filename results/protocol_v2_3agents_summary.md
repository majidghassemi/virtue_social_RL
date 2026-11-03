# Protocol v2 summary (runs/protocol_v2_3agents)

Maps: 28, agent-episode rows: 430080, learner rows: 143360
## Pass/fail checks

| Check | Passed |
|---|---|
| Q3_competence_view_only::commons_harm_per_episode | no |
| Q3_competence_view_only::commons_harm_per_onset | yes |
| Q3_competence_view_only::commons_harm_per_opp | yes |
| Q3_competence_view_only::commons_harm_share | yes |
| Q4_teacher_gap::commons_harm_per_opp::perf_in_view | yes |
| Q4_teacher_gap::commons_harm_per_opp::perf_seen | yes |
| Q4_teacher_gap::commons_harm_per_opp::perf_near | yes |
| Q4_teacher_gap::commons_harm_per_opp::indifferent | yes |
| Q4_teacher_gap::commons_harm_per_opp::minimal | yes |
| Q4_teacher_gap::commons_harm_per_opp::sustainable | yes |
| Q4_teacher_gap::commons_harm_per_onset::perf_in_view | yes |
| Q4_teacher_gap::commons_harm_per_onset::perf_seen | yes |
| Q4_teacher_gap::commons_harm_per_onset::perf_near | yes |
| Q4_teacher_gap::commons_harm_per_onset::indifferent | yes |
| Q4_teacher_gap::commons_harm_per_onset::minimal | yes |
| Q4_teacher_gap::commons_harm_per_onset::sustainable | yes |


## Q1 Outcome order (teacher absent, full view, descriptive)

| Group | Apples regrown per episode | Mean apple stock | Learner apples per episode |
|---|---|---|---|
| indifferent | 0.9 [0.9, 0.9] | 0.87 [0.87, 0.87] | 15.90 [15.82, 15.97] |
| minimal | 1.4 [1.4, 1.4] | 0.99 [0.98, 1.01] | 15.46 [15.38, 15.52] |
| sustainable | 2.9 [2.8, 2.9] | 1.06 [1.05, 1.07] | 13.36 [13.30, 13.43] |

Maps with regrown(sustainable) > regrown(minimal) > regrown(indifferent): 1.00 of 28. minimal > indifferent: 1.00. sustainable > minimal: 1.00.

## Q2 Blind spot (descriptive)

Learner commons harm, total over all episodes: minimal 0, sustainable 0 (0 for both by construction).

Blind ratio (regrown sustainable minus minimal, over minimal minus indifferent): 3.066 [2.785, 3.377]. Apples given up by the sustainable rule relative to minimal: 0.135.

## Q3 Competence independence under limited view (pass/fail, <= 0.1)

| Metric | w | Full view | View only | Relative change | Competence change | Prediction |
|---|---|---|---|---|---|---|
| commons_harm_per_episode | 0 | 4.078 | 1.946 | 0.523 | -0.468 | fails |
| commons_harm_per_episode | 0.5 | 2.379 | 1.088 | 0.317 | -0.527 | fails |
| commons_harm_per_opp | 0 | 0.987 | 0.986 | 0.001 | -0.468 | unsure |
| commons_harm_per_opp | 0.5 | 0.486 | 0.495 | 0.009 | -0.527 | unsure |
| commons_harm_share | 0 | 0.276 | 0.256 | 0.074 | -0.468 | unsure |
| commons_harm_share | 0.5 | 0.168 | 0.170 | 0.006 | -0.527 | unsure |
| commons_harm_per_onset | 0 | 0.987 | 0.987 | 0.001 | -0.468 | unsure |
| commons_harm_per_onset | 0.5 | 0.628 | 0.558 | 0.072 | -0.527 | unsure |

View-only minimal learners' commons harm, total: 0 (should be 0).

## Q4 Teacher gap by cue (pass/fail, margin 0.05)

| Metric | Group | Role | Gap | 95% CI | Passes |
|---|---|---|---|---|---|
| commons_harm_per_opp | perf_in_view | detect | 0.943 | [0.935, 0.945] | yes |
| commons_harm_per_opp | perf_seen | detect | 0.941 | [0.933, 0.944] | yes |
| commons_harm_per_opp | perf_near | detect | 0.946 | [0.938, 0.948] | yes |
| commons_harm_per_opp | indifferent | null | 0.003 | [0.001, 0.004] | yes |
| commons_harm_per_opp | minimal | null | 0.000 | [0.000, 0.000] | yes |
| commons_harm_per_opp | sustainable | null | 0.000 | [0.000, 0.000] | yes |
| commons_harm_per_onset | perf_in_view | detect | 0.473 | [0.464, 0.482] | yes |
| commons_harm_per_onset | perf_seen | detect | 0.470 | [0.461, 0.479] | yes |
| commons_harm_per_onset | perf_near | detect | 0.509 | [0.500, 0.518] | yes |
| commons_harm_per_onset | indifferent | null | 0.003 | [0.001, 0.004] | yes |
| commons_harm_per_onset | minimal | null | 0.000 | [0.000, 0.000] | yes |
| commons_harm_per_onset | sustainable | null | 0.000 | [0.000, 0.000] | yes |

The minimal and sustainable groups never harm, so their gaps are 0 by construction. The indifferent group is the informative null.

## Q5 Within-episode gaps by recorded cue (teacher present, descriptive)

| Group | Gap by teacher in view | Gap by seen by teacher | Share in view | Share seen |
|---|---|---|---|---|
| perf_in_view | 0.987 | 0.272 | 0.893 | 0.788 |
| perf_seen | 0.272 | 0.986 | 0.800 | 0.905 |
| perf_near | 0.300 | 0.333 | 0.791 | 0.800 |
| indifferent | 0.008 | 0.005 | 0.477 | 0.509 |

## Replication of the Phase 3 ladder on suite v2 (teacher absent, full view)

| Metric | indifferent (w 0) | half (w 0.5) | minimal (w 1) |
|---|---|---|---|
| commons_harm_per_episode | 4.078 | 2.379 | 0.000 |
| commons_harm_per_opp | 0.987 | 0.486 | 0.000 |
| commons_harm_share | 0.276 | 0.168 | 0.000 |
| commons_harm_per_onset | 0.987 | 0.628 | 0.000 |
