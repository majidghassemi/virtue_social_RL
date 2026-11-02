# Protocol v2 summary (runs/protocol_v2)

Maps: 28, agent-episode rows: 501760, learner rows: 215040
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
| indifferent | 0.5 [0.5, 0.6] | 0.52 [0.51, 0.52] | 5.28 [5.24, 5.32] |
| minimal | 0.7 [0.7, 0.7] | 0.59 [0.58, 0.60] | 4.62 [4.58, 4.66] |
| sustainable | 1.2 [1.1, 1.2] | 0.61 [0.60, 0.62] | 3.78 [3.75, 3.82] |

Maps with regrown(sustainable) > regrown(minimal) > regrown(indifferent): 0.93 of 28. minimal > indifferent: 0.93. sustainable > minimal: 1.00.

## Q2 Blind spot (descriptive)

Learner commons harm, total over all episodes: minimal 0, sustainable 0 (0 for both by construction).

Blind ratio (regrown sustainable minus minimal, over minimal minus indifferent): 2.834 [2.231, 3.784]. Apples given up by the sustainable rule relative to minimal: 0.181.

## Q3 Competence independence under limited view (pass/fail, <= 0.1)

| Metric | w | Full view | View only | Relative change | Competence change | Prediction |
|---|---|---|---|---|---|---|
| commons_harm_per_episode | 0 | 1.294 | 0.758 | 0.415 | -0.386 | fails |
| commons_harm_per_episode | 0.5 | 0.804 | 0.372 | 0.334 | -0.460 | fails |
| commons_harm_per_opp | 0 | 0.930 | 0.944 | 0.015 | -0.386 | unsure |
| commons_harm_per_opp | 0.5 | 0.466 | 0.477 | 0.011 | -0.460 | unsure |
| commons_harm_share | 0 | 0.266 | 0.259 | 0.023 | -0.386 | unsure |
| commons_harm_share | 0.5 | 0.175 | 0.158 | 0.066 | -0.460 | unsure |
| commons_harm_per_onset | 0 | 0.930 | 0.944 | 0.016 | -0.386 | unsure |
| commons_harm_per_onset | 0.5 | 0.585 | 0.539 | 0.049 | -0.460 | unsure |

View-only minimal learners' commons harm, total: 0 (should be 0).

## Q4 Teacher gap by cue (pass/fail, margin 0.05)

| Metric | Group | Role | Gap | 95% CI | Passes |
|---|---|---|---|---|---|
| commons_harm_per_opp | perf_in_view | detect | 0.836 | [0.802, 0.841] | yes |
| commons_harm_per_opp | perf_seen | detect | 0.846 | [0.812, 0.849] | yes |
| commons_harm_per_opp | perf_near | detect | 0.841 | [0.801, 0.844] | yes |
| commons_harm_per_opp | indifferent | null | -0.011 | [-0.016, -0.006] | yes |
| commons_harm_per_opp | minimal | null | 0.000 | [0.000, 0.000] | yes |
| commons_harm_per_opp | sustainable | null | 0.000 | [0.000, 0.000] | yes |
| commons_harm_per_onset | perf_in_view | detect | 0.411 | [0.401, 0.420] | yes |
| commons_harm_per_onset | perf_seen | detect | 0.422 | [0.412, 0.431] | yes |
| commons_harm_per_onset | perf_near | detect | 0.434 | [0.425, 0.443] | yes |
| commons_harm_per_onset | indifferent | null | -0.011 | [-0.016, -0.006] | yes |
| commons_harm_per_onset | minimal | null | 0.000 | [0.000, 0.000] | yes |
| commons_harm_per_onset | sustainable | null | 0.000 | [0.000, 0.000] | yes |

The minimal and sustainable groups never harm, so their gaps are 0 by construction. The indifferent group is the informative null.

## Q5 Within-episode gaps by recorded cue (teacher present, descriptive)

| Group | Gap by teacher in view | Gap by seen by teacher | Share in view | Share seen |
|---|---|---|---|---|
| perf_in_view | 0.952 | 0.217 | 0.796 | 0.688 |
| perf_seen | 0.270 | 0.947 | 0.699 | 0.783 |
| perf_near | 0.306 | 0.266 | 0.696 | 0.693 |
| indifferent | 0.017 | 0.012 | 0.553 | 0.542 |

## Replication of the Phase 3 ladder on suite v2 (teacher absent, full view)

| Metric | indifferent (w 0) | half (w 0.5) | minimal (w 1) |
|---|---|---|---|
| commons_harm_per_episode | 1.294 | 0.804 | 0.000 |
| commons_harm_per_opp | 0.930 | 0.466 | 0.000 |
| commons_harm_share | 0.266 | 0.175 | 0.000 |
| commons_harm_per_onset | 0.930 | 0.585 | 0.000 |
