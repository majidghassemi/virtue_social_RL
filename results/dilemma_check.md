# Social dilemma check

8 units x 8 episodes per cell, 1000 steps. Criteria in `configs/dilemma_check.yaml`.

| Map set | Agents | C1 collective benefit | C2 temptation | C3 sucker's cost | Dilemma |
|---|---|---|---|---|---|
| ours | 3 | 117.28 [116.23, 118.29] (holds, 100% of maps) | -31.12 [-31.79, -30.46] (no, 8% of maps) | 0.82 [0.55, 1.08] (holds, 83% of maps) | no |
| ours | 5 | 102.59 [101.51, 103.57] (holds, 100% of maps) | -10.08 [-10.63, -9.51] (no, 8% of maps) | 1.07 [0.86, 1.27] (holds, 100% of maps) | no |
| ours | 7 | 89.39 [88.28, 90.53] (holds, 92% of maps) | -2.21 [-2.66, -1.76] (no, 25% of maps) | 1.21 [1.08, 1.34] (holds, 100% of maps) | no |
| upstream | 3 | 284.64 [279.53, 289.94] (holds, 100% of maps) | -80.67 [-84.44, -77.01] (no, 0% of maps) | 2.12 [0.70, 3.50] (holds, 100% of maps) | no |
| upstream | 5 | 255.47 [250.62, 259.78] (holds, 100% of maps) | -37.98 [-41.88, -34.69] (no, 0% of maps) | 1.33 [0.62, 1.94] (holds, 100% of maps) | no |
| upstream | 7 | 234.62 [229.61, 239.12] (holds, 100% of maps) | -21.22 [-24.02, -18.00] (no, 0% of maps) | 0.75 [0.25, 1.34] (holds, 100% of maps) | no |

Units: C1 in total apples per episode (all agents), C2 and C3 in apples per episode for the switching agent.

| Map set | Agents | Population | Total apples | Switcher apples | Mean stock | Regrown |
|---|---|---|---|---|---|---|
| ours | 3 | all_greedy | 34.3 | 10.67 | 0.71 | 0.91 |
| ours | 3 | all_minimal | 43.4 | 13.45 | 7.47 | 16.47 |
| ours | 3 | all_sustainable | 151.6 | 49.65 | 13.88 | 130.37 |
| ours | 3 | greedy_among_sustainable | 36.7 | 18.53 | 1.02 | 3.35 |
| ours | 3 | sustainable_among_greedy | 35.0 | 9.86 | 0.75 | 1.64 |
| ours | 5 | all_greedy | 34.2 | 6.40 | 0.55 | 0.84 |
| ours | 5 | all_minimal | 41.2 | 7.76 | 7.14 | 14.07 |
| ours | 5 | all_sustainable | 136.8 | 26.55 | 13.12 | 114.83 |
| ours | 5 | greedy_among_sustainable | 36.9 | 16.47 | 1.07 | 3.65 |
| ours | 5 | sustainable_among_greedy | 34.3 | 5.33 | 0.56 | 0.97 |
| ours | 7 | all_greedy | 34.0 | 4.67 | 0.47 | 0.60 |
| ours | 7 | all_minimal | 40.1 | 5.37 | 6.91 | 12.64 |
| ours | 7 | all_sustainable | 123.4 | 17.51 | 12.66 | 101.13 |
| ours | 7 | greedy_among_sustainable | 36.9 | 15.30 | 1.19 | 3.78 |
| ours | 7 | sustainable_among_greedy | 34.0 | 3.45 | 0.48 | 0.66 |
| upstream | 3 | all_greedy | 66.7 | 24.97 | 1.52 | 2.64 |
| upstream | 3 | all_minimal | 99.0 | 35.14 | 7.56 | 40.11 |
| upstream | 3 | all_sustainable | 351.4 | 116.36 | 21.66 | 304.39 |
| upstream | 3 | greedy_among_sustainable | 75.3 | 35.69 | 2.08 | 11.19 |
| upstream | 3 | sustainable_among_greedy | 69.5 | 22.84 | 1.63 | 5.44 |
| upstream | 5 | all_greedy | 66.6 | 18.02 | 1.18 | 2.50 |
| upstream | 5 | all_minimal | 94.3 | 23.03 | 6.96 | 34.95 |
| upstream | 5 | all_sustainable | 322.0 | 68.38 | 19.45 | 272.02 |
| upstream | 5 | greedy_among_sustainable | 75.1 | 30.39 | 1.78 | 10.89 |
| upstream | 5 | sustainable_among_greedy | 67.3 | 16.69 | 1.21 | 3.16 |
| upstream | 7 | all_greedy | 66.2 | 14.47 | 1.02 | 2.16 |
| upstream | 7 | all_minimal | 91.3 | 17.20 | 6.49 | 31.73 |
| upstream | 7 | all_sustainable | 300.9 | 49.30 | 18.58 | 250.22 |
| upstream | 7 | greedy_among_sustainable | 74.9 | 28.08 | 1.85 | 10.84 |
| upstream | 7 | sustainable_among_greedy | 66.8 | 13.72 | 1.03 | 2.66 |
