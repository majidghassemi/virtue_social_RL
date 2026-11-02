# Social dilemma check v2

8 units x 8 episodes per cell, 1000 steps, gamma 0.99. Criteria in `configs/dilemma_check_v2.yaml`.

| Map set | Agents | Defector | Value | C1 collective benefit | C2 temptation | C3 sucker's cost | Dilemma |
|---|---|---|---|---|---|---|---|
| ours | 3 | greedy | undiscounted | 117.28 [116.23, 118.29] holds | -31.12 [-31.79, -30.46] no | 0.82 [0.55, 1.08] holds | no |
| ours | 3 | greedy | discounted | 1.08 [0.89, 1.25] holds | 4.53 [4.36, 4.71] holds | 0.68 [0.46, 0.89] holds | yes |
| ours | 3 | minimal | undiscounted | 108.19 [107.15, 109.22] holds | -21.24 [-21.93, -20.56] no | 5.64 [5.36, 5.90] holds | no |
| ours | 3 | minimal | discounted | 4.77 [4.62, 4.92] holds | 1.58 [1.43, 1.75] holds | 1.15 [1.02, 1.29] holds | yes |
| ours | 5 | greedy | undiscounted | 102.59 [101.57, 103.61] holds | -10.08 [-10.65, -9.50] no | 1.07 [0.87, 1.29] holds | no |
| ours | 5 | greedy | discounted | -0.01 [-0.16, 0.15] no | 6.36 [6.21, 6.51] holds | 0.92 [0.75, 1.09] holds | no |
| ours | 5 | minimal | undiscounted | 95.60 [94.56, 96.62] holds | -1.80 [-2.45, -1.16] no | 3.42 [3.23, 3.62] holds | no |
| ours | 5 | minimal | discounted | 4.33 [4.17, 4.48] holds | 3.04 [2.91, 3.16] holds | 0.91 [0.81, 1.00] holds | yes |
| ours | 7 | greedy | undiscounted | 89.39 [88.26, 90.51] holds | -2.21 [-2.67, -1.73] no | 1.21 [1.08, 1.34] holds | no |
| ours | 7 | greedy | discounted | -1.03 [-1.19, -0.87] no | 7.01 [6.87, 7.16] holds | 1.06 [0.94, 1.17] holds | no |
| ours | 7 | minimal | undiscounted | 83.31 [82.17, 84.50] holds | 4.84 [4.29, 5.36] holds | 2.48 [2.29, 2.65] holds | yes |
| ours | 7 | minimal | discounted | 3.58 [3.43, 3.73] holds | 3.49 [3.36, 3.61] holds | 0.75 [0.67, 0.83] holds | yes |
| upstream | 3 | greedy | undiscounted | 284.64 [279.39, 290.06] holds | -80.67 [-84.23, -77.11] no | 2.12 [0.69, 3.59] holds | no |
| upstream | 3 | greedy | discounted | 9.88 [9.18, 10.67] holds | 2.28 [1.00, 3.46] holds | 1.64 [0.60, 2.65] holds | yes |
| upstream | 3 | minimal | undiscounted | 252.33 [245.95, 258.02] holds | -48.84 [-53.16, -45.31] no | 10.73 [9.17, 12.17] holds | no |
| upstream | 3 | minimal | discounted | 10.53 [9.42, 11.81] holds | 0.84 [-0.21, 1.96] no | 1.66 [0.77, 2.59] holds | no |
| upstream | 5 | greedy | undiscounted | 255.47 [250.80, 259.84] holds | -37.98 [-41.95, -34.50] no | 1.33 [0.62, 1.94] holds | no |
| upstream | 5 | greedy | discounted | 8.73 [8.01, 9.42] holds | 5.30 [4.64, 5.95] holds | 1.07 [0.58, 1.52] holds | yes |
| upstream | 5 | minimal | undiscounted | 227.70 [223.14, 231.69] holds | -8.59 [-12.58, -5.19] no | 6.66 [4.84, 8.05] holds | no |
| upstream | 5 | minimal | discounted | 10.36 [9.74, 10.94] holds | 3.78 [3.11, 4.45] holds | 1.11 [0.35, 1.81] holds | yes |
| upstream | 7 | greedy | undiscounted | 234.62 [229.55, 239.25] holds | -21.22 [-24.00, -18.19] no | 0.75 [0.27, 1.33] holds | no |
| upstream | 7 | greedy | discounted | 7.52 [6.84, 8.16] holds | 6.59 [5.71, 7.37] holds | 0.63 [0.24, 1.10] holds | yes |
| upstream | 7 | minimal | undiscounted | 209.52 [204.25, 214.31] holds | 4.72 [1.59, 7.94] holds | 4.02 [3.36, 4.55] holds | yes |
| upstream | 7 | minimal | discounted | 9.36 [8.89, 9.83] holds | 5.18 [4.58, 5.69] holds | 0.84 [0.32, 1.32] holds | yes |

C1 in total value of all agents, C2 and C3 in value of the switching agent (apples, discounted or not).
