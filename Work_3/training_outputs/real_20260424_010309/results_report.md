# Results Summary

Brief note: training results for one saved iSFS logistic run on the real dataset.

## Run Setup

| Item | Value |
|---|---|
| Dataset | `real` |
| Outer max iter | 10 |
| Block A max iter | 40 |
| Alpha/Beta subproblem max iter | 25 / 25 |
| Learning rate alpha | 0.01 |
| Learning rate beta | 0.01 |
| Lambda beta | 0.01 |
| Alpha l1 radius | 1.0 |
| Decision threshold | 0.5 |
| Initial cost vector | `0.5,0.5` |
| Baseline cost vector | `0.5,0.5` |
| Final learned cost vector | `[0.7650, 0.9240]` |

## Overall Performance

| Metric | Value |
|---|---:|
| Samples | 500 |
| Valid predictions | 500 |
| TP | 179 |
| TN | 190 |
| FP | 75 |
| FN | 56 |
| Accuracy | 0.7380 |
| Precision | 0.7047 |
| Recall / Sensitivity | 0.7617 |
| Specificity | 0.7170 |
| F1 score | 0.7321 |
| Balanced accuracy | 0.7393 |
| Negative predictive value | 0.7724 |
| False positive rate | 0.2830 |
| False negative rate | 0.2383 |
| MCC | 0.4779 |
| Brier score | 0.1755 |
| Log loss | 0.5244 |
| Probability range | `0.0161 to 0.9878` |

## BCD Summary

| Recorded iterations | Accepted cost steps | Final cost vector |
|---:|---:|---|
| 4 | 3 | `[0.7650, 0.9240]` |

| Iter | Accepted | Pred. increase | Actual increase | Ratio | Radius | Current objective | Trial objective |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 1 | True | 0.0801 | 0.1315 | 1.6418 | 0.0200 | 0.3018 | 0.4333 |
| 2 | True | 0.1403 | 0.0412 | 0.2935 | 0.0400 | 0.4333 | 0.4745 |
| 3 | True | 0.0115 | 0.0107 | 0.9310 | 0.0400 | 0.4745 | 0.4852 |
| 4 | False | 0.0000 | 0.0000 | -inf | 0.0800 | 0.4852 | 0.4852 |

## Learned Alpha Weights

| Profile | source1 | source2 | source3 |
|---|---:|---:|---:|
| source1 | 0.1937 |  |  |
| source1, source2 | 0.1466 | 0.6848 |  |
| source1, source2, source3 | 0.0641 | 0.0000 | 0.9359 |
| source1, source3 | 0.0752 |  | 0.9248 |
| source2 |  | 0.8908 |  |
| source2, source3 |  | 0.0263 | 0.9737 |
| source3 |  |  | 1.0000 |

## Beta Summary

| Source | Features | L1 norm | L2 norm | Nonzero coeffs |
|---|---:|---:|---:|---:|
| source1 | 10 | 0.4716 | 0.2755 | 7 |
| source2 | 8 | 0.2890 | 0.1620 | 4 |
| source3 | 25 | 2.0169 | 0.6417 | 14 |

## Exact Profile Results

| Code | Samples | Accuracy | Precision | Recall | Specificity | F1 | BCE | Cost-BCE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 001 | 27 | 0.7407 | 0.6667 | 0.7273 | 0.7500 | 0.6957 | 0.5149 | 0.4263 |
| 010 | 27 | 0.6667 | 0.6250 | 0.7692 | 0.5714 | 0.6897 | 0.6685 | 0.5622 |
| 011 | 61 | 0.7705 | 0.8065 | 0.7576 | 0.7857 | 0.7812 | 0.5001 | 0.4290 |
| 100 | 41 | 0.6585 | 0.6087 | 0.7368 | 0.5909 | 0.6667 | 0.6013 | 0.5004 |
| 101 | 56 | 0.7143 | 0.6786 | 0.7308 | 0.7000 | 0.7037 | 0.5375 | 0.4492 |
| 110 | 75 | 0.6800 | 0.6098 | 0.7576 | 0.6190 | 0.6757 | 0.5904 | 0.4896 |
| 111 | 213 | 0.7793 | 0.7573 | 0.7800 | 0.7788 | 0.7685 | 0.4728 | 0.3966 |

## Optimization Group Results

| Code | Samples | Accuracy | Precision | Recall | Specificity | F1 | BCE | Cost-BCE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 001 | 357 | 0.7703 | 0.7558 | 0.7647 | 0.7754 | 0.7602 | 0.4955 | 0.4171 |
| 010 | 376 | 0.6356 | 0.6105 | 0.6480 | 0.6244 | 0.6287 | 0.6436 | 0.5399 |
| 011 | 274 | 0.7774 | 0.7727 | 0.7669 | 0.7872 | 0.7698 | 0.4825 | 0.4073 |
| 100 | 385 | 0.6390 | 0.5882 | 0.7303 | 0.5604 | 0.6516 | 0.6336 | 0.5271 |
| 101 | 269 | 0.7695 | 0.7462 | 0.7698 | 0.7692 | 0.7578 | 0.4862 | 0.4074 |
| 110 | 288 | 0.6840 | 0.6280 | 0.7744 | 0.6065 | 0.6936 | 0.6092 | 0.5069 |
| 111 | 213 | 0.7793 | 0.7573 | 0.7800 | 0.7788 | 0.7685 | 0.4728 | 0.3966 |
