### Table 8c — Training, inner-validation and held-out test E_Π (means over folds and seeds)

Training: the four training cases of each fold. Inner validation: the mean leave-one-out error of the selected configuration during tuning; it is used for model selection and is NaN for untuned models. Test: the held-out case. The last column is the exact-reproducibility check from refitting.

| model            |   E_Pi_train |   E_Pi_inner_val |   E_Pi_test |   max |test refit − recorded| |
|:-----------------|-------------:|-----------------:|------------:|------------------------------:|
| ssg_published    |       0.4274 |         nan      |      0.4322 |                        0.0000 |
| ssg_refit        |       0.1675 |         nan      |      0.2003 |                        0.0000 |
| lrr_published    |       0.4257 |         nan      |      0.4350 |                        0.0000 |
| lrr_refit        |       0.2549 |         nan      |      0.2990 |                        0.0000 |
| lrr_gl_published |       0.3808 |         nan      |      0.3949 |                        0.0000 |
| lrr_gl_refit     |       0.2482 |         nan      |      0.2982 |                        0.0000 |
| ridge            |       0.0812 |           0.2222 |      0.1514 |                        0.0000 |
| rf               |       0.0067 |           0.1068 |      0.0922 |                        0.0000 |
| lgbm             |       0.0053 |           0.0954 |      0.0841 |                        0.0000 |
| mlp              |       0.0145 |           0.0711 |      0.0869 |                        0.0000 |
