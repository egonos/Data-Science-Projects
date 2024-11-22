# What I have learned?

1. Filling the missing values with summary statistics can boost the performance of the tree algorithms.
2. Always write codes with a small part of the data. When you see everything works fine, then run the code on the whole dataset.
3. Autogluon can be useful especially when it runs for a long time with GPU.
4. Labels can be encoded and oof prodicted probabilities can be used for meta learner.
5. Optuna LGBM integration makes the optimization procedure much more easier.
6. Stick with Python or Scikit Learn API from the beginning.
7. Consider the models having the best CV and ensemble the predictions. Usually this is the way to win the competition.  
8. Use `tree_method = 'hist'` to fasten up XGBoost trees. It makes the XGB tree similart to LGB but still they are different from each other. The perfomrance could be lower, this could be solved by increasing `max_bin` parameter.


## Things to Keep In Mind

**LGBM trees**

1. GOSS
> Samples based on the magnitude of the gradient
>
> **Upside:** Highly accurate and fast on large datasets.
>
> **Downside:** Can be unstable
2. DART
> Gradient Boosting with Dropout
>
> **Upside:** Reduces overfitting
>
> **Downside:** Painfully slow. Not optimal for small datasets.

3. GBDT
> Classical Gradient Boosting algorithm
>
> **Upside:** High accuracy
>
> **Downside:** Slow on the big datasets
4. RF
> Random forests
>
> **Upside:** Stable predictions (good for noisy data)
>
> **Downside:** Predictions may not be that accurate.

