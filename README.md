# Auto-sklearn Fail

The code in `fail.py` runs
[auto-sklearn](https://automl.github.io/auto-sklearn/master/) on a dataset for 5
minutes. The model it finds after that time is *worse* than just using a random
forest with default hyperparameters.

Find out what's going on, why auto-sklearn's performance is so bad, and how to
fix it. If you cannot run auto-sklearn yourself, you can use [this
colab](https://colab.research.google.com/drive/1goMxbe5P9TA9V6qzpA-RB01TmPiZB8AC?usp=sharing)
as a starting point.

## Submission

Add your report detailing your findings and code to this repository. Be specific
when you explain why auto-sklearn fails here -- what does auto-sklearn do that
results in bad performance, why is this a problem on this particular dataset,
and what can you do to fix it?
