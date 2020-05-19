本人的*Spark*练习代码。

主要是**分布式优化**及**分布式机器学习**。


* `Admm`、`Fista`、`ProximalAlgorithm` 为分布式优化求解齐次线性方程组算法。

* `Gradient`： 基于`RDD`，优化logistics regression模型，使用的库为`spark.mllib`

* `Classify`与`Boosting`： 基于`DataFrame`，使用的为`spark.ml`
