package binary

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.sql.SparkSession

object LogisticRegressionWithElasticNetExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local[4]")
      .appName("LogisticRegressionWithElasticNetExample")
      .getOrCreate()

    // $example on$
    // Load training data
    val training = spark.read.format("libsvm").load("E:\\Datasets\\work3\\data\\gisette")
    val Array(trainingData, testData) = training.randomSplit(Array(0.8, 0.2), seed=0)
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit the model
    val lrModel = lr.fit(training)

    // Print the coefficients and intercept for logistic regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // We can also use the multinomial family for binary classification
    val mlr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setFamily("multinomial")

    val mlrModel = mlr.fit(trainingData)
    val predictions = mlrModel.transform(testData)
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("prediction").setMetricName("areaUnderROC")


    // 选择想要展示的预测结果，(label, prediction)
    println("选择想要展示的预测结果：")
    predictions.select("label", "prediction").show(5)
    // 调用相关的评估函数，计算评价指标
    val areaUnderROC = evaluator.setMetricName("areaUnderROC").evaluate(predictions)
    val areaUnderPR = evaluator.setMetricName("areaUnderPR").evaluate(predictions)


    val evaluator1 = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction")
    val accuracy = evaluator1.setMetricName("accuracy").evaluate(predictions)
    val f1 = evaluator1.setMetricName("f1").evaluate(predictions)
    val weightedPrecision  = evaluator1.setMetricName("weightedPrecision").evaluate(predictions)
    val weightedRecall  = evaluator1.setMetricName("weightedRecall").evaluate(predictions)

    // 输出评价指标
    println("输出评价指标：")
    println(s"accuracy = ${accuracy}")
    println(s"f1 = ${f1}")
    println(s"weightedPrecision = ${weightedPrecision}")
    println(s"weightedRecall = ${weightedRecall}")
    println(s"areaUnderROC = ${areaUnderROC}")
    println(s"areaUnderPR = ${areaUnderPR}")
    // Print the coefficients and intercepts for logistic regression with multinomial family
    println(s"Multinomial coefficients: ${mlrModel.coefficientMatrix}")
    println(s"Multinomial intercepts: ${mlrModel.interceptVector}")
    // $example off$

    spark.stop()
  }
}
