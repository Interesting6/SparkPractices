package binary

import java.lang.System.currentTimeMillis
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{LBFGS, LogisticGradient, SquaredL2Updater}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.{LabeledPoint => LP}

/**
 * -*- coding: utf-8 -*-
 *
 * @File : LbfgsBinaryExample
 * @set_param training : The training data in form of LibSVM
 * @set_param test: The label of data in form of LibSVM
 * @set_param InitialWeight : The initial weight in form of Vector
 * @return weightsWithIntercept : The solving weights after iteration.
 * @return model : LogisticRegressionModel after training.
 * @return metrixs  : The matrix for evaluations on test data.
 * @Author: Yummy Chen
 * @Date : 2020/5/3
 * @Desc : The binary classification
 */
object LbfgsBinaryExample  {
  def main(args : Array[String]){
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.apache.eclipse.jetty.server").setLevel(Level.OFF)
    val conf =new SparkConf().setAppName("WordCountScala").set("spark.driver.maxResultSize","500g").setMaster("local[*]");
    val sc = new SparkContext(conf);
    def score2label(x:Double):Double={
      if (x < 0.5) 0
      else 1
    }
    val train_path =  "E:\\Datasets\\work3\\data\\gisette"
    val train_data = MLUtils.loadLibSVMFile(sc, train_path)
    val test_path = "E:\\Datasets\\work3\\data\\gisette.t"
    val test_data = MLUtils.loadLibSVMFile(sc, test_path)
    val train_m = train_data.count().toDouble
    val test_m = test_data.count().toDouble
    println(s"Number of training samples: $train_m, Number of testing samples: $test_m")

    val firstElem = train_data.take(1)(0)
    val numFeatures = firstElem.features.size
    println(s"Number of Features: $numFeatures")

    val tic1 = currentTimeMillis();

    val numCorrections = 10
    val convergenceTol = 1e-5
    val maxNumIterations = 100
    val regParam = 0.1

    val initialWeightsWithIntercept = Vectors.dense(new Array[Double](numFeatures + 1))
    val train_data_ = train_data.map{ x => (x.label, MLUtils.appendBias(x.features)) }

    val (weightsWithIntercept, loss) = LBFGS.runLBFGS(train_data_,
      new LogisticGradient(),
      new SquaredL2Updater(),
      numCorrections,
      convergenceTol,
      maxNumIterations,
      regParam,
      initialWeightsWithIntercept)

    println("Loss of each step in training process")
    loss.foreach(println)
    println()

    val model = new LogisticRegressionModel(
      Vectors.dense(weightsWithIntercept.toArray.slice(0, weightsWithIntercept.size - 1)), // features的权重
      weightsWithIntercept(weightsWithIntercept.size - 1)) // 这个是bias的权重

    // Clear the default threshold.
    model.clearThreshold()

    // Compute raw scores on the test set.
    val scorePredAndLabels = test_data.map { point =>
      val score = model.predict(point.features)
      (score, score2label(score), point.label)
    }.cache()

    // 自己写的计算准确率
//    val correct = scorePredAndLabels.filter{ x => x._2 == x._3 }.count()
//    val test_m = test.count().toDouble
//    val accuracy = correct / test_m
//    println(s"Accuracy is $accuracy")

    // Get evaluation metrics.
    val scoreAndLabels = scorePredAndLabels.map { x => (x._1, x._3) }
//    scoreAndLabels.foreach(i => println(i))
    val metrics2 = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics2.areaUnderROC()
    val auPR = metrics2.areaUnderPR()
    println(s"Area under ROC = $auROC")
    println(s"Area under PR = $auPR")

    val predAndLabels = scorePredAndLabels.map { x => (x._2, x._3) }
    // Get evaluation metrics.
    val metrics = new MulticlassMetrics(predAndLabels)
    val acc = metrics.accuracy
    val recall = metrics.weightedRecall
    val precision = metrics.weightedPrecision
    val f1Score = metrics.weightedFMeasure
    println(s"The accuracy is $acc")
    println(s"recall = $recall")
    println(s"precision = $precision")
    println(s"f1 = $f1Score")

    val tic2 = currentTimeMillis();
    println("Runing time is : "+(tic2-tic1)/1000.0)
  }
}
