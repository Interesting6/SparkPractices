package binary

import java.lang.System.currentTimeMillis

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics,MulticlassMetrics}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/** -*- coding: utf-8 -*-
 *
 * @File SGDExample Example for solving logistic regression using splash SGD
 * @input training  : The training data in form of RDD[(Double, Vector)].
 * @input test      : The test data in form of RDD[LabeledPoint].
 * @input initialWeights : The initial weights in form of Vector.
 * @set_param setGradient : Set gradient for problem includes LeastSquareGradient,LogisticGradient,HingeGradient and so on.
 * @set_param setNumIterations : Set max of iteration number default 10.
 * @return weights : The solving weights after iteration.
 * @return splashGrad : The current gradient.
 * @return metrics    : The evaluation matrix on test dataset.
 * @Author: linasun
 * @Date : 2020/4/1
 *
 */



object SplashBinaryExample {
  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.apache.eclipse.jetty.server").setLevel(Level.OFF)
    val conf =new SparkConf().setAppName("WordCountScala").set("spark.driver.maxResultSize","500g").setMaster("local[*]");
    val sc = new SparkContext(conf);
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



    // Train a logistic regression model
    val tic1 = currentTimeMillis();
    val NumIterations = 100
    val initialWeightsWithIntercept = Vectors.dense(new Array[Double](numFeatures + 1))
    val train_data_ = train_data.map{ x => (x.label, MLUtils.appendBias(x.features)) }

    val weightsWithIntercept = (new  splash.optimization.StochasticGradientDescent())
      .setGradient(new  splash.optimization.LogisticGradient())
      .setNumIterations(NumIterations)
      .setPrintDebugInfo(true)
      .optimize(train_data_, initialWeightsWithIntercept)


    val model = new LogisticRegressionModel(
      Vectors.dense(weightsWithIntercept.toArray.slice(0, weightsWithIntercept.size - 1)),
      weightsWithIntercept(weightsWithIntercept.size - 1))

    // Clear the default threshold.
    model.clearThreshold()

    def score2label(x:Double):Double={
      if (x < 0.5) 0
      else 1
    }
    // Compute raw scores on the test set.
    val scorePredAndLabels = test_data.map { point =>
      val score = model.predict(point.features)
      (score, score2label(score), point.label)
    }.cache()


    // Get evaluation metrics.
    val scoreAndLabels = scorePredAndLabels.map { x => (x._1, x._3) }
    val metrics2 = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics2.areaUnderROC()
    val auPR = metrics2.areaUnderPR()
    println(s"Area under ROC = $auROC")
    println(s"Area under PR = $auPR")

    // Get evaluation metrics.
    val predAndLabels = scorePredAndLabels.map { x => (x._2, x._3) }
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