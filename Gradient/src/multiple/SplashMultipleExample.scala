package multiple
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.log4j.{Level, Logger}
import java.lang.System.currentTimeMillis

import org.apache.spark.mllib.optimization.{GradientDescent, LBFGS, LogisticGradient, SquaredL2Updater}


object SplashMultipleExample {
  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.apache.eclipse.jetty.server").setLevel(Level.OFF)
    val path = "Eacoustic.t"
    val conf = new SparkConf().setAppName("logistic regression").setMaster("local[4]")
    val sc = new SparkContext(conf)

//    val train_path =  "E:\\Datasets\\work3\\data\\seismic"
//    val train_data = MLUtils.loadLibSVMFile(sc, train_path).cache()
//    val test_path = "E:\\Datasets\\work3\\data\\seismic.t"
//    val test_data = MLUtils.loadLibSVMFile(sc, test_path)
//    val train_m = train_data.count().toDouble
//    val test_m = test_data.count().toDouble
//    println(s"Number of training samples: $train_m, Number of testing samples: $test_m")
//    val firstElem = train_data.take(1)(0)
//    val numFeatures = firstElem.features.size
//    println(s"Number of Features: $numFeatures")

    val numFeatures = 784
    val train_path =  "E:\\Datasets\\work3\\data\\mnist.scale"
    val train_data = MLUtils.loadLibSVMFile(sc, train_path, numFeatures)
    val test_path = "E:\\Datasets\\work3\\data\\mnist.t"
    val test_data = MLUtils.loadLibSVMFile(sc, test_path, numFeatures)
    val train_m = train_data.count().toDouble
    val test_m = test_data.count().toDouble
    println(s"Number of training samples: $train_m, Number of testing samples: $test_m")
    println(s"Number of Features: $numFeatures")

    val numClass = train_data.map { x => x.label }.distinct().count().toInt
    println(s"Number of Class: $numClass")

    // Train a logistic regression model
    val NumIterations = 1

    val tic1 = currentTimeMillis();


    val train_data_ = train_data.map{ x => (x.label, MLUtils.appendBias(x.features)) }
    val initialWeightsWithIntercept = Vectors.dense(new Array[Double]( (numFeatures + 1)*(numClass-1)) )

//    println("init weight "+initialWeightsWithIntercept)
//    val weightsWithIntercept = (new splash.optimization.StochasticGradientDescent())
//      .setGradient(new splash.optimization.MultiClassLogisticGradient(numClass))
//      .setPrintDebugInfo(true)
//      .setNumIterations(NumIterations)
//      .optimize(train_data_, initialWeightsWithIntercept)


    val numCorrections = 10
    val convergenceTol = 1e-5
    val maxNumIterations = 200
    val regParam = 0.001  // 正则参数
    val miniBatchFraction = 0.01  // 设置SGD每次迭代的数据采样比，若其小于1，则因convergenceTol不稳定
    val stepSize = 10.0

    val (weightsWithIntercept, loss) = GradientDescent.runMiniBatchSGD(
      data = train_data_,
      gradient = new LogisticGradient(numClass),
      updater = new SquaredL2Updater,
      stepSize = stepSize,
      numIterations = maxNumIterations,
      regParam = regParam,
      miniBatchFraction = miniBatchFraction,
      initialWeights = initialWeightsWithIntercept,
      convergenceTol = convergenceTol
    )
    println("Loss of each step in training process")
    loss.foreach(println)
    println()

//    val (weightsWithIntercept, loss) = LBFGS.runLBFGS(train_data_,
//      new LogisticGradient(numClass),
//      new SquaredL2Updater(),
//      numCorrections,
//      convergenceTol,
//      maxNumIterations,
//      regParam,
//      initialWeightsWithIntercept)
//    val weightsWithIntercept = initialWeightsWithIntercept

    val weightsSizeWithIntercept = weightsWithIntercept.size  // numFeatures*numClass + numClass
    println(s"weights with intercept size is $weightsSizeWithIntercept")

    val model = new LogisticRegressionModel(
      Vectors.dense(weightsWithIntercept.toArray.slice(0, weightsWithIntercept.size)), // numFeatures*numClass+numClass-numFeatures-1
      0, // 未用到
      numFeatures, numClass)

    // Clear the default threshold.
    model.clearThreshold()
    train_data.unpersist()


    // Compute raw scores on the test set.
    val scoreAndLabels = test_data.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    // Get evaluation metrics.
    val metrics = new MulticlassMetrics(scoreAndLabels)
    val acc = metrics.accuracy
    val recall = metrics.weightedRecall
    val precision = metrics.weightedPrecision
    val f1Score = metrics.weightedFMeasure
    println("The accuracy is "+acc)
    println(s"recall = $recall")
    println(s"precision = $precision")
    println(s"f1 = $f1Score")

    val tic2 = currentTimeMillis();
    println("Runing time is : "+(tic2-tic1)/1000.0)

  }
}