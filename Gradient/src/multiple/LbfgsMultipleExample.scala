package multiple
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
//import org.apache.spark.mllib.linalg.Vectors
//import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{LBFGS, LogisticGradient, SquaredL2Updater}
import org.apache.spark.mllib.util.MLUtils
import java.lang.System.currentTimeMillis
import org.apache.spark.mllib.regression.{LabeledPoint=>LP}
//import org.apache.spark.ml.feature.LabeledPoint

object LbfgsMultipleExample  {
  def main(args : Array[String]){
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.apache.eclipse.jetty.server").setLevel(Level.OFF)
    val conf =new SparkConf().setAppName("WordCountScala").set("spark.driver.maxResultSize","500g").setMaster("local[*]");
    val sc = new SparkContext(conf);
    def score2label(x:Double):Double={
      if (x < 0.5) 0
      else 1
    }
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
    println(s"Number of Label: $numClass")

    val tic1 = currentTimeMillis();
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(numClass)
      .run(train_data)


    // Compute raw scores on the test set.
    val scoreAndLabels = test_data.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }.cache()
//    scoreAndLabels.foreach(i => println(i))

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