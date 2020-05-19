import java.lang.System.currentTimeMillis
import org.apache.log4j.{ Level, Logger }
import breeze.linalg.{ DenseVector => BDV };
import org.apache.spark.{ SparkConf, SparkContext }

object RegressionExample {
  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.apache.eclipse.jetty.server").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("TestMyFista").setMaster("local[4]")
    //      .setSparkHome(System.getenv("SPARK_HOME"));
    val spark = new SparkContext(conf);
    val input_data_path="E:\\Datasets\\data\\pm.txt"
    //    val ground_truth_x_path="E:\\data\\xT3000.txt"
    val data2 = spark.textFile(input_data_path); //load data [b,A] by rows
    val spliter = ','; // spliter :space
    val n = data2.first().split(spliter).map(_.toDouble).length - 1;
    val m = data2.count().toInt;
    val data = data2.map { line =>
    {
      val total = line.split(spliter).map(_.toDouble);
      (total.apply(0), BDV(total.slice(1, n + 1)))
    }}
    val split = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val train = split(0).cache()
    val test = split(1).cache()
    val initialx = BDV.zeros[Double](n) //x0
    val xT = initialx
    //    val xT = BDV(spark.textFile(ground_truth_x_path, 1).map(x=>x.toDouble).collect())
    val maxi = 100;
    val key = 2;
    val lambda = 1e-3
    val scalar = 2;
    val step =1e-6
    val prox = new proximal(data,initialx);
    prox.set_opts(alpha=0.01,maxit=maxi,step=step)

    val tic0 = currentTimeMillis()
    val (x, stepsize) = prox.run();
    val tic1 = currentTimeMillis();
//    val predict = test.map(f=>if (f._2.dot(x)>0.5)1.0 else 0.0)
    val predict = test.map(f=> f._2.dot(x) )
    val err = predict.zip(test).map(f=>math.abs(f._1.-(f._2._1))).reduce(_+_)
    println("The real error is " + err)
    val test_label = test.map { point => point._1 }
    val label_Norm = test_label.map(f => f.*(f)).reduce(_ + _)
    val relativeErr = err / label_Norm
    println("The relative error is "+relativeErr)

    println("The recovery signal  is:" + x)
    println("the size of data is :"+m+"(dimension) *  "+n+"(number of data)");
    println("-------------------- start at:"+tic0);
    println("-----------------------end at:"+tic1);
    println("----use----- "+maxi+" iterators to recovery the signal and Spark spend totally:"+(tic1-tic0)/1000.0+" seconds")
    spark.stop();

  }
}