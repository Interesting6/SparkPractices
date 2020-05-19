import java.lang.System.currentTimeMillis
import org.apache.log4j.{ Level, Logger }
import breeze.linalg.{ DenseVector => BDV };
import org.apache.spark.{ SparkConf, SparkContext }

object ProximalExample {
  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.apache.eclipse.jetty.server").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("TestMyFista").setMaster("local[4]")
    //      .setSparkHome(System.getenv("SPARK_HOME"));
    val spark = new SparkContext(conf);
    val input_data_path="E:\\data\\data3000.txt"
    val ground_truth_x_path="E:\\data\\xT3000.txt"
    val data2 = spark.textFile(input_data_path); //load data [b,A] by rows
    val spliter = ','; // spliter :space
    val n = data2.first().split(spliter).map(_.toDouble).length - 1;
    val m = data2.count().toInt;
    val data = data2.map { line =>
    {
      val total = line.split(spliter).map(_.toDouble);
      (total.apply(0), BDV(total.slice(1, n + 1)))
    }}

    data.cache;
    val xT = BDV(spark.textFile(ground_truth_x_path, 1).map(x=>x.toDouble).collect())
    val maxi = 100;
    val key = 2;
    val alpha = 0.9
    val scalar = 2;
    val step =0.1
    val initialx = BDV.zeros[Double](n) //x0
    val prox = new proximal(data,initialx);
    prox.set_opts(alpha=0.9,step=1e-3)
    val tic0 = currentTimeMillis()
    val (x, stepsize) = prox.run();
    val tic1 = currentTimeMillis();
    val realErr = (x.-(xT)).map(f=>f.*(f)).reduce(_+_)/xT.map(f=>f.*(f)).reduce(_+_)
    println("The real error is "+realErr)
    val xTNorm = xT.map(f => f.*(f)).reduce(_ + _)
    val relativeErr = realErr/xTNorm
    println("The relative error is "+ relativeErr)

    println("The recovery signal  is:" + x)
    println("The true x is "+xT)
    println("the size of data is :"+m+"(dimension) *  "+n+"(number of data)");
    println("-------------------- start at:"+tic0);
    println("-----------------------end at:"+tic1);
    println("----use----- "+maxi+" iterators to recovery the signal and Spark spend totally:"+(tic1-tic0)/1000.0+" seconds")
    spark.stop();

  }
}