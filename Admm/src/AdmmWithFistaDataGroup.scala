/*
 * ADMM of dada Grouping
 * min |(|𝐴𝑥−𝑏|)|_2^2+𝜆|(|𝑥|)|1  , 𝐴∈𝑅^(𝑚∗𝑛),
 * min 𝜆|x|1+∑_(𝑖=1)^𝑁|𝐴_𝑖 𝑥-b_i|
 *  s.t. 𝐴_𝑖 𝑥 −b_i=0,   𝑖=1,2,…, 𝑁, 𝑛_1+𝑛_2+…+𝑛_𝑁=𝑛
 */
import org.apache.spark.{SparkConf, SparkContext}
import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.rdd.RDD
import org.apache.log4j.{Level, Logger}
import scala.math._
import scala.util.control.Breaks._
object AdmmWithFistaDataGroup {
  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel {
      Level.WARN
    }
    Logger.getLogger("org.apache.eclipse.jetty.server").setLevel(Level.OFF)
    val conf = new SparkConf()
      .setAppName("ADMM")
      .setMaster("local[4]")
      .setJars(Seq(System.getenv("JARS")))
      .set("spark.driver.maxResultSize", "0g")
    val sc = new SparkContext(conf)
    def soft(x: Double, thld: Double): Double = {
      if (Math.abs(x) > thld) x - Math.signum(x) * thld;
      else 0;
    }
    def half(x: Double, thld: Double): Double = {
      val thildt = Math.pow(54.0, 1.0 / 3.0) / 4.0 * Math.pow(thld, 2.0 / 3.0);
      if (Math.abs(x) > thildt) {
        val phi = Math.acos(thld / 8.0 * Math.pow(Math.abs(x) / 3, -1.5));
        2.0 / 3.0 * x * (1 + Math.cos(2.0 / 3.0 * (Math.PI - phi)))
      } else 0;
    }

    def L2(x: Double, thld: Double):Double = {
      x/(1.0+thld)
    }

    val numPartitions = 3;
    val n = 300
    val A0 = sc.textFile("Ab.txt", numPartitions).map { line =>
      val total = line.split(",").map(_.toDouble);
      (total.apply(0), BDV(total.slice(1, n + 1)))

    }
    val A: RDD[BDV[Double]] = A0.map(f => f._2)
    val b = A0.map(f => f._1)

    val xT = BDV(sc.textFile("xT.txt", numPartitions).map(x=>x.toDouble).collect())
    val bNorm = b.map(f=>f.*(f)).reduce(_+_)

    val m = A.count().toInt
    println("n----" + n + "m----" + m)
    var relativeErr=0.0
    val lambda = 1.1
    val beta = 0.1
    val u = 0.001
    val maxit = 100
    val tol=1e-4
    val key = 1
    var tnew=1.0
    var flag: Boolean = false
    var x = BDV.zeros[Double](n)
    var p = b.map(f => 0.0)
    var y = b.map(f => 0.0)
    var z=x
    var i = 0
    while (i < maxit && !flag) {
      i += 1
      println("-----Iter-----" + i)
      var xold = x
      val pold = p
      val t=tnew
      val Ax =A.map(f=>f.dot(z)).cache()
      val Axy = Ax.zip(y).map(f=>f._1.-(f._2)).cache()
      p = pold.zip(Axy).map(f=>f._1.+(f._2.*(beta)))
      var sumbp: RDD[Double] = b.zip(p).map(f=>(f._1.+(f._2))./(beta))
      var tmp= Ax.zip(sumbp).map(f=>f._1.+(f._2))
      y = tmp.map(f=>f.*(beta)./(1.0+beta))
      y.cache()
      var res = Ax.zip(y).map(f=>f._1.-(f._2))
      var resp = res.zip(p).map(f=>f._1.+(f._2./(beta)))
      var grad: BDV[Double] = A.zip(resp).map(f => (f._1.map(fn => fn * (f._2)))).reduce(_ + _)
      var tmp2 = z.-(grad.map(fn => fn.*(u)))
      var thld = lambda*u/beta
//      x = tmp2.map(fn => soft(fn, thld))
      if (key == 1){
        x = tmp2.map(fn => soft(fn, thld))
      }
      if (key == 2){
        x = tmp2.map(fn => L2(fn, thld))
      }
      if (key == 1/2){
        x = tmp2.map(fn => half(fn, thld))
      }
      tnew = 0.5*(1.0+math.sqrt(1.0+4.0*t*t))
      z=x.+(x.-(xold))*(t-0.1)/tnew
      relativeErr =(x.-(xT)).map(f => f.*(f)).reduce(_ + _)
      if (relativeErr < tol) {
        break;
      }
    }
    val xTNorm = xT.map(f => f.*(f)).reduce(_ + _)
    val RealErr = math.sqrt(relativeErr/xTNorm)
    println("The algorithm iterations is "+i)
    println("we get the solution is x= " + x)
    println("The real error is "+RealErr)
    println("The relative error is "+ relativeErr)
  }

}