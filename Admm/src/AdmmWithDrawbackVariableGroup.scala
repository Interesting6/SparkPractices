/*
 * ADMM of variable Grouping
 * min |(|𝐴𝑥−𝑏|)|_2^2+𝜆|(|𝑥|)|1  , 𝐴∈𝑅^(𝑚∗𝑛),
 * min 𝜆∑_(𝑖=1)^𝑁(A𝑥_𝑖 |)|_1+||𝑦−𝑏||_2^2 〗
 *  ∑_(𝑖=1)^𝑁 𝐴_𝑖 𝑥_𝑖 −𝑦=0,   𝑖=1,2,…, 𝑁, 𝑛_1+𝑛_2+…+𝑛_𝑁=𝑛
 */
import org.apache.spark.{SparkConf, SparkContext}
import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.rdd.RDD
import org.apache.log4j.{Level, Logger}
import scala.math._
import scala.util.control.Breaks._

object AdmmWithDrawbackVariableGroup {
  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel {
      Level.WARN
    }
    Logger.getLogger("org.apache.eclipse.jetty.server").setLevel(Level.OFF)
    val conf = new SparkConf()
      .setAppName("ADMM")
      .setMaster("local[4]")
      .setJars(Seq(System.getenv("JARS")))
      .set("spark.driver.maxResultSize", "1g")
      .set("spark.driver.extraJavaOptions","-Xss10m")
    val sc = new SparkContext(conf)

    def L2(x: Double, thld: Double):Double = {
      x/(1.0+thld)
    }
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

    val numPartitions = 3;
    val AT = sc.textFile("ATNorm.txt", numPartitions).map { line =>
      val total = line.split(",").map(_.toDouble);
      BDV(total)

    }.cache()
    val xT = BDV(sc.textFile("xTNorm.txt", numPartitions).map(x=>x.toDouble).collect())
    val n = AT.count().toInt
    val m=AT.first().toArray.length
    var x = AT.map(f=>0.0)
    val b = BDV(sc.textFile("bNorm.txt").map(f=>f.toDouble).collect())
    val bNorm=b.dot(b)
    var relativeErr=0.0
    val lambda = 1e-6
    val beta = 1.5
    var u = 0.1
    val maxit = 80
    val tol=1e-6
    var tnew=1.0
    val bufferNum=20
    val scala=2.0
    var flag: Boolean = false
    var p =BDV.zeros[Double](m)
    var y =BDV.zeros[Double](m)
    var z=x
    var i = 0
    breakable {
      while (i < maxit && !flag) {
        i += 1
        println("-----Iter-----" + i)
        val t = tnew
        val xold = x.cache()

        var pold = p


        val Ax = AT.zip(z).map(f => (f._1.map(fn => fn * (f._2)))).reduce(_ + _)

        val Axy = Ax.-(y)
        p = pold.+(Axy.*(beta))

        var sumbp = (b + p)./(beta)
        var tmp = Ax.+(sumbp)
        y = tmp.map(f => f.*(beta)./(1.0 + beta))
        val resp = Ax.-(y).+(p./(beta))
        val grad = AT.map(f => f.dot(resp))
        var tmp2 = z.zip(grad).map(f => f._1.-(f._2.*(u)))
        var thld = lambda * u / beta
        x = tmp2.map(fn => soft(fn, thld))
        x.cache()

        val resX = AT.zip(x).map(f => (f._1.map(fn => fn * (f._2)))).reduce(_ + _) .-(b)
        val resZ = AT.zip(z).map(f => (f._1.map(fn => fn * (f._2)))).reduce(_ + _) .-(b)
        val gradZ = AT.map(f => f.dot(resZ))
        var bosh = 0
        breakable {
          while (true) {

            if (bosh > bufferNum) {
              flag = true
              break
            }
            bosh += 1;
            val h1 = resX.dot(resX)
            val h2 = resZ.dot(resZ)
            val z_x = z.zip(x).map(f=>f._1.-(f._2))
            val h3 = z_x.zip(gradZ).map(f=>f._1.*(f._2)).reduce(_+_)*2.0
            val h4 = z_x.map(f=>f.*(f)).reduce(_+_)* (1.0 / (2.0 * u))
            if (h1 < h2 + h3 + h4) {
              break
            }
            u = u / scala
            if (flag) {
              break
            }
            if (i % 10 == 0) {
              u = 2.0 * u
            }
          }
        }
        tnew = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        z = x.zip(xold).map(f => f._1.+((f._1.-(f._2)).*(t - 1.0)./(tnew)) )
        z.cache()
      }
    }
    val resXNorm = (BDV(x.collect()).-(xT)).map(f => f.*(f)).reduce(_ + _)
    val xTNorm = xT.map(f => f.*(f)).reduce(_ + _)
    val RealErr = math.sqrt(resXNorm/xTNorm)
    println("The algorithm iterations is "+i)
    println("we get the solution is x= " + BDV(x.collect()))
    println("The real error is "+RealErr)
    println("The relative error is "+ relativeErr)



  }
}