import scala.util.control.Breaks._;
import org.apache.spark.internal.Logging
import scala.math._;
import org.apache.spark.rdd.RDD;
import breeze.linalg.{ DenseVector => BDV };
class FISTA {
  private var lambda: Double = 2.8e-4; //regular parameter
  private var maxit: Int = 500; //max iterations
  private var step: Double = 1; // the step size
  private var key: Int = 1; // 1 for L1 and 1/2 for L_half
  private var scalar: Double = 2.0; // scalar for drawBack of step size

  // newly update
  private var bufferNum: Int = 20;
  def setBufferNum(bufferNum: Int): this.type = {
    this.bufferNum = bufferNum
    this
  }

  def setMaxit(maxit: Int): this.type = {
    this.maxit = maxit
    this
  }

  def setLambda(lambda: Double): this.type = {
    this.lambda = lambda
    this
  }

  def setStep(step: Double): this.type = {
    this.step = step
    this
  }

  def setKey(key: Int): this.type = {
    this.key = key
    this
  }

  def setScalar(scalar: Double): this.type = {
    this.scalar = scalar
    this
  }
  def optimize(
                data: RDD[(Double, BDV[Double])], //[b,A]
                xTrue: BDV[Double],
                initialx: BDV[Double]// initial value of x
              ): (BDV[Double], Double) = {
    val (x, stepsize) = FISTA.runFista(data, xTrue ,initialx, lambda, maxit, step, key, scalar, bufferNum);
    (x, stepsize)
  }
}

object FISTA extends Logging {

  def soft(x: Double, thld: Double): Double = {
    // if (Math.abs(x) > thld) x - Math.signum(x) * thld;
    if (Math.abs(x) > thld) (Math.abs(x)-thld)*Math.signum(x)
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
  def formatByRow(data: RDD[(Double, BDV[Double])]): RDD[(Double, BDV[Double])] = {
    val data_squareSumByRow = data.map(x => {
      val row = BDV(x._2.toArray);
      (x, Math.sqrt(row.dot(row)))
    }).map(y => {
      val formatRow = BDV(y._1._2.toArray) / y._2;
      (y._1._1 / y._2, formatRow);
    })
    data_squareSumByRow
  }

  def runFista(
                  data: RDD[(Double, BDV[Double])], //[b,A]
                  xTrue:BDV[Double],
                  initialx: BDV[Double], // initial value of x
                  lambda: Double,
                  maxit: Int,
                  step: Double,
                  key: Int,
                  scalar: Double,
                  bufferNum: Int): (BDV[Double], Double) =
  {
    var z = initialx
    var x = z;
    var tnew = 1.0;
    var iter: Int = 0;
    var stepsize = step;
    var breezedata: RDD[(Double, BDV[Double])] = data.map(x => (x._1, BDV(x._2.toArray))).cache()
    val xT = BDV(xTrue.toArray);
    var endflag = false;

      while (iter < maxit ) {
        iter = iter + 1;
        val res_z = breezedata.map{line=>
          line._2.dot(z) - line._1
        }
        val temp = breezedata zip res_z;
        val grad = temp.map(line => {                //grad=AT(Ax-b)
          line._1._2 * line._2
        }).reduce(_ + _);
        var t = tnew;
        var xold = x;
        var sumi = 0;
        breakable {
          do {
            val tempx = z - stepsize * grad;        //x=x-step*grad
            val thld = lambda * stepsize;

            if (key == 1) {
              x = tempx.map(a => soft(a, thld));
            } else if (key == 1/2) {
              x = tempx.map(a => half(a, thld));
            } else if (key==2) {
              x = tempx.map(a => L2(a, thld));
            }
            val res_x = breezedata.map{line => {
              line._2.dot(x) - line._1
            }};
            val localRes_x = BDV(res_x.collect())
            val localRes_z = BDV(res_z.collect())
            val h1 = localRes_x.dot(localRes_x);
            val h2 = localRes_z.dot(localRes_z);
            val bias = x - z;
            val h3 = 2 * (bias).dot(grad);
            val h4 = (bias).dot(bias) / (2*stepsize);
            if (h1 < h2 + h3 + h4) { break; }
            stepsize = stepsize / scalar;

            sumi+=1;
            if(sumi==bufferNum) endflag = true;

          } while (true && sumi < bufferNum)
        }
        if ((iter % 10) == 0){ stepsize = stepsize * 2;}
        tnew = 0.5 * (1 + sqrt(1 + 4 * t * t));
        z = x + (t - 1) / tnew * (x - xold);
        println("----------------------iter"+iter)
      }

    val error=sqrt((xT-z).dot(xT-z)/(xT.dot(xT)))
    val mol =sqrt(breezedata.map{p=>{
      (p._2.dot(z)-p._1)*(p._2.dot(z)-p._1)
    }}.reduce(_+_))
    val den =sqrt(breezedata.map(x=>{
      x._1.*(x._1)
    }).reduce(_+_))
    val err=mol/den
    (z, stepsize)
  }

}