import org.apache.spark.sql.DataFrame
import breeze.linalg.{*, DenseVector => BDV}
import org.apache.spark.rdd.RDD

import scala.util.control.Breaks.breakable
import java.util.Date

import scala.math.sqrt

/** -*- coding: utf-8 -*-
 *
 * @File :Proximal gradient method
 * @setparam ADataFrame    The feature of data in form of DataFrame
 * @setparam bDataFrame    The label of data in form of DataFrame
 * @param InitialWeight The initial weight in form of Vector
 * @set_param alpha : Line search increase parameter.
 * @set_param theta : Current Lipschitz estimate.
 * @set_param L     : The initial Lipschitz estimate.
 * @set_param maxit : The maximum number of iterations to run the optimization algorithm.
 * @set_param step  : The step size in iteration default 1e-3
 * @set_param key   : Proximal operation ,key can be 1,2,1/2
 * @return iter : The current iteration number.
 * @return relativeErr: The relative error
 * @return time : The total time after iteration.
 * @return x    : The solving weights after iteration.
 * @Author: linasun
 * @Date : 2020/4/8
 * @see : On accelerated proximal gradient methods for convex-concave optimization,2008.
 * @see : Templates for convex cone problems with applications to sparse signal recovery,2010.
 *      min |(|𝐴𝑥−𝑏|)|_2^2+𝜆|(|𝑥|)|1  , 𝐴∈𝑅^(𝑚∗𝑛),
 */
class proximal(data: RDD[(Double, BDV[Double])], InitialWeight: BDV[Double]) extends java.io.Serializable {

  private var flag: Boolean = false;
  protected var alpha: Double = 0.9;
  private var maxit: Int = 100; //max iterations
  protected var step: Double = 1e-3; // step size
  private var key: Int = 1; // 1 for L1，1/2 for L_half，and 2 for L2
  private var tol: Double = 1e-5; //tolerance parameter
  private var theta: Double = Double.PositiveInfinity; // scale of step size for back
  private var L: Double = 1;
  //Main variable
  protected var x0 = InitialWeight: BDV[Double]
  protected var x = x0
  protected var z = x
  protected var xnew = x
  protected var znew = z
  protected var y = x0

  def set_opts(alpha: Double = 0.9, maxit: Int = 100,
               step: Double = 1e-3, key: Int = 1, tol: Double = 1e-5, theta: Double = Double.PositiveInfinity) {
    this.alpha = alpha
    this.maxit = maxit
    this.step = step
    this.key = key
    this.tol = tol
    this.theta = theta
  }

//  def compute_resid_primal(): RDD[Double] = {
//    val residual = new Residual()
//    val bRes = residual.compute(distMatrixA, xnew, b).cache()
//    bRes
//  }
//
//  def compute_multiply(): (RDD[Double], RDD[Double], RDD[Double], RDD[Double]) = {
//    val Ax = distMatrixA.multiplyVector(x).cache()
//    val Axnew = distMatrixA.multiplyVector(xnew).cache()
//    val Az = distMatrixA.multiplyVector(z).cache()
//    val Aznew = distMatrixA.multiplyVector(znew).cache()
//    (Ax, Axnew, Az, Aznew)
//  }
//
//
//  def update_z(tmp: BDV[Double],key:Int): BDV[Double] = {
//    val prox = new Proximity()
//    if (key ==1 ){
//      znew = tmp.map(fn => prox.soft(fn,  step))
//    }
//    if (key == 1/2){
//      znew = tmp.map(fn => prox.half(fn,  step))
//    }
//    if (key == 2){
//      znew = tmp.map(fn => prox.L2(fn, step))
//    }
//    znew
//  }
//
//  def update_y(tmp1: BDV[Double], tmp2: BDV[Double]): BDV[Double] = {
//    val ynew = tmp1 * (1 - theta) + tmp2 * theta
//    ynew
//  }
//
//  def gradient(ay: RDD[Double]): BDV[Double] = {
//    val gradLS = new LeastSquaresGradient()
//    val grad = gradLS.compute(distMatrixA, b, y)
//    grad
//  }
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

  def run(): (BDV[Double], Double) = {
    var i = 0
    var relativeErr = 0.0
    val residual = data.map{line=>
      line._2.dot(xnew) - line._1
    }
    var cntrAy = 0
    var cntrAx = 0
    val cntrReset = 50
    var cntrTol = 0
    var Ax = data.map{line=>line._2.dot(x)}
    var Axnew =data.map{line=>line._2.dot(xnew)}
    var Az =data.map{line=>line._2.dot(z)}
    var Aznew =data.map{line=>line._2.dot(znew)}
    val iterStartTime = (new Date).getTime
    breakable {
      while (i < maxit && !flag) {
        i += 1
        val L_old = L
        L = L * alpha
        val theta_old = theta
        x = xnew
        z = znew
        Ax = Axnew
        Az = Aznew
        var isBackTracking = true
        while (isBackTracking) {
          theta = 2.0 / (1.0 + math.sqrt(1.0 + 4.0 * (L / L_old) / (theta_old * theta_old)))
          y= x * (1 - theta) + z * theta
          val a_y = if (cntrAy >= cntrReset) {
            cntrAy = 0
            data.map{line=>
              line._2.dot(y)
            }
          } else {
            cntrAy = cntrAy + 1
            Ax.zip(Az).map(f=>f._1*(1.0-theta)+f._2*theta)
          }
//          val grad = gradient(a_y)
          val ay_b =  data.map{line=>line._2.dot(y).-(line._1)}
          val grad = (data.zip(ay_b)).map(line => {                //grad=AT(Ax-b)
            line._1._2 * line._2
          }).reduce(_ + _);
          var desGrad = z.-(grad.map(fn => fn.*(step)))
//          znew = update_z(desGrad,key)
          znew = desGrad.map(a => soft(a, step))

          Aznew = data.map{line=>line._2.dot(znew)}

          xnew = (1.0 - theta) * x + theta * znew
          Axnew = if (cntrAx >= cntrReset) {
            cntrAx = 0
//            distMatrixA.multiplyVector(xnew)
            data.map{line=>line._2.dot(xnew)}
          } else {
            cntrAx = cntrAx + 1
//            localVector.axpy(Ax, Aznew, 1.0 - theta, theta)
            Ax.zip(Aznew).map(f=>f._1*(1.0-theta)+f._2*theta)
          }
          isBackTracking = false
          val tmp = grad.dot(xnew - x)
          if (tmp > 0) {
            znew = xnew
            Aznew = Axnew
          }
        }
//        relativeErr = localVector.norm2(compute_resid_primal())
        val mol =sqrt(data.map{p=>{
          (p._2.dot(z)-p._1)*(p._2.dot(z)-p._1)
        }}.reduce(_+_))
        val den =sqrt(data.map(x=>{
          x._1.*(x._1)
        }).reduce(_+_))
        val relativeErr=mol/den
        var timeElapsed = ((new Date).getTime - iterStartTime).toDouble / 1000
        println("Iter Number =%d; Loss = %5.8f; Time = %5.3f;  ".format(i, relativeErr, timeElapsed))
      }
    }
    (xnew, relativeErr)
  }
}
