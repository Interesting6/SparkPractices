package covType

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession

object covType_Xgboost_CV {
  def main(args: Array[String]) {

    //屏蔽不必要的日志显示在终端上
//    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
//    Logger.getLogger("org.apache.eclipse.jetty.server").setLevel(Level.OFF)

    // 启动环境
    val spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("xbgoost")
      .getOrCreate()

    //  记录开始时间
    var beg = System.currentTimeMillis()

    // 读入数据
    val dataset0 = spark.read.option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ",")
      .csv("data/covtype/covtype.data.csv")

    // 查看当前数据前10行
    println("查看当前数据的前10行：")
    dataset0.limit(10).show()
    // 查看数据的列和类型
    println("查看数据的列和类型：")
    dataset0.printSchema()

    //    删除无用的列，此处为用户ID,去除缺失值
//    val dataset = dataset0.drop("user_id").na.drop

    //标签列
    var Labelcols = Array("Cover_Type")

    // 标签列重命名
    val dataset = dataset0.withColumnRenamed(Labelcols(0), "label")

    // 将标签列改为修改后的名字
    Labelcols = Array("label")

    // 获取类别列
    val categorical = dataset.dtypes.filter(_._2 == "StringType") map (_._1)
    //  从中去除掉标签列
    val categoricals = categorical.filter(!_.contains(Labelcols(0)))

    // 数值列
    val colArray_numerics = dataset.dtypes.filter(_._2 != "StringType") map (_._1)
    //  从中去除掉标签列
    val colArray_numeric = colArray_numerics.filter(!_.contains(Labelcols(0)))

    //.字符特征转换成数字索引编码，OneHot编码，注意setDropLast设置为false
    //  将待预测的label字符列转换成数字索引编码
//    val labelIndexer = new StringIndexer()
//      .setInputCol(Labelcols(0))
//      .setOutputCol("label")

    // 字符特征转换成数字索引编码
    val indexers = categoricals.map(
      c => new StringIndexer().setInputCol(c).setOutputCol(s"${c}_idx").setHandleInvalid("keep")
    )
    // OneHot编码，注意setDropLast设置为true
    val encoders = categoricals.map(
      c => new OneHotEncoderEstimator().setInputCols(Array(s"${c}_idx")).setOutputCols(Array(s"${c}_enc")).setDropLast(true)
    )


    // 字符转换后的新列
    val colArray_enc = categoricals.map(x => x + "_enc")

    // 最终的所有的特征列
    val final_colArray = (colArray_numeric ++ colArray_enc).filter(!_.contains(Labelcols(0)))

    //将字段组合成向量feature
    val vectorAssembler = new VectorAssembler().setInputCols(final_colArray).setOutputCol("features")

    // 创建一个XGBoost分类器,设置分类数为7
    val xgb = new XGBoostClassifier().setLabelCol("label").setFeaturesCol("features").setMissing(0).setNumClass(7)

    // XGBoost paramater grid
    val xgbParamGrid = new ParamGridBuilder()
      .addGrid(xgb.maxDepth, Array(4,8))
      .addGrid(xgb.minChildWeight, Array(0.1))
      .addGrid(xgb.gamma, Array(0.1))
      .addGrid(xgb.alpha, Array(0.0))
      .addGrid(xgb.lambda, Array(0.6))
      .addGrid(xgb.eta, Array(0.4))
      .addGrid(xgb.objective, Array("multi:softprob"))
//      .addGrid(xgb.numRound,Array(100))
      .build()

    // 创建一个xgboost的pipeline
    val pipeline = new Pipeline().setStages(indexers ++ encoders ++ Array(vectorAssembler,xgb))

    // 创建一个多分类的评估器
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("f1")

    //创建交叉验证的Pipeline，用 XGBoost作为estimator，用MulticlassClassificationEvaluator作为evaluator，用xgbParamGrid作为hyperparameters
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(xgbParamGrid)
      .setNumFolds(3)
      .setSeed(0)

    // 划分数据集
    val Array(trainingData, testData) = dataset.randomSplit(Array(0.8, 0.2), seed=0)

    // 训练模型
    val cvModel = cv.fit(trainingData)

    // 获取最好的模型
    val bestModel: PipelineModel = cvModel.bestModel match {
      case model: PipelineModel => model
      case _ => null
    }

    // 模型保存
    bestModel.write.overwrite().save("./saveModel/spark-xgboost-classifier-model-covType")

    // 模型读取
    val loadModel = PipelineModel.load ("./saveModel/spark-xgboost-classifier-model-covType")

    // 在测试集上做预测
    val predictions = loadModel.transform(testData)

    // 选择想要展示的预测结果，(label, prediction)
    println("选择想要展示的预测结果：")
    predictions.select("label", "prediction").show(5)

    // 调用相关的评估函数，计算评价指标
    val accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    val f1 = evaluator.setMetricName("f1").evaluate(predictions)
    val weightedPrecision  = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    val weightedRecall  = evaluator.setMetricName("weightedRecall").evaluate(predictions)

    // 输出评价指标
    println("输出评价指标：")
    println(s"accuracy = ${accuracy}")
    println(s"f1 = ${f1}")
    println(s"weightedPrecision = ${weightedPrecision}")
    println(s"weightedRecall = ${weightedRecall}")

    //结束时间
    var end = System.currentTimeMillis()

    //耗时时间
    var castTime = end - beg
    println("整个程序耗时: " + castTime / 1000 + "s")

  }
  }
