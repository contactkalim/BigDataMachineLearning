import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{CountVectorizer,StringIndexer,RegexTokenizer,StopWordsRemover, HashingTF,IDF, RFormula}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, BinaryLogisticRegressionSummary}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.Row

object Tweet_Classification{
  def main(args: Array[String]) {
val rootLogger = Logger.getRootLogger()
rootLogger.setLevel(Level.ERROR)
Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
Logger.getLogger("org.spark-project").setLevel(Level.ERROR)
val spark = SparkSession.builder.master("yarn").appName("Tweet_Classification_Qts11").getOrCreate()

println("Read the CSV file into spark Dataframe to perform data frame related operations. Keep the actual header and schema.")

val dftweet = spark.read.format("csv").option("header", "true").option("inferSchema", "true").option("mode", "PERMISSIVE").load("/user/username/Tweet_Classification/train_E6oV3lV.csv")

println("Print the top 5 rows of the data frame (without truncate) to get some initial understanding of the data.")
dftweet.show(5, false)
println("Get the total counts of rows.")
println(dftweet.count)
println("Print the schema of the data in tree format.")
dftweet.printSchema

println("Find out the distinct values of the label.")
import spark.implicits._
dftweet.createOrReplaceTempView("tweetstrain")
spark.sql(""" 
select distinct label as Distinct_Labels from tweetstrain 
""").show(false); 

println("Find out the count of instances for each type of label value.")
spark.sql(""" 
select label, count(*) as label_Count from tweetstrain group by label 
""").show(false); 

println("Data Preprocessing and Cleaning and Feature Engineering - Clean the tweet text, Tokenizing, Stop words removal")
val dftweetlower=dftweet.withColumn("tweet", lower(col("tweet")));
val dftweets_nop = dftweetlower.withColumn("tweet_noPunct", regexp_replace(col("tweet"), "\\pP", ""))


val regexTokenizer = new RegexTokenizer().setInputCol("tweet_noPunct").setOutputCol("tweetsclean").setPattern("\\W")
val sremover = new StopWordsRemover().setInputCol("tweetsclean").setOutputCol("tweets_nostop")

val Array(train, test) = dftweets_nop.randomSplit(Array(0.7, 0.3))
println("Building Model 1 using TF-IDF:")
val tf = new HashingTF().setInputCol("tweets_nostop").setOutputCol("tweets_tf").setNumFeatures(400)
val idf = new IDF().setInputCol("tweets_tf").setOutputCol("tweets_tfidf").setMinDocFreq(2)
val rForm = new RFormula()
val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features")

val stages = Array(regexTokenizer,sremover,tf, idf, rForm, lr)
val pipelineLR = new Pipeline().setStages(stages)
val params = new ParamGridBuilder()
			.addGrid(rForm.formula, Array("label ~ tweets_tfidf"))
			.addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
			.addGrid(lr.regParam, Array(0.1, 0.2))
			.build()
val evaluatorLR = new BinaryClassificationEvaluator().setMetricName("areaUnderROC").setRawPredictionCol("prediction").setLabelCol("label")
val crossval = new CrossValidator().setEstimatorParamMaps(params).setEstimator(pipelineLR).setEvaluator(evaluatorLR).setNumFolds(3)
val crossvalFitted = crossval.fit(train)
val predictions = crossvalFitted.transform(test)
import spark.implicits._
val selectPrediction = predictions.select("label", "features", "rawPrediction","prediction")
//selectPrediction.show(10)

val accuracym1 = evaluatorLR.evaluate(predictions)
println("Classification accuracy: " + accuracym1)    

// Compute other performence metrices
val predictionAndLabels = predictions
.selectExpr("cast(prediction as double) prediction", "cast(label as double) label")
.rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))

val metrics = new BinaryClassificationMetrics(predictionAndLabels)
val metricsMM = new MulticlassMetrics(predictionAndLabels)

val areaUnderPR = metrics.areaUnderPR
println("Area under the precision-recall curve: " + areaUnderPR)

val areaUnderROC = metrics.areaUnderROC
println("Area under the receiver operating characteristic (ROC) curve: " + areaUnderROC)
val lp = predictions.select("label", "prediction")
val counttotal = predictions.count()
val correct = lp.filter($"label" === $"prediction").count()
val wrong = lp.filter(not($"label" === $"prediction")).count()
val truep = lp.filter($"prediction" === 0.0).filter($"label" === $"prediction").count()
val truen = lp.filter($"prediction" === 1.0).filter($"label" === $"prediction").count()
val falsep = lp.filter($"prediction" === 1.0).filter(not($"label" === $"prediction")).count()
val falsen = lp.filter($"prediction" === 0.0).filter(not($"label" === $"prediction")).count()

val labels = metricsMM.labels
labels.foreach { l =>
  println(s"F1-Score($l) = " + metricsMM.fMeasure(l))
}

println("Total Count: " + counttotal)
println("Correct: " + correct)
println("Wrong: " + wrong)

println("Printing the Confusion Matrix")
println("=============================")
println("true positive: " + truep)
println("false positive: " + falsep)
println("true negative: " + truen)
println("false negative: " + falsen)
println("=============================")

println("Building Model 2 using Count Vectorizer:")
val countvec = new CountVectorizer().setInputCol("tweets_nostop").setOutputCol("tweets_cvec").setMinTF(1).setMinDF(2).setVocabSize(5000)
val stagescv = Array(regexTokenizer,sremover,countvec, rForm, lr)
val pipelineLRcv = new Pipeline().setStages(stagescv)
val paramscv = new ParamGridBuilder()
			.addGrid(rForm.formula, Array("label ~ tweets_cvec"))
			.addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
			.addGrid(lr.regParam, Array(0.1, 0.2))
			.build()
val evaluatorLRCV = new BinaryClassificationEvaluator().setMetricName("areaUnderROC").setRawPredictionCol("prediction").setLabelCol("label")
val crossvalcv = new CrossValidator().setEstimatorParamMaps(paramscv).setEstimator(pipelineLRcv).setEvaluator(evaluatorLRCV).setNumFolds(3)
val crossvalFittedcv = crossvalcv.fit(train)
val predictionscv = crossvalFittedcv.transform(test)
import spark.implicits._
val selectPredictioncv = predictionscv.select("label", "features", "rawPrediction","prediction")
//selectPrediction.show(10)

val accuracym2 = evaluatorLRCV.evaluate(predictionscv)
println("Classification accuracy: " + accuracym2)
if(accuracym1 > accuracym2)
{
	println("Model 1 with TF-IDF accuracy is better")
}
else
{
	println("Model 2 with Count Vectorizer accuracy is better")
}

  }
}
