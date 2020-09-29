import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, StandardScaler, VectorAssembler}
import org.apache.spark.ml.classification.{RandomForestClassifier}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrameStatFunctions

object Churn_Classification{
  def main(args: Array[String]) {
val rootLogger = Logger.getRootLogger()
rootLogger.setLevel(Level.ERROR)
Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
Logger.getLogger("org.spark-project").setLevel(Level.ERROR)
val spark = SparkSession.builder.master("yarn").appName("Churn_Classification_Qts11").getOrCreate()

println("Read the CSV file into spark Dataframe to perform data frame related operations. Keep the actual header and schema.")

val dfchurn = spark.read.format("csv").option("header", "true").option("inferSchema", "true").option("mode", "PERMISSIVE").load("/user/username/Churn_Classification/WA_Fn-UseC_-Telco-Customer-Churn.csv")

println("Print the top 5 rows of the data frame (without truncate) to get some initial understanding of the data.")
dfchurn.show(5, false)
println("Get the total counts of rows.")
println(dfchurn.count)
println("Print the schema of the data in tree format.")
dfchurn.printSchema

println("Compute basic statistics for numeric columns - count, mean, standard deviation, min and max.")

println("Try to describe the columns and check how statistics are shown for numerical andcategorical columns.")
println("Describe numeric columns:")
dfchurn.describe("tenure","MonthlyCharges","TotalCharges").show()

println("Describe non-numeric columns:")
dfchurn.describe("customerID","gender","SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
"DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","Churn").show()

println("Desribe TotalCharges after removing space")
val dfchurnclean = dfchurn.withColumn("TotalCharges", when(col("TotalCharges") === " ", null).otherwise(col("TotalCharges")))
dfchurnclean.describe("TotalCharges").show()

println("Drop null values from TotalCharges after removing space and describe, print schema")
val dfchurndropnull = dfchurnclean.na.drop("all", Seq("TotalCharges"))
dfchurndropnull.describe()
dfchurndropnull.describe("TotalCharges").show()
dfchurndropnull.printSchema

println("Convert TotalCharges to double and descibe, print schema")
val dfchurntoDouble = dfchurndropnull.selectExpr("customerID","gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
"DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","cast(TotalCharges as double) as TotalCharges","Churn")
dfchurntoDouble.describe("TotalCharges").show()
dfchurntoDouble.printSchema

val dfchurnCat= dfchurntoDouble.withColumn("OnlineSecurity", when(col("OnlineSecurity") === "No internet service", "No").otherwise(col("OnlineSecurity")))
                          .withColumn("OnlineBackup", when(col("OnlineBackup") === "No internet service", "No").otherwise(col("OnlineBackup")))
						  .withColumn("DeviceProtection", when(col("DeviceProtection") === "No internet service", "No").otherwise(col("DeviceProtection")))
						  .withColumn("TechSupport", when(col("TechSupport") === "No internet service", "No").otherwise(col("TechSupport")))
						  .withColumn("StreamingTV", when(col("StreamingTV") === "No internet service", "No").otherwise(col("StreamingTV")))
						  .withColumn("StreamingMovies", when(col("StreamingMovies") === "No internet service", "No").otherwise(col("StreamingMovies")))

println("Find out the total number of customers churned and not churned.")
import spark.implicits._
dfchurnCat.createOrReplaceTempView("churn")
spark.sql(""" 
select Churn, count(*) as Churn_Count from churn group by Churn 
""").show(false); 

println("Show individual distribution in cross tabs.")
dfchurnCat.stat.crosstab("Churn", "gender").show
dfchurnCat.stat.crosstab("Churn", "SeniorCitizen").show
dfchurnCat.stat.crosstab("Churn", "Dependents").show
dfchurnCat.stat.crosstab("Churn", "InternetService").show
dfchurnCat.stat.crosstab("Churn", "OnlineSecurity").show
dfchurnCat.stat.crosstab("Churn", "TechSupport").show
dfchurnCat.stat.crosstab("Churn", "Contract").show
dfchurnCat.stat.crosstab("Churn", "PaperlessBilling").show

val dfChurnYes =  dfchurnCat.filter("Churn = 'Yes'").selectExpr("MonthlyCharges","TotalCharges")
val dfChurnNo =  dfchurnCat.filter("Churn = 'No'").selectExpr("MonthlyCharges","TotalCharges")
println("Basic statistics for churned customers")
dfChurnYes.describe("MonthlyCharges","TotalCharges").show()
println("Basic statistics for not churned customers")
dfChurnNo.describe("MonthlyCharges","TotalCharges").show()

println("Find out how “MonthlyCharges” and “TotalCharges” are correlated.")
println(dfchurnCat.stat.corr("MonthlyCharges", "TotalCharges", "pearson"))

//Select the categorical column contract, string index and one hot encode this column. Execute the string indexing and one-hot encoding using a pipeline.
val lblindxr = new StringIndexer().setInputCol("Contract").setOutputCol("Contract_idx")
val ohe = new OneHotEncoder().setInputCol("Contract_idx").setOutputCol("Contract_ohe")
val stages = Array(lblindxr,ohe)
val pipeline = new Pipeline().setStages(stages)
val inx_ohe_model =  pipeline.fit(dfchurnCat)
val dfChurnTrnfmd = inx_ohe_model.transform(dfchurnCat)

//Select the numerical column "TotalCharges" and scale the values using standard scaling.
val vecassTC = new VectorAssembler().setInputCols(Array("TotalCharges")).setOutputCol("TotalCharges_vec")
val dfTCVec = vecassTC.transform(dfChurnTrnfmd)
val stdscaler = new StandardScaler().setInputCol("TotalCharges_vec").setOutputCol("TotalCharges_Scld")
val dfChurnTS = stdscaler.fit(dfTCVec).transform(dfTCVec)

//Feature and label
val vecass = new VectorAssembler().setInputCols(Array("MonthlyCharges", "TotalCharges_Scld")).setOutputCol("features")
val dfwithfeatures = vecass.transform(dfChurnTS)
val churnindxr = new StringIndexer().setInputCol("Churn").setOutputCol("label")
val dfChurnFL = churnindxr.fit(dfwithfeatures).transform(dfwithfeatures)

// Split the data frame into a training set and test set with a 70:30 ratio
val seed = 1123
val Array(train, test) = dfChurnFL.randomSplit(Array(0.7, 0.3), seed)

// train Random Forest model with training data set
val rfclassifier = new RandomForestClassifier()
  .setImpurity("gini")
  .setMaxDepth(10)
  .setNumTrees(50)
  .setFeatureSubsetStrategy("auto")
  .setSeed(seed)

val randomForestModel = rfclassifier.fit(train)
val predictions = randomForestModel.transform(test)

import spark.implicits._
val selectPrediction = predictions.select("label", "features", "rawPrediction","prediction")
   

// Compute other performence metrices
val predictionAndLabels = predictions
.selectExpr("cast(prediction as double) prediction", "cast(label as double) label")
.rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))

val lp = predictions.select("label", "prediction")
val counttotal = predictions.count()
val correct = lp.filter($"label" === $"prediction").count()
val wrong = lp.filter(not($"label" === $"prediction")).count()
val truep = lp.filter($"prediction" === 0.0).filter($"label" === $"prediction").count()
val truen = lp.filter($"prediction" === 1.0).filter($"label" === $"prediction").count()
val falsep = lp.filter($"prediction" === 1.0).filter(not($"label" === $"prediction")).count()
val falsen = lp.filter($"prediction" === 0.0).filter(not($"label" === $"prediction")).count()

lp.show()
println("  ") 
println("Printing the Confusion Matrix")
println("=============================")
println("true positive: " + truep)
println("false positive: " + falsep)
println("true negative: " + truen)
println("false negative: " + falsen)
println("=============================")

val evaluatorMC = new MulticlassClassificationEvaluator().setMetricName("accuracy").setPredictionCol("prediction").setLabelCol("label")
val accuracy = evaluatorMC.evaluate(predictions)
println("  ") 
println("Classification accuracy: " + accuracy) 

  }
}
