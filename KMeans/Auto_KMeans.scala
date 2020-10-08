import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, StandardScaler, VectorAssembler}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrameStatFunctions

object Auto_KMeans{
  def main(args: Array[String]) {
val rootLogger = Logger.getRootLogger()
rootLogger.setLevel(Level.ERROR)
Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
Logger.getLogger("org.spark-project").setLevel(Level.ERROR)
val spark = SparkSession.builder.master("yarn").appName("Auto_mpeg").getOrCreate()

println("Data Exploration:")
println("Read the CSV file into spark Dataframe to perform data frame related operations. Keep the actual header and schema.")

val dfauto = spark.read.format("csv").option("header", "true").option("inferSchema", "true").option("mode", "PERMISSIVE").load("/user/edureka_766323/auto-mpg.csv")

println("Print the top 5 rows of the data frame to get some initial understanding of the data.")
dfauto.show(5, false)
println("Get the total counts of rows.")
println(dfauto.count)
println("Print the schema of the data in tree format.")
dfauto.printSchema

println("Compute basic statistics for numeric columns - count, mean, standard deviation, min and max.")

println("Describe numeric columns:")
dfauto.describe("mpg","cylinders","displacement","horsepower","weight","acceleration","model year","origin").show()

println("Describe non-numeric columns:")
dfauto.describe("car name").show()

println("Data Cleaning:")
println("Find out the type of feature 'HORSEPOWER'")
println(dfauto.dtypes(3))
dfauto.describe("horsepower").show()

val dfautoclean = dfauto.withColumn("horsepower", when(col("horsepower") === "?", null).otherwise(col("horsepower")))
val dfautotoDouble = dfautoclean.selectExpr("mpg","cylinders","displacement","cast(horsepower as double) as horsepower","weight","acceleration","`model year`","origin","`car name`")
dfautotoDouble.describe("horsepower").show()
println("Now show how many (count) null values are there in this column:")
println(dfautotoDouble.filter("horsepower is null").count)

//As imputer is not working in Spark 2.1.0 version in Cloudlab (exception - error: object Imputer is not a member of package org.apache.spark.ml.feature), i have used avg and fillna
//import org.apache.spark.ml.feature.Imputer
//val imputer = new Imputer().setInputCol("horsepower").setOutputCol("horsepower_imputed").setStrategy("mean")
//val dfautoimp = imputer.fit(dfautotoDouble).transform(dfautotoDouble)
val meanhp = dfautotoDouble.select(avg("horsepower")).collect().map(x => x.get(0)).mkString.toDouble
val dfautoimp = dfautotoDouble.na.fill(meanhp)

val vaAuto= new VectorAssembler().setInputCols(Array("mpg","cylinders","displacement","horsepower","weight","acceleration","model year")).setOutputCol("features_va")
val dfautovec = vaAuto.transform(dfautoimp)
val stdscaler = new StandardScaler().setInputCol("features_va").setOutputCol("features")
val dfautostd = stdscaler.fit(dfautovec).transform(dfautovec)
val kmauto = new KMeans().setK(4).setFeaturesCol("features").setPredictionCol("prediction")
val kmautomodel = kmauto.fit(dfautostd)
val autoSummary = kmautomodel.summary
autoSummary.clusterSizes
val WSSSE = kmautomodel.computeCost(dfautostd)
println(s"Within Set Sum of Squared Errors = $WSSSE")

println("Get the prediction from the k-means model as cluster value.")
val predictDf = kmautomodel.transform(dfautostd)
predictDf.show(20)

println("Find out the count of data points in each cluster.")
predictDf.groupBy("prediction").count().show()

println("Find out the profile of the clusters. To do so, group the data points based on the cluster number and show the mean of each feature.")
predictDf.groupBy("prediction").avg("mpg","cylinders","displacement","horsepower","weight","acceleration","model year").show(false)

println("Cluster centers:")
kmautomodel.clusterCenters.foreach(println)

  }
}

/*
//From above analysis we can see that clusters 0 consists of cars with aroud 6 cylinders and cluster 2 consists of cars of 8 cylinders on avg, cluster 0 is suggestive of SUVs
//and cluster 2 is suggestive of sports cars. Cluster 1 and 3 however have similar mean features with avg 4 cylinders suggestive of economy Sedans.
//From above analysis its suggestive of 3 actual cluaters. We should  perform K-means again with K=3.

Find out the profile of the clusters. To do so, group the data points based on the cluster number and show the mean of each feature.
+----------+------------------+-----------------+------------------+------------------+------------------+------------------+-----------------+
|prediction|avg(mpg)          |avg(cylinders)   |avg(displacement) |avg(horsepower)   |avg(weight)       |avg(acceleration) |avg(model year)  |
+----------+------------------+-----------------+------------------+------------------+------------------+------------------+-----------------+
|1         |32.72719298245613 |4.052631578947368|112.02631578947368|76.41997851772287 |2320.3859649122805|16.60087719298246 |80.07894736842105|
|3         |25.055999999999997|4.01             |110.125           |83.20469387755102 |2331.98           |16.461000000000002|73.59            |
|2         |14.429787234042553|8.0              |350.0425531914894 |162.39361702127658|4157.978723404255 |12.576595744680853|73.46808510638297|
|0         |19.621111111111112|6.177777777777778|225.51111111111112|103.12743764172335|3262.8555555555554|16.392222222222223|76.2             |
+----------+------------------+-----------------+------------------+------------------+------------------+------------------+-----------------+*/
