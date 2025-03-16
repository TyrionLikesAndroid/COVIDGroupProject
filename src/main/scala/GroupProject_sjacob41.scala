package com.cs7265.homework

import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.{DecisionTree, GradientBoostedTrees, RandomForest}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object GroupProject_sjacob41 {

  val spark: SparkSession = SparkSession.builder().appName("CS7265 Group Project")
    .master("local").getOrCreate()

  def main(args: Array[String]): Unit = {

    println("")
    println("*****************************************")
    println("***  Group Project - Stephen Jacobs    ***")
    println("***  KSU CS7265 - Big Data Analytics  ***")
    println("*****************************************")
    println("")

    // Determine the working directory so we understand where to put the text file
    println("Working Directory: " + System.getProperty("user.dir"))

    spark.sparkContext.setLogLevel("ERROR")

    // Load the csv file and return a data frame with its contents.  Grab and print the row count
    val csvFile = spark.read.option("header","true").option("inferSchema", "true").csv("covid_dataset_brazil.csv")
    println("covid_dataset_brazil row count: " + csvFile.count())

    // Load the dataframe and see how the format looks
    //csvFile.take(10).foreach(println)

    // Drop the patient IDs since they are not interesting from a analytics POV
    //val dsTrim = csvFile.drop("Patient ID")

    // Map the COVID diagnosis labels from string values to doubles
    val labelName = "SARS-Cov-2 exam result"
    var dsTrim2 = csvFile.withColumn(labelName,
      when(col(labelName).equalTo("positive"),1.0).otherwise(col(labelName)))
    dsTrim2 = dsTrim2.withColumn(labelName,
      when(col(labelName).equalTo("negative"),0.0).otherwise(col(labelName)))
    dsTrim2 = dsTrim2.withColumn(labelName, col(labelName).cast("Double"))

    // Get a summary to determine how sparsely the table is populated
    println("\nInitial data summary")
    val dsSummary = dsTrim2.summary()
    dsSummary.foreach(println)

    // Get the column names and iterate through them dropping any columns that have no data
    val colDataCounts = dsSummary.take(1).mkString
    val colDataCountArr = colDataCounts.substring(1, colDataCounts.length - 1).split(",").tail.map(_.toInt)
    for (i <- colDataCountArr.indices)
    {
      // Drop any columns that are less than 5% populated
      if (colDataCountArr(i) < dsTrim2.count()*0.05)
      {
          //println("Drop column " + dsSummary.schema.fields(i+1).name + " with count = " + colDataCountArr(i))
          dsTrim2 = dsTrim2.drop(dsSummary.schema.fields(i+1).name)
      }
    }

    // Get a new summary to determine how sparsely the table is populated
    //println("\nDataset summary after dropping sparse columns")
    //val dsSummary2 = dsTrim2.summary()
    //dsSummary2.foreach(println)

    // Determine which rows are sparsely populated by creating an array with row id and null count
    val dsTrimRDD = dsTrim2.rdd
    val rowNullCntArr = dsTrimRDD.map { row =>
      var emptyCount = 0
      for(i <- 0 until row.length)
        emptyCount = emptyCount + (if(row.isNullAt(i)) 1 else 0)

      (row.get(0), emptyCount)
    }

    //println("\nNull count by row")
    //rowArray.take(10).foreach(println)

    // Remove any row that has more than half of the columns empty, then map down to just the healthy ids
    val minColumnThreshold = dsTrim2.columns.length/2
    val populatedRows = rowNullCntArr.filter { aRow => aRow._2 < minColumnThreshold  }
    val healthyIdsOnly = populatedRows.map { aRow => aRow._1 }

    //println("\nHealthy remaining rows - top 10")
    //healthyIdsOnly.take(10).foreach(println)
    //println("Total healthy rows = " + healthyIdsOnly.count())

    // Filter out just the healthy rows from our data set, this will get rid of the sparse ones
    val healthyPopulatedRows = healthyIdsOnly.collect().toSet
    val dsTrimHealthy = dsTrimRDD.filter { aRow => healthyPopulatedRows.contains(aRow.get(0)) }

    println("\nFiltered dataset rows - top 10")
    dsTrimHealthy.take(10).foreach(println)
    println("Total filtered dataset rows = " + dsTrimHealthy.count())

    // Determine how many blanks are in each column
    //println(dsTrimHealthy.first().schema)
    val healthyDf = spark.createDataFrame(dsTrimHealthy, dsTrimHealthy.first().schema)

    //println("\nDataset summary after dropping sparse rows")
    //val healthyDfSummary = healthyDf.summary()
    //healthyDfSummary.foreach(println)

    // Get the header names and types so we can start repairing the missing data
    val healthyColumns = healthyDf.dtypes
    //healthyColumns.foreach(println)

    // Custom function that will look at a given column where data is type Double, determine
    // the average value for the non null data in the column, then fill in the blanks based
    // on that average
    def fillInBlanks_Double(index: Integer, frame: DataFrame): DataFrame = {

      var colDataSum = 0.0
      var colNotNullCount = 0.0
      val tuple = dsTrimHealthy.map { row =>
        colNotNullCount = colNotNullCount + (if (row.isNullAt(index)) 0 else 1)
        colDataSum = colDataSum + (if (row.isNullAt(index)) 0 else row.getDouble(index))
        (colDataSum, colNotNullCount, colDataSum / colNotNullCount)
      }

      // The last line has the final totals for the whole dataframe
      val lastTuple = tuple.collect().last
      //println("tuple for index[" + index + " is " + lastTuple)

      // Replace the null column entries with our calculated average value
      val healthWithValueDf = frame.withColumn(healthyColumns(index)._1,
        when(col(healthyColumns(index)._1).isNull, lastTuple._3).otherwise(col(healthyColumns(index)._1)))

      healthWithValueDf
    }

    // Fix missing elements with type Double on columns 6 thru 19
    val myArray1 = Array.range(6, 20)
    var repairedDataframe = fillInBlanks_Double(6, healthyDf)
    for (element <- myArray1) {
      repairedDataframe = fillInBlanks_Double(element, repairedDataframe)
    }

    // Fix missing elements with type Double on columns 37 thru 42
    val myArray2 = Array.range(37, 43)
    for (element <- myArray2) {
      repairedDataframe = fillInBlanks_Double(element, repairedDataframe)
    }

    //println("\nAfter double repair in rows 6 thru 19, 37 thru 42")
    //val healthyDfSummary2 = repairedDataframe.summary()
    //healthyDfSummary2.foreach(println)

    // Custom function that will look at a given column where data is an enumerated String,
    // determine the most used value for the non null data in the column, then fill in the
    // blanks based on that most used value
    def fillInBlanks_String(index: Integer, frame: DataFrame): DataFrame = {

      var colNotDetectedCount = 0.0
      var colDetectedCount = 0.0

      val tuple = dsTrimHealthy.map { row =>
        colNotDetectedCount = colNotDetectedCount + (if (! row.isNullAt(index) && row.getString(index).equals("not_detected")) 1 else 0)
        colDetectedCount = colDetectedCount + (if (! row.isNullAt(index) && row.getString(index).equals("detected")) 1 else 0)
        val isNDLarger = colNotDetectedCount > colDetectedCount
        (colNotDetectedCount, colDetectedCount, if(isNDLarger) "not_detected" else "detected")
      }
      val lastTuple = tuple.collect().last
      //println("tuple for index[" + index + " is " + lastTuple)

      val healthWithValueDf = frame.withColumn(healthyColumns(index)._1,
        when(col(healthyColumns(index)._1).isNull, lastTuple._3).otherwise(col(healthyColumns(index)._1)))

      healthWithValueDf
    }

    // Fix missing elements with type String on columns 20 thru 36
    val myArray3 = Array.range(20, 37)
    for (element <- myArray3) {
      repairedDataframe = fillInBlanks_String(element, repairedDataframe)
    }

    //println("\nAfter string repair in rows 20 thru 36")
    //val healthyDfSummary3 = repairedDataframe.summary()
    //healthyDfSummary3.foreach(println)

    // Drop three remaining rows that just dont have enough data to duplicate
    repairedDataframe = repairedDataframe.drop("Influenza B, rapid test")
    repairedDataframe = repairedDataframe.drop("Influenza A, rapid test")
    repairedDataframe = repairedDataframe.drop("Strepto A")
    var healthyColDrop = healthyColumns.filterNot(array => array._1.equals("Influenza B, rapid test"))
    healthyColDrop = healthyColDrop.filterNot(array => array._1.equals("Influenza A, rapid test"))
    healthyColDrop = healthyColDrop.filterNot(array => array._1.equals("Strepto A"))
    healthyColDrop = healthyColDrop.filterNot(array => array._1.equals("Patient ID"))

    // Map the "detected" value to 1 and "not_detected" value to 0
    val strColumns = healthyColDrop.filter( array => array._2.equals("StringType"))
    for(strLabel <- strColumns) {

      repairedDataframe = repairedDataframe.withColumn(strLabel._1,
        when(col(strLabel._1).equalTo("detected"), 1.0).otherwise(col(strLabel._1)))
      repairedDataframe = repairedDataframe.withColumn(strLabel._1,
        when(col(strLabel._1).equalTo("not_detected"), 0.0).otherwise(col(strLabel._1)))
      repairedDataframe = repairedDataframe.withColumn(strLabel._1,
        when(col(strLabel._1).equalTo("positive"), 1.0).otherwise(col(strLabel._1)))
      repairedDataframe = repairedDataframe.withColumn(strLabel._1,
        when(col(strLabel._1).equalTo("negative"), 0.0).otherwise(col(strLabel._1)))
      repairedDataframe = repairedDataframe.withColumn(strLabel._1, col(strLabel._1).cast("Double"))
    }

    //SMJ Note - Comment out above for loop and comment in the line below to drop the string attributes
    //for (strLabel <- strColumns) { repairedDataframe = repairedDataframe.drop(strLabel._1) }

    // Review the dataset to see how it looks
    //repairedDataframe.take(20).foreach(println)

    // The data looks viable here. Let's trim off the non trainable attributes and setup our label
    repairedDataframe = repairedDataframe.drop("Patient ID")
    repairedDataframe = repairedDataframe.drop("Patient addmited to regular ward (1=yes, 0=no)")
    repairedDataframe = repairedDataframe.drop("Patient addmited to semi-intensive unit (1=yes, 0=no)")
    repairedDataframe = repairedDataframe.drop("Patient addmited to intensive care unit (1=yes, 0=no)")

    // Review the dataset to see how it looks
    //repairedDataframe.take(20).foreach(println)

    def anyToDouble(value: Any): Option[Double] = {
      value match {
        case d: Double => Some(d)
        case i: Int => Some(i.toDouble)
        case l: Long => Some(l.toDouble)
        case f: Float => Some(f.toDouble)
        case _ => None // Return None for other types
      }
    }

    // Convert our data into an RDD with LabeledPoints so we can start to analyze
    val finalDataRdd = repairedDataframe.rdd
    val testDataset = finalDataRdd.map { aRow =>
      val defaultDouble = 0.0
      val label = anyToDouble(aRow.get(1)).getOrElse(defaultDouble)
      val rowList = aRow.toSeq.zipWithIndex.filter(_._2 != 1).map(_._1)
      val rowDoubles = rowList.map( value => anyToDouble(value).getOrElse(defaultDouble))

      LabeledPoint(label, Vectors.dense(rowDoubles.toArray))
    }

    println("\nLabeled rows ready to analyze")
    testDataset.take(10).foreach(println)

    println("\nLabel ratios from the prepared data")
    val labelSummary = testDataset.map(aLine => aLine.label).countByValue()
    labelSummary.foreach(println)

    // Create a common dataset for analysis that we will use repeatedly for equal comparison
    // across the different algorithms
    val Array(trainData, testData) = testDataset.randomSplit(Array(0.60, 0.40))
    trainData.cache()
    testData.cache()

    // Common method for evaluating binary metrics
    def evaluateBinaryMetrics(data: RDD[(Double, Double)]): Unit = {

      val metrics = new BinaryClassificationMetrics(data)
      println("Area under ROC = " + metrics.areaUnderROC())
      val metricsSum = metrics.scoreAndLabels.map(aRow => aRow.toString).countByValue()
      metricsSum.foreach(println)

      var (correct, incorrect) = (0.0, 0.0)
      for ((key, value) <- metricsSum) {
        correct = correct + (if (key.equals("(1.0,1.0)") || key.equals("(0.0,0.0)")) value else 0.0)
        incorrect = incorrect + (if (key.equals("(0.0,1.0)") || key.equals("(1.0,0.0)")) value else 0.0)
      }
      println("Predictions correct [" + correct + "] incorrect[" + incorrect + "]")
      println("Accuracy = " + (correct / (correct + incorrect)))
    }

    println
    println("*************************************************")
    println("** DECISION TREE ANALYSIS                      **")
    println("*************************************************")

    // Train the model with our training data portion
    val modelDT = DecisionTree.trainClassifier(
      trainData, 2, Map[Int, Int](), "gini", 4, 100)

    val predictionsDT = testData.map { point => (modelDT.predict(point.features), point.label) }
    evaluateBinaryMetrics(predictionsDT)

    println
    println("*************************************************")
    println("** LOGISTIC REGRESSION ANALYSIS                **")
    println("*************************************************")

    // Train the logistic regression model with the training data from our original dataset.
    val modelLR = new LogisticRegressionWithLBFGS().run(trainData)
    val predictionsLR = testData.map { point => (modelLR.predict(point.features), point.label) }
    evaluateBinaryMetrics(predictionsLR)

    println
    println("*************************************************")
    println("** RANDOM FOREST ANALYSIS                      **")
    println("*************************************************")

    // Train the random forest model with the training data from our original dataset.
    val modelRF = RandomForest.trainClassifier(trainData, 2, Map[Int, Int](),
      2, "auto", "gini", 5, 32, 11)

    val predictionsRF = testData.map { point => (modelRF.predict(point.features), point.label) }
    evaluateBinaryMetrics(predictionsRF)

    println
    println("*************************************************")
    println("** GRADIENT BOOSTED TREE ANALYSIS              **")
    println("*************************************************")

    // Train a GradientBoostedTrees model with the training data from our original dataset.
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.numIterations = 10 // Number of iterations (trees)
    boostingStrategy.treeStrategy.maxDepth = 5 // Maximum depth of each tree

    val modelGBT = GradientBoostedTrees.train(trainData, boostingStrategy)
    val predictionsGBT = testData.map { point => (modelGBT.predict(point.features), point.label) }
    evaluateBinaryMetrics(predictionsGBT)

    println
    println("*************************************************")
    println("** SUPPORT VECTOR MACHINE ANALYSIS             **")
    println("*************************************************")

    // Train a SVN model with the training data from our original dataset.
    val modelSVM = SVMWithSGD.train(trainData, 100)
    val predictionsSVM = testData.map { point => (modelSVM.predict(point.features), point.label) }
    evaluateBinaryMetrics(predictionsSVM)

    println
    println("*************************************************")
    println("** SMJ IMPROVEMENT ANALYSIS                    **")
    println("*************************************************")

    def normalizeData(notNormData: RDD[LabeledPoint]): RDD[LabeledPoint] = {

      // Normalize the dataset and see what changes
      val numAttribs = notNormData.first().features.size
      val featureValues = (0 until numAttribs).map { featureIndex =>
        notNormData.map(_.features(featureIndex))
      }

      val minMaxValues = featureValues.map { values =>
        (values.min(), values.max())
      }

      val normalizedTrainData = notNormData.map { lp =>
        val normalizedFeatures = (0 until numAttribs).map { featureIndex =>
          val originalFeature = lp.features(featureIndex)
          val (min, max) = minMaxValues(featureIndex)
          if (max == min) 0.0 else (originalFeature - min) / (max - min)
        }.toArray

        LabeledPoint(lp.label, Vectors.dense(normalizedFeatures))
      }
      normalizedTrainData
    }

    val normTrainData = normalizeData(trainData)
    val normTestData = normalizeData(testData)
    normTestData.take(10).foreach(println)

    println
    println("*************************************************")
    println("** LOGISTIC REGRESSION ANALYSIS - NORM         **")
    println("*************************************************")

    // Train the logistic regression model with the training data from our original dataset.
    val modelLR_N = new LogisticRegressionWithLBFGS().run(normTrainData)
    val predictionsLR_N = normTestData.map { point => (modelLR_N.predict(point.features), point.label) }
    evaluateBinaryMetrics(predictionsLR_N)

    println
    println("*************************************************")
    println("** RANDOM FOREST ANALYSIS - NORM               **")
    println("*************************************************")

    // Train the random forest model with the training data from our original dataset.
    val modelRF_N = RandomForest.trainClassifier(normTrainData, 2, Map[Int, Int](),
      2, "auto", "gini", 5, 32, 11)

    val predictionsRF_N = normTestData.map { point => (modelRF_N.predict(point.features), point.label) }
    evaluateBinaryMetrics(predictionsRF_N)

    println
    println("*************************************************")
    println("** GRADIENT BOOSTED TREE ANALYSIS              **")
    println("*************************************************")

    // Train a GradientBoostedTrees model with the training data from our original dataset.
    val modelGBT_N = GradientBoostedTrees.train(normTrainData, boostingStrategy)
    val predictionsGBT_N = normTestData.map { point => (modelGBT_N.predict(point.features), point.label) }
    evaluateBinaryMetrics(predictionsGBT_N)

    // Restore INFO level verbosity so we can get time duration for the total run
    spark.sparkContext.setLogLevel("INFO")
  }
}