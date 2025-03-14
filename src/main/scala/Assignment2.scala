package com.cs7265.homework

import org.apache.spark._
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd.RDD

object Assignment2 {

  def main(args: Array[String]): Unit = {

    println("")
    println("*****************************************")
    println("***  ASSIGNMENT 2 - Stephen Jacobs    ***")
    println("***  KSU CS7265 - Big Data Analytics  ***")
    println("*****************************************")
    println("")

    // Determine the working directory so we understand where to put the text file
    println("Working Directory: " + System.getProperty("user.dir"))

    // Create a spark context, which is required for interacting with the spark libraries
    val context = new SparkContext(new SparkConf().setAppName("CS7265 HW2").setMaster("local"))
    context.setLogLevel("ERROR")

    // Load the text file and return an RDD with its contents.  Save the line count
    val file = context.textFile("covtype.data")
    val totalRows = file.count().toDouble
    println("covtype.data row count: " + totalRows)

    // Debug log to look at the first ten if we need to debug raw data
    // val firstRawTenRdd = file.take(10)
    // firstRawTenRdd.foreach(println)

    // Operate on each line of the RDD, splitting at the comma delimiter and converting each
    // integer value to a double.  Create vectors for all of the data except the last value
    // which will be used as a label for the each vector
    val data = file.map { line =>
      val values = line.split(",").map(_.toDouble)
      val featureVector = Vectors.dense(values.init)
      val label = values.last - 1
      LabeledPoint(label, featureVector) }

    // Randomly split the data into 80% training, 10% cross validation, 10% test data.
    // Cache saves the RDD at the default storage level, which is to memory
    val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))
    trainData.cache()
    cvData.cache()
    testData.cache()

    // Debug log to look at the training data before modelling if needed
    // val firstTenRdd = trainData.take(10)
    // trainData.foreach(println)

    // Function declaration to determine metrics from the trained model for a given data
    // set.  We will use this after training the model to see how well it works predicting
    // results for a new data set
    def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]) :
      MulticlassMetrics = {
      val predictionsAndLabels = data.map(example =>
        (model.predict(example.features), example.label)
      )
      new MulticlassMetrics(predictionsAndLabels)
    }

    // Train the model with the training data from our original dataset.  Hyper
    // parameters designate how we want drive the model design with depth and classifiers
    val model = DecisionTree.trainClassifier(
      trainData, 7, Map[Int,Int](), "gini", 4, 100)

    // Print the full trained model to the debug output.  This will show us all of the
    // decision points and we can export it to a graphics file manually
    println()
    println("TRAINED MODEL:")
    print(model.toDebugString)
    println()

    // Use of function declaration above to get the predict metrics from our cross validation
    // portion of the original dataset
    val cvMetrics = getMetrics(model, cvData)

    // Output the precision values for each class from the cvData
    for (i <- 0 to 6) {
      println("MulticlassMetrics (cvData): precision of label(" + i + ") = " + cvMetrics.precision(i.toDouble))
    }

    // Print the overall accuracy and weighted precision from the cvData
    println("MulticlassMetrics (cvData): accuracy of model = " + cvMetrics.accuracy)
    println("MulticlassMetrics (cvData): weighted precision of model = " + cvMetrics.weightedPrecision)
    println

    // Prepare data for manual calculation of the overall precision.  Collect the labels
    // from the cross validation data to count the number of times each label is used
    val totalCvRows = cvData.count().toDouble
    val filteredLabels = cvData.map(line => line.label)
    val labelCounts = filteredLabels.countByValue()

    // Output the number of elements for each label in our cross validation data
    for (i <- 0 to 6) {
      println("Number of elements for cvData label(" + i + ") = " + labelCounts(i.toDouble))
    }

    // Create an array for manually calculating the weighted precision of the metrics.
    // Map the array and transform it with our precision calculation
    val labelArray = (0 to 6).map(_.toDouble).toArray
    val precisionByLabel = labelArray.map { x =>
      (labelCounts(x).toDouble/totalCvRows) * cvMetrics.precision(x) }

    // Debug print for the precision if needed
    // precisionByLabel.foreach(println)

    // Sum the precision by label to calculate our overall precision
    var overallPrecision2 = 0.0
    for (i <- 0 until precisionByLabel.length) {
      overallPrecision2 = overallPrecision2 + precisionByLabel(i)
    }

    println("Manually calculated overall precision (cvData) = " + overallPrecision2)
    println

    // Run the model with the test data also since we have it and see if there is
    // any significant variance from the results of the cvData
    val testMetrics = getMetrics(model, testData)
    for (i <- 0 to 6) {
      println("MulticlassMetrics (testData): precision of label(" + i + ") = " + testMetrics.precision(i.toDouble))
    }
    println("MulticlassMetrics (testData): accuracy of model = " + testMetrics.accuracy)
    println("MulticlassMetrics (testData): weighted precision of model = " + testMetrics.weightedPrecision)
    println

    // Restore INFO level verbosity so we can get time duration for the total run
    context.setLogLevel("INFO")
  }
}