package com.cs7265.homework

import org.apache.spark._
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object Assignment3 {

  def main(args: Array[String]): Unit = {

    println("")
    println("*****************************************")
    println("***  ASSIGNMENT 3 - Stephen Jacobs    ***")
    println("***  KSU CS7265 - Big Data Analytics  ***")
    println("*****************************************")
    println("")

    // Create a spark context, which is required for interacting with the spark libraries.  For this
    // homework assignment we only use it to manage the log verbosity
    val context = new SparkContext(new SparkConf().setAppName("CS7265 HW3").setMaster("local"))
    context.setLogLevel("ERROR")

    // Create a Spark session, which is required to load the libsvm data
    val spark = SparkSession.builder().appName("CS7265 HW3").master("local").getOrCreate()

    // Load the text file and return a data frame with its contents.  Grab and print the row count
    val regFilePath = "C:/spark/data/mllib/sample_linear_regression_data.txt"
    val regTraining = spark.read.format("libsvm").load(regFilePath)
    val count = regTraining.count()
    println("sample_linear_regression_data count: " + count)

    // Debug log to look at the first ten lines of the regression data so we can understand it
    //val firstTen = regTraining.take(10)
    //firstTen.foreach(println)

    // Make the structures to use in our training loops.  The first array is our elastic net values,
    // the second list is for holding the calculated  values, stuffing tuples in lists
    val elasticNetArray = Array(0.0, 0.13, 1.0)  // 0 = L2, 1 = L1, 0.5 = ElasticNet
    var finalOutputList : List[List[(Double, Double, Double)]] = List.empty[List[(Double, Double, Double)]]

    // Create a double loop to collect our R2 data.  Outer loop will change the elastic net parameter
    // based on the values in our input array
    for (elasticNetValue <- elasticNetArray) {

      // Make an empty list for holding the tuples for this elastic net iteration. The first value is
      // the elasticNetValue, second value is the lambda value, third is the R2 value
      var iterationOutput : List[(Double, Double, Double)] = List.empty[(Double, Double, Double)]

      val startLambda = 0.0
      val endLambda = 10.05  // Pad it a little or we won't always get an iteration for 10
      val incrLambda = 0.1
      var currentLambda = startLambda

      // Increment out inner list based on our incremental lambda value and run a while loop until
      // we hit the maximum value
      while(currentLambda <= endLambda) {

        //println("Starting iteration for ElasticNetParm(" + elasticNetValue + ") Lambda(" + currentLambda +")")

        // Create a linear regression and set the hyperparameters that denote its configuration
        val linearRegression = new LinearRegression().setMaxIter(10).
          setRegParam(currentLambda).setElasticNetParam(elasticNetValue)

        // Generate the linear regression model derived from the training set
        val linRegressionModel = linearRegression.fit(regTraining);

        // Summarize the model and gather supporting metrics
        val linRegTrainingSummary = linRegressionModel.summary

        // Stuff the current tuple at the end of the output list
        iterationOutput = iterationOutput :+ (elasticNetValue, currentLambda, linRegTrainingSummary.r2)

        // Increment our lambda value
        currentLambda += incrLambda
      }

      // Stuff the list for the first elasticNet values into our final output list
      finalOutputList = finalOutputList :+ iterationOutput
    }

    // Print out our table values.  Iterate through the elastic net lists and print the values
    // for each lambda increment
    var i = 0
    for(eNetList <- finalOutputList)
    {
      println("\nTable for ElasticNet Value = " + elasticNetArray(i))
      for(lambdaTuple <- eNetList)
      {
        // Format our doubles for consistent output
        println("Lambda = " + "%.1f".format(lambdaTuple._2) + " R2 = " + "%.8f".format(lambdaTuple._3))
      }
      i += 1
    }

    // Restore INFO level verbosity so we can get time duration for the total run
    context.setLogLevel("INFO")
  }
}