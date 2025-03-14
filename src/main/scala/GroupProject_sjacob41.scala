package com.cs7265.homework

import org.apache.spark.sql.{SparkSession, DataFrame, Row}
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
          println("Drop column " + dsSummary.schema.fields(i+1).name + " with count = " + colDataCountArr(i))
          dsTrim2 = dsTrim2.drop(dsSummary.schema.fields(i+1).name)
      }
    }

    // Get a new summary to determine how sparsely the table is populated
    println("\nDataset summary after dropping sparse columns")
    val dsSummary2 = dsTrim2.summary()
    dsSummary2.foreach(println)

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
    println("Total healthy rows = " + healthyIdsOnly.count())

    // Filter out just the healthy rows from our data set, this will get rid of the sparse ones
    val healthyPopulatedRows = healthyIdsOnly.collect().toSet
    val dsTrimHealthy = dsTrimRDD.filter { aRow => healthyPopulatedRows.contains(aRow.get(0)) }

    println("\nFiltered dataset rows - top 10")
    dsTrimHealthy.take(10).foreach(println)
    println("Total filtered dataset rows = " + dsTrimHealthy.count())

    // Determine how many blanks are in each column
    //println(dsTrimHealthy.first().schema)
    val healthyDf = spark.createDataFrame(dsTrimHealthy, dsTrimHealthy.first().schema)

    println("\nDataset summary after dropping sparse rows")
    val healthyDfSummary = healthyDf.summary()
    healthyDfSummary.foreach(println)

    // Get the header names and types so we can start repairing the missing data
    val healthyColumns = healthyDf.dtypes
    healthyColumns.foreach(println)

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

    println("\nafter double repair in rows 6 thru 19, 37 thru 42")
    val healthyDfSummary2 = repairedDataframe.summary()
    healthyDfSummary2.foreach(println)

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
      println("tuple for index[" + index + " is " + lastTuple)

      val healthWithValueDf = frame.withColumn(healthyColumns(index)._1,
        when(col(healthyColumns(index)._1).isNull, lastTuple._3).otherwise(col(healthyColumns(index)._1)))

      healthWithValueDf
    }

    // Fix missing elements with type String on columns 20 thru 36
    val myArray3 = Array.range(20, 37)
    for (element <- myArray3) {
      repairedDataframe = fillInBlanks_String(element, repairedDataframe)
    }

    println("\nafter string repair in rows 20 thru 36")
    val healthyDfSummary3 = repairedDataframe.summary()
    healthyDfSummary3.foreach(println)




    // Restore INFO level verbosity so we can get time duration for the total run
    spark.sparkContext.setLogLevel("INFO")
  }
}