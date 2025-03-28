package com.cs7265.homework

import com.jmatio.io.MatFileReader
import com.jmatio.types.MLDouble
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.sql.SparkSession

object Assignment6 {

  val spark: SparkSession = SparkSession.builder().appName("CS7265 Assignment 6")
    .master("local").getOrCreate()

  def main(args: Array[String]): Unit = {

    println("")
    println("*****************************************")
    println("***  ASSIGNMENT 6 - Stephen Jacobs    ***")
    println("***  KSU CS7265 - Big Data Analytics  ***")
    println("*****************************************")
    println("")

    // Determine the working directory so we understand where to put the text file
    println("Working Directory: " + System.getProperty("user.dir"))

    // Suppress log level to ERROR so our output will not be as verbose
    spark.sparkContext.setLogLevel("ERROR")

    // Helper function to print out double array contents
    def printArrayData(title : String, number : Int, data : Array[Array[Double]]): Unit = {
      println("\n" + title + " - Top[" + number + "]")
      data.take(number).foreach { anArray =>
        anArray.foreach(element => print(element + ","))
        println
      } }

    // Load our MAT file into memory and take a peek at the contents
    val matFile = new MatFileReader("exPCA.mat")
    val mfData = matFile.getContent.get("X").asInstanceOf[MLDouble].getArray
    printArrayData("Original Data", 5, mfData)

    // Normalize the data around zero by calculating the average for each feature and then
    // subtracting it from the dataset in a map function.  We will need to run one iteration
    // to find the averages, and then a second iteration to perform the normalization
    var totalX = 0.0
    var totalY = 0.0
    mfData.foreach{ anArray =>
      totalX = totalX + anArray(0)
      totalY = totalY + anArray(1)
    }
    val averageX = totalX/mfData.length
    val averageY = totalY/mfData.length
    println("\nAverage X=" + averageX + " Average Y=" + averageY)

    val normMfData = mfData.map { anArray => Array(anArray(0) - averageX, anArray(1) - averageY) }
    printArrayData("Normalized Data", 5, normMfData)

    // Map our normalized data into dense vectors that will work with PCA analysis
    val mfRowData = normMfData.map { anArray => Vectors.dense(anArray(0), anArray(1)) }

    // Create a Row matrix from the MAT file RDD in the proper format and calculate
    // the top principal component
    val mfMatrix = new RowMatrix(spark.sparkContext.parallelize(mfRowData))
    val mfPrincipalComps = mfMatrix.computePrincipalComponents(1)
    println("\nPrincipal Components:\n" + mfPrincipalComps.toString())

    // project the rows based on our top principal component and print out the numbers
    // so we can graph them for the report
    val projRows = mfMatrix.multiply(mfPrincipalComps)
    println("\nLinear Projection:")
    projRows.rows.collect().foreach(row => println(row))

    // Restore INFO level verbosity so we can get time duration for the total run
    spark.sparkContext.setLogLevel("INFO")
  }
}