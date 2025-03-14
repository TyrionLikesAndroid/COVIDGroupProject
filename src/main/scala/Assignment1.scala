package com.cs7265.homework

import org.apache.spark._

object Assignment1 {

  def main(args: Array[String]): Unit = {

    println("")
    println("*****************************************")
    println("***  ASSIGNMENT 1 - Stephen Jacobs    ***")
    println("***  KSU CS7265 - Big Data Analytics  ***")
    println("*****************************************")
    println("")

    // Determine the working directory so we understand where to put the text file
    println("Working Directory: " + System.getProperty("user.dir"))

    // Create a spark context, which is required for interacting with the spark libraries
    val context = new SparkContext(new SparkConf().setAppName("CS7265 HW1").setMaster("local"))
    context.setLogLevel("ERROR")

    // Load the text file and return an RDD for it's contents.  Log the word count for grins
    val file = context.textFile("news-2016-2017.txt")
    println("*** news-2016-2017.txt word count:" + file.count() + " ***")

    // Split the original RDD using empty space as a delimiter
    val wordList = file.flatMap(x => x.split(" "))

    // Convert RDD to map of KV pairs that shows word frequency count from the unstructured blob
    val wordSums = wordList.countByValue()

    // Log that we have counted the keys, leave a print statement for debugging
    println("*** COUNTED, NOT SORTED ***")
    //wordSums.foreach(println)

    // Convert the Map back to RDD and sort because it will be much faster than sorting the Map
    val kvRDD = context.parallelize(wordSums.toSeq)
    val sortedSums = kvRDD.sortBy(_._2, false)

    // Log that we have sorted the keys, leave a print statement for debugging
    println("*** SORTED ***")
    //sortedSums.foreach(println)

    // Grab and print the first 500 items in the RDD, which will be our top 500 words
    val top = sortedSums.take(500)
    println("*** TOP 500 TRUNCATED ***")
    top.foreach(println)

    // Restore INFO level verbosity so we can get time duration for the total run
    context.setLogLevel("INFO")
  }
}