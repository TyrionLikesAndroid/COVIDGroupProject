package com.cs7265.homework

import scala.math._
import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

object Assignment4 {

  def main(args: Array[String]): Unit = {

    println("")
    println("*****************************************")
    println("***  ASSIGNMENT 4 - Stephen Jacobs    ***")
    println("***  KSU CS7265 - Big Data Analytics  ***")
    println("*****************************************")
    println("")

    // Determine the working directory so we understand where to put the text file
    println("Working Directory: " + System.getProperty("user.dir"))

    // Create a spark context, which is required for interacting with the spark libraries
    val context = new SparkContext(new SparkConf().setAppName("CS7265 HW4").setMaster("local"))
    context.setLogLevel("ERROR")

    // Load the text file and return an RDD with its contents.  Write the line count
    val file = context.textFile("iris.data")
    println("iris.data raw row count: " + file.count())

    // Debug log to take a look at the loaded iris data for the first sample of 10
    //file.take(10).foreach(println)

    // Split each line with a comma delimiter and save off all of the attributes except
    // the last one as the feature vector.  Convert the label strings into doubles so they
    // will work with the labeled point class
    val data = file.map { line =>
      val values = line.split(",")

      val irisAttribs = values.init.map(_.toDouble)
      val irisAttribVector = Vectors.dense(irisAttribs)

      val label = values.last
      val lableDbl = {
        if(label == "Iris-virginica") 2.0
        else if (label == "Iris-versicolor") 1.0
        else if (label == "Iris-setosa") 0.0
        else 4.0
      }
      LabeledPoint(lableDbl, irisAttribVector)
    }

    // Remove any rows with value 4.0 since they must be data fragments
    val cleanData = data.filter( labeledPoint => labeledPoint.getLabel != 4.0)
    println("iris.data clean row count: " + cleanData.count())

    // Create our two-class subsets to review linear separability.  Just filter out
    // the one we don't want to see in the list
    val labels0and2 = cleanData.filter(labeledPoint => labeledPoint.getLabel != 1.0)
    val labels0and1 = cleanData.filter(labeledPoint => labeledPoint.getLabel != 2.0)
    val labels1and2 = cleanData.filter(labeledPoint => labeledPoint.getLabel != 0.0)

    // Remap the labels for our training since it will give an error if we classify
    // anything with classes other than 0 and 1.
    val labels0and2SVM = labels0and2.map { aPoint =>
      val newLabel = if(aPoint.label == 2.0) 1.0 else 0.0
      LabeledPoint(newLabel, aPoint.features)
    }
    val labels1and2SVM = labels1and2.map { aPoint =>
      val newLabel = if (aPoint.label == 2.0) 0.0 else 1.0
      LabeledPoint(newLabel, aPoint.features)
    }

    // Review our counts and make sure there are no surprises.  Each set should be 100 count
    println("labels 0+1 count: " + labels0and1.count())
    println("labels 0+2 count: " + labels0and2SVM.count())
    println("labels 1+2 count: " + labels1and2SVM.count())

    // Write a function for our test series that we can reuse multiple times
    def testTwoClasses(datasetLabel: String, dataset: RDD[LabeledPoint]): Unit = {

      println()
      println("Linear SVM Results For " + datasetLabel)

      // Split the data for testing and save off the training and test datasets
      val splits = dataset.randomSplit(Array(0.6, 0.4), seed = 14L)
      val training = splits(0)
      val test = splits(1)

      // Run our training data in the SVM model.  Clearing the threshold will let us
      // see how close our data is to the best fit line
      val model = SVMWithSGD.train(training, 100)
      model.clearThreshold()

      // Run predictions for our test data and keep track of the results and the mismatches
      val scoresAndLabelsAndMatches = test.map { point =>
        val score = model.predict(point.features)
        val correct = ((score < 0.0) && (point.label == 0.0)) ||
          ((score > 0.0) && (point.label == 1.0))
        (score, point.label, correct)
      }

      // Print our correct and incorrect test data counts
      val results = scoresAndLabelsAndMatches.map { row => row._3 }.countByValue()
      println(datasetLabel + " predictions correct [" + results.getOrElse(true, 0) +
        "] incorrect[" + results.getOrElse(false, 0) +"]")

      // Run the classification metrics and print our area under the ROC curve
      val scoresAndLabels = scoresAndLabelsAndMatches.map { row => (row._1, row._2)}
      val metrics = new BinaryClassificationMetrics(scoresAndLabels)
      println(datasetLabel + " area under ROC [" + metrics.areaUnderROC() + "]")
    }

    // Run all of our two-class combinations through the test function and observe results
    testTwoClasses("[Labels 0 and 1]", labels0and1)
    testTwoClasses("[Labels 0 and 2]", labels0and2SVM)
    testTwoClasses("[Labels 1 and 2]", labels1and2SVM)

    // We observe that labels 1 and 2 ARE NOT linearly separable.  We will do our kernel
    // expansion based on the provided function and see if that improves the accuracy
    val expandedLabels1and2 = labels1and2SVM.map { aLabeledPoint =>

      // Apply our kernel expansion equation to each of the values in the feature set
      var expandedAttribs = Array.empty[Double]
      for(i <- 0 to 3)
      {
        val attribValue = aLabeledPoint.features(i)
        val expandedValue = Array(attribValue * (1 / sqrt(3)), math.pow(attribValue, 2),
          math.pow(attribValue, 3))
        expandedAttribs = expandedAttribs ++ expandedValue
      }

      // Return our new label with our expanded dataset
      LabeledPoint(aLabeledPoint.label, Vectors.dense(expandedAttribs))
    }

    // Test labels 1 and 2 again and see if we have improved results due to expansion
    testTwoClasses("[Kernel Expanded Labels 1 and 2]", expandedLabels1and2)

    // Restore INFO level verbosity so we can get time duration for the total run
    context.setLogLevel("INFO")
  }
}