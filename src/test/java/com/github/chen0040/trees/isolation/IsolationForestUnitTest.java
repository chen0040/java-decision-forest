package com.github.chen0040.trees.isolation;


import com.github.chen0040.data.evaluators.BinaryClassifierEvaluator;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataQuery;
import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.data.frame.Sampler;
import com.github.chen0040.trees.utils.FileUtils;
import org.testng.annotations.Test;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.testng.Assert.*;


/**
 * Created by xschen on 28/5/2017.
 */
public class IsolationForestUnitTest {
   @Test
   public void testFindOutliers(){
      String[] filenames_X = {"X1.txt", "X2.txt"};
      String[] filenames_outliers = { "outliers1.txt", "outliers2.txt"};
      double[] thresholds = { 0.2943, 0.45};

      for(int k=0; k < filenames_X.length; ++k){

         String filename_X = filenames_X[k];
         String filename_outliers = filenames_outliers[k];
         double threshold = thresholds[k];

         InputStream inputStream = FileUtils.getResource(filename_X);
         DataFrame data = DataQuery.csv().from(inputStream)
                 .selectColumn(0).asNumeric().asInput("x1")
                 .selectColumn(1).asNumeric().asInput("x2")
                 .build();

         IsolationForest algorithm = new IsolationForest();
         algorithm.setThreshold(threshold);


         List<Integer> predicted_outliers = new ArrayList<>();

         DataFrame learnedData = algorithm.fitAndTransform(data);


         inputStream = FileUtils.getResource(filename_outliers);
         DataFrame outliers = DataQuery.csv().from(inputStream)
                 .selectColumn(0).asNumeric().asInput("outlier")
                 .build();

         BinaryClassifierEvaluator evaluator = new BinaryClassifierEvaluator();

         for(int i = 0; i < learnedData.rowCount(); ++i){
            DataRow tuple = learnedData.row(i);
            boolean predicted = tuple.getCategoricalTargetCell("anomaly").equals("1");
            final int outlier_index = i;
            boolean actual = outliers.filter(row -> row.getCell("outlier") == outlier_index).rowCount() > 0;
            System.out.println("predicted: " + predicted +"\tactual: "+actual);
            evaluator.evaluate(actual, predicted);
         }

         System.out.println(evaluator.getSummary());
      }
   }


   private static Random random = new Random();

   public static double rand(){
      return random.nextDouble();
   }

   public static double rand(double lower, double upper){
      return rand() * (upper - lower) + lower;
   }

   public static double randn(){
      double u1 = rand();
      double u2 = rand();
      double r = Math.sqrt(-2.0 * Math.log(u1));
      double theta = 2.0 * Math.PI * u2;
      return r * Math.sin(theta);
   }


   // unit testing based on example from http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html#
   @Test
   public void testSimple(){


      DataQuery.DataFrameQueryBuilder schema = DataQuery.blank()
              .newInput("c1")
              .newInput("c2")
              .newOutput("anomaly")
              .end();

      Sampler.DataSampleBuilder negativeSampler = new Sampler()
              .forColumn("c1").generate((name, index) -> randn() * 0.3 + (index % 2 == 0 ? -2 : 2))
              .forColumn("c2").generate((name, index) -> randn() * 0.3 + (index % 2 == 0 ? -2 : 2))
              .forColumn("anomaly").generate((name, index) -> 0.0)
              .end();

      Sampler.DataSampleBuilder positiveSampler = new Sampler()
              .forColumn("c1").generate((name, index) -> rand(-4, 4))
              .forColumn("c2").generate((name, index) -> rand(-4, 4))
              .forColumn("anomaly").generate((name, index) -> 1.0)
              .end();

      DataFrame data = schema.build();

      data = negativeSampler.sample(data, 20);
      data = positiveSampler.sample(data, 20);

      System.out.println(data.head(10));

      IsolationForest algorithm = new IsolationForest();
      algorithm.setThreshold(0.38);

      DataFrame learnedData = algorithm.fitAndTransform(data);

      BinaryClassifierEvaluator evaluator = new BinaryClassifierEvaluator();

      for(int i = 0; i < learnedData.rowCount(); ++i){
         DataRow tuple = learnedData.row(i);
         boolean predicted = tuple.getCategoricalTargetCell("anomaly").equals("1");
         boolean actual = data.row(i).getTargetCell("anomaly") == 1.0;
         System.out.println("predicted: " + predicted +"\tactual: "+actual);

         evaluator.evaluate(actual, predicted);
      }

      System.out.println(evaluator.getSummary());



   }
}
