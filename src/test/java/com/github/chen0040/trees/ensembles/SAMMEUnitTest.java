package com.github.chen0040.trees.ensembles;


import com.github.chen0040.data.evaluators.ClassifierEvaluator;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataQuery;
import com.github.chen0040.data.utils.TupleTwo;
import com.github.chen0040.trees.utils.FileUtils;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.InputStream;

import static org.testng.Assert.*;


/**
 * Created by xschen on 9/6/2017.
 */
public class SAMMEUnitTest {
   @Test
   public void test_iris() throws IOException {
      InputStream irisStream = FileUtils.getResource("iris.data");
      DataFrame irisData = DataQuery.csv(",")
              .from(irisStream)
              .selectColumn(0).asNumeric().asInput("Sepal Length")
              .selectColumn(1).asNumeric().asInput("Sepal Width")
              .selectColumn(2).asNumeric().asInput("Petal Length")
              .selectColumn(3).asNumeric().asInput("Petal Width")
              .selectColumn(4).asCategory().asOutput("Iris Type")
              .build();

      TupleTwo<DataFrame, DataFrame> parts = irisData.shuffle().split(0.9);

      DataFrame trainingData = parts._1();
      DataFrame crossValidationData = parts._2();

      System.out.println(crossValidationData.head(10));

      SAMME multiClassClassifier = new SAMME();
      multiClassClassifier.fit(trainingData);

      ClassifierEvaluator evaluator = new ClassifierEvaluator();

      for(int i=0; i < crossValidationData.rowCount(); ++i) {
         String predicted = multiClassClassifier.classify(crossValidationData.row(i));
         String actual = crossValidationData.row(i).categoricalTarget();
         System.out.println("predicted: " + predicted + "\tactual: " + actual);
         evaluator.evaluate(actual, predicted);
      }

      evaluator.report();
   }
}
