package com.github.chen0040.trees.ensembles;


import com.github.chen0040.data.evaluators.ClassifierEvaluator;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataQuery;
import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.data.utils.TupleTwo;
import com.github.chen0040.trees.id3.ID3;
import com.github.chen0040.trees.utils.FileUtils;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.InputStream;

import static org.testng.Assert.*;


/**
 * Created by xschen on 2/6/2017.
 */
public class BaggingUnitTest {

   @Test
   public void test() {
      InputStream inputStream = FileUtils.getResource("heart_scale");

      DataFrame dataFrame = DataQuery.libsvm().from(inputStream).build();

      dataFrame.unlock();
      for(int i=0; i < dataFrame.rowCount(); ++i){
         DataRow row = dataFrame.row(i);
         row.setCategoricalTargetCell("category-label", "" + row.target());
      }
      dataFrame.lock();

      Bagging classifier = new Bagging();
      classifier.setTreeCount(30);
      classifier.setTrainingSizePerTree(10);

      classifier.fit(dataFrame);

      ClassifierEvaluator evaluator = new ClassifierEvaluator();

      for(int i = 0; i < dataFrame.rowCount(); ++i){
         DataRow tuple = dataFrame.row(i);
         String predicted = classifier.classify(tuple);
         String actual = tuple.categoricalTarget();
         System.out.println("predicted: "+predicted+"\tactual: "+actual);
         evaluator.evaluate(actual, predicted);
      }

      System.out.println(evaluator.getSummary());

   }

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

      Bagging multiClassClassifier = new Bagging();
      multiClassClassifier.setTreeCount(30);
      multiClassClassifier.setTrainingSizePerTree(10);
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
