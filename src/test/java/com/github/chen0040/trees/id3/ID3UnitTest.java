package com.github.chen0040.trees.id3;


import com.github.chen0040.data.evaluators.BinaryClassifierEvaluator;
import com.github.chen0040.data.evaluators.ClassifierEvaluator;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataQuery;
import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.trees.utils.FileUtils;
import org.testng.annotations.Test;

import java.io.InputStream;

import static org.testng.Assert.*;


/**
 * Created by xschen on 29/5/2017.
 */
public class ID3UnitTest {
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

     ID3 classifier = new ID3();

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
}
