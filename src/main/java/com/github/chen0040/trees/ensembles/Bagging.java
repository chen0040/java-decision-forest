package com.github.chen0040.trees.ensembles;


import com.github.chen0040.data.frame.BasicDataFrame;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.trees.id3.ID3;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * Created by xschen on 2/6/2017.
 */
@Getter
@Setter
public class Bagging {
   private final List<ID3> trees = new ArrayList<>();
   private int treeCount = 100;
   private int trainingSizePerTree = 100;

   public void fit(DataFrame batch){
      trees.clear();

      int count = Math.min(trainingSizePerTree, batch.rowCount());
      for(int i=0; i < treeCount; ++i) {
         ID3 tree = new ID3();

         batch = batch.shuffle();
         DataFrame frame = new BasicDataFrame();
         for(int j=0; j < count; ++j) {
            frame.addRow(batch.row(j).makeCopy());
         }
         frame.lock();

         tree.fit(frame);

         trees.add(tree);
      }
   }

   public String classify(DataRow row){
      Map<String, Integer> candidates = new HashMap<>();
      for(int i=0; i < trees.size(); ++i){
         String label = trees.get(i).classify(row);
         candidates.put(label, candidates.getOrDefault(label, 0) + 1);
      }

      String predicted = null;
      int maxCount = 0;
      for(Map.Entry<String, Integer> entry : candidates.entrySet()){
         if(entry.getValue() > maxCount){
            maxCount = entry.getValue();
            predicted = entry.getKey();
         }
      }
      return predicted;
   }


}
