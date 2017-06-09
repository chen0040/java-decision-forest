package com.github.chen0040.trees.ensembles;


import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.data.utils.TupleTwo;
import com.github.chen0040.data.utils.discretizers.KMeansDiscretizer;
import com.github.chen0040.trees.id3.ID3;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;


/**
 * Created by xschen on 9/6/2017.
 */
public class SAMME {
   private final List<ID3> classifiers = new ArrayList<>();
   private final List<TupleTwo<Integer, Double>> model = new ArrayList<>();

   @Getter
   @Setter
   private int treeCount = 100;

   private KMeansDiscretizer discretizer=new KMeansDiscretizer();

   @Getter
   private final List<String> classLabels = new ArrayList<>();

   @Setter
   @Getter
   public double dataSampleRate = 0.2; // value between 0 and 1

   public SAMME(){
   }

   public void fit(DataFrame frame){

      frame = discretizer.fitAndTransform(frame);

      classifiers.clear();
      classLabels.clear();
      for(int m = 0; m < treeCount; ++m) {
         ID3 classifier = new ID3(false);
         classifier.fit(frame.shuffle().split(0.2)._1());
         classifiers.add(classifier);
      }

      final int N = frame.rowCount();
      double[] weights = new double[N];
      Set<String> labels = new HashSet<>();
      for(int i=0; i < N; ++i){
         weights[i] = 1.0 / N;
         labels.add(frame.row(i).categoricalTarget());
      }
      classLabels.addAll(labels);
      int K = classLabels.size();

      for(int t = 0; t < treeCount; ++t) {

         double min_err = Double.MAX_VALUE;
         int M = -1;
         for (int m = 0; m < treeCount; ++m) {
            ID3 classifier_m = classifiers.get(m);
            double err_m = 0;
            for (int i = 0; i < N; ++i) {
               DataRow row = frame.row(i);
               String predicted = classifier_m.classify(row);

               if (!predicted.equals(row.categoricalTarget())) {
                  err_m += weights[i];
               }
            }

            if (min_err > err_m) {
               min_err = err_m;
               M = m;
            }
         }

         // Add next classifier
         ID3 classifier_t = classifiers.get(M);
         double alpha_t = 0.5 * Math.log((1-min_err) / min_err) + Math.log(K - 1);
         model.add(new TupleTwo<>(M, alpha_t));

         // Update weight
         double sum = 0;
         for(int i=0; i < N; ++i){
            DataRow row_i = frame.row(i);
            String predicted = classifier_t.classify(row_i);
            double II = predicted.equals(row_i.categoricalTarget()) ? 0 : 1;
            weights[i] = weights[i] * Math.exp(alpha_t * II);
            sum += weights[i];
         }

         // Normalize weight
         for(int i=0; i < N; ++i) {
            weights[i] /= sum;
         }
      }
   }

   public String classify(DataRow row) {
      row = discretizer.transform(row);

      double max_sum_k = Double.NEGATIVE_INFINITY;
      int K = -1;
      for(int k =0; k < classLabels.size(); ++k){
         String candidate = classLabels.get(k);

         double sum_k = 0;
         for(int m = 0; m < treeCount; ++m) {
            TupleTwo<Integer, Double> t = model.get(m);
            ID3 classifier_t = classifiers.get(t._1());
            double alpha_t = t._2();
            String predicted = classifier_t.classify(row);
            double II = predicted.equals(candidate) ? 1 : 0;
            sum_k += alpha_t * II;
         }

         if(sum_k > max_sum_k) {
            max_sum_k =sum_k;
            K = k;
         }
      }

      return classLabels.get(K);
   }
}
