package com.github.chen0040.trees.id3;

import com.github.chen0040.data.frame.BasicDataFrame;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.data.utils.CountRepository;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;


/**
 * Created by xschen on 17/8/15.
 */
public class ID3TreeNode implements Cloneable {
    private int rowCount;
    private int splitAttributeIndex;
    private String attributeValue;
    private ArrayList<ID3TreeNode> childNodes;
    private String classLabel;
    private List<String> columns = new ArrayList<>();

    public void copy(ID3TreeNode rhs){
        rowCount = rhs.rowCount;
        splitAttributeIndex = rhs.splitAttributeIndex;
        attributeValue = rhs.attributeValue;
        childNodes.clear();
        for(int i=0; i < rhs.childNodes.size(); ++i){
            childNodes.add((ID3TreeNode)rhs.childNodes.get(i).clone());
        }
        classLabel = rhs.classLabel;
    }

    @Override
    public Object clone(){
        ID3TreeNode clone = new ID3TreeNode();
        clone.copy(this);

        return clone;
    }

    public ID3TreeNode(){
        childNodes = new ArrayList<ID3TreeNode>();
    }

    public ID3TreeNode(DataFrame batch, Random random, int height, int maxHeight, List<String> columns){
        childNodes = new ArrayList<>();
        this.columns.addAll(columns);

        rowCount = batch.rowCount();
        splitAttributeIndex = -1;
        attributeValue = "";
        classLabel = "";
        updateClassLabel(batch);

        if(rowCount <= 1 || height == maxHeight){
            return;
        }


        int n = columns.size();

        CountRepository[] counts = new CountRepository[n];
        CountRepository counts2 = new CountRepository();

        for(int i=0; i < n; ++i){
            String category = String.format("field%d", i);
            counts[i] = new CountRepository(category);
        }


        for(int i=0; i < rowCount; ++i){
            DataRow tuple = batch.row(i);
            String label = tuple.categoricalTarget();
            String classEventName = "ClassLabel="+label;

            for(int j=0; j < n; ++j){
                String category_value = columns.get(j);
                counts[j].addSupportCount(category_value, classEventName);
                counts[j].addSupportCount(category_value);
                counts[j].addSupportCount();
            }
            counts2.addSupportCount(classEventName);
            counts2.addSupportCount();
        }


        double entropy_S = 0;
        for(String classEventName : counts2.getSubEventNames()){
            double p_class = counts2.getProbability(classEventName);
            entropy_S += (-p_class * log2(p_class));
        }


        if(entropy_S == 0){ // perfectly classified
            return;
        }

        splitAttributeIndex =  -1;

        HashMap<Integer, Double> candidates = new HashMap<Integer, Double>();
        for(int i = 0; i < n; ++i){
            List<String> T = counts[i].getSubEventNames();
            double entropy_reduced = 0;
            for(int j=0; j < T.size(); ++j) {

                String t = T.get(j);
                double p_t = counts[i].getProbability(t);
                List<String> classNames = counts[i].getSubEventNames(t);
                double entropy_t = 0;
                for(int k=0; k < classNames.size(); ++k) {
                    double p_class_in_t = counts[i].getConditionalProbability(T.get(j), classNames.get(k));
                    entropy_t += (-p_class_in_t * log2(p_class_in_t));
                }
                entropy_reduced += p_t * entropy_t;
            }
            double information_gain = entropy_S - entropy_reduced;

            if(information_gain > 0){
                candidates.put(i, information_gain);
            }
        }

        if(candidates.isEmpty()){
            return;
        }

        double max_information_gain = 0;
        for(Integer candidateFeatureIndex : candidates.keySet()){
            double information_gain = candidates.get(candidateFeatureIndex);
            if(information_gain > max_information_gain){
                max_information_gain = information_gain;
                splitAttributeIndex = candidateFeatureIndex;
            }
        }

        List<String> T = counts[splitAttributeIndex].getSubEventNames();

        DataFrame[] batches = new DataFrame[T.size()];

        for(int i=0; i < batches.length; ++i){
            batches[i] = new BasicDataFrame();
        }

        for(int i=0; i < rowCount; ++i){
            DataRow row = batch.row(i);
            String attribute_value = columns.get(splitAttributeIndex);
            batches[T.indexOf(attribute_value)].addRow(row);
        }


        for(int i=0; i < batches.length; ++i){
            batches[i].lock();

            childNodes.add(new ID3TreeNode(batches[i], random, height+1, maxHeight, columns));
            childNodes.get(i).attributeValue = T.get(i);
        }
    }

    public static double heuristicCost(double n){
        if(n <= 1.0) return 0;
        return 2 * (Math.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n);
    }

    private double log2(double val){
        return Math.log(val) / Math.log(2);
    }

    private void updateClassLabel(DataFrame batch){
        HashMap<String, Integer> classLabelCounts = new HashMap<String, Integer>();
        for(int i = 0; i < batch.rowCount(); ++i){
            String label = batch.row(i).categoricalTarget();
            classLabelCounts.put(label, classLabelCounts.containsKey(label) ? classLabelCounts.get(label)+1 : 1);
        }
        int maxCount = 0;
        for(String label : classLabelCounts.keySet()){
            if(classLabelCounts.get(label) > maxCount){
                maxCount = classLabelCounts.get(label);
                classLabel = label;
            }
        }
        //System.out.println("label: "+classLabel+"\tcount: "+maxCount);
    }

    public String predict(DataRow row){
        if(childNodes != null){
            String value = this.columns.get(splitAttributeIndex);

            for(ID3TreeNode child : childNodes){

                if(child.attributeValue.equals(value)){
                    return child.predict(row);
                }
            }
        }
        return classLabel;
    }


    protected double pathLength(DataRow row){
        if(childNodes!=null){
            String value = columns.get(splitAttributeIndex);
            for(ID3TreeNode child : childNodes){
                if(child.attributeValue.equals(value)){
                    return child.pathLength(row)+1.0;
                }
            }
        }

        return heuristicCost(rowCount);
    }
}
