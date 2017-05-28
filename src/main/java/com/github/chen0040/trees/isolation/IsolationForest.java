package com.github.chen0040.trees.isolation;


import com.github.chen0040.data.frame.BasicDataFrame;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;


/**
 * Created by xschen on 17/8/15.
 */
@Getter
@Setter
public class IsolationForest implements Cloneable {

    private double threshold = 0.5;
    private int treeCount = 100;

    private static final Random random = new Random();
    private static final double log2 = Math.log(2);

    @Setter(AccessLevel.NONE)
    private List<IFTreeNode> trees;

    @Setter(AccessLevel.NONE)
    private int rowCount;




    public void copy(IsolationForest rhs2) throws CloneNotSupportedException {

        rowCount = rhs2.rowCount;

        trees = null;
        if(rhs2.trees != null) {

            trees = new ArrayList<>();

            for (int i = 0; i < rhs2.trees.size(); ++i) {
                trees.add((IFTreeNode)rhs2.trees.get(i).clone());
            }
        }
    }

    @Override
    public Object clone() throws CloneNotSupportedException {
        IsolationForest clone = (IsolationForest)super.clone();
        clone.copy(this);

        return clone;
    }

    public IsolationForest(){

    }

    private static double log2(double n){
        return Math.log(n) / log2;
    }


    public boolean isAnomaly(DataRow tuple) {
        return evaluate(tuple) > threshold;
    }

    public void fit(DataFrame batch) {
        trees = new ArrayList<>();
        rowCount = batch.rowCount();
        int maxHeight = (int)Math.ceil(log2(rowCount));
        for(int i=0; i < treeCount; ++i){
            DataFrame treeBatch = randomize(batch);
            IFTreeNode tree = new IFTreeNode(treeBatch, random, 0, maxHeight);
            trees.add(tree);
        }
    }

    private DataFrame randomize(DataFrame dataFrame){
        DataFrame treeBatch = new BasicDataFrame();

        List<DataRow> list =new ArrayList<DataRow>();
        for(int i = 0; i < dataFrame.rowCount(); ++i){
            list.add(dataFrame.row(i));
        }
        Collections.shuffle(list);
        for(int i=0; i < list.size(); ++i){
            treeBatch.addRow(list.get(i));
        }

        treeBatch.lock();
        return treeBatch;
    }

    public double[] getDistributionScores(DataRow tuple){
        double[] scores = new double[2];
        scores[0] = evaluate(tuple);
        scores[1] = 1- scores[0];

        return scores;
    }

    public double evaluate(DataRow tuple) {
        double avgPathLength = 0;
        for(int i=0; i < trees.size(); ++i){
            avgPathLength += trees.get(i).pathLength(tuple);
        }
        avgPathLength /= trees.size();

        return Math.pow(2, - avgPathLength / IFTreeNode.heuristicCost(rowCount));
    }


    public DataFrame fitAndTransform(DataFrame data) {
        fit(data);
        data = data.makeCopy();
        for(int i=0; i < data.rowCount(); ++i) {
            DataRow row = data.row(i);
            boolean anomaly = isAnomaly(row);
            row.setCategoricalTargetCell("anomaly", anomaly ? "1" : "0");
        }

        return data;
    }
}
