package com.github.chen0040.trees.isolation;


import com.github.chen0040.data.frame.BasicDataFrame;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;


/**
 * Created by xschen on 17/8/15.
 */
public class IsolationForest implements Cloneable {

    private double threshold = 0.5;
    private int treeCount = 100;
    private static final double log2 = Math.log(2);
    private ArrayList<IFTreeNode> trees;
    private int batchSize;
    private static Random random = new Random();


    public void copy(IsolationForest rhs2) throws CloneNotSupportedException {

        batchSize = rhs2.batchSize;

        trees = null;
        if(rhs2.trees != null) {

            trees = new ArrayList<IFTreeNode>();

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

    public int getBatchSize(){
        return batchSize;
    }

    public void fit(DataFrame batch) {
        trees = new ArrayList<>();
        batchSize = batch.rowCount();
        int maxHeight = (int)Math.ceil(log2(batchSize));
        for(int i=0; i < treeCount; ++i){
            DataFrame treeBatch = randomize(batch);
            IFTreeNode tree = new IFTreeNode(treeBatch, random, 0, maxHeight);
            trees.add(tree);
        }
    }

    private DataFrame randomize(DataFrame batch){
        DataFrame treeBatch = new BasicDataFrame();

        List<DataRow> list =new ArrayList<DataRow>();
        for(int i = 0; i < batch.rowCount(); ++i){
            list.add(batch.row(i));
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

        return Math.pow(2, - avgPathLength / IFTreeNode.heuristicCost(batchSize));
    }
}
