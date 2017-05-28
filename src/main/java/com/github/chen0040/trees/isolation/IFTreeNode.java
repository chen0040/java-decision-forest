package com.github.chen0040.trees.isolation;

import com.github.chen0040.data.frame.BasicDataFrame;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.UUID;


/**
 * Created by xschen on 17/8/15.
 */
public class IFTreeNode implements Cloneable {
    private int rowCount;
    private int featureIndex;
    private double splitPoint;
    private List<IFTreeNode> childNodes;
    private String nodeId;

    public void copy(IFTreeNode rhs) throws CloneNotSupportedException {
        rowCount = rhs.rowCount;
        featureIndex = rhs.featureIndex;
        splitPoint = rhs.splitPoint;
        nodeId = rhs.nodeId;

        childNodes =  null;
        if(rhs.childNodes != null){
            childNodes = new ArrayList<IFTreeNode>();
            for(int i=0; i < rhs.childNodes.size(); ++i){
                childNodes.add((IFTreeNode)rhs.childNodes.get(i).clone());
            }
        }
    }

    @Override
    public Object clone() throws CloneNotSupportedException {
        IFTreeNode clone = (IFTreeNode)super.clone();
        clone.copy(this);
        
        return clone;
    }

    public IFTreeNode(){
        childNodes = new ArrayList<>();
        nodeId = UUID.randomUUID().toString();
    }

    private static double epsilon(){
        return 0.00000000001;
    }

    public IFTreeNode(DataFrame batch, Random random, int height, int maxHeight){
        rowCount = batch.rowCount();
        nodeId = UUID.randomUUID().toString();

        if(rowCount <= 1 || height == maxHeight){
            return;
        }

        int n = batch.row(0).toArray().length;

        double[] minValues = new double[n];
        double[] maxValues = new double[n];

        for(int i=0; i < rowCount; ++i){
            DataRow tuple = batch.row(i);
            double[] x = tuple.toArray();
            int n_min = Math.min(n, x.length);
            for(int j=0; j < n_min; ++j){
                minValues[j] = Math.min(minValues[j], x[j]);
                maxValues[j] = Math.max(maxValues[j], x[j]);
            }
        }

        List<Integer> featureList = new ArrayList<>();
        for(int i=0; i < n; ++i){
            if(minValues[i] < maxValues[i] && (maxValues[i] - minValues[i]) > epsilon()){
                featureList.add(i);
            }
        }

        if(featureList.isEmpty()){
            return;
        }else{
            featureIndex =  random.nextInt(featureList.size());
            splitPoint = minValues[featureIndex] + (maxValues[featureIndex] - minValues[featureIndex]) * random.nextDouble();

            DataFrame[] batches = new DataFrame[2];
            childNodes = new ArrayList<>();

            for(int i=0; i < batches.length; ++i){
                batches[i]= new BasicDataFrame();
            }

            for(int i=0; i < rowCount; ++i){
                DataRow tuple = batch.row(i);
                double[] x = tuple.toArray();
                double featureValue = x[featureIndex];
                if(featureValue < splitPoint){
                    batches[0].addRow(tuple);
                }else{
                    batches[1].addRow(tuple);
                }
            }

            for(int i=0; i < batches.length; ++i){
                batches[i].lock();
                childNodes.add(new IFTreeNode(batches[i], random, height+1, maxHeight));
            }

        }
    }

    public static double heuristicCost(double n){
        if(n <= 1.0) return 0;
        return 2 * (Math.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n);
    }

    public int getTupleCount() {
        return rowCount;
    }

    public void setTupleCount(int rowCount) {
        this.rowCount = rowCount;
    }

    public int getFeatureIndex() {
        return featureIndex;
    }

    public void setFeatureIndex(int featureIndex) {
        this.featureIndex = featureIndex;
    }

    public double getSplitPoint() {
        return splitPoint;
    }

    public void setSplitPoint(double splitPoint) {
        this.splitPoint = splitPoint;
    }

    public List<IFTreeNode> getChildNodes() {
        return childNodes;
    }

    public void setChildNodes(ArrayList<IFTreeNode> childNodes) {
        this.childNodes = childNodes;
    }

    protected double pathLength(DataRow tuple){
        if(childNodes==null){
            return heuristicCost(rowCount);
        }else{
            double[] x = tuple.toArray();
            double featureValue = x[featureIndex];
            if(featureValue < splitPoint){
                return childNodes.get(0).pathLength(tuple)+1.0;
            }else{
                return childNodes.get(1).pathLength(tuple)+1.0;
            }
        }
    }
}
