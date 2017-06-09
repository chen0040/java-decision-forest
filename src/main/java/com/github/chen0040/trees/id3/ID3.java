package com.github.chen0040.trees.id3;


import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.data.utils.discretizers.KMeansDiscretizer;
import lombok.Getter;
import lombok.Setter;

import java.util.List;
import java.util.Random;


/**
 * Created by xschen on 23/8/15.
 */
@Getter
@Setter
public class ID3 {
    private static Random rand = new Random();
    private KMeansDiscretizer discretizer=new KMeansDiscretizer();
    private ID3TreeNode tree;
    private int maxHeight = 1000;
    private boolean discretized;

    public ID3(){
        discretized = true;
    }

    public ID3(boolean discretized){
        this.discretized = discretized;
    }

    public String classify(DataRow tuple) {
        if(discretized) {
            tuple = discretizer.transform(tuple);
        }
        return tree.predict(tuple);
    }

    public void fit(DataFrame batch) {

        if(discretized) {
            batch = discretizer.fitAndTransform(batch);
        }

        List<String> columns = batch.row(0).getCategoricalColumnNames();
        tree = new ID3TreeNode(batch, rand, 0, maxHeight, columns);
    }
}
