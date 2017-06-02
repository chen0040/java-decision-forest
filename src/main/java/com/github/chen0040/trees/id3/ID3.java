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
public class ID3 implements Cloneable {
    private static Random rand = new Random();
    private KMeansDiscretizer discretizer=new KMeansDiscretizer();
    private ID3TreeNode tree;
    private int maxHeight = 1000;

    public void copy(ID3 rhs2) throws CloneNotSupportedException {
        tree = rhs2.tree==null ? null : (ID3TreeNode)rhs2.tree.clone();
        maxHeight = rhs2.maxHeight;
        discretizer = rhs2.discretizer == null ? null : rhs2.discretizer.makeCopy();
    }

    @Override
    public Object clone() throws CloneNotSupportedException {
        ID3 clone = (ID3)super.clone();
        clone.copy(this);

        return clone;
    }

    public ID3(){

    }

    public String classify(DataRow tuple) {
        tuple = discretizer.transform(tuple);
        return tree.predict(tuple);
    }

    public void fit(DataFrame batch) {

        batch = discretizer.fitAndTransform(batch);

        List<String> columns = batch.row(0).getCategoricalColumnNames();
        tree = new ID3TreeNode(batch, rand, 0, maxHeight, columns);
    }
}
