# java-decision-forest

Package implements decision tree and ensemble methods

[![Build Status](https://travis-ci.org/chen0040/java-decision-forest.svg?branch=master)](https://travis-ci.org/chen0040/java-decision-forest) [![Coverage Status](https://coveralls.io/repos/github/chen0040/java-decision-forest/badge.svg?branch=master)](https://coveralls.io/github/chen0040/java-decision-forest?branch=master) 

# Features

* ID3 Decision Tree with both numerical and categorical inputs 
* Isolation Forest for Anomaly Detection

# Install 

# Usage

### Anomaly Detection

The problem that we will be using as demo is the following anomaly detection problem:

![scki-learn example for one-class](http://scikit-learn.org/stable/_images/sphx_glr_plot_oneclass_001.png)


Below is the sample code which illustrates how to use Isolation Forest to detect outliers in the above problem:

```java
DataQuery.DataFrameQueryBuilder schema = DataQuery.blank()
      .newInput("c1")
      .newInput("c2")
      .newOutput("anomaly")
      .end();

Sampler.DataSampleBuilder negativeSampler = new Sampler()
      .forColumn("c1").generate((name, index) -> randn() * 0.3 + (index % 2 == 0 ? -2 : 2))
      .forColumn("c2").generate((name, index) -> randn() * 0.3 + (index % 2 == 0 ? -2 : 2))
      .forColumn("anomaly").generate((name, index) -> 0.0)
      .end();

Sampler.DataSampleBuilder positiveSampler = new Sampler()
      .forColumn("c1").generate((name, index) -> rand(-4, 4))
      .forColumn("c2").generate((name, index) -> rand(-4, 4))
      .forColumn("anomaly").generate((name, index) -> 1.0)
      .end();

DataFrame data = schema.build();

data = negativeSampler.sample(data, 20);
data = positiveSampler.sample(data, 20);

System.out.println(data.head(10));

IsolationForest method = new IsolationForest();
method.setThreshold(0.38);
DataFrame learnedData = method.fitAndTransform(data);

BinaryClassifierEvaluator evaluator = new BinaryClassifierEvaluator();

for(int i = 0; i < learnedData.rowCount(); ++i){
 boolean predicted = learnedData.row(i).categoricalTarget().equals("1");
 boolean actual = data.row(i).target() == 1.0;
 evaluator.evaluate(actual, predicted);
 logger.info("predicted: {}\texpected: {}", predicted, actual);
}
```
