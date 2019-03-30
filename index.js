// require need to tell tensor flow what to use for calucations
// - using cpu right now
require('@tensorflow/tfjs-node');

// require('@tensorflow/tfjs-node-gpu');

const tf = require('@tensorflow/tfjs');

const loadCSV = require('./load-csv');

let { features, labels, testFeatures, testLabels } = loadCSV(
  './kc_house_data.csv',
  {
    shuffle: true,
    splitTest: 10,
    dataColumns: ['lat', 'long'],
    labelColumns: ['price']
  }
);

function knn(features, labels, predictionPoint, k) {
  return (
    features
      .sub(predictionPoint)
      .pow(2)
      .sum(1)
      .pow(0.5)
      .expandDims(1)
      .concat(labels, 1)
      .unstack()
      .sort((a, b) => (a.get(0) > b.get(0) ? 1 : -1))
      .slice(0, k)
      .reduce((acc, pair) => acc + pair.get(1), 0) / k
  );
}

features = tf.tensor(features);
labels = tf.tensor(labels);
testFeatures = tf.tensor(testFeatures);
testLabels = tf.tensor(testLabels);

const result = knn(features, labels, tf.tensor(testFeatures[0]), 10);

console.log('guess', result, testLabels[0][0]);
