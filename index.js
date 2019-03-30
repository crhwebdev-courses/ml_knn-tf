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
      .sort((a, b) => (a.arraySync()[0] > b.arraySync()[0] ? 1 : -1))
      .slice(0, k)
      .reduce((acc, pair) => acc + pair.arraySync()[1], 0) / k
  );
}

const predictionPoint = tf.tensor(testFeatures[0]);

const featuresTensor = tf.tensor(features);
const labelsTensor = tf.tensor(labels);
const testFeaturesTensor = tf.tensor(testFeatures);
const testLabelsTensor = tf.tensor(testLabels);

const result = knn(featuresTensor, labelsTensor, predictionPoint, 10);

console.log('guess', result, 'result', testLabels[0][0]);
