require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");


function knn(locationFeatures, labels, predictionPoint, k){
    const {mean, variance} = tf.moments(locationFeatures, 0);
    const scaledPred = predictionPoint.sub(mean).div(variance.pow(0.5))

return locationFeatures
.sub(mean)
.div(variance.pow(0.5))
.sub(scaledPred)
.pow(2)
.sum(1)
.pow(0.5)
.expandDims(1)
.concat(labels,1 )
.unstack() 
.sort((a,b)=>{
 a.get(0)>b.get(0) ?1:-1
}) 
.slice(0,k).reduce((acc, obj)=>
   acc+ obj.get(1), 0)/k

}


let {features, labels, testFeatures, testLabels} = loadCSV('kc_house_data.csv',
{
    shuffle:true,
    splitTest: 10,
    dataColumns: ['lat', 'long','sqft_lot','sqft_living'],
    labelColumns:['price']
});

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testPoint, index )=> {
    const result = knn(
        features,
        labels,
        tf.tensor(testPoint), 10
       );
       const err = ((testLabels[index][0]-result) / testLabels[index][0])*100; // error = expctdvalue- predictedvalue/expectedvalue
       console.log("predicted", result,testLabels[index][0])
       console.log(err);
});


