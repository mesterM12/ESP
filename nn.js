import {MnistData} from './data.js'; // load the data class 
import {TensorLord} from './tensor.js' ; // load the custom TensorFlow controller class 
const TL = new TensorLord ; // new Tensor Lord
const inputdims = [28,28,1] ; // Dimensions of the input pictures 
const NUM_OUTPUT_CLASSES = 10;  // number of outputs 
function getModel() {
tf.setbackend('cpu');
const model = tf.sequential(); 
model.add(tf.layers.dense({inputShape: inputdims , units: 32, activation: 'relu'})); //  Input Layer
model.add(tf.layers.dense({units: 16 , activation: 'relu'})); // 1st Hidden layer 
model.add(tf.layers.dense({units: 16, activation: 'relu'})); // 2ed Hidden layer 
    model.add(tf.layers.flatten()); // flatten the model
    model.add(tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
      })); // output using softmax to output probability values as binary
    const optimizer = tf.train.adam(); // using adam to optimize. 
    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
    return model;
  }
async function run() {  
    const data = new MnistData();
    await data.load();
    const model = getModel();
  tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, model);
  await TL.train(model, data);
  await TL.showAccuracy(model, data);
  await TL.showConfusion(model, data);
}
let trainBTN = document.getElementById("train")  ; 
trainBTN.addEventListener("click" , ()=>{
run();
});
