import {MnistData} from './data.js';
import {TensorLord} from './tensor.js' ;
const TL = new TensorLord ; 
const inputdims = [28,28,1] ; 
tf.setBackend('webgl');
function getModel() {
    const model = tf.sequential(); 
    model.add(tf.layers.conv2d({
      inputShape: inputdims,
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    model.add(tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    model.add(tf.layers.flatten());
    const NUM_OUTPUT_CLASSES = 10;
    model.add(tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax'
    }));
    const optimizer = tf.train.adam();
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


