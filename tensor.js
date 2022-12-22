const options = {
    classNames:['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'],
    BATCH_SIZE:512,
    TRAIN_DATA_SIZE:5500,
    TEST_DATA_SIZE:1000,
    shuffle:true,
    epochs:10,
    xShapeTest:[1000, 28, 28, 1],
    xShapeTrain:[5500 , 28, 28 , 1 ]
}
export class TensorLord {
    async train(model, data) {
        const metrics=['loss', 'val_loss', 'acc', 'val_acc'];
        const container = {
            name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
        };
        const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
        const [trainXs, trainYs] = tf.tidy(() => {
            const d = data.nextTrainBatch(options.TRAIN_DATA_SIZE);
            return [
                d.xs.reshape(options.xShapeTrain),
                d.labels
            ];
        });
        const [testXs, testYs] = tf.tidy(() => {
            const d = data.nextTestBatch(options.TEST_DATA_SIZE);
            return [
                d.xs.reshape(options.xShapeTest),
                d.labels
            ];
        });
        return model.fit(trainXs, trainYs, {
            batchSize: options.BATCH_SIZE,
            validationData: [testXs, testYs],
            epochs: options.epochs,
            shuffle: options.shuffle,
            callbacks: fitCallbacks
        });
    }
    doPrediction(model, data) {
        const testData = data.nextTestBatch(options.TEST_DATA_SIZE);
        const testxs = testData.xs.reshape(options.xShapeTest);
        const labels = testData.labels.argMax(-1);
        const preds = model.predict(testxs).argMax(-1);
        testxs.dispose();
        return [preds, labels];
    }
    async showAccuracy(model, data) {
        const [preds, labels] = this.doPrediction(model, data);
        const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
        const container = { name: 'Accuracy', tab: 'Evaluation' };
        tfvis.show.perClassAccuracy(container, classAccuracy, options.classNames);
        labels.dispose();
    }
    async showConfusion(model, data) {
        const [preds, labels] = this.doPrediction(model, data);
        const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
        const container = { name: 'Confusion Matrix', tab: 'Evaluation' };
        tfvis.render.confusionMatrix(container, { values: confusionMatrix, tickLabels: options.classNames });
        labels.dispose();
    }
}
