//Create a function
async function getPrediction() {
    //Create a sequential model
    const model = tf.sequential();
    //Add the hidden layer
    model.add(tf.layers.dense({inputShape: [4], units: 1}));
    //Compile our model
    model.compile(
      {
        optimizer: 'sgd',
        loss: 'meanSquaredError',
        metrics: ['accuracy']
      }
    )
    //Train our model
    const data = tf.tensor([1, 3, 7, 5], [1, 4]);
    const labels = tf.tensor([4], [1, 1]);
    await model.fit(data, labels, {epochs: 100});
    //Make prediction
    const testData = tf.tensor([2, 3, 8, 12], [1, 4]);
    //Get the our target location on the web page
    document.getElementById('content').innerText = model.predict(testData);
  }
  //Call the function
  getPrediction();