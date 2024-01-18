const webcam = new Webcam(document.getElementById('wc'));
let isPredicting = true;
const MODEL_URL = './sign_language_model.json';


//Converts Images to greyscale to match model expected input
function convertToGreyscale(image) {
  return tf.tidy(() => {
    const greyscale = tf.image.rgbToGrayscale(image);
    return greyscale;
  });
}


async function predict() {
  //Load tfjs converted model
  const model = await tf.loadLayersModel(MODEL_URL);
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      //Get webcam captured image, convert to grey, and predict
      const img = webcam.capture();
      const predictions = model.predict(convertToGreyscale(img));
      return predictions.as1D().argMax();
    });
    const classId = await predictedClass.arraySync();
    console.log(classId)
    var predictionText = "";
      //Categorical classification and DOM update (Shifted to ASCII 'A')
      document.getElementById("prediction").innerHTML = `
        <span style="font-size: 48px; color: blue;">${String.fromCharCode(65 + classId)}</span><br>`;
    await tf.nextFrame();
  }
}

async function init() {
  await webcam.setup();
  const webcamElement = document.getElementById('wc');
  webcamElement.classList.add('mirrored');
  predict();
}

init();