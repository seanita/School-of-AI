const classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');

let net;

async function setupWebcam() {
    return new Promise((resolve, reject) => {
        const navigatorAny = navigator;
        navigator.getUserMedia = navigator.getUserMedia ||
            navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
            navigatorAny.msGetUserMedia;
        if (navigator.getUserMedia) {
            navigator.getUserMedia({ video: true },
                stream => {
                    webcamElement.srcObject = stream;
                    webcamElement.addEventListener('loadeddata', () => resolve(), false);
                },
                error => reject());
        } else {
        reject();
        }
    });
}

async function app() {
    console.log("Loading mobilenet...");

    // Load the model
    net = await mobilenet.load();
    console.log("Successfully loaded model");

    await setupWebcam();

    //Reads an image from webcam and associates with a class index
    const addExample = classId => {
        //Get intermediate MobileNet activation 'conv_preds' and pass to KNN classifire
        const activation = net.infer(webcamElement, 'conv_preds');

        //Passing activation to classifer
        classifier.addExample(activation, classId);
    };


    
    //When clicking a button, add an example for that class.
    document.getElementById('class-a').addEventListener('click', () => addExample(0));
    document.getElementById('class-b').addEventListener('click', () => addExample(1));
    document.getElementById('class-c').addEventListener('click', () => addExample(2));

    while (true) {
        //const result = await net.classify(webcamElement);
        if (classifier.getNumClasses() > 0) {
            //Get teh activation fro mobilenet from the webcam
            const activation = net.infer(webcamElement, 'conv_preds');
            //Get the most likely class and confidences from the classifier module
            const result = await classifier.predictClass(activation);
            
            const classes = ['A', 'B', 'C'];
            document.getElementById('console').innerText = `
                prediction: ${classes[result.classIndex]}\n
                probability: ${result.confidences[result.classIndex]}
                `;
        }  
    await tf.nextFrame();
    }
}
app();