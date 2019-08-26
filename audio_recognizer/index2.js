let recognizer;

const NUM_FRAMES = 3;
let examples = [];

function collect(label) {
    if(recognizer.isListening()) {
        return recognizer.stopListening();
    }
    if(label==null) {
        return;
    }
    recognizer.listen(async ({spectrogram: {frameSize,data}}) => {
        let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
        examples.push({vals, label});
        document.querySelector('#console').textContent = `${examples.length} examples collected`;
    },
    {
        overlapFactor: 0.999,
        includeSpectrogram: true,
        involveCallbackOnNoiseAndUnknown: true
    }); 
}

function normalize(x){
    const mean = -100;
    const std = 10;
    return x.map(x => (x-mean)/std);
}

async function app() {
    recognizer = speechCommands.create('BROWSER_FFT');
    await recognizer.ensureModeLoaded();
}
