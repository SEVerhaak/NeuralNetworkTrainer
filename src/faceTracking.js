import {FaceMesh} from '@mediapipe/face_mesh';
import {Hands} from '@mediapipe/hands';
import {Camera} from '@mediapipe/camera_utils';
import {FACEMESH_TESSELATION, FACEMESH_RIGHT_EYE, FACEMESH_LEFT_EYE, FACEMESH_FACE_OVAL} from '@mediapipe/face_mesh';
import {HAND_CONNECTIONS} from '@mediapipe/hands';
import {drawConnectors, drawLandmarks} from '@mediapipe/drawing_utils';

// vars
const video = document.getElementById('input_video');
const canvas = document.getElementById('face_canvas');
const ctx = canvas.getContext('2d');
const startWebcamButton = document.getElementById('start-webcam')
const toggleVideoButton = document.getElementById('toggle-video')
const addTrainingsDataButton = document.getElementById('add-to-training-data-button');
const trainButton = document.getElementById('train-button')
const detectButton = document.getElementById('detect-button')
const saveButton = document.getElementById('save-button')
const testButton = document.getElementById('test-button')

const emotionSelector = document.getElementById('emotions');

let faceLandmarks = [];
let handLandmarks = [];

let trainingsData = {}
let testData

// NeuralNetwork init
ml5.setBackend("webgl");
const neuralNetwork = ml5.neuralNetwork({task: 'classification', debug: true})

// Camera var
const camera = new Camera(video, {
    onFrame: async () => {
        await faceMesh.send({image: video});
        await hands.send({image: video});
    },
    width: 640,
    height: 480,
});

// event listeners
// start button
startWebcamButton.addEventListener('click', () => {
    init();
})
// toggle video backdrop button
toggleVideoButton.addEventListener('click', () => {
    if (video.style.display === 'block') {
        video.style.display = 'none';
    } else {
        video.style.display = 'block';
    }
})

testButton.addEventListener('click', () => {
    testTraining();
})

// add current pose to training data
addTrainingsDataButton.addEventListener('click', () => {
    trainButton.disabled = false;

    // adds a key to an object based on the current selected value
    const key = emotionSelector.value;
    const data = normalizeHandData();

    // if it doesnt exist create the key entry
    if (!trainingsData[key]) {
        trainingsData[key] = [];
    }

    //push data into the big array
    trainingsData[key].push(data);

    console.log('trainingsData object:');
    console.log(trainingsData);

    //neuralNetwork.addData(normalizeHandData(), {label: `${emotionSelector.value}`});
});

function splitTrainingData(dataObject, trainRatio = 0.8) {
    const trainData = {};
    const testData = {};

    for (const key in dataObject) {
        if (dataObject.hasOwnProperty(key)) {
            const allSamples = dataObject[key];
            const shuffled = [...allSamples].sort(() => Math.random() - 0.5); // Shuffle randomly

            const splitIndex = Math.floor(shuffled.length * trainRatio);
            trainData[key] = shuffled.slice(0, splitIndex);
            testData[key] = shuffled.slice(splitIndex);
        }
    }

    return { trainData, testData };
}


// start AI Training Model (Takes some time to load)
trainButton.addEventListener('click', () => {
    startTraining().then(finishedTraining);
}, false);

// detect the current pose
detectButton.addEventListener('click', () => {
    detectPose();
})

saveButton.addEventListener('click', () => {
    saveTraining();
})


async function startTraining() {
    let splitData = splitTrainingData(trainingsData);

    testData = splitData.testData;
    console.log(testData);

    await addDataSet(splitData.trainData);

    await neuralNetwork.normalizeData();
    neuralNetwork.train({epochs: 20}, () => alert('Done!'));
}


async function detectPose() {
    const results = await neuralNetwork.classify(normalizeHandData())
    console.log(results)
}

async function saveTraining() {
    neuralNetwork.save("model", () => console.log("model was saved!"))
}

function finishedTraining() {
    saveButton.disabled = false;
}

async function testTraining(){
    for (const key in testData) {
        if (testData.hasOwnProperty(key)) {
            const samples = testData[key];
            for (let i = 0; i < samples.length; i++) {
                const sample = samples[i];
                const results = await neuralNetwork.classify(sample)

                const bestPrediction = results.reduce((prev, current) =>
                    current.confidence > prev.confidence ? current : prev
                ).label;

                if (bestPrediction === key) {
                    console.log('correct')
                } else {
                    console.log('incorrect')
                }
                //console.log(bestPrediction);
            }
        }
    }
}

// init
async function init() {
    await camera.start();
    toggleVideoButton.disabled = false;
    addTrainingsDataButton.disabled = false;
    emotionSelector.disabled = false;
}

// ==== FaceMesh Setup ====
const faceMesh = new FaceMesh({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
});

faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
});

// ==== Hands Setup ====
const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
});

hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
});

// make data ready to be learned
function normalizeHandData() {
    const flatHandArray = [];

    if (handLandmarks[0].length > 0) {
        const wrist = handLandmarks[0][0]; // Wrist is always index 0

        for (let i = 0; i < handLandmarks[0].length; i++) {
            const landmark = handLandmarks[0][i];

            const x = landmark.x - wrist.x;
            const y = landmark.y - wrist.y;
            const z = landmark.z - wrist.z;

            flatHandArray.push(x, y, z);
        }
        return flatHandArray;
    }
    return null;
}

function normalizeFaceData() {
    const flatFaceArray = [];

    if (faceLandmarks[0].length > 0) {
        const noseTip = faceLandmarks[0][1]; // Nose tip as reference (index 1)

        for (let i = 0; i < faceLandmarks[0].length; i++) {
            const landmark = faceLandmarks[0][i];

            const x = landmark.x - noseTip.x;
            const y = landmark.y - noseTip.y;
            const z = landmark.z - noseTip.z;

            flatFaceArray.push(x, y, z);
        }

        console.log('Normalized flatFaceArray', flatFaceArray);
        return flatFaceArray;
    }
    return null;
}

async function addDataSet(trainData){
    for (const key in trainData) {
        if (trainData.hasOwnProperty(key)) {
            const samples = trainData[key];
            for (let i = 0; i < samples.length; i++) {
                const sample = samples[i];
                neuralNetwork.addData(sample, { label: `${key}` });
            }
        }
    }
}


// Draw loop
function drawLandmarksFunc() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw face landmarks with MediaPipe style
    for (const landmarks of faceLandmarks) {
        drawConnectors(ctx, landmarks, FACEMESH_TESSELATION, {color: '#C0C0C070', lineWidth: 1});
        drawConnectors(ctx, landmarks, FACEMESH_RIGHT_EYE, {color: 'red'});
        drawConnectors(ctx, landmarks, FACEMESH_LEFT_EYE, {color: 'red'});
        drawConnectors(ctx, landmarks, FACEMESH_FACE_OVAL, {color: 'white'});
        drawLandmarks(ctx, landmarks, {color: 'white', lineWidth: 1, radius: 1});
    }

    for (const hand of handLandmarks) {
        drawConnectors(ctx, hand, HAND_CONNECTIONS, {
            color: 'blue',
            lineWidth: 2
        });

        drawLandmarks(ctx, hand, {
            color: 'green',
            lineWidth: 1
        });
    }
}

// FaceMesh callback
/*
faceMesh.onResults((results) => {
    faceLandmarks = results.multiFaceLandmarks || [];
    drawLandmarksFunc();
});
 */

// Hands callback
hands.onResults((results) => {
    handLandmarks = results.multiHandLandmarks || [];
    drawLandmarksFunc();
});