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

const emotionSelector = document.getElementById('emotions');

let faceLandmarks = [];
let handLandmarks = [];

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

// add current pose to training data
addTrainingsDataButton.addEventListener('click', () => {
    trainButton.disabled = false;
    console.log('adding data:');
    console.log(normalizeHandData());
    neuralNetwork.addData(normalizeHandData(), {label: `${emotionSelector.value}`});
});

// start AI Training Model (Takes some time to load)
trainButton.addEventListener('click', () => {
    startTraining();
})

// detect the current pose
detectButton.addEventListener('click', () => {
    detectPose();
})

saveButton.addEventListener('click', () => {
    saveTraining();
})

async function startTraining() {
    await neuralNetwork.normalizeData();
    neuralNetwork.train({epochs: 20}, () => finishedTraining());
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
    alert('done');
}

// init
async function init() {
    await camera.start();
    toggleVideoButton.disabled = false;
    addTrainingsDataButton.disabled = false;
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
        console.log('Normalized flatHandArray', flatHandArray);
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