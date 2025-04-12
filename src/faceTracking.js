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
const addTestDataButton = document.getElementById('add-to-test-data-button');
const trainButton = document.getElementById('train-button')
const detectButton = document.getElementById('detect-button')
const saveButton = document.getElementById('save-button')
const testButton = document.getElementById('test-button')
const downloadButton = document.getElementById('download-json-button')

const emotionSelector = document.getElementById('emotions');

let faceLandmarks = [];
let handLandmarks = [];

let trainingsData = {}
let testData
let splitData

// NeuralNetwork init
ml5.setBackend("webgl");

const options = {
    task: 'classification',
    debug: true,
    layers: [
        {
            type: 'dense',
            units: 32,
            activation: 'relu',
        }, {
            type: 'dense',
            units: 32,
            activation: 'relu',
        }, {
            type: 'dense',
            units: 32,
            activation: 'relu',
        },
        {
            type: 'dense',
            activation: 'softmax',
        },
    ]
}

const neuralNetwork = ml5.neuralNetwork(options)

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
    downloadButton.disabled = false;

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

addTestDataButton.addEventListener('click', () => {
    const key = emotionSelector.value;
    const data = normalizeHandData();

    if(splitData.testData[key]) {
        splitData.testData[key].push(data);
    }

    console.log(splitData.testData);
})

downloadButton.addEventListener('click', () => {
    downloadObjectAsJson(trainingsData, 'JSON_DATA');
})

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


document.getElementById('jsonFileInput').addEventListener('change', function(event) {
    const file = event.target.files[0];

    if (file) {
        const reader = new FileReader();

        reader.onload = function(e) {
            try {
                trainingsData = JSON.parse(e.target.result);
                console.log('Parsed JSON Object:', trainingsData);
                trainButton.disabled = false;
                downloadButton.disabled = false;
            } catch (err) {
                console.error('Invalid JSON file:', err);
                alert('The uploaded file is not valid JSON.');
            }
        };

        reader.readAsText(file);
    }
});


async function startTraining() {
    addTestDataButton.disabled = false;
    detectButton.disabled = false;
    testButton.disabled = false;

    splitData = splitTrainingData(trainingsData);

    testData = splitData.testData;
    console.log(testData);

    await addDataSet(splitData.trainData);

    await neuralNetwork.normalizeData();
    neuralNetwork.train({epochs: 50}, () => alert('Done!'));
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

async function testTraining() {
    const correctArray = [];
    const wrongArray = [];
    const confusionMatrix = {};
    const allLabels = new Set();

    // Build confusion matrix
    for (const key in testData) {
        if (testData.hasOwnProperty(key)) {
            const samples = testData[key];

            // Loop through the array of each key
            for (let i = 0; i < samples.length; i++) {
                const sample = samples[i];
                const results = await neuralNetwork.classify(sample);

                let bestPrediction = results[0];
                for (let j = 1; j < results.length; j++) {
                    if (results[j].confidence > bestPrediction.confidence) {
                        bestPrediction = results[j];
                    }
                }

                allLabels.add(key);
                allLabels.add(bestPrediction.label);

                if (!confusionMatrix[key]) {
                    confusionMatrix[key] = {};
                }
                if (!confusionMatrix[key][bestPrediction.label]) {
                    confusionMatrix[key][bestPrediction.label] = 0;
                }
                confusionMatrix[key][bestPrediction.label]++;

                if (bestPrediction.label === key) {
                    correctArray.push(bestPrediction.label);
                    console.log('correct');
                } else {
                    wrongArray.push(bestPrediction.label);
                    console.log('incorrect');
                }
            }
        }
    }

    const total = correctArray.length + wrongArray.length;
    const accuracy = ((correctArray.length / total) * 100).toFixed(2);

    // Build HTML table
    const labels = Array.from(allLabels).sort();
    let html = `<h3>Accuracy and Confusion Matrix</h3>`;
    html += `<p>Correct: ${correctArray.length}, Wrong: ${wrongArray.length}, Accuracy: ${accuracy}%</p>`;
    html += `<table class="confusion-table"><thead><tr><th>Actual \\ Predicted</th>`;

    // Add headers for each label
    for (const label of labels) {
        html += `<th>${label}</th>`;
    }
    html += `</tr></thead><tbody>`;

    // Add rows for confusion matrix
    for (const actual of labels) {
        html += `<tr><td class="row-label">${actual}</td>`;
        for (const predicted of labels) {
            const count = confusionMatrix[actual]?.[predicted] || 0;

            // Set color: green for correct, red for incorrect
            let bgColor = "";
            if (count > 0) {
                if (actual === predicted) {
                    bgColor = "rgba(0, 200, 0, 0.5)"; // Green for correct
                } else {
                    bgColor = "rgba(200, 0, 0, 0.5)"; // Red for incorrect
                }
            }

            html += `<td style="background-color: ${bgColor}">${count}</td>`;
        }
        html += `</tr>`;
    }

    html += `</tbody></table>`;

    // Inject into the page
    document.getElementById("confusion-matrix").innerHTML = html;
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

function downloadObjectAsJson(exportObj, exportName){
    let dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportObj));
    let downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href",     dataStr);
    downloadAnchorNode.setAttribute("download", exportName + ".json");
    document.body.appendChild(downloadAnchorNode); // required for firefox
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
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
    console.log('Training data:', trainData);
    console.log('Test data:', testData);

    return { trainData, testData };
}

// Hands callback
hands.onResults((results) => {
    handLandmarks = results.multiHandLandmarks || [];
    drawLandmarksFunc();
});