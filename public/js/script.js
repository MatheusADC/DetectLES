// const MODEL_PATH = "/modelo-lupus2/";
// const MODEL_PATH = "/model-teach/";
const MODEL_PATH = "/modelo-lupus4/";

const imageUpload = document.getElementById('imageUpload');
const uploadLabel = document.getElementById('upload-label');
const imagePreview = document.getElementById('image-preview');
const previewContainer = document.getElementById('preview-container');
const imageInfo = document.getElementById('image-info');
const resetButton = document.getElementById('reset-button');
const classifyButton = document.getElementById('classify-button');
const labelContainer = document.getElementById('label-container');
const noResults = document.getElementById('no-results');

let model, maxPredictions;

// const illnessCategories = {
//     'lupus' : { icon: '🧬', color: '#3498DB' },      
//     'saudavel': { icon: '✅', color: '#27AE60' },    
// };

const illnessCategories = {
    'lupus':    { icon: '🧬', color: '#3498DB' },
    'lúpus':    { icon: '🧬', color: '#3498DB' }, // se quiser suportar acento também
    'saudavel': { icon: '✅', color: '#27AE60' },
    'saudável': { icon: '✅', color: '#27AE60' }  // idem
};


const defaultCategory = { icon: '❓', color: '#BDC3C7' };

const MAX_WIDTH = 800;
const MAX_HEIGHT = 600;

window.onload = async () => {
    try {
        await loadModel();
        console.log("Model loaded successfully");
    } catch (error) {
        console.error("Error loading model:", error);
    }
};

async function loadModel() {
    const modelURL = MODEL_PATH + "model.json";
    const metadataURL = MODEL_PATH + "metadata.json";

    try {
        if (typeof tmImage === 'undefined') {
            console.error("Teachable Machine library not loaded");
            throw new Error("Teachable Machine library not loaded");
        }
        
        model = await tmImage.load(modelURL, metadataURL);
        maxPredictions = model.getTotalClasses();
        console.log(`Model loaded with ${maxPredictions} classes`);
    } catch (error) {
        console.error("Failed to load model:", error);
        throw error;
    }
}

imageUpload.addEventListener('change', function (event) {
    const file = event.target.files[0];
    if (file) {
        processAndDisplayImage(file);
    }
});

function processAndDisplayImage(file) {
    if (!file.type.match('image.*')) {
        alert("Please select an image file.");
        return;
    }
    
    if (file.size > 10 * 1024 * 1024) {
        alert("File is too large. Please select an image smaller than 10MB.");
        return;
    }
    
    const reader = new FileReader();

    reader.onload = function (e) {
        const img = new Image();
        
        img.onload = function () {
            const width = img.width;
            const height = img.height;
        
            let resizedImage;
            let resizedWidth = width;
            let resizedHeight = height;
            
            if (width > MAX_WIDTH || height > MAX_HEIGHT) {
                try {
                    resizedImage = resizeImage(img, MAX_WIDTH, MAX_HEIGHT);
                    const scaleFactor = Math.min(MAX_WIDTH / width, MAX_HEIGHT / height);
                    resizedWidth = Math.round(width * scaleFactor);
                    resizedHeight = Math.round(height * scaleFactor);
                } catch (error) {
                    console.error("Error resizing image:", error);
                    resizedImage = e.target.result;
                }
            } else {
                resizedImage = e.target.result;
            }
            
            imagePreview.src = resizedImage;
            
            previewContainer.style.display = 'block';
            uploadLabel.style.display = 'none';
            
            classifyButton.disabled = false;
            
            const fileSize = formatFileSize(file.size);
            const fileName = file.name;
            imageInfo.innerHTML = `${fileName} (${fileSize}) <span class="image-dimensions">${width}×${height}px${width !== resizedWidth ? ` → ${resizedWidth}×${resizedHeight}px` : ''}</span>`;
            
            noResults.style.display = 'flex';
            labelContainer.innerHTML = '';
            labelContainer.appendChild(noResults);
        };
        
        img.onerror = function() {
            alert("Error loading image. Please try another file.");
            resetUpload();
        };
        
        img.src = e.target.result;
    };
    
    reader.onerror = function() {
        alert("Error reading file. Please try another file.");
        resetUpload();
    };

    reader.readAsDataURL(file);
}

function resizeImage(img, maxWidth, maxHeight) {
    try {
        const canvas = document.createElement('canvas');
        let width = img.width;
        let height = img.height;
    
        const scaleFactor = Math.min(maxWidth / width, maxHeight / height);
        width = Math.round(width * scaleFactor);
        height = Math.round(height * scaleFactor);
        
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        
        ctx.drawImage(img, 0, 0, width, height);
        
        return canvas.toDataURL('image/jpeg', 0.9);
    } catch (error) {
        console.error("Error in resizeImage:", error);
        throw error;
    }
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' bytes';
    else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    else return (bytes / 1048576).toFixed(1) + ' MB';
}

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    uploadLabel.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    uploadLabel.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    uploadLabel.addEventListener(eventName, unhighlight, false);
});

function highlight() {
    uploadLabel.classList.add('highlight');
}

function unhighlight() {
    uploadLabel.classList.remove('highlight');
}

uploadLabel.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const file = dt.files[0];

    if (file && file.type.match('image.*')) {
        imageUpload.files = dt.files;
        processAndDisplayImage(file);
    }
}

resetButton.addEventListener('click', function () {
    resetUpload();
    classifyButton.textContent = 'Classificar';
});

function resetUpload() {
    imagePreview.src = '#';
    previewContainer.style.display = 'none';
    uploadLabel.style.display = 'flex';
    classifyButton.disabled = true;
    imageInfo.textContent = '';
    imageUpload.value = '';
    noResults.style.display = 'flex';
    labelContainer.innerHTML = '';
    labelContainer.appendChild(noResults);
}

classifyButton.addEventListener('click', async function () {
    if (!model) {
        try {
            await loadModel();
        } catch (error) {
            alert("Failed to load the model. Please try again later.");
            return;
        }
    }

    classifyButton.disabled = true;
    classifyButton.innerHTML = '<svg class="spinner" viewBox="0 0 50 50"><circle class="path" cx="25" cy="25" r="20" fill="none" stroke-width="5"></circle></svg> Classificando...';

    noResults.style.display = 'none';

    try {
        const img = document.getElementById('image-preview');
        const predictions = await model.predict(img);
        // const imgBase64 = img.src;
        // await postImageToBackend(imgBase64);


        
        // Achar a classe com maior probabilidade
        const bestPrediction = predictions.sort((a, b) => b.probability - a.probability)[0];

        const resultado = bestPrediction.className;       // "lupus" ou "saudavel"
        const probabilidade = bestPrediction.probability; // ex: 0.9823

        // Enviar tudo para o backend
        await postImageToBackend(img.src, resultado, probabilidade);

        displayResults(predictions);
    } catch (error) {
        console.error("Error during classification:", error);
        labelContainer.innerHTML = `<div class="error-message">Error during classification: ${error.message}</div>`;
    } finally {
        classifyButton.textContent = 'Classificar novamente';
        classifyButton.disabled = false;
    }
});

async function postImageToBackend(imgBase64, resultado, probabilidade) {
    try {
        const response = await fetch('/classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: imgBase64, resultado, probabilidade,
                timestamp: new Date().toISOString()
            })
        });

        if (!response.ok) {
            throw new Error('Failed to upload the image');
        }

        const result = await response.json();
        console.log('Image uploaded successfully:', result);
    } catch (error) {
        console.error('Error uploading image:', error);
    }
}

function displayResults(predictions) {
    labelContainer.innerHTML = '';

    predictions.sort((a, b) => b.probability - a.probability);

    predictions.forEach((prediction, index) => {
        // Verificar se prediction.className existe, senão usar o índice
        let className = prediction.className || `Class_${index}`;
        if (className == 'saudavel') {
            className = 'Saudável';
        } else {
            className = 'Lúpus';
        }
        const category = className.toLowerCase();
        const { icon, color } = illnessCategories[category] || defaultCategory;
        const probabilityBarra = prediction.probability * 100;
        const probability = (prediction.probability * 100).toFixed(2).replace(".", ",");;

        const resultItem = document.createElement('div');
        resultItem.className = 'result-item';
        resultItem.style.animationDelay = `${index * 0.1}s`;

        resultItem.innerHTML = `
      <div class="result-icon" style="background-color: ${color}20; color: ${color}">
        ${icon}
      </div>
      <div class="result-details">
        <div class="result-name">${className}</div>
        <div class="result-probability">${probability}% confiança</div>
        <div class="progress-bar">
          <div class="progress-fill" style="width: ${probabilityBarra}%; background-color: ${color}"></div>
        </div>
      </div>
    `;
        labelContainer.appendChild(resultItem);
    });
}
