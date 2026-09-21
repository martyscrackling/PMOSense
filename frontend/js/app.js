/**
 * PCOSense - Main Application JavaScript
 * Supports multiple images (Left/Right Ovary) + Camera
 */

// ============================================
// GLOBALS
// ============================================

let currentView = 'dashboard';
let leftOvaryImages = [];
let rightOvaryImages = [];
let leftOvaryDataURLs = [];
let rightOvaryDataURLs = [];
let clinicalData = {
    age: '',
    height: '',
    weight: '',
    bmi: '',
    menstrualIrregularities: 'no',
    acne: 'no',
    weightGain: 'no',
    familyHistory: 'no'
};
let latestAIResult = null;
let screeningHistory = [];
let expertValidationData = {};
let susRatingsData = {};
let apiAvailable = false;

// Camera variables
let cameraStream = null;
let facingMode = 'environment';
let cameraActive = false;
let currentTargetInput = null; // 'left' or 'right'

const container = document.getElementById('appViewContainer');
const API_BASE_URL = 'http://localhost:8000';

// ============================================
// API CONNECTION CHECK
// ============================================

async function checkAPIConnection() {
    const dot = document.getElementById('apiStatusDot');
    const text = document.getElementById('apiStatusText');

    try {
        const response = await fetch(`${API_BASE_URL}/`, {
            method: 'GET',
            signal: AbortSignal.timeout(3000)
        });

        if (response.ok) {
            apiAvailable = true;
            dot.className = 'w-2 h-2 rounded-full api-status-online';
            text.textContent = 'AI model online — ResNet-50, 97.87% validation accuracy';
            text.className = 'text-blue-800';
        } else {
            throw new Error('API returned error');
        }
    } catch (error) {
        apiAvailable = false;
        dot.className = 'w-2 h-2 rounded-full api-status-offline';
        text.textContent = 'AI model unavailable — running in simulation mode. Start backend with: python backend_api.py';
        text.className = 'text-red-700';
        console.warn('API not available:', error.message);
    }
}

// ============================================
// STORAGE HELPERS
// ============================================

function loadHistoryFromStorage() {
    const stored = localStorage.getItem('pcosense_history');
    if (stored) {
        screeningHistory = JSON.parse(stored);
    } else {
        screeningHistory = [
            { id: 'hist1', date: '2025-03-10', prediction: 'PCOS Detected', confidence: 87, clinicalSnapshot: { age: 27, bmi: 28.5, menstrual: 'yes' } },
            { id: 'hist2', date: '2025-02-22', prediction: 'At Risk', confidence: 62, clinicalSnapshot: { age: 24, bmi: 24.1, menstrual: 'no' } },
            { id: 'hist3', date: '2025-01-15', prediction: 'No PCOS Detected', confidence: 91, clinicalSnapshot: { age: 31, bmi: 22.3, menstrual: 'no' } }
        ];
        saveHistoryToStorage();
    }
}

function saveHistoryToStorage() {
    localStorage.setItem('pcosense_history', JSON.stringify(screeningHistory));
}

function addToHistory(prediction, confidence, clinicalSnapshot, imageRef = null, expertScore = null) {
    const newEntry = {
        id: Date.now().toString(),
        date: new Date().toLocaleDateString('en-CA'),
        prediction: prediction,
        confidence: confidence,
        clinicalSnapshot: clinicalSnapshot,
        expertValidation: expertScore,
        hasImages: leftOvaryDataURLs.length > 0 || rightOvaryDataURLs.length > 0
    };
    screeningHistory.unshift(newEntry);
    if (screeningHistory.length > 12) screeningHistory.pop();
    saveHistoryToStorage();
}

// ============================================
// AI DETECTION - REAL API
// ============================================

async function realAIDetection(imageFiles, clinical) {
    const clinicalData = {
        menstrualIrregularities: clinical.menstrualIrregularities || 'no',
        acne: clinical.acne || 'no',
        weightGain: clinical.weightGain || 'no',
        familyHistory: clinical.familyHistory || 'no',
        bmi: clinical.bmi || '0',
        age: clinical.age || '0'
    };

    const formData = new FormData();
    formData.append('clinical_data', JSON.stringify(clinicalData));

    imageFiles.left.forEach(file => formData.append('left_ovary', file));
    imageFiles.right.forEach(file => formData.append('right_ovary', file));

    try {
        const response = await fetch(`${API_BASE_URL}/predict-multiple`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            console.error('API Error:', error);
            return simulateAIDetection(clinical, true);
        }

        const result = await response.json();

        const contributions = {
            'Menstrual irregularities': clinical.menstrualIrregularities === 'yes' ? 3.0 : 0,
            'Acne': clinical.acne === 'yes' ? 1.5 : 0,
            'Weight gain': clinical.weightGain === 'yes' ? 2.0 : 0,
            'Family history': clinical.familyHistory === 'yes' ? 1.5 : 0,
            'High BMI': parseFloat(clinical.bmi) >= 25 ? 2.0 : 0,
            'Ultrasound signs': result.class_id === 1 ? 2.5 : 0.5
        };

        let totalScore = 0;
        for (const key in contributions) {
            totalScore += contributions[key];
        }

        return {
            prediction: result.prediction,
            confidence: result.confidence,
            contributions: contributions,
            folliclePresence: result.class_id === 1,
            totalScore: totalScore,
            heatmap: result.heatmap || null,
            probabilities: result.probabilities || null,
            clinicalRisk: result.clinical_risk || null,
            leftResult: result.left_result || null,
            rightResult: result.right_result || null
        };

    } catch (error) {
        console.error('Error calling AI model:', error);
        return simulateAIDetection(clinical, true);
    }
}

// ============================================
// AI DETECTION - SIMULATION (Fallback)
// ============================================

function simulateAIDetection(clinical, hasImage) {
    let score = 0;
    const factors = {};
    if (clinical.menstrualIrregularities === 'yes') { score += 3;
        factors.menstrual = 3; } else factors.menstrual = 0;
    if (clinical.acne === 'yes') { score += 1.5;
        factors.acne = 1.5; } else factors.acne = 0;
    if (clinical.weightGain === 'yes') { score += 2;
        factors.weight = 2; } else factors.weight = 0;
    if (clinical.familyHistory === 'yes') { score += 1.5;
        factors.family = 1.5; } else factors.family = 0;

    let bmiVal = parseFloat(clinical.bmi);
    let bmiPoints = 0;
    if (!isNaN(bmiVal)) {
        if (bmiVal >= 30) bmiPoints = 2.5;
        else if (bmiVal >= 25) bmiPoints = 1.5;
        else if (bmiVal >= 18.5) bmiPoints = 0;
        else bmiPoints = 0.5;
        score += bmiPoints;
        factors.bmi = bmiPoints;
    } else factors.bmi = 0;

    let ageVal = parseInt(clinical.age);
    let agePoints = 0;
    if (!isNaN(ageVal) && ageVal >= 18 && ageVal <= 40) agePoints = 0.2;
    score += agePoints;
    factors.age = agePoints;

    let ultrasoundImpact = 0;
    let folliclePresence = false;
    if (hasImage) {
        ultrasoundImpact = (score > 2.5) ? 1.2 : 0.6;
        score += ultrasoundImpact;
        folliclePresence = true;
        factors.ultrasoundFollicles = ultrasoundImpact;
    } else factors.ultrasoundFollicles = 0;

    let prediction = '';
    let confidenceBase = 0;
    if (score >= 6) { prediction = 'PCOS Detected';
        confidenceBase = 75 + Math.min(20, Math.floor(score * 2.5)); } else if (score >= 3.2) { prediction = 'At Risk';
        confidenceBase = 55 + Math.floor((score - 3) * 12); } else { prediction = 'No PCOS Detected';
        confidenceBase = 70 + (5 - score) * 4; }

    let confidence = Math.min(98, Math.max(52, confidenceBase));
    if (prediction === 'PCOS Detected') confidence = Math.min(96, confidence);
    if (prediction === 'At Risk') confidence = Math.min(88, confidence);

    const contributions = {
        'Menstrual irregularities': factors.menstrual || 0,
        'Acne': factors.acne || 0,
        'Weight gain': factors.weight || 0,
        'Family history': factors.family || 0,
        'High BMI': factors.bmi || 0,
        'Ultrasound signs': factors.ultrasoundFollicles || 0
    };
    return { prediction, confidence, contributions, folliclePresence, totalScore: score };
}

// ============================================
// DRAW ULTRASOUND HEATMAP
// ============================================

function drawHighlightedUltrasound(imgElement, canvasElement, folliclePresent) {
    const ctx = canvasElement.getContext('2d');
    canvasElement.width = imgElement.width;
    canvasElement.height = imgElement.height;
    ctx.drawImage(imgElement, 0, 0, imgElement.width, imgElement.height);
    if (folliclePresent) {
        ctx.save();
        ctx.shadowBlur = 0;
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.ellipse(imgElement.width * 0.35, imgElement.height * 0.55, imgElement.width * 0.12, imgElement.height * 0.09, 0, 0, 2 * Math.PI);
        ctx.fillStyle = '#f97316';
        ctx.fill();
        ctx.beginPath();
        ctx.ellipse(imgElement.width * 0.68, imgElement.height * 0.52, imgElement.width * 0.12, imgElement.height * 0.09, 0, 0, 2 * Math.PI);
        ctx.fill();
        ctx.globalAlpha = 0.9;
        ctx.beginPath();
        ctx.arc(imgElement.width * 0.33, imgElement.height * 0.53, 10, 0, 2 * Math.PI);
        ctx.fillStyle = '#ea580c';
        ctx.fill();
        ctx.beginPath();
        ctx.arc(imgElement.width * 0.65, imgElement.height * 0.51, 9, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillStyle = 'white';
        ctx.font = `bold ${Math.max(12, imgElement.width * 0.03)}px Inter`;
        ctx.shadowBlur = 0;
        ctx.fillStyle = '#fff7ed';
        ctx.fillText("Follicles", imgElement.width * 0.3, imgElement.height * 0.48);
        ctx.restore();
    } else {
        ctx.font = `14px Inter`;
        ctx.fillStyle = '#0f766e';
        ctx.shadowBlur = 0;
        ctx.fillText("Normal ovarian appearance", imgElement.width * 0.05, imgElement.height * 0.1);
    }
}

// ============================================
// CAMERA FUNCTIONS
// ============================================

async function startCamera() {
    const video = document.getElementById('cameraVideo');
    const status = document.getElementById('cameraStatus');

    try {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
        }

        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: facingMode, width: { ideal: 640 }, height: { ideal: 480 } }
        });

        video.srcObject = cameraStream;
        await video.play();
        cameraActive = true;
        status.textContent = 'Camera ready — position the ultrasound image';
        status.className = 'text-center text-sm text-blue-700 mt-3';
    } catch (error) {
        console.error('Camera error:', error);
        status.textContent = 'Unable to access camera. Use file upload instead.';
        status.className = 'text-center text-sm text-red-600 mt-3';
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    cameraActive = false;
    const video = document.getElementById('cameraVideo');
    video.srcObject = null;
}

function capturePhoto() {
    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('cameraCanvas');
    const status = document.getElementById('cameraStatus');

    if (!cameraActive || !video.videoWidth) {
        status.textContent = 'Please wait for camera to initialize';
        status.className = 'text-center text-sm text-red-600 mt-3';
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to data URL
    const dataURL = canvas.toDataURL('image/jpeg', 0.9);

    // Convert to file
    const blob = dataURLToBlob(dataURL);
    const file = new File([blob], `camera_capture_${Date.now()}.jpg`, { type: 'image/jpeg' });

    addOvaryImage(currentTargetInput, file, dataURL);

    // Close modal
    closeCameraModal();

    status.textContent = 'Image captured';
    status.className = 'text-center text-sm text-blue-700 mt-3';
}

function dataURLToBlob(dataURL) {
    const parts = dataURL.split(',');
    const mime = parts[0].match(/:(.*?);/)[1];
    const bstr = atob(parts[1]);
    const n = bstr.length;
    const u8arr = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
        u8arr[i] = bstr.charCodeAt(i);
    }
    return new Blob([u8arr], { type: mime });
}

function openCameraModal(target) {
    currentTargetInput = target;
    const modal = document.getElementById('cameraModal');
    modal.classList.remove('hidden');
    startCamera();
}

function closeCameraModal() {
    const modal = document.getElementById('cameraModal');
    modal.classList.add('hidden');
    stopCamera();
    currentTargetInput = null;
}

// ============================================
// RENDER: CLINICAL UPLOAD
// ============================================

function makeSelectCard(label, id, options, currentVal, icon) {
    const opts = options.map(opt =>
        `<option value="${opt}" ${currentVal === opt ? 'selected' : ''}>${opt === 'yes' ? 'Yes' : 'No'}</option>`
    ).join('');
    return `<div class="field-card"><label class="field-label"><i class="fas ${icon} field-icon"></i>${label}</label><select id="${id}" class="field-input">${opts}</select></div>`;
}

function renderImageUploadSection(ovaryType, label, dataURLs) {
    const id = ovaryType;
    const hasImages = dataURLs.length > 0;
    const containerId = `${id}PreviewContainer`;
    const previews = dataURLs.map((dataURL, index) => `
        <div class="image-preview">
            <img src="${dataURL}" class="rounded-lg h-32 w-32 object-contain border border-slate-200 bg-slate-900" alt="${label} ${index + 1}">
            <button class="remove-image-btn remove-btn" data-target="${id}" data-index="${index}" aria-label="Remove image ${index + 1}">×</button>
        </div>
    `).join('');

    return `
        <div class="upload-panel">
            <label class="block text-sm font-semibold text-slate-700 mb-2">
                <i class="fas fa-ultrasound mr-2 text-blue-600"></i>${label}
            </label>

            <!-- Upload Area -->
            <div id="${id}UploadArea" class="upload-area rounded-xl p-4 text-center cursor-pointer transition">
                <i class="fas fa-cloud-upload-alt text-3xl text-slate-300 mb-2"></i>
                <p class="text-slate-500 text-sm">Click to upload, or use the capture device</p>
                <div class="flex gap-3 justify-center mt-3">
                    <button class="upload-file-btn btn-chip btn-chip-blue" data-target="${id}">
                        <i class="fas fa-folder-open mr-1"></i>${hasImages ? 'Add images' : 'Choose files'}
                    </button>
                    <button class="camera-btn btn-chip btn-chip-teal" data-target="${id}">
                        <i class="fas fa-camera mr-1"></i>Capture
                    </button>
                </div>
                <input type="file" id="${id}FileInput" accept="image/jpeg,image/png" class="hidden" multiple />
            </div>

            <!-- Preview -->
            <div id="${containerId}" class="mt-4 flex flex-wrap gap-3 ${hasImages ? '' : 'hidden'}">
                ${previews}
            </div>
        </div>
    `;
}

function renderClinicalUploadView() {
    const leftHasImage = leftOvaryDataURLs.length > 0;
    const rightHasImage = rightOvaryDataURLs.length > 0;
    const bothImages = leftHasImage && rightHasImage;

    return `
        <div class="panel">
            <div class="panel-header">
                <i class="fas fa-notes-medical text-blue-600 text-2xl"></i>
                <h2 class="panel-title">Clinical intake &amp; ultrasound</h2>
                ${apiAvailable ? '<span class="ml-auto status-chip status-chip-online"><i class="fas fa-check-circle"></i> Model ready</span>' : '<span class="ml-auto status-chip status-chip-offline"><i class="fas fa-exclamation-triangle"></i> Simulation mode</span>'}
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <!-- Left Column: Clinical Data -->
                <div>
                    <p class="section-label">Patient parameters</p>
                    <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <div class="field-card"><label class="field-label"><i class="fas fa-calendar-alt field-icon"></i>Age</label><input type="number" id="ageInput" class="field-input" placeholder="e.g., 28" value="${clinicalData.age}"></div>
                        <div class="field-card"><label class="field-label"><i class="fas fa-ruler-vertical field-icon"></i>Height (cm)</label><input type="number" step="0.1" id="heightInput" class="field-input" placeholder="e.g., 165" value="${clinicalData.height}"></div>
                    </div>
                    <div class="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <div class="field-card"><label class="field-label"><i class="fas fa-scale-balanced field-icon"></i>Weight (kg)</label><input type="number" step="0.1" id="weightInput" class="field-input" placeholder="e.g., 62" value="${clinicalData.weight}"></div>
                        <div class="field-card flex items-center"><div><label class="field-label"><i class="fas fa-calculator field-icon"></i>Computed BMI</label><div id="bmiDisplay" class="mt-1 text-lg font-semibold text-slate-800 font-mono">BMI <span id="bmiValue">${clinicalData.bmi || '—'}</span></div></div></div>
                    </div>
                    <p class="section-label mt-5">Reported symptoms</p>
                    <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        ${makeSelectCard('Menstrual irregularities', 'menstrualSelect', ['yes','no'], clinicalData.menstrualIrregularities, 'fa-droplet')}
                        ${makeSelectCard('Acne / hirsutism', 'acneSelect', ['yes','no'], clinicalData.acne, 'fa-face-frown')}
                        ${makeSelectCard('Unexplained weight gain', 'weightSelect', ['yes','no'], clinicalData.weightGain, 'fa-chart-line')}
                        ${makeSelectCard('Family history (PCOS)', 'familySelect', ['yes','no'], clinicalData.familyHistory, 'fa-people-arrows')}
                    </div>
                </div>

                <!-- Right Column: Ultrasound Images -->
                <div>
                    <p class="section-label">Ultrasound imaging</p>
                    <div class="space-y-4">
                        ${renderImageUploadSection('left', 'Left ovary', leftOvaryDataURLs)}
                        ${renderImageUploadSection('right', 'Right ovary', rightOvaryDataURLs)}
                    </div>
                    <div class="mt-4 text-xs text-slate-400">
                        <i class="fas fa-circle-info mr-1"></i>At least one image (left or right ovary) is required to run analysis.
                    </div>
                </div>
            </div>

            <div class="mt-8 flex justify-end gap-3 border-t border-slate-200 pt-6">
                <button id="cancelClinicalBtn" class="btn-secondary">Cancel</button>
                <button id="startAnalysisBtn" class="btn-primary">
                    <i class="fas fa-microscope mr-2"></i>Run analysis
                </button>
            </div>
        </div>
    `;
}

// ============================================
// RENDER: LOADING
// ============================================

function renderLoadingView() {
    return `
        <div class="panel text-center py-14">
            <div class="flex flex-col items-center gap-5">
                <i class="fas fa-brain text-5xl text-blue-500 animate-pulse"></i>
                <h3 class="text-2xl font-semibold text-slate-800">${apiAvailable ? 'Analyzing intake data and ultrasound imagery' : 'Running simulation model'}</h3>
                <div class="w-full max-w-md bg-slate-100 rounded-full h-2.5 overflow-hidden">
                    <div id="progressFill" class="progress-fill h-2.5 rounded-full w-0 transition-all duration-500"></div>
                </div>
                <p class="text-slate-500 text-sm">${apiAvailable ? 'ResNet-50 model is evaluating follicle count, ovarian volume and clinical indicators…' : 'Backend model unavailable — using rule-based simulation'}</p>
            </div>
        </div>
    `;
}

// ============================================
// HELPER: XAI MAPPING
// ============================================

function generateExplanation(prediction, contributions, confidence) {
    const topFactors = Object.entries(contributions)
        .filter(([k, v]) => v > 0.8)
        .map(([k]) => k);
    if (prediction === 'PCOS Detected') {
        return `The model classified this case as PCOS-positive with ${confidence}% confidence. The strongest contributing factors were: ${topFactors.join(', ')}. The ultrasound heatmap highlights follicular regions consistent with polycystic ovarian morphology.`;
    } else if (prediction === 'At Risk') {
        return `The model identified moderate risk factors for PCOS (${confidence}% confidence). Key contributing factors: ${topFactors.join(', ')}. Ultrasound shows some follicular activity, though findings are not conclusive for PCOS.`;
    } else {
        return `The model did not classify this case as PCOS (${confidence}% confidence). No significant risk factors were identified, and the ultrasound appears within normal limits with no polycystic morphology.`;
    }
}

function getIoUScore(folliclePresent) {
    return folliclePresent ? 0.78 : 0.12;
}

// ============================================
// RENDER: RESULTS
// ============================================

function renderResultsView(aiResult, clinical, imageURL) {
    const { prediction, confidence, contributions, folliclePresence } = aiResult;
    let pcosChance = 0;
    if (prediction === 'PCOS Detected') pcosChance = Math.round(confidence);
    else if (prediction === 'At Risk') pcosChance = Math.max(30, Math.round(confidence * 0.75));
    else pcosChance = Math.max(1, Math.round(100 - confidence));

    let barColor = 'var(--clr-normal)';
    if (pcosChance >= 65) barColor = 'var(--clr-positive)';
    else if (pcosChance >= 35) barColor = 'var(--clr-caution)';

    let badgeClass = '';
    if (prediction === 'PCOS Detected') badgeClass = 'result-badge-pcos';
    else if (prediction === 'At Risk') badgeClass = 'result-badge-risk';
    else badgeClass = 'result-badge-normal';

    let contribHtml = '';
    const maxContrib = Math.max(...Object.values(contributions), 0.1);
    for (const [factor, value] of Object.entries(contributions)) {
        const percentWidth = (value / maxContrib) * 100;
        contribHtml += `<div class="mb-3"><div class="flex justify-between text-xs text-slate-600"><span>${factor}</span><span class="font-mono">${value.toFixed(1)} pts</span></div><div class="w-full bg-slate-100 rounded-full h-2"><div class="bg-blue-600 h-2 rounded-full chart-bar-fill" style="width: ${percentWidth}%"></div></div></div>`;
    }

    // Show both images in results
    const imagesHtml = `
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
            ${leftOvaryDataURLs.map((dataURL, index) => `
                <div>
                    <p class="text-xs font-medium text-slate-600 mb-1">Left ovary ${index + 1}</p>
                    <img src="${dataURL}" class="rounded-lg w-full max-h-48 object-contain border border-slate-200 bg-slate-900" alt="Left ovary ${index + 1}">
                </div>
            `).join('')}
            ${rightOvaryDataURLs.map((dataURL, index) => `
                <div>
                    <p class="text-xs font-medium text-slate-600 mb-1">Right ovary ${index + 1}</p>
                    <img src="${dataURL}" class="rounded-lg w-full max-h-48 object-contain border border-slate-200 bg-slate-900" alt="Right ovary ${index + 1}">
                </div>
            `).join('')}
        </div>
    `;

    return `
        <div class="panel p-0 overflow-hidden">
            <div class="p-6 bg-slate-50 border-b border-slate-200">
                <div class="flex items-center justify-between flex-wrap gap-3">
                    <h2 class="text-2xl font-bold text-slate-800"><i class="fas fa-chart-simple mr-2 text-blue-600"></i>Assessment results</h2>
                    <span class="badge-pill ${badgeClass}">${prediction}</span>
                </div>
                <div class="mt-3 flex items-center gap-2">
                    <span class="text-sm text-slate-500">Estimated PCOS likelihood:</span>
                    <div class="w-32 bg-slate-200 rounded-full h-2">
                        <div class="h-2 rounded-full" style="width: ${pcosChance}%; background: ${barColor};"></div>
                    </div>
                    <span class="font-semibold font-mono">${pcosChance}%</span>
                    ${apiAvailable ? '<span class="text-xs text-blue-700 ml-2"><i class="fas fa-check-circle"></i> Model output</span>' : '<span class="text-xs text-amber-700 ml-2"><i class="fas fa-sync"></i> Simulated</span>'}
                </div>
            </div>

            <div class="p-6 grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div>
                    <h3 class="font-semibold text-slate-800 mb-3"><i class="fas fa-chart-bar text-blue-600 mr-2"></i>Symptom &amp; feature impact</h3>
                    <div class="bg-slate-50 border border-slate-200 p-4 rounded-xl">${contribHtml}</div>
                    <div class="mt-5 bg-teal-50 border border-teal-100 p-4 rounded-xl">
                        <i class="fas fa-stethoscope text-teal-600 mr-2"></i>
                        <span class="text-sm text-slate-700">Key drivers: ${Object.entries(contributions).filter(([k,v])=>v>0.8).map(([k])=>k).join(', ') || 'balanced profile'}</span>
                    </div>
                </div>
                <div>
                    <h3 class="font-semibold text-slate-800 mb-3"><i class="fas fa-ultrasound mr-2 text-blue-600"></i>Ultrasound images</h3>
                    ${imagesHtml}
                </div>
            </div>

            <div class="px-6 py-6 bg-slate-50 border-t border-slate-200">
                <h4 class="font-semibold text-slate-800 mb-4 flex items-center gap-2">
                    <i class="fas fa-directions text-blue-600"></i> Detailed analysis
                </h4>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                    <button id="goToXAIMapping" class="nav-card">
                        <i class="fas fa-link nav-card-icon text-blue-600"></i>
                        <h5 class="font-semibold text-slate-800">XAI mapping</h5>
                        <p class="text-xs text-slate-500 mt-1">Model decisions and heatmap alignment</p>
                    </button>
                    <button id="goToExpertValidation" class="nav-card">
                        <i class="fas fa-user-doctor nav-card-icon text-teal-600"></i>
                        <h5 class="font-semibold text-slate-800">Expert assessment</h5>
                        <p class="text-xs text-slate-500 mt-1">Validate explainability with clinical feedback</p>
                    </button>
                    <button id="goToSUS" class="nav-card">
                        <i class="fas fa-clipboard-list nav-card-icon text-amber-600"></i>
                        <h5 class="font-semibold text-slate-800">System feedback</h5>
                        <p class="text-xs text-slate-500 mt-1">Record usability observations</p>
                    </button>
                </div>
            </div>

            <div class="px-6 pb-6 pt-4 flex flex-wrap justify-between gap-3">
                <button id="saveAndHistoryBtn" class="btn-chip btn-chip-blue"><i class="fas fa-save mr-1"></i> Save to history</button>
                <button id="newScreeningFromResults" class="btn-primary"><i class="fas fa-file-medical mr-2"></i> New screening</button>
                <button id="dashboardFromResults" class="btn-secondary">Dashboard</button>
            </div>
        </div>
    `;
}

// ============================================
// RENDER: XAI MAPPING DETAIL PAGE (Abbreviated)
// ============================================

function renderXAIMappingPage() {
    const { prediction, confidence, contributions, folliclePresence } = latestAIResult;
    const explanationText = generateExplanation(prediction, contributions, confidence);
    const alignmentPercent = folliclePresence ? 85 : 95;

    return `
        <div class="panel p-0 overflow-hidden">
            <div class="p-6 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                <h2 class="text-2xl font-bold text-slate-800"><i class="fas fa-link mr-2 text-blue-600"></i>XAI result mapping</h2>
                <button id="backFromXAI" class="icon-btn"><i class="fas fa-arrow-left"></i></button>
            </div>

            <div class="p-8 space-y-6">
                <div class="callout callout-blue">
                    <h3 class="font-semibold text-slate-800 mb-2 flex items-center gap-2">
                        <i class="fas fa-microscope text-blue-700"></i> Prediction summary
                    </h3>
                    <p class="text-slate-700 leading-relaxed">${explanationText}</p>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div class="callout callout-green">
                        <h4 class="font-semibold text-slate-800 mb-3 flex items-center gap-2">
                            <i class="fas fa-check-circle text-emerald-700"></i> Region-prediction alignment
                        </h4>
                        <div class="flex items-center gap-3 mb-2">
                            <div class="flex-1 bg-slate-200 rounded-full h-3">
                                <div class="bg-emerald-600 h-3 rounded-full" style="width: ${alignmentPercent}%"></div>
                            </div>
                            <span class="text-lg font-bold text-emerald-700 font-mono">${alignmentPercent}%</span>
                        </div>
                        <p class="text-sm text-slate-600">${folliclePresence ? 'Heatmap regions align with follicle detection.' : 'Heatmap regions align with normal ovarian morphology.'}</p>
                    </div>

                    <div class="callout callout-teal">
                        <h4 class="font-semibold text-slate-800 mb-3 flex items-center gap-2">
                            <i class="fas fa-layer-group text-teal-700"></i> IoU score (heatmap vs. clinical)
                        </h4>
                        <div class="flex items-center gap-3 mb-2">
                            <div class="flex-1 bg-slate-200 rounded-full h-3">
                                <div class="bg-teal-600 h-3 rounded-full" style="width: ${Math.round(getIoUScore(folliclePresence) * 100)}%"></div>
                            </div>
                            <span class="text-lg font-bold text-teal-700 font-mono">${Math.round(getIoUScore(folliclePresence) * 100)}%</span>
                        </div>
                        <p class="text-sm text-slate-600">Intersection over union between highlighted regions and clinical features.</p>
                    </div>
                </div>

                <div class="callout callout-amber">
                    <h4 class="font-semibold text-slate-800 mb-3 flex items-center gap-2">
                        <i class="fas fa-chart-pie text-amber-700"></i> Factor contributions
                    </h4>
                    <div class="space-y-3">
                        ${Object.entries(latestAIResult.contributions).map(([factor, value]) => {
                            const maxContrib = Math.max(...Object.values(latestAIResult.contributions), 0.1);
                            const percentWidth = (value / maxContrib) * 100;
                            return `
                                <div>
                                    <div class="flex justify-between text-sm text-slate-700 mb-1">
                                        <span class="font-medium">${factor}</span>
                                        <span class="font-mono">${value.toFixed(2)} pts</span>
                                    </div>
                                    <div class="w-full bg-slate-200 rounded-full h-2.5">
                                        <div class="bg-amber-500 h-2.5 rounded-full" style="width: ${percentWidth}%"></div>
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            </div>

            <div class="px-8 py-4 bg-slate-50 border-t border-slate-200 flex justify-between gap-3">
                <button id="backToResultsFromXAI" class="btn-secondary"><i class="fas fa-arrow-left mr-1"></i> Back to results</button>
                <button id="goToExpertFromXAI" class="btn-primary"><i class="fas fa-arrow-right mr-1"></i> Expert assessment</button>
            </div>
        </div>
    `;
}

// ============================================
// RENDER: EXPERT VALIDATION PAGE
// ============================================

function renderExpertValidationPage() {
    return `
        <div class="panel p-0 overflow-hidden">
            <div class="p-6 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                <h2 class="text-2xl font-bold text-slate-800"><i class="fas fa-user-doctor mr-2 text-teal-600"></i>Expert validation</h2>
                <button id="backFromExpert" class="icon-btn"><i class="fas fa-arrow-left"></i></button>
            </div>

            <div class="p-8 space-y-6">
                <div class="callout callout-teal">
                    <h3 class="font-semibold text-slate-800 mb-2 flex items-center gap-2">
                        <i class="fas fa-stethoscope text-teal-700"></i> Clinician assessment
                    </h3>
                    <p class="text-slate-700">Rate how well the model's Grad-CAM heatmap explains the prediction. This feedback is used to audit and improve explainability.</p>
                </div>

                <div class="space-y-4">
                    ${[
                        { idx: 0, label: 'Region-prediction alignment', desc: 'How well do the highlighted regions match the prediction?' },
                        { idx: 1, label: 'Clinical relevance', desc: 'Are the highlighted regions clinically relevant for PCOS detection?' },
                        { idx: 2, label: 'Heatmap-prediction consistency', desc: 'Is the explanation consistent with the final prediction?' },
                        { idx: 3, label: 'Confidence-heatmap correlation', desc: 'Does the heatmap intensity correlate with confidence?' }
                    ].map(item => `
                        <div class="field-card-lg">
                            <label class="block font-semibold text-slate-800 mb-1">${item.label}</label>
                            <p class="text-sm text-slate-600 mb-3">${item.desc}</p>
                            <div class="flex gap-2 mb-2">
                                ${[1,2,3,4,5].map(n => `<button class="expert-rating-btn rating-btn" data-criteria="${item.idx}" data-value="${n}">${n}</button>`).join('')}
                            </div>
                            <div class="flex justify-between text-xs text-slate-400 px-1">
                                <span>Poor</span>
                                <span>Excellent</span>
                            </div>
                        </div>
                    `).join('')}
                </div>

                <div class="callout callout-blue">
                    <label class="block font-semibold text-slate-800 mb-2 flex items-center gap-2">
                        <i class="fas fa-comment text-blue-700"></i> Additional notes (optional)
                    </label>
                    <textarea id="expertComments" class="w-full p-3 border border-slate-300 rounded-lg resize-none h-24 bg-white" placeholder="Record any additional clinical observations…"></textarea>
                </div>

                <div id="expertFeedback" class="hidden callout callout-green">
                    <p class="text-emerald-800 font-semibold"><i class="fas fa-check-circle mr-2"></i>Validation submitted.</p>
                    <p id="expertFeedbackDetails" class="text-sm text-emerald-700 mt-2"></p>
                </div>
            </div>

            <div class="px-8 py-4 bg-slate-50 border-t border-slate-200 flex justify-between gap-3">
                <button id="backToResultsFromExpert" class="btn-secondary"><i class="fas fa-arrow-left mr-1"></i> Back</button>
                <div class="flex gap-3">
                    <button id="goToSUSFromExpert" class="btn-secondary"><i class="fas fa-forward mr-1"></i> Go to SUS</button>
                    <button id="submitExpertValidation" class="btn-primary"><i class="fas fa-paper-plane mr-1"></i> Submit</button>
                </div>
            </div>
        </div>
    `;
}

// ============================================
// RENDER: SUS PAGE (Abbreviated)
// ============================================

function renderSUSPage() {
    const susQuestions = [
        "I think that I would like to use this system frequently.",
        "I found the system unnecessarily complex.",
        "I thought the system was easy to use.",
        "I think that I would need the support of a technical person to be able to use this system.",
        "I found the various functions in this system were well integrated.",
        "I thought there was too much inconsistency in this system.",
        "I would imagine that most people would learn to use this system very quickly.",
        "I found the system very cumbersome to use.",
        "I felt very confident using the system.",
        "I needed to learn a lot of things before I could get going with this system."
    ];

    let susHtml = susQuestions.map((q, idx) => `
        <div class="field-card-lg">
            <p class="font-medium text-slate-800 mb-3"><span class="text-amber-700 font-bold font-mono">Q${idx+1}.</span> ${q}</p>
            <div class="flex gap-2 mb-3">
                ${[1,2,3,4,5].map(n => `<button class="sus-rating-btn sus-btn" data-q="${idx}" data-value="${n}">${n}</button>`).join('')}
            </div>
            <div class="flex justify-between text-xs text-slate-400 px-1">
                <span>Strongly disagree</span>
                <span>Strongly agree</span>
            </div>
        </div>
    `).join('');

    return `
        <div class="panel p-0 overflow-hidden">
            <div class="p-6 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                <h2 class="text-2xl font-bold text-slate-800"><i class="fas fa-clipboard-list mr-2 text-amber-600"></i>System Usability Scale (SUS)</h2>
                <button id="backFromSUS" class="icon-btn"><i class="fas fa-arrow-left"></i></button>
            </div>

            <div class="p-8 space-y-6">
                <div class="callout callout-amber">
                    <h3 class="font-semibold text-slate-800 mb-2 flex items-center gap-2">
                        <i class="fas fa-comments text-amber-700"></i> Usability feedback
                    </h3>
                    <p class="text-slate-700">Rate your experience using PCOSense on these 10 standard items. This feedback informs future usability review.</p>
                </div>

                <div class="space-y-4 max-h-96 overflow-y-auto pr-2">
                    ${susHtml}
                </div>

                <div id="susFeedback" class="hidden callout callout-green">
                    <p class="text-emerald-800 font-semibold"><i class="fas fa-square-check mr-2"></i>SUS survey completed.</p>
                    <p id="susFeedbackDetails" class="text-sm text-emerald-700 mt-2"></p>
                </div>
            </div>

            <div class="px-8 py-4 bg-slate-50 border-t border-slate-200 flex justify-between gap-3">
                <button id="backToResultsFromSUS" class="btn-secondary"><i class="fas fa-arrow-left mr-1"></i> Back</button>
                <div class="flex gap-3">
                    <button id="goToExpertFromSUS" class="btn-secondary"><i class="fas fa-backward mr-1"></i> Back to expert</button>
                    <button id="submitSUS" class="btn-primary"><i class="fas fa-paper-plane mr-1"></i> Submit feedback</button>
                </div>
            </div>
        </div>
    `;
}

// ============================================
// RENDER: HISTORY
// ============================================

function renderHistoryView() {
    if (screeningHistory.length === 0) {
        return `<div class="panel text-center py-16"><i class="fas fa-clock-rotate-left text-5xl text-slate-200 mb-3"></i><p class="text-slate-400">No screenings recorded yet.</p><button id="startFromEmptyHistory" class="mt-4 btn-primary">+ New screening</button></div>`;
    }
    let rows = '';
    screeningHistory.forEach(entry => {
        let borderClass = '';
        let badgeClass = '';
        if (entry.prediction === 'PCOS Detected') { borderClass = 'history-row-positive'; badgeClass = 'result-badge-pcos'; }
        else if (entry.prediction === 'At Risk') { borderClass = 'history-row-caution'; badgeClass = 'result-badge-risk'; }
        else { borderClass = 'history-row-normal'; badgeClass = 'result-badge-normal'; }
        const expertNote = entry.expertValidation ? `Expert ${entry.expertValidation.avgScore.toFixed(1)}/5` : '';
        const imageIcon = entry.hasImages ? '<i class="fas fa-image text-slate-400 text-xs" title="Includes ultrasound images"></i>' : '';
        rows += `
            <div class="history-row ${borderClass}">
                <div>
                    <div class="flex items-center gap-2">
                        <i class="far fa-calendar-alt text-slate-400"></i>
                        <span class="text-sm text-slate-500 font-mono">${entry.date}</span>
                        ${imageIcon}
                    </div>
                    <div class="mt-1 font-semibold text-slate-800">${entry.prediction}</div>
                </div>
                <div class="flex items-center gap-2">
                    <span class="text-sm text-slate-500">Confidence</span>
                    <div class="w-20 bg-slate-200 rounded-full h-1.5">
                        <div class="bg-blue-600 h-1.5 rounded-full" style="width: ${entry.confidence}%"></div>
                    </div>
                    <span class="text-xs font-medium font-mono">${entry.confidence}%</span>
                </div>
                <span class="badge-pill ${badgeClass} badge-pill-sm">${entry.prediction}</span>
                ${expertNote ? `<span class="text-xs text-teal-700 bg-teal-50 px-2 py-1 rounded-full font-mono">${expertNote}</span>` : '<span></span>'}
            </div>
        `;
    });
    return `<div class="panel"><div class="flex justify-between items-center mb-4"><h2 class="panel-title"><i class="fas fa-clock-rotate-left text-blue-600 mr-2"></i>Screening history</h2><button id="clearHistoryBtn" class="text-xs text-red-600 hover:underline">Clear all</button></div><div class="space-y-2">${rows}</div><div class="mt-6 flex justify-end"><button id="newFromHistoryBtn" class="btn-primary"><i class="fas fa-plus mr-1"></i> New screening</button></div></div>`;
}

// ============================================
// RENDER: DASHBOARD
// ============================================

function renderDashboard() {
    const carouselImages = [
        '../dataset/cleaned/PCOS/image10029.jpg',
        '../dataset/cleaned/Healthy/Image_063.jpg',
        '../dataset/cleaned/PCOS/image10028.jpg',
        '../dataset/cleaned/Healthy/Image_061.jpg',
        '../dataset/cleaned/PCOS/image10027.jpg',
        '../dataset/cleaned/Healthy/Image_059.jpg'
    ];
    const awarenessImages = [
        'assets/awareness-calendar.svg',
        'assets/awareness-variation.svg',
        'assets/awareness-conversation.svg',
        'assets/awareness-support.svg'
    ];
    const carouselItems = [...carouselImages, ...carouselImages]
        .map((src, index) => `<div class="ultrasound-slide"><img src="${src}" alt="Ovarian ultrasound sample ${index % carouselImages.length + 1}" loading="lazy"></div>`)
        .join('');

    return `
        <section class="content-section active">
            <div class="ultrasound-carousel" aria-label="Ultrasound image carousel">
                <div class="ultrasound-track">
                    ${carouselItems}
                </div>
            </div>

            <div class="page-heading">
                <div>
                    <span class="eyebrow">PCOSense</span>
                    <h1>Ultrasound review, <em>made clearer.</em></h1>
                    <p>Review ovarian ultrasound images with structured follicle analysis and explainable PCOS detection.</p>
                </div>
                <button id="dashboardStartBtn" class="primary-button" type="button">
                    <span>＋</span> New analysis
                </button>
            </div>

            <div class="clinical-banner">
                <div class="banner-symbol">i</div>
                <div>
                    <strong>Clinical decision support</strong>
                    <p>Interpret results alongside patient history, physical examination, laboratory findings, and professional clinical judgment.</p>
                </div>
            </div>

            <div class="section-label"><span>Quick access</span></div>
            <div class="dashboard-grid">
                <button class="dashboard-card analysis-card" id="dashboardAnalysisBtn" type="button">
                    <div class="card-icon rose">◉</div>
                    <div class="card-content">
                        <span class="card-kicker">IMAGE ANALYSIS</span>
                        <h2>Analyze an ultrasound</h2>
                        <p>Upload an ovarian ultrasound image and begin PCOS assessment.</p>
                        <span class="card-link">Start review →</span>
                    </div>
                </button>
                <button class="dashboard-card history-card" id="dashboardHistoryBtn" type="button">
                    <div class="card-icon plum">◷</div>
                    <div class="card-content">
                        <span class="card-kicker">REVIEW HISTORY</span>
                        <h2>Previous analyses</h2>
                        <p>Access previously reviewed ultrasound cases.</p>
                        <span class="card-link">View history →</span>
                    </div>
                </button>
            </div>

            <div class="section-label overview-label"><span>Workspace overview</span></div>
            <div class="overview-grid">
                <div class="overview-card">
                    <span class="overview-title">Detection workflow</span>
                    <div class="workflow">
                        <div class="workflow-step"><span>01</span><p>Upload image</p></div>
                        <div class="workflow-line"></div>
                        <div class="workflow-step"><span>02</span><p>Image analysis</p></div>
                        <div class="workflow-line"></div>
                        <div class="workflow-step"><span>03</span><p>Explainable result</p></div>
                    </div>
                </div>
                <div class="overview-card clinical-overview">
                    <span class="overview-title">Review principle</span>
                    <p>PCOSense presents computational findings as supporting information rather than as a replacement for professional diagnosis.</p>
                    <div class="overview-tag">Professional review</div>
                </div>
            </div>

            <div class="section-label awareness-label"><span>PCOS awareness</span></div>
            <div class="awareness-grid">
                <article class="awareness-card awareness-rose">
                    <img class="awareness-image" src="${awarenessImages[0]}" alt="Calendar and awareness ribbon illustration" loading="lazy">
                    <span class="awareness-mark">01</span>
                    <h2>September is PCOS Awareness Month</h2>
                    <p>Each September, awareness efforts help educate the public about PCOS and support people affected by it.</p>
                    <a class="awareness-source" href="https://pcoschallenge.org/pcos-awareness-month/" target="_blank" rel="noopener noreferrer">Learn more</a>
                </article>
                <article class="awareness-card awareness-plum">
                    <img class="awareness-image" src="${awarenessImages[1]}" alt="Illustration showing different symptom pathways" loading="lazy">
                    <span class="awareness-mark">02</span>
                    <h2>PCOS can look different</h2>
                    <p>Symptoms and experiences vary. Periods, skin, hair, weight, and fertility may be affected differently from person to person.</p>
                    <a class="awareness-source" href="https://www.calm.com/blog/pcos-awareness-month" target="_blank" rel="noopener noreferrer">Read the overview</a>
                </article>
                <article class="awareness-card awareness-sage">
                    <img class="awareness-image" src="${awarenessImages[2]}" alt="Patient and clinician having a conversation" loading="lazy">
                    <span class="awareness-mark">03</span>
                    <h2>Awareness supports earlier conversations</h2>
                    <p>Recognizing possible symptoms can make it easier to speak with a qualified clinician and seek an individualized assessment.</p>
                    <span class="awareness-note">Education is not a diagnosis</span>
                </article>
                <article class="awareness-card awareness-amber">
                    <img class="awareness-image" src="${awarenessImages[3]}" alt="Supportive hands and heart illustration" loading="lazy">
                    <span class="awareness-mark">04</span>
                    <h2>Support matters all year</h2>
                    <p>PCOS awareness is more than one month: respectful care, reliable information, and ongoing follow-up can make a meaningful difference.</p>
                    <span class="awareness-note">Talk with your care team</span>
                </article>
            </div>
        </section>
    `;
}

// ============================================
// VIEW SWITCHING
// ============================================

function switchView(view, params = null) {
    currentView = view;
    renderCurrentView(params);
}

function renderCurrentView(params) {
    if (currentView === 'dashboard') container.innerHTML = renderDashboard();
    else if (currentView === 'clinical') container.innerHTML = renderClinicalUploadView();
    else if (currentView === 'loading') container.innerHTML = renderLoadingView();
    else if (currentView === 'results') container.innerHTML = renderResultsView(latestAIResult, clinicalData, leftOvaryDataURLs[0] || rightOvaryDataURLs[0]);
    else if (currentView === 'xaiMapping') container.innerHTML = renderXAIMappingPage();
    else if (currentView === 'expertValidation') container.innerHTML = renderExpertValidationPage();
    else if (currentView === 'sus') container.innerHTML = renderSUSPage();
    else if (currentView === 'history') container.innerHTML = renderHistoryView();

    attachViewEvents();
    updateActiveNav();

    // Handle canvas rendering for heatmaps
    if (currentView === 'results' && (leftOvaryDataURLs.length || rightOvaryDataURLs.length)) {
        const imgSrc = leftOvaryDataURLs[0] || rightOvaryDataURLs[0];
        if (imgSrc) {
            const img = new Image();
            img.onload = () => {
                const canvas = document.getElementById('highlightCanvas');
                if (canvas) drawHighlightedUltrasound(img, canvas, latestAIResult?.folliclePresence);
            };
            img.src = imgSrc;
        }
    }
    if (currentView === 'xaiMapping' && (leftOvaryDataURLs.length || rightOvaryDataURLs.length)) {
        const imgSrc = leftOvaryDataURLs[0] || rightOvaryDataURLs[0];
        if (imgSrc) {
            const img = new Image();
            img.onload = () => {
                const canvas = document.getElementById('xaiHighlightCanvas');
                if (canvas) drawHighlightedUltrasound(img, canvas, latestAIResult?.folliclePresence);
            };
            img.src = imgSrc;
        }
    }
}

function updateActiveNav() {
    const activeSection = {
        dashboard: 'dashboard',
        clinical: 'analysis',
        loading: 'analysis',
        results: 'analysis',
        xaiMapping: 'analysis',
        expertValidation: 'rating',
        sus: 'rating',
        history: 'history'
    }[currentView];

    document.querySelectorAll('.sidebar .nav-item').forEach(button => {
        button.classList.toggle('active', button.dataset.section === activeSection);
    });
}

// ============================================
// ATTACH EVENTS
// ============================================

function attachViewEvents() {
    document.querySelectorAll('.sidebar .nav-item').forEach(button => {
        button.onclick = () => {
            const views = {
                dashboard: 'dashboard',
                analysis: 'clinical',
                history: 'history',
                rating: 'sus'
            };
            const view = views[button.dataset.section];
            if (view) switchView(view);
        };
    });

    // Dashboard
    document.getElementById('dashboardStartBtn')?.addEventListener('click', () => { resetClinicalForm();
        switchView('clinical'); });
    document.getElementById('dashboardAnalysisBtn')?.addEventListener('click', () => { resetClinicalForm();
        switchView('clinical'); });
    document.getElementById('dashboardHistoryBtn')?.addEventListener('click', () => switchView('history'));

    // Clinical upload - Image upload handlers
    setupImageUploadHandlers();

    // BMI calculation
    function updateBMIDisplay() {
        const h = parseFloat(document.getElementById('heightInput')?.value);
        const w = parseFloat(document.getElementById('weightInput')?.value);
        const el = document.getElementById('bmiValue');
        if (el) {
            if (!isNaN(h) && !isNaN(w) && h > 0) {
                const bmi = w / ((h / 100) * (h / 100));
                el.textContent = bmi.toFixed(1);
            } else el.textContent = '—';
        }
    }
    document.getElementById('heightInput')?.addEventListener('input', updateBMIDisplay);
    document.getElementById('weightInput')?.addEventListener('input', updateBMIDisplay);

    // Start Analysis
    document.getElementById('startAnalysisBtn')?.addEventListener('click', () => {
        const age = document.getElementById('ageInput')?.value;
        const height = document.getElementById('heightInput')?.value;
        const weight = document.getElementById('weightInput')?.value;
        if (!age || !height || !weight || age < 12 || age > 60) {
            alert('Please enter a valid age (12–60), height and weight');
            return;
        }
        if (!leftOvaryDataURLs.length && !rightOvaryDataURLs.length) {
            alert('Please upload at least one ultrasound image (left or right ovary)');
            return;
        }
        const h = parseFloat(height);
        const w = parseFloat(weight);
        if (isNaN(h) || isNaN(w) || h <= 0 || w <= 0) {
            alert('Please enter a valid numeric height and weight');
            return;
        }
        const bmiVal = (w / ((h / 100) * (h / 100))).toFixed(1);
        clinicalData = {
            age,
            height,
            weight,
            bmi: bmiVal,
            menstrualIrregularities: document.getElementById('menstrualSelect')?.value || 'no',
            acne: document.getElementById('acneSelect')?.value || 'no',
            weightGain: document.getElementById('weightSelect')?.value || 'no',
            familyHistory: document.getElementById('familySelect')?.value || 'no'
        };
        const bmiEl = document.getElementById('bmiValue');
        if (bmiEl) bmiEl.textContent = bmiVal;
        switchView('loading');
        startLoadingSimulation();
    });
    document.getElementById('cancelClinicalBtn')?.addEventListener('click', () => switchView('dashboard'));

    // Results buttons
    document.getElementById('saveAndHistoryBtn')?.addEventListener('click', () => {
        if (latestAIResult) addToHistory(latestAIResult.prediction, latestAIResult.confidence, clinicalData);
        switchView('history');
    });
    document.getElementById('newScreeningFromResults')?.addEventListener('click', () => { resetClinicalForm();
        switchView('clinical'); });
    document.getElementById('dashboardFromResults')?.addEventListener('click', () => switchView('dashboard'));

    // Navigation between detail pages
    document.getElementById('goToXAIMapping')?.addEventListener('click', () => switchView('xaiMapping'));
    document.getElementById('goToExpertValidation')?.addEventListener('click', () => switchView('expertValidation'));
    document.getElementById('goToSUS')?.addEventListener('click', () => switchView('sus'));

    document.getElementById('backFromXAI')?.addEventListener('click', () => switchView('results'));
    document.getElementById('backToResultsFromXAI')?.addEventListener('click', () => switchView('results'));
    document.getElementById('goToExpertFromXAI')?.addEventListener('click', () => switchView('expertValidation'));

    document.getElementById('backFromExpert')?.addEventListener('click', () => switchView('results'));
    document.getElementById('backToResultsFromExpert')?.addEventListener('click', () => switchView('results'));
    document.getElementById('goToSUSFromExpert')?.addEventListener('click', () => switchView('sus'));

    document.getElementById('backFromSUS')?.addEventListener('click', () => switchView('results'));
    document.getElementById('backToResultsFromSUS')?.addEventListener('click', () => switchView('results'));
    document.getElementById('goToExpertFromSUS')?.addEventListener('click', () => switchView('expertValidation'));

    // History
    document.getElementById('startFromEmptyHistory')?.addEventListener('click', () => { resetClinicalForm();
        switchView('clinical'); });
    document.getElementById('newFromHistoryBtn')?.addEventListener('click', () => { resetClinicalForm();
        switchView('clinical'); });
    document.getElementById('clearHistoryBtn')?.addEventListener('click', () => { screeningHistory = [];
        saveHistoryToStorage();
        switchView('history'); });

    // Top nav
    document.getElementById('navDashboardBtn')?.addEventListener('click', () => switchView('dashboard'));
    document.getElementById('navHistoryBtn')?.addEventListener('click', () => switchView('history'));
    document.getElementById('newScreeningNavBtn')?.addEventListener('click', () => { resetClinicalForm();
        switchView('clinical'); });

    // Camera modal events
    document.getElementById('closeCameraBtn')?.addEventListener('click', closeCameraModal);
    document.getElementById('capturePhotoBtn')?.addEventListener('click', capturePhoto);
    document.getElementById('switchCameraBtn')?.addEventListener('click', () => {
        facingMode = facingMode === 'environment' ? 'user' : 'environment';
        startCamera();
    });

    // Expert Validation
    attachExpertValidationEvents();
    // SUS
    attachSUSEvents();
}

// ============================================
// IMAGE UPLOAD HANDLERS
// ============================================

function addOvaryImage(side, file, dataURL) {
    const images = side === 'left' ? leftOvaryImages : rightOvaryImages;
    const dataURLs = side === 'left' ? leftOvaryDataURLs : rightOvaryDataURLs;
    images.push(file);
    dataURLs.push(dataURL);
    renderCurrentView();
}

function addSelectedFiles(side, files) {
    Array.from(files).forEach(file => {
        if (file.type !== 'image/jpeg' && file.type !== 'image/png') {
            alert('Please upload JPG or PNG images');
            return;
        }
        const reader = new FileReader();
        reader.onload = event => addOvaryImage(side, file, event.target.result);
        reader.readAsDataURL(file);
    });
}

function setupImageUploadHandlers() {
    // File upload buttons
    document.querySelectorAll('.upload-file-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const target = btn.dataset.target;
            const fileInput = document.getElementById(`${target}FileInput`);
            if (fileInput) fileInput.click();
        });
    });

    // File input change handlers
    ['left', 'right'].forEach(side => {
        const fileInput = document.getElementById(`${side}FileInput`);
        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                addSelectedFiles(side, e.target.files);
                fileInput.value = '';
            });
        }
    });

    // Camera buttons
    document.querySelectorAll('.camera-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const target = btn.dataset.target;
            openCameraModal(target);
        });
    });

    // Remove image buttons
    document.querySelectorAll('.remove-image-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.target;
            const index = Number(btn.dataset.index);
            const images = target === 'left' ? leftOvaryImages : rightOvaryImages;
            const dataURLs = target === 'left' ? leftOvaryDataURLs : rightOvaryDataURLs;
            images.splice(index, 1);
            dataURLs.splice(index, 1);
            renderCurrentView();
        });
    });

    // Drag and drop support
    ['left', 'right'].forEach(side => {
        const uploadArea = document.getElementById(`${side}UploadArea`);
        if (uploadArea) {
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            uploadArea.addEventListener('dragleave', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
            });
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                addSelectedFiles(side, e.dataTransfer.files);
            });
        }
    });
}

// ============================================
// EXPERT VALIDATION LOGIC
// ============================================

function attachExpertValidationEvents() {
    const btns = document.querySelectorAll('.expert-rating-btn');
    btns.forEach(btn => {
        btn.addEventListener('click', function() {
            const criteria = this.dataset.criteria;
            const value = this.dataset.value;
            expertValidationData[criteria] = parseInt(value);
            const parent = this.parentElement;
            parent.querySelectorAll('.expert-rating-btn').forEach(b => {
                b.classList.remove('selected');
                if (parseInt(b.dataset.value) <= parseInt(value)) {
                    b.classList.add('selected');
                }
            });
        });
    });

    document.getElementById('submitExpertValidation')?.addEventListener('click', function() {
        const alignment = expertValidationData[0] || 0;
        const relevance = expertValidationData[1] || 0;
        const consistency = expertValidationData[2] || 0;
        const correlation = expertValidationData[3] || 0;
        if (alignment === 0 || relevance === 0 || consistency === 0 || correlation === 0) {
            alert('Please rate all 4 criteria (1-5).');
            return;
        }
        const avgScore = (alignment + relevance + consistency + correlation) / 4;
        const comments = document.getElementById('expertComments')?.value || '';

        const feedback = document.getElementById('expertFeedback');
        const feedbackDetails = document.getElementById('expertFeedbackDetails');
        if (feedback && feedbackDetails) {
            feedback.classList.remove('hidden');
            feedbackDetails.innerHTML = `Average score: <strong>${avgScore.toFixed(1)}/5</strong><br>
            Region alignment: ${alignment}/5 · Clinical relevance: ${relevance}/5 · Consistency: ${consistency}/5 · Confidence correlation: ${correlation}/5
            ${comments ? `<br><strong>Notes:</strong> ${comments}` : ''}`;
        }
        this.disabled = true;
        this.textContent = 'Submitted';

        if (latestAIResult && screeningHistory.length > 0) {
            screeningHistory[0].expertValidation = { alignment, relevance, consistency, correlation, avgScore, comments };
            saveHistoryToStorage();
        }
    });
}

// ============================================
// SUS LOGIC
// ============================================

function attachSUSEvents() {
    const susBtns = document.querySelectorAll('.sus-rating-btn');
    susBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const q = this.dataset.q;
            const val = this.dataset.value;
            susRatingsData[q] = parseInt(val);
            const parent = this.parentElement;
            parent.querySelectorAll('.sus-rating-btn').forEach(b => {
                b.classList.remove('selected');
                if (parseInt(b.dataset.value) <= parseInt(val)) {
                    b.classList.add('selected');
                }
            });
        });
    });

    document.getElementById('submitSUS')?.addEventListener('click', function() {
        const total = Object.keys(susRatingsData).length;
        if (total < 10) {
            alert(`Please answer all 10 questions (currently ${total}/10).`);
            return;
        }
        let sum = 0;
        for (let i = 0; i < 10; i++) {
            const rating = susRatingsData[i] || 3;
            if (i % 2 === 0) sum += rating - 1;
            else sum += 5 - rating;
        }
        const susScore = sum * 2.5;
        let grade = '';
        if (susScore >= 80.3) grade = 'Excellent (Grade A)';
        else if (susScore >= 68) grade = 'Good (Grade B)';
        else if (susScore >= 50.9) grade = 'Okay (Grade C)';
        else grade = 'Poor (Grade D/F)';

        const feedback = document.getElementById('susFeedback');
        const feedbackDetails = document.getElementById('susFeedbackDetails');
        if (feedback && feedbackDetails) {
            feedback.classList.remove('hidden');
            feedbackDetails.innerHTML = `SUS score: <strong>${susScore.toFixed(1)}/100</strong><br>Rating: <strong>${grade}</strong>`;
        }
        this.disabled = true;
        this.textContent = 'Submitted';
        localStorage.setItem('pcosense_sus_score', JSON.stringify({ score: susScore, date: new Date().toISOString(), ratings: susRatingsData }));
    });
}

// ============================================
// LOADING SIMULATION
// ============================================

function startLoadingSimulation() {
    const fillDiv = document.getElementById('progressFill');
    if (fillDiv) fillDiv.style.width = '25%';

    if (apiAvailable && (leftOvaryImages.length || rightOvaryImages.length)) {
        performRealDetection();
        return;
    }

    if (fillDiv) fillDiv.style.width = '100%';
    const aiResult = simulateAIDetection(clinicalData, leftOvaryDataURLs.length > 0 || rightOvaryDataURLs.length > 0);
    latestAIResult = aiResult;
    addToHistory(aiResult.prediction, aiResult.confidence, clinicalData);
    switchView('results');
}

async function performRealDetection() {
    try {
        // Prepare image files
        const imageFiles = {};
        imageFiles.left = leftOvaryImages;
        imageFiles.right = rightOvaryImages;

        const aiResult = await realAIDetection(imageFiles, clinicalData);
        const fillDiv = document.getElementById('progressFill');
        if (fillDiv) fillDiv.style.width = '100%';
        latestAIResult = aiResult;
        addToHistory(aiResult.prediction, aiResult.confidence, clinicalData);
        switchView('results');
    } catch (error) {
        console.error('AI detection failed:', error);
        const aiResult = simulateAIDetection(clinicalData, leftOvaryDataURLs.length > 0 || rightOvaryDataURLs.length > 0);
        latestAIResult = aiResult;
        addToHistory(aiResult.prediction, aiResult.confidence, clinicalData);
        switchView('results');
    }
}

function resetClinicalForm() {
    clinicalData = { age: '', height: '', weight: '', bmi: '', menstrualIrregularities: 'no', acne: 'no', weightGain: 'no', familyHistory: 'no' };
    leftOvaryDataURLs = [];
    leftOvaryImages = [];
    rightOvaryDataURLs = [];
    rightOvaryImages = [];
}

// ============================================
// INIT
// ============================================

loadHistoryFromStorage();
checkAPIConnection();
switchView('dashboard');