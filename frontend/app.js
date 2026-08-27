// HAR Frontend Application Logic

const API_BASE_URL = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : window.location.origin;

const ACTIVITIES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
];

let isSensorActive = false;
let sensorBuffer = [];
const BUFFER_SIZE = 128;          // UCI standard window: 2.56 s @ 50 Hz
const MIN_SAMPLES = 64;           // backend accepts >= 64 samples
const SAMPLE_INTERVAL_MS = 20;    // throttle sampling to ~50 Hz
const PREDICTION_INTERVAL = 2000; // predict every 2 s on the rolling window
let lastPredictionTime = 0;
let lastSampleTime = 0;
let predictionHistory = [];
let simulationTimer = null;
let simulationTick = 0;

// DOM Elements
const systemStatusDot = document.getElementById("system-status-dot");
const systemStatusText = document.getElementById("system-status-text");
const btnToggleSensors = document.getElementById("btn-toggle-sensors");
const sensorStatusBadge = document.getElementById("sensor-status-badge");
const currentActivityEl = document.getElementById("current-activity");
const currentConfidenceEl = document.getElementById("current-confidence");
const confidenceBarEl = document.getElementById("confidence-bar");
const probabilityContainer = document.getElementById("probability-container");
const neuralLogTerminal = document.getElementById("neural-log-terminal");
const logCounter = document.getElementById("log-counter");

const accelXEl = document.getElementById("accel-x");
const accelYEl = document.getElementById("accel-y");
const accelZEl = document.getElementById("accel-z");
const gyroXEl = document.getElementById("gyro-x");
const gyroYEl = document.getElementById("gyro-y");
const gyroZEl = document.getElementById("gyro-z");

const btnPredictDescriptive = document.getElementById("btn-predict-descriptive");
const demoResultContainer = document.getElementById("demo-result-container");

function init() {
    renderProbabilityBarsDefault();
    simulateLoadingSequence();
    checkSystemHealth();
    fetchModelInfo();

    setInterval(checkSystemHealth, 5000);

    btnToggleSensors.addEventListener("click", toggleMotionSensors);
    btnPredictDescriptive.addEventListener("click", runDescriptivePrediction);
    document.getElementById("btn-scroll-demo").addEventListener("click", () => {
        document.getElementById("demo-section").scrollIntoView({ behavior: "smooth" });
    });
}

async function simulateLoadingSequence() {
    const fill = document.getElementById("loading-progress-fill");
    const statusText = document.getElementById("loading-status-text");
    const screen = document.getElementById("loading-screen");

    const steps = [
        { pct: 25, text: "Loading Ensemble Classifier (RF + GB)..." },
        { pct: 60, text: "Calibrating Pocket Inertial Filters..." },
        { pct: 90, text: "Establishing Secure Backend Tunnel..." },
        { pct: 100, text: "System Ready." }
    ];

    for (const step of steps) {
        if (fill) fill.style.width = `${step.pct}%`;
        if (statusText) statusText.textContent = step.text;
        await new Promise(r => setTimeout(r, 180));
    }

    try {
        await fetch(`${API_BASE_URL}/health`);
    } catch (e) {}

    await new Promise(r => setTimeout(r, 150));
    if (screen) screen.classList.add("fade-out");
}

function renderProbabilityBarsDefault() {
    probabilityContainer.innerHTML = "";
    ACTIVITIES.forEach(act => {
        const item = document.createElement("div");
        item.className = "prob-item";
        item.innerHTML = `
            <div class="prob-info">
                <span class="prob-name">${act}</span>
                <span class="prob-percentage" id="prob-pct-${act}">0%</span>
            </div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" id="prob-bar-${act}" style="width: 0%"></div>
            </div>
        `;
        probabilityContainer.appendChild(item);
    });
}

async function checkSystemHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            if (data.model_loaded) {
                systemStatusDot.className = "status-dot online";
                systemStatusText.textContent = "SYSTEM ONLINE (READY)";
            } else {
                systemStatusDot.className = "status-dot";
                systemStatusText.textContent = "MODEL DEGRADED";
            }
        } else {
            setOfflineStatus();
        }
    } catch (err) {
        setOfflineStatus();
    }
}

function setOfflineStatus() {
    systemStatusDot.className = "status-dot offline";
    systemStatusText.textContent = "API OFFLINE";
}

async function fetchModelInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/model-info`);
        if (response.ok) {
            const info = await response.json();
            document.getElementById("stat-model-name").textContent = info.model_name || "Ensemble";
            document.getElementById("stat-accuracy").textContent = `${(info.accuracy * 100).toFixed(1)}%`;
            document.getElementById("stat-features").textContent = info.features_count || 561;
            document.getElementById("stat-samples").textContent = (info.training_samples || 7352).toLocaleString();
        }
    } catch (err) {
        console.error("Failed to fetch model info:", err);
    }
}

// Motion Sensors Handling (UCI-faithful: total acceleration in g, gyro in rad/s)
async function toggleMotionSensors() {
    if (isSensorActive) {
        stopSensors();
    } else {
        await startSensors();
    }
}

async function startSensors() {
    const hasDeviceMotion = typeof window.DeviceMotionEvent !== "undefined";
    if (hasDeviceMotion) {
        if (typeof DeviceMotionEvent.requestPermission === "function") {
            try {
                const permissionState = await DeviceMotionEvent.requestPermission();
                if (permissionState !== "granted") {
                    startSimulation();
                    return;
                }
            } catch (error) {
                console.error("Permission error:", error);
                startSimulation();
                return;
            }
        }
        window.addEventListener("devicemotion", handleDeviceMotion, true);
        isSensorActive = true;
        setSensorUI("POCKET SENSORS ACTIVE", true, "var(--danger)");
    } else {
        // Desktop browsers have no motion sensors -> activity simulation mode
        startSimulation();
    }
}

function stopSensors() {
    if (simulationTimer !== null) {
        clearInterval(simulationTimer);
        simulationTimer = null;
    }
    window.removeEventListener("devicemotion", handleDeviceMotion, true);
    isSensorActive = false;
    sensorBuffer = [];
    setSensorUI("SENSORS IDLE", false, "");
    document.getElementById("sim-controls").style.display = "none";
}

function setSensorUI(badgeText, active, buttonColor) {
    sensorStatusBadge.textContent = badgeText;
    sensorStatusBadge.style.color = active ? "var(--accent-green)" : "";
    sensorStatusBadge.style.borderColor = active ? "rgba(16, 185, 129, 0.3)" : "";
    btnToggleSensors.innerHTML = active
        ? `DISABLE MOTION SENSORS`
        : `<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6"></path></svg> ENABLE MOTION SENSORS`;
    btnToggleSensors.style.background = buttonColor;
}

function handleDeviceMotion(event) {
    const now = Date.now();
    if (now - lastSampleTime < SAMPLE_INTERVAL_MS) return;
    lastSampleTime = now;

    // The UCI pipeline models TOTAL acceleration (gravity + body motion).
    // event.acceleration has gravity REMOVED, so prefer accelerationIncludingGravity.
    const acc = event.accelerationIncludingGravity || event.acceleration || { x: 0, y: 0, z: 0 };
    const gyro = event.rotationRate || { alpha: 0, beta: 0, gamma: 0 };

    // Devices report m/s^2 -> convert to g (UCI convention); gyro deg/s -> rad/s.
    const G = 9.80665;
    const ax = clampValue((acc.x || 0) / G, 6);
    const ay = clampValue((acc.y || 0) / G, 6);
    const az = clampValue((acc.z || 0) / G, 6);
    const gx = clampValue((gyro.alpha || 0) * (Math.PI / 180), 35);
    const gy = clampValue((gyro.beta || 0) * (Math.PI / 180), 35);
    const gz = clampValue((gyro.gamma || 0) * (Math.PI / 180), 35);

    pushSample(ax, ay, az, gx, gy, gz);
}

function clampValue(v, limit) {
    if (!isFinite(v)) return 0;
    return Math.max(-limit, Math.min(limit, v));
}

function pushSample(ax, ay, az, gx, gy, gz) {
    accelXEl.textContent = ax.toFixed(2);
    accelYEl.textContent = ay.toFixed(2);
    accelZEl.textContent = az.toFixed(2);
    gyroXEl.textContent = gx.toFixed(2);
    gyroYEl.textContent = gy.toFixed(2);
    gyroZEl.textContent = gz.toFixed(2);

    sensorBuffer.push({
        timestamp: Date.now(),
        accelerometer: { x: ax, y: ay, z: az },
        gyroscope: { x: gx, y: gy, z: gz }
    });

    if (sensorBuffer.length > BUFFER_SIZE) {
        sensorBuffer.shift();
    }

    const now = Date.now();
    if (sensorBuffer.length >= MIN_SAMPLES && (now - lastPredictionTime > PREDICTION_INTERVAL)) {
        lastPredictionTime = now;
        sendSensorWindowForPrediction([...sensorBuffer]);
    }
}

// ---------------------------------------------------------------------------
// Desktop simulation mode: activity-coherent inertial patterns so the full
// pipeline (window -> 561 UCI features -> model) can be exercised anywhere.
// ---------------------------------------------------------------------------
const SIM_GRAVITY = {
    WALKING: [0.0, 0.15, 0.99],
    WALKING_UPSTAIRS: [0.0, 0.55, 0.83],
    WALKING_DOWNSTAIRS: [0.0, -0.5, 0.86],
    SITTING: [0.1, 0.95, 0.3],
    STANDING: [0.05, 0.1, 0.99],
    LAYING: [0.98, 0.1, 0.15]
};
const SIM_DYNAMIC = {
    WALKING: { freq: 1.8, amp: 0.16, tilt: 0.0 },
    WALKING_UPSTAIRS: { freq: 1.4, amp: 0.26, tilt: 0.25 },
    WALKING_DOWNSTAIRS: { freq: 1.7, amp: 0.22, tilt: -0.25 }
};

function startSimulation() {
    simulationTick = 0;
    document.getElementById("sim-controls").style.display = "block";
    isSensorActive = true;
    setSensorUI("SIMULATION MODE ACTIVE", true, "var(--danger)");
    simulationTimer = setInterval(simulationSample, SAMPLE_INTERVAL_MS);
}

function simulationSample() {
    const activity = document.getElementById("sim-activity").value;
    simulationTick += 1;
    const t = simulationTick / 50; // seconds @ 50 Hz

    const [gvx, gvy, gvz] = SIM_GRAVITY[activity] || SIM_GRAVITY.STANDING;
    const dyn = SIM_DYNAMIC[activity];
    let ax = gvx, ay = gvy, az = gvz, gx = 0, gy = 0, gz = 0;

    if (dyn) {
        const tilt = dyn.tilt * Math.sin(2 * Math.PI * 0.35 * t);
        const swing = Math.sin(2 * Math.PI * dyn.freq * t);
        const swing2 = Math.sin(2 * Math.PI * dyn.freq * 2 * t);
        ax = dyn.amp * swing + tilt;
        ay = gvy + 0.6 * dyn.amp * swing2 + 0.5 * tilt;
        az = gvz + dyn.amp * Math.sin(2 * Math.PI * dyn.freq * t + 1.0) + tilt;
        gx = 0.7 * swing;
        gy = 0.5 * Math.cos(2 * Math.PI * dyn.freq * t);
        gz = 0.4 * Math.sin(2 * Math.PI * dyn.freq * t + 0.7);
    }

    const noise = () => (Math.random() - 0.5) * 0.02;
    pushSample(
        clampValue(ax + noise(), 6), clampValue(ay + noise(), 6), clampValue(az + noise(), 6),
        clampValue(gx + noise(), 35), clampValue(gy + noise(), 35), clampValue(gz + noise(), 35)
    );
}

async function sendSensorWindowForPrediction(windowData) {
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ window: windowData })
        });

        if (response.ok) {
            const result = await response.json();
            updatePredictionUI(result);
        }
    } catch (err) {
        console.error("Sensor prediction connection error:", err);
    }
}

function updatePredictionUI(result) {
    const activity = result.activity;
    const confidence = result.confidence * 100;
    const probabilities = result.probabilities;

    currentActivityEl.textContent = activity;
    currentConfidenceEl.textContent = `${confidence.toFixed(1)}% Confidence`;
    if (confidenceBarEl) confidenceBarEl.style.width = `${confidence}%`;

    for (const [act, prob] of Object.entries(probabilities)) {
        const pct = (prob * 100).toFixed(1);
        const bar = document.getElementById(`prob-bar-${act}`);
        const pctText = document.getElementById(`prob-pct-${act}`);
        if (bar) bar.style.width = `${pct}%`;
        if (pctText) pctText.textContent = `${pct}%`;
    }

    addElonMuskLogEntry(activity, confidence, result.top_predictions);
}

function addElonMuskLogEntry(activity, confidence, topPreds) {
    const timeStr = new Date().toLocaleTimeString();
    predictionHistory.unshift({ time: timeStr, activity, confidence, topPreds });

    if (neuralLogTerminal.querySelector(".system")) {
        neuralLogTerminal.innerHTML = "";
    }

    neuralLogTerminal.innerHTML = "";
    predictionHistory.slice(0, 15).forEach((entry) => {
        const div = document.createElement("div");
        div.className = `log-entry ${entry.activity.toLowerCase()}`;
        div.innerHTML = `
            <div class="log-info">
                <span class="log-time">${entry.time}</span>
                <span class="log-activity">🚶 ${entry.activity}</span>
                <span class="log-metrics">Top: ${entry.top_predictions ? entry.top_predictions.map(p => `${p.activity} (${(p.probability*100).toFixed(0)}%)`).join(' | ') : ''}</span>
            </div>
            <div class="log-conf">${entry.confidence.toFixed(1)}%</div>
        `;
        neuralLogTerminal.appendChild(div);
    });

    logCounter.textContent = `${predictionHistory.length} Events Stored`;
}

// Descriptive Demo Mode (Matching exact prompt format)
async function runDescriptivePrediction() {
    const intensity = document.getElementById("desc-intensity").value;
    const stability = document.getElementById("desc-stability").value;
    const orientation = document.getElementById("desc-orientation").value;
    const rotation = document.getElementById("desc-rotation").value;
    const pattern = document.getElementById("desc-pattern").value;

    try {
        btnPredictDescriptive.textContent = "Analyzing Movement Vector...";
        btnPredictDescriptive.disabled = true;

        const response = await fetch(`${API_BASE_URL}/predict/descriptive`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ intensity, stability, orientation, rotation, pattern })
        });

        if (response.ok) {
            const data = await response.json();
            const topPreds = data.top_predictions || [];

            demoResultContainer.innerHTML = `
                <div class="result-header">RESULT</div>
                <div class="result-row"><span class="result-label">Movement Intensity</span><span class="result-val">${intensity}</span></div>
                <div class="result-row"><span class="result-label">Movement Stability</span><span class="result-val">${stability}</span></div>
                <div class="result-row"><span class="result-label">Body Orientation</span><span class="result-val">${orientation}</span></div>
                <div class="result-row"><span class="result-label">Rotation</span><span class="result-val">${rotation}</span></div>
                <div class="result-row"><span class="result-label">Movement Pattern</span><span class="result-val">${pattern}</span></div>
                
                <div class="result-header" style="margin-top: 16px;">DETECTED ACTIVITY</div>
                <div class="result-activity-display">
                    <div class="result-activity-title">🚶 ${data.activity}</div>
                    <div class="result-activity-conf">Confidence: ${(data.confidence * 100).toFixed(2)}%</div>
                </div>

                <div style="font-size: 12px; color: #94a3b8; margin-top: 8px;">Top predictions:</div>
                ${topPreds.map(p => `
                    <div style="font-size: 12px; color: #f8fafc; margin-top: 2px;">
                      &nbsp;&nbsp;🚶 ${p.activity.padEnd(22, ' ')} ${(p.probability * 100).toFixed(2)}%
                    </div>
                `).join('')}
                
                <div style="text-align: center; color: #4ade80; font-size: 11px; margin-top: 16px; border-top: 1px dashed rgba(255,255,255,0.15); pt: 8px;">
                    Prediction Complete
                </div>
            `;

            updatePredictionUI(data);
        } else {
            const err = await response.json();
            alert(err.detail || "Prediction failed.");
        }
    } catch (err) {
        console.error("Descriptive prediction error:", err);
        alert("Failed to communicate with backend API.");
    } finally {
        btnPredictDescriptive.textContent = "Run Descriptive Prediction Processing";
        btnPredictDescriptive.disabled = false;
    }
}

document.addEventListener("DOMContentLoaded", init);
