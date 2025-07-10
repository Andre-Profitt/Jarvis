// JARVIS Dashboard JavaScript

// Global variables
let socket = null;
let responseChart = null;
let responseData = [];
const MAX_HISTORY_ITEMS = 50;
const MAX_CHART_POINTS = 20;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeSocket();
    initializeChart();
    setupEventListeners();
});

// Initialize Socket.IO connection
function initializeSocket() {
    socket = io();
    
    // Connection events
    socket.on('connect', function() {
        console.log('Connected to JARVIS Dashboard');
        updateConnectionStatus(true);
    });
    
    socket.on('disconnect', function() {
        console.log('Disconnected from JARVIS Dashboard');
        updateConnectionStatus(false);
    });
    
    // JARVIS events
    socket.on('initial_state', function(data) {
        updateDashboard(data);
    });
    
    socket.on('status_update', function(data) {
        updateJarvisStatus(data.status);
    });
    
    socket.on('listening_update', function(data) {
        updateListeningStatus(data.listening);
    });
    
    socket.on('command_update', function(data) {
        addCommandToHistory(data);
    });
    
    socket.on('metrics_update', function(data) {
        updateMetrics(data);
    });
    
    socket.on('predictions_update', function(data) {
        updatePredictions(data.predictions);
    });
    
    socket.on('voice_activity', function(data) {
        updateVoiceActivity(data.active);
    });
}

// Initialize response time chart
function initializeChart() {
    const ctx = document.getElementById('response-chart').getContext('2d');
    
    responseChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Response Time (ms)',
                data: [],
                borderColor: '#00d4ff',
                backgroundColor: 'rgba(0, 212, 255, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                pointRadius: 3,
                pointBackgroundColor: '#00d4ff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    display: false
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#999'
                    }
                }
            }
        }
    });
}

// Setup event listeners
function setupEventListeners() {
    // Command form
    document.getElementById('command-form').addEventListener('submit', function(e) {
        e.preventDefault();
        sendCommand();
    });
    
    // Listening toggle
    document.getElementById('listening-toggle').addEventListener('change', function() {
        socket.emit('toggle_listening');
    });
    
    // Clear history
    document.getElementById('clear-history').addEventListener('click', function() {
        clearCommandHistory();
    });
}

// Update connection status
function updateConnectionStatus(connected) {
    const statusEl = document.getElementById('connection-status');
    
    if (connected) {
        statusEl.classList.remove('bg-danger');
        statusEl.classList.add('bg-success', 'connected');
        statusEl.innerHTML = '<i class="fas fa-circle"></i> Connected';
    } else {
        statusEl.classList.remove('bg-success', 'connected');
        statusEl.classList.add('bg-danger');
        statusEl.innerHTML = '<i class="fas fa-circle"></i> Disconnected';
    }
}

// Update JARVIS status
function updateJarvisStatus(status) {
    const statusEl = document.getElementById('jarvis-status');
    
    statusEl.classList.remove('bg-secondary', 'bg-success', 'bg-warning');
    
    switch(status) {
        case 'online':
            statusEl.classList.add('bg-success', 'online');
            statusEl.innerHTML = '<i class="fas fa-check-circle"></i> Online';
            break;
        case 'busy':
            statusEl.classList.add('bg-warning', 'busy');
            statusEl.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing';
            break;
        default:
            statusEl.classList.add('bg-secondary');
            statusEl.innerHTML = '<i class="fas fa-power-off"></i> Offline';
    }
}

// Update listening status
function updateListeningStatus(listening) {
    const toggle = document.getElementById('listening-toggle');
    const status = document.getElementById('listening-status');
    
    toggle.checked = listening;
    status.textContent = listening ? 'Active' : 'Inactive';
    
    if (listening) {
        status.classList.add('text-success');
    } else {
        status.classList.remove('text-success');
    }
}

// Update voice activity indicator
function updateVoiceActivity(active) {
    const indicator = document.getElementById('voice-indicator');
    
    if (active) {
        indicator.classList.add('active');
    } else {
        indicator.classList.remove('active');
    }
}

// Send command
function sendCommand() {
    const input = document.getElementById('command-input');
    const command = input.value.trim();
    
    if (command) {
        // Send via socket
        socket.emit('send_command', { command: command });
        
        // Clear input
        input.value = '';
        
        // Show processing
        updateJarvisStatus('busy');
    }
}

// Add command to history
function addCommandToHistory(data) {
    const historyEl = document.getElementById('command-history');
    
    // Remove placeholder if exists
    const placeholder = historyEl.querySelector('.text-center');
    if (placeholder) {
        placeholder.remove();
    }
    
    // Create command entry
    const entry = document.createElement('div');
    entry.className = 'command-entry new';
    entry.innerHTML = `
        <div class="command-text">
            <i class="fas fa-user"></i> ${escapeHtml(data.command)}
        </div>
        <div class="response-text">
            <i class="fas fa-robot"></i> ${escapeHtml(data.response)}
        </div>
        <div class="command-time">
            ${formatTime(data.timestamp)}
        </div>
    `;
    
    // Add to top of history
    historyEl.insertBefore(entry, historyEl.firstChild);
    
    // Remove animation class after animation completes
    setTimeout(() => {
        entry.classList.remove('new');
    }, 500);
    
    // Limit history items
    while (historyEl.children.length > MAX_HISTORY_ITEMS) {
        historyEl.removeChild(historyEl.lastChild);
    }
    
    // Update status
    updateJarvisStatus('online');
}

// Clear command history
function clearCommandHistory() {
    const historyEl = document.getElementById('command-history');
    historyEl.innerHTML = `
        <div class="text-center text-muted py-5">
            <i class="fas fa-comments fa-3x mb-3"></i>
            <p>No commands yet. Start talking to JARVIS!</p>
        </div>
    `;
}

// Update metrics
function updateMetrics(metrics) {
    // CPU Usage
    if (metrics.cpu_usage !== undefined) {
        const cpuBar = document.getElementById('cpu-bar');
        const cpuPercent = Math.min(100, metrics.cpu_usage);
        cpuBar.style.width = cpuPercent + '%';
        cpuBar.textContent = cpuPercent + '%';
    }
    
    // Memory Usage
    if (metrics.memory_usage !== undefined) {
        const memoryBar = document.getElementById('memory-bar');
        const memoryPercent = Math.min(100, (metrics.memory_usage / 1000) * 100);
        memoryBar.style.width = memoryPercent + '%';
        memoryBar.textContent = metrics.memory_usage + 'MB';
    }
    
    // Active Agents
    if (metrics.active_agents !== undefined) {
        document.getElementById('active-agents').textContent = metrics.active_agents;
    }
    
    // Response Time
    if (metrics.response_time !== undefined) {
        document.getElementById('response-time').textContent = metrics.response_time;
        updateResponseChart(metrics.response_time);
    }
}

// Update response time chart
function updateResponseChart(responseTime) {
    const now = new Date().toLocaleTimeString();
    
    responseData.push(responseTime);
    responseChart.data.labels.push(now);
    
    // Limit data points
    if (responseData.length > MAX_CHART_POINTS) {
        responseData.shift();
        responseChart.data.labels.shift();
    }
    
    responseChart.data.datasets[0].data = responseData;
    responseChart.update('none'); // Update without animation
}

// Update predictions
function updatePredictions(predictions) {
    const listEl = document.getElementById('predictions-list');
    
    if (!predictions || predictions.length === 0) {
        listEl.innerHTML = '<p class="text-muted">No predictions available</p>';
        return;
    }
    
    listEl.innerHTML = '';
    
    predictions.forEach(pred => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.innerHTML = `
            <div class="prediction-confidence">${Math.round(pred.confidence * 100)}%</div>
            <div class="prediction-type">
                <i class="fas fa-lightbulb"></i> ${pred.type}
            </div>
            <div class="prediction-desc">${escapeHtml(pred.description)}</div>
        `;
        listEl.appendChild(item);
    });
}

// Update dashboard with initial state
function updateDashboard(state) {
    updateJarvisStatus(state.status);
    updateListeningStatus(state.listening);
    
    if (state.metrics) {
        updateMetrics(state.metrics);
    }
    
    if (state.last_command && state.last_response) {
        addCommandToHistory({
            command: state.last_command,
            response: state.last_response,
            timestamp: new Date().toISOString()
        });
    }
}

// Utility functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatTime(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;
    
    if (diff < 60000) { // Less than 1 minute
        return 'Just now';
    } else if (diff < 3600000) { // Less than 1 hour
        const minutes = Math.floor(diff / 60000);
        return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
    } else if (diff < 86400000) { // Less than 1 day
        const hours = Math.floor(diff / 3600000);
        return `${hours} hour${hours > 1 ? 's' : ''} ago`;
    } else {
        return date.toLocaleString();
    }
}

// Periodic updates
setInterval(() => {
    if (socket && socket.connected) {
        socket.emit('request_metrics');
    }
}, 5000);