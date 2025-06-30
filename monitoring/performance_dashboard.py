"""
JARVIS Phase 10: Performance Monitoring Dashboard
Real-time visualization of performance optimizations
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any
import logging
from aiohttp import web
import aiohttp_cors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    WebSocket server for real-time performance monitoring
    """
    
    def __init__(self, jarvis_core):
        self.jarvis = jarvis_core
        self.app = web.Application()
        self.websockets = set()
        self.monitoring_active = True
        
        # Setup routes
        self.setup_routes()
        
        # Setup CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*"
            )
        })
        
        # Configure CORS on all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    def setup_routes(self):
        """Setup web routes"""
        self.app.router.add_get('/', self.index)
        self.app.router.add_get('/ws', self.websocket_handler)
        self.app.router.add_get('/api/stats', self.get_stats)
        self.app.router.add_post('/api/benchmark', self.run_benchmark)
        self.app.router.add_post('/api/optimize', self.optimize_workload)
    
    async def index(self, request):
        """Serve the dashboard HTML"""
        html_content = self.get_dashboard_html()
        return web.Response(text=html_content, content_type='text/html')
    
    async def websocket_handler(self, request):
        """Handle WebSocket connections"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.websockets.add(ws)
        
        try:
            # Send initial data
            stats = self.jarvis.get_performance_report()
            await ws.send_json({
                'type': 'initial',
                'data': stats
            })
            
            # Keep connection alive
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    # Handle client messages if needed
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
        except Exception as e:
            logger.error(f'WebSocket handler error: {e}')
        finally:
            self.websockets.discard(ws)
            
        return ws
    
    async def broadcast_stats(self):
        """Broadcast stats to all connected clients"""
        while self.monitoring_active:
            if self.websockets:
                stats = self.jarvis.get_performance_report()
                
                # Add timestamp
                stats['timestamp'] = time.time()
                
                # Broadcast to all clients
                disconnected = set()
                for ws in self.websockets:
                    try:
                        await ws.send_json({
                            'type': 'update',
                            'data': stats
                        })
                    except ConnectionResetError:
                        disconnected.add(ws)
                
                # Remove disconnected clients
                self.websockets -= disconnected
            
            await asyncio.sleep(1)  # Update every second
    
    async def get_stats(self, request):
        """Get current performance statistics"""
        stats = self.jarvis.get_performance_report()
        return web.json_response(stats)
    
    async def run_benchmark(self, request):
        """Run performance benchmark"""
        results = await self.jarvis.run_performance_benchmark()
        return web.json_response(results)
    
    async def optimize_workload(self, request):
        """Optimize for specific workload"""
        data = await request.json()
        workload_type = data.get('workload_type', 'real_time')
        
        await self.jarvis.optimize_for_workload(workload_type)
        
        return web.json_response({
            'status': 'success',
            'message': f'Optimized for {workload_type} workload'
        })
    
    def get_dashboard_html(self):
        """Generate dashboard HTML"""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>JARVIS Phase 10 - Performance Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #0a0a0a;
            color: #fff;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        h1 {
            font-size: 2.5em;
            margin: 0;
            background: linear-gradient(45deg, #00d4ff, #0099ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .subtitle {
            color: #666;
            font-size: 1.2em;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }
        
        .panel {
            background: #1a1a1a;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.5);
            border: 1px solid #333;
        }
        
        .panel h2 {
            margin-top: 0;
            color: #00d4ff;
            font-size: 1.3em;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 10px;
            background: #2a2a2a;
            border-radius: 8px;
        }
        
        .metric-value {
            font-weight: bold;
            color: #0099ff;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 20px;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        
        .status-good { background: #00ff00; }
        .status-warning { background: #ffaa00; }
        .status-critical { background: #ff0000; }
        
        .control-panel {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        
        button {
            background: linear-gradient(45deg, #00d4ff, #0099ff);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,212,255,0.3);
        }
        
        .optimization-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 15px;
        }
        
        .optimization-item {
            padding: 15px;
            background: #2a2a2a;
            border-radius: 8px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .optimization-item:hover {
            background: #3a3a3a;
            transform: translateY(-2px);
        }
        
        .optimization-value {
            font-size: 2em;
            font-weight: bold;
            color: #00d4ff;
        }
        
        .optimization-label {
            color: #888;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #2a2a2a;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #0099ff);
            transition: width 0.3s ease;
        }
        
        .loading {
            text-align: center;
            padding: 50px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>JARVIS Performance Monitor</h1>
        <div class="subtitle">Phase 10: Real-time Performance Optimization</div>
    </div>
    
    <div class="dashboard">
        <!-- System Metrics -->
        <div class="panel">
            <h2>System Metrics</h2>
            <div class="metric">
                <span>CPU Usage</span>
                <span class="metric-value" id="cpu-usage">--</span>
            </div>
            <div class="metric">
                <span>Memory Usage</span>
                <span class="metric-value" id="memory-usage">--</span>
            </div>
            <div class="metric">
                <span>Active Threads</span>
                <span class="metric-value" id="active-threads">--</span>
            </div>
            <div class="metric">
                <span>Response Time</span>
                <span class="metric-value" id="response-time">--</span>
            </div>
            <div class="chart-container">
                <canvas id="system-chart"></canvas>
            </div>
        </div>
        
        <!-- Cache Performance -->
        <div class="panel">
            <h2>Cache Performance</h2>
            <div class="metric">
                <span>Hit Rate</span>
                <span class="metric-value" id="cache-hit-rate">--</span>
            </div>
            <div class="metric">
                <span>Total Hits</span>
                <span class="metric-value" id="cache-hits">--</span>
            </div>
            <div class="metric">
                <span>Memory Usage</span>
                <span class="metric-value" id="cache-memory">--</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="cache-efficiency"></div>
            </div>
            <div class="chart-container">
                <canvas id="cache-chart"></canvas>
            </div>
        </div>
        
        <!-- Parallel Processing -->
        <div class="panel">
            <h2>Parallel Processing</h2>
            <div class="metric">
                <span>Tasks Executed</span>
                <span class="metric-value" id="parallel-tasks">--</span>
            </div>
            <div class="metric">
                <span>Average Speedup</span>
                <span class="metric-value" id="parallel-speedup">--</span>
            </div>
            <div class="optimization-grid">
                <div class="optimization-item">
                    <div class="optimization-value" id="thread-pool">--</div>
                    <div class="optimization-label">Thread Pool</div>
                </div>
                <div class="optimization-item">
                    <div class="optimization-value" id="process-pool">--</div>
                    <div class="optimization-label">Process Pool</div>
                </div>
                <div class="optimization-item">
                    <div class="optimization-value" id="gpu-usage">--</div>
                    <div class="optimization-label">GPU Usage</div>
                </div>
                <div class="optimization-item">
                    <div class="optimization-value" id="async-tasks">--</div>
                    <div class="optimization-label">Async Tasks</div>
                </div>
            </div>
        </div>
        
        <!-- JIT Compilation -->
        <div class="panel">
            <h2>JIT Compilation</h2>
            <div class="metric">
                <span>Compiled Functions</span>
                <span class="metric-value" id="jit-compiled">--</span>
            </div>
            <div class="metric">
                <span>Average Speedup</span>
                <span class="metric-value" id="jit-speedup">--</span>
            </div>
            <div class="metric">
                <span>Hot Functions</span>
                <span class="metric-value" id="hot-functions">--</span>
            </div>
            <div id="hot-functions-list" style="margin-top: 15px;"></div>
        </div>
        
        <!-- Module Management -->
        <div class="panel">
            <h2>Module Management</h2>
            <div class="metric">
                <span>Loaded Modules</span>
                <span class="metric-value" id="loaded-modules">--</span>
            </div>
            <div class="metric">
                <span>Memory Saved</span>
                <span class="metric-value" id="memory-saved">--</span>
            </div>
            <div class="metric">
                <span>Lazy Load Efficiency</span>
                <span class="metric-value" id="lazy-efficiency">--</span>
            </div>
            <div id="module-list" style="margin-top: 15px;"></div>
        </div>
        
        <!-- Controls -->
        <div class="panel">
            <h2>Performance Controls</h2>
            <div class="control-panel">
                <button onclick="runBenchmark()">Run Benchmark</button>
                <button onclick="optimizeWorkload('real_time')">Real-time Mode</button>
                <button onclick="optimizeWorkload('batch_processing')">Batch Mode</button>
                <button onclick="optimizeWorkload('memory_constrained')">Low Memory</button>
            </div>
            <div id="benchmark-results" style="margin-top: 20px;"></div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let systemChart = null;
        let cacheChart = null;
        
        // Initialize WebSocket connection
        function connectWebSocket() {
            ws = new WebSocket('ws://localhost:8889/ws');
            
            ws.onopen = () => {
                console.log('Connected to JARVIS Performance Monitor');
                document.body.classList.remove('loading');
            };
            
            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                if (message.type === 'initial' || message.type === 'update') {
                    updateDashboard(message.data);
                }
            };
            
            ws.onclose = () => {
                console.log('Disconnected from JARVIS');
                setTimeout(connectWebSocket, 3000); // Reconnect after 3 seconds
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }
        
        // Update dashboard with new data
        function updateDashboard(data) {
            // System metrics
            if (data.system) {
                updateElement('cpu-usage', data.system.cpu_usage + '%');
                updateElement('memory-usage', data.system.memory_usage + '%');
                updateElement('active-threads', data.system.active_threads);
            }
            
            if (data.metrics) {
                updateElement('response-time', 
                    (data.metrics.average_response_time * 1000).toFixed(1) + 'ms');
            }
            
            // Cache metrics
            if (data.cache) {
                updateElement('cache-hit-rate', 
                    (data.cache.hit_rate * 100).toFixed(1) + '%');
                updateElement('cache-hits', data.cache.total_hits);
                updateElement('cache-memory', data.cache.memory_usage);
                
                // Update progress bar
                const efficiency = data.cache.hit_rate * 100;
                document.getElementById('cache-efficiency').style.width = efficiency + '%';
            }
            
            // Parallel processing
            if (data.metrics) {
                updateElement('parallel-tasks', data.metrics.parallel_tasks);
            }
            
            if (data.parallel && data.parallel.strategy_performance) {
                let totalTasks = 0;
                let avgTime = 0;
                
                Object.values(data.parallel.strategy_performance).forEach(stat => {
                    totalTasks += stat.total_tasks;
                    avgTime += stat.average_time * stat.total_tasks;
                });
                
                if (totalTasks > 0) {
                    avgTime /= totalTasks;
                    const speedup = 1 / avgTime; // Simplified speedup calculation
                    updateElement('parallel-speedup', speedup.toFixed(2) + 'x');
                }
            }
            
            // JIT compilation
            if (data.jit) {
                updateElement('jit-compiled', data.jit.successful);
                updateElement('jit-speedup', 
                    data.jit.average_speedup.toFixed(2) + 'x');
                
                // Hot functions list
                if (data.jit.hot_functions) {
                    const listHtml = data.jit.hot_functions
                        .slice(0, 5)
                        .map(func => `
                            <div class="metric">
                                <span style="font-size: 0.9em">${func.name.split('.').pop()}</span>
                                <span class="metric-value">${func.speedup.toFixed(2)}x</span>
                            </div>
                        `).join('');
                    document.getElementById('hot-functions-list').innerHTML = listHtml;
                }
            }
            
            // Module management
            if (data.modules) {
                updateElement('loaded-modules', 
                    `${data.modules.loaded_modules}/${data.modules.total_modules}`);
                updateElement('memory-saved', data.modules.memory_usage.available_memory);
                
                // Calculate lazy load efficiency
                const efficiency = (1 - data.modules.loaded_modules / data.modules.total_modules) * 100;
                updateElement('lazy-efficiency', efficiency.toFixed(1) + '%');
            }
            
            // Update charts
            updateCharts(data);
        }
        
        function updateElement(id, value) {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        }
        
        // Initialize charts
        function initCharts() {
            // System performance chart
            const systemCtx = document.getElementById('system-chart').getContext('2d');
            systemChart = new Chart(systemCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'CPU %',
                        data: [],
                        borderColor: '#00d4ff',
                        backgroundColor: 'rgba(0, 212, 255, 0.1)',
                        tension: 0.4
                    }, {
                        label: 'Memory %',
                        data: [],
                        borderColor: '#0099ff',
                        backgroundColor: 'rgba(0, 153, 255, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: { color: '#fff' }
                        }
                    },
                    scales: {
                        x: {
                            ticks: { color: '#666' },
                            grid: { color: '#333' }
                        },
                        y: {
                            ticks: { color: '#666' },
                            grid: { color: '#333' },
                            max: 100
                        }
                    }
                }
            });
            
            // Cache performance chart
            const cacheCtx = document.getElementById('cache-chart').getContext('2d');
            cacheChart = new Chart(cacheCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Hits', 'Misses'],
                    datasets: [{
                        data: [0, 0],
                        backgroundColor: ['#00d4ff', '#333'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: { color: '#fff' }
                        }
                    }
                }
            });
        }
        
        // Update charts with new data
        function updateCharts(data) {
            // Update system chart
            if (systemChart && data.system) {
                const time = new Date().toLocaleTimeString();
                
                systemChart.data.labels.push(time);
                systemChart.data.datasets[0].data.push(data.system.cpu_usage);
                systemChart.data.datasets[1].data.push(data.system.memory_usage);
                
                // Keep only last 20 points
                if (systemChart.data.labels.length > 20) {
                    systemChart.data.labels.shift();
                    systemChart.data.datasets.forEach(dataset => dataset.data.shift());
                }
                
                systemChart.update('none');
            }
            
            // Update cache chart
            if (cacheChart && data.cache) {
                cacheChart.data.datasets[0].data = [
                    data.cache.total_hits,
                    data.cache.total_misses
                ];
                cacheChart.update('none');
            }
        }
        
        // Control functions
        async function runBenchmark() {
            const resultsDiv = document.getElementById('benchmark-results');
            resultsDiv.innerHTML = '<div class="loading">Running benchmark...</div>';
            
            try {
                const response = await fetch('/api/benchmark', { method: 'POST' });
                const results = await response.json();
                
                let html = '<h3>Benchmark Results</h3>';
                for (const [test, time] of Object.entries(results)) {
                    html += `
                        <div class="metric">
                            <span>${test.replace(/_/g, ' ')}</span>
                            <span class="metric-value">${(time * 1000).toFixed(1)}ms</span>
                        </div>
                    `;
                }
                
                resultsDiv.innerHTML = html;
            } catch (error) {
                resultsDiv.innerHTML = '<div style="color: #ff6666">Benchmark failed</div>';
            }
        }
        
        async function optimizeWorkload(type) {
            try {
                const response = await fetch('/api/optimize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ workload_type: type })
                });
                
                const result = await response.json();
                console.log(result.message);
            } catch (error) {
                console.error('Optimization failed:', error);
            }
        }
        
        // Initialize on load
        window.onload = () => {
            initCharts();
            connectWebSocket();
        };
    </script>
</body>
</html>
        '''
    
    async def start(self, port: int = 8889):
        """Start the monitoring server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', port)
        
        # Start broadcast task
        asyncio.create_task(self.broadcast_stats())
        
        await site.start()
        logger.info(f"Performance Monitor running on http://localhost:{port}")


async def start_performance_monitor(jarvis_core, port: int = 8889):
    """Start the performance monitoring dashboard"""
    monitor = PerformanceMonitor(jarvis_core)
    await monitor.start(port)
    
    return monitor