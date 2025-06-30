"""
Self-Healing Dashboard Server
Provides real-time monitoring and control interface for the self-healing system
"""

import asyncio
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import logging

from .self_healing_integration import self_healing_jarvis

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    """Serve the main dashboard page"""
    return render_template("self_healing_dashboard.html")


@app.route("/api/status")
async def get_status():
    """Get current self-healing system status"""
    try:
        status = await self_healing_jarvis.get_healing_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/metrics/current")
async def get_current_metrics():
    """Get current system metrics"""
    try:
        if not self_healing_jarvis._metrics_buffer:
            return jsonify({"metrics": None})

        latest_metrics = self_healing_jarvis._metrics_buffer[-1]
        return jsonify(
            {
                "timestamp": latest_metrics.timestamp.isoformat(),
                "cpu_usage": latest_metrics.cpu_usage,
                "memory_usage": latest_metrics.memory_usage,
                "disk_io": latest_metrics.disk_io,
                "network_latency": latest_metrics.network_latency,
                "error_rate": latest_metrics.error_rate,
                "request_rate": latest_metrics.request_rate,
                "response_time": latest_metrics.response_time,
                "active_connections": latest_metrics.active_connections,
                "custom_metrics": latest_metrics.custom_metrics,
            }
        )
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/metrics/history")
async def get_metrics_history():
    """Get metrics history"""
    try:
        # Get last hour of metrics
        cutoff = datetime.now() - timedelta(hours=1)
        history = []

        for metric in self_healing_jarvis._metrics_buffer:
            if metric.timestamp > cutoff:
                history.append(
                    {
                        "timestamp": metric.timestamp.isoformat(),
                        "cpu": metric.cpu_usage,
                        "memory": metric.memory_usage,
                        "error_rate": metric.error_rate,
                        "neural_efficiency": metric.custom_metrics.get(
                            "neural_efficiency", 0
                        ),
                    }
                )

        return jsonify({"history": history})
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/anomalies")
async def get_anomalies():
    """Get recent anomalies"""
    try:
        anomalies = []
        for anomaly in self_healing_jarvis.healing_system.anomaly_buffer[-20:]:
            anomalies.append(
                {
                    "id": anomaly.id,
                    "type": anomaly.type.value,
                    "severity": anomaly.severity,
                    "confidence": anomaly.confidence,
                    "detected_at": anomaly.detected_at.isoformat(),
                    "affected_components": anomaly.affected_components,
                    "metrics": anomaly.metrics,
                }
            )
        return jsonify({"anomalies": anomalies})
    except Exception as e:
        logger.error(f"Error getting anomalies: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/fixes")
async def get_fixes():
    """Get recent fixes"""
    try:
        fixes = []
        for fix in self_healing_jarvis.healing_system.fix_history[-20:]:
            fixes.append(
                {
                    "id": fix.id,
                    "anomaly_id": fix.anomaly_id,
                    "strategy": fix.strategy,
                    "confidence": fix.confidence,
                    "applied_at": (
                        fix.applied_at.isoformat()
                        if hasattr(fix, "applied_at")
                        else None
                    ),
                    "success": getattr(fix, "success", None),
                    "recovery_time": str(fix.estimated_recovery_time),
                }
            )
        return jsonify({"fixes": fixes})
    except Exception as e:
        logger.error(f"Error getting fixes: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/healing/enable", methods=["POST"])
async def enable_healing():
    """Enable self-healing"""
    try:
        self_healing_jarvis.enable_healing()
        return jsonify({"status": "enabled"})
    except Exception as e:
        logger.error(f"Error enabling healing: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/healing/disable", methods=["POST"])
async def disable_healing():
    """Disable self-healing"""
    try:
        self_healing_jarvis.disable_healing()
        return jsonify({"status": "disabled"})
    except Exception as e:
        logger.error(f"Error disabling healing: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/healing/manual", methods=["POST"])
async def apply_manual_fix():
    """Apply a manual fix"""
    try:
        data = request.json
        strategy = data.get("strategy", "restart")
        component = data.get("component", "unknown")

        result = await self_healing_jarvis.apply_manual_fix(strategy, component)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error applying manual fix: {e}")
        return jsonify({"error": str(e)}), 500


def create_dashboard_app():
    """Create and configure the dashboard Flask app"""
    return app


async def run_dashboard_server(host="localhost", port=5000):
    """Run the dashboard server"""
    logger.info(f"Starting Self-Healing Dashboard on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
