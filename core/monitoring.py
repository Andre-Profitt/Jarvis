"""
Minimal Monitoring Module
"""
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SystemMonitor:
    """Minimal monitoring implementation"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.metrics = {}
        
    def log_metric(self, name, value):
        """Log a metric"""
        self.metrics[name] = value
        logger.info(f"Metric {name}: {value}")
        
    def get_metrics(self):
        """Get all metrics"""
        return self.metrics

# Create global monitor instance
monitor = SystemMonitor()
