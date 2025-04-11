# backend/utils/resource_monitor.py
import psutil
import threading
import time
import os
import csv
import numpy as np
from datetime import datetime
import platform
from config.settings import ON_RASPBERRY

class ResourceMonitor:
    def __init__(self, sampling_interval=1.0, log_dir="logs"):
        self.sampling_interval = sampling_interval  # seconds
        self.monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process(os.getpid())
        self.log_dir = log_dir
        self.stats = {
            "timestamp": [],
            "cpu_percent": [],
            "memory_percent": [],
            "memory_mb": [],
            "disk_io_read_mb": [],
            "disk_io_write_mb": [],
            "network_sent_mb": [],
            "network_recv_mb": [],
            "temperature": []
        }
        self.prev_disk_read = 0
        self.prev_disk_write = 0
        self.prev_net_sent = 0
        self.prev_net_recv = 0
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
    def start(self):
        """Start monitoring system resources"""
        if self.monitoring:
            print("Resource monitoring is already running")
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("Resource monitoring started")
        
    def stop(self):
        """Stop monitoring and save results"""
        if not self.monitoring:
            print("Resource monitoring is not running")
            return
            
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            
        # Save collected statistics
        self._save_stats()
        print("Resource monitoring stopped and data saved")
        
    def _monitor_resources(self):
        """Monitor system resources periodically"""
        while self.monitoring:
            try:
                # Record timestamp
                self.stats["timestamp"].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
                
                # CPU usage
                self.stats["cpu_percent"].append(self.process.cpu_percent())
                
                # Memory usage
                memory_info = self.process.memory_info()
                self.stats["memory_percent"].append(self.process.memory_percent())
                self.stats["memory_mb"].append(memory_info.rss / (1024 * 1024))  # Convert to MB
                
                # Disk I/O
                io_counters = self.process.io_counters() if hasattr(self.process, 'io_counters') else None
                if io_counters:
                    # Calculate delta from previous measurement
                    disk_read = io_counters.read_bytes / (1024 * 1024)  # MB
                    disk_write = io_counters.write_bytes / (1024 * 1024)  # MB
                    
                    if self.prev_disk_read > 0:
                        self.stats["disk_io_read_mb"].append(disk_read - self.prev_disk_read)
                    else:
                        self.stats["disk_io_read_mb"].append(0)
                        
                    if self.prev_disk_write > 0:
                        self.stats["disk_io_write_mb"].append(disk_write - self.prev_disk_write)
                    else:
                        self.stats["disk_io_write_mb"].append(0)
                        
                    self.prev_disk_read = disk_read
                    self.prev_disk_write = disk_write
                else:
                    self.stats["disk_io_read_mb"].append(0)
                    self.stats["disk_io_write_mb"].append(0)
                
                # Network usage
                net_io = psutil.net_io_counters()
                net_sent = net_io.bytes_sent / (1024 * 1024)  # MB
                net_recv = net_io.bytes_recv / (1024 * 1024)  # MB
                
                if self.prev_net_sent > 0:
                    self.stats["network_sent_mb"].append(net_sent - self.prev_net_sent)
                else:
                    self.stats["network_sent_mb"].append(0)
                    
                if self.prev_net_recv > 0:
                    self.stats["network_recv_mb"].append(net_recv - self.prev_net_recv)
                else:
                    self.stats["network_recv_mb"].append(0)
                    
                self.prev_net_sent = net_sent
                self.prev_net_recv = net_recv
                
                # Temperature monitoring (Raspberry Pi only)
                if ON_RASPBERRY:
                    try:
                        temp = float(os.popen("vcgencmd measure_temp").readline().replace("temp=","").replace("'C\n",""))
                        self.stats["temperature"].append(temp)
                    except:
                        self.stats["temperature"].append(0)
                else:
                    self.stats["temperature"].append(0)
                
            except Exception as e:
                print(f"Error monitoring resources: {e}")
                
            time.sleep(self.sampling_interval)
            
    def _save_stats(self):
        """Save collected statistics to CSV file"""
        if not any(self.stats["timestamp"]):
            print("No statistics collected")
            return
            
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        platform_name = "raspberry" if ON_RASPBERRY else "laptop"
        filename = f"{self.log_dir}/resource_stats_{platform_name}_{timestamp}.csv"
        
        # Write stats to CSV
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(self.stats.keys())
            
            # Write data rows
            for i in range(len(self.stats["timestamp"])):
                row = [self.stats[key][i] for key in self.stats.keys()]
                writer.writerow(row)
                
        print(f"Statistics saved to {filename}")
        return filename
        
    def get_summary_stats(self):
        """Calculate summary statistics"""
        if not any(self.stats["timestamp"]):
            return None
            
        summary = {}
        
        # Calculate statistics for each metric
        for key in self.stats.keys():
            if key != "timestamp":
                values = [x for x in self.stats[key] if isinstance(x, (int, float))]
                if values:
                    summary[f"{key}_mean"] = np.mean(values)
                    summary[f"{key}_max"] = np.max(values)
                    summary[f"{key}_min"] = np.min(values)
                    summary[f"{key}_std"] = np.std(values)
                    
        # Add system info
        summary["platform"] = platform.platform()
        summary["python_version"] = platform.python_version()
        summary["cpu_count"] = psutil.cpu_count()
        summary["total_memory_gb"] = psutil.virtual_memory().total / (1024**3)
        summary["monitoring_duration_sec"] = len(self.stats["timestamp"]) * self.sampling_interval
        
        return summary
    
    def generate_report(self):
        """
        Generate a structured report as a JSON-friendly dictionary
        Return the stats since the last resource monitoring reset
        
        """
        summary = self.get_summary_stats()
        if not summary:
            return {"error": "No statistics available to generate report."}
        
        report = {
            "system": {
                "platform": summary['platform'],
                "python_version": summary['python_version'],
                "cpu_cores": summary['cpu_count'],
                "total_memory_gb": round(summary['total_memory_gb'], 2),
                "monitoring_duration_sec": round(summary['monitoring_duration_sec'], 2)
            },
            "resource_usage": {
                "cpu": {
                    "average": round(summary['cpu_percent_mean'], 2),
                    "min": round(summary['cpu_percent_min'], 2),
                    "max": round(summary['cpu_percent_max'], 2),
                    "unit": "%"
                },
                "memory": {
                    "average": round(summary['memory_mb_mean'], 2),
                    "min": round(summary['memory_mb_min'], 2),
                    "max": round(summary['memory_mb_max'], 2),
                    "unit": "MB"
                },
                "network": {
                    "sent": round(summary['network_sent_mb_mean'], 4),
                    "received": round(summary['network_recv_mb_mean'], 4),
                    "unit": "MB/s"
                }
            }
        }
        
        # Add temperature data for Raspberry Pi
        if ON_RASPBERRY:
            report["resource_usage"]["temperature"] = {
                "average": round(summary['temperature_mean'], 2),
                "min": round(summary['temperature_min'], 2),
                "max": round(summary['temperature_max'], 2),
                "unit": "°C"
            }
        else:
            report["resource_usage"]["temperature"] = {
                "average": None,
                "min": None,
                "max": None,
                "unit": "°C"
            }
        
        return report