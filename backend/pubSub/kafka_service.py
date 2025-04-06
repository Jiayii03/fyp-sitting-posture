from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import json
import subprocess
import time
import os
from config.settings import ON_RASPBERRY

class KafkaService:
    def __init__(self):
        self.producer = None
        self.initialize()
        
    def initialize(self):
        """Initialize the Kafka service with retries."""
        self.start_docker_containers()
        self.connect_with_retries()
        
    def connect_with_retries(self, max_retries=5, retry_delay=5):
        """Attempt to connect to Kafka broker with retries."""
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Attempting to connect to Kafka broker (attempt {attempt}/{max_retries})...")
                self.producer = KafkaProducer(
                    bootstrap_servers='localhost:9092',
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    # Set shorter timeouts to fail faster during retry
                    api_version_auto_timeout_ms=5000,
                    request_timeout_ms=10000
                )
                print("✅ Kafka producer initialized successfully")
                # Test the connection
                self.producer.bootstrap_connected()
                return  # Connection successful, exit the retry loop
            except NoBrokersAvailable:
                if attempt < max_retries:
                    print(f"⚠️ Kafka broker not available yet. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("❌ Failed to connect to Kafka broker after multiple attempts")
                    self.producer = None
            except Exception as e:
                print(f"❌ Failed to initialize Kafka producer: {e}")
                self.producer = None
                break  # Exit on unexpected errors
        
    def start_docker_containers(self):
        """Start Kafka and Zookeeper containers if they are not already running."""
        try:
            # Determine if we need to use sudo (typically on Raspberry Pi)
            sudo_prefix = ["sudo"] if ON_RASPBERRY else []
            
            # Check if Kafka and Zookeeper containers are running
            kafka_cmd = sudo_prefix + ["docker", "ps", "-q", "-f", "name=kafka"]
            zookeeper_cmd = sudo_prefix + ["docker", "ps", "-q", "-f", "name=zookeeper"]
            
            kafka_running = subprocess.run(kafka_cmd, capture_output=True, text=True).stdout.strip()
            zookeeper_running = subprocess.run(zookeeper_cmd, capture_output=True, text=True).stdout.strip()

            if not kafka_running or not zookeeper_running:
                print("Kafka or Zookeeper container not running. Starting containers...")

                # Determine if we should use docker-compose V1 or V2
                compose_v2_check_cmd = ["docker", "compose", "version"]
                use_compose_v2 = subprocess.run(compose_v2_check_cmd, capture_output=True, text=True).returncode == 0
                
                if use_compose_v2:
                    compose_cmd = sudo_prefix + ["docker", "compose"]
                else:
                    compose_cmd = sudo_prefix + ["docker-compose"]
                
                # Construct the full command to start containers
                full_cmd = compose_cmd + ["-f", "pubSub/docker-compose.yml", "up", "-d"]
                
                # Run Docker Compose to start the containers
                subprocess.run(full_cmd, check=True)

                # Give containers time to start, actual connection attempts handled by connect_with_retries
                print("Docker containers for Kafka starting up...")
                time.sleep(5)  # Short initial wait before retry attempts
            else:
                print("Kafka and Zookeeper containers are already running.")
        except Exception as e:
            print(f"Error while starting Docker containers: {e}")
            
    def is_connected(self):
        """Check if the producer is connected to the Kafka broker."""
        if not self.producer:
            return False
        try:
            return self.producer.bootstrap_connected()
        except:
            return False
        
    def send_posture_event(self, posture, message):
        try:
            self.producer.send('posture_events', {
                'posture': posture,
                'message': message
            })
            return True
        except Exception as e:
            print(f"Failed to send posture event: {e}")
            return False
        
    def send_alert_event(self, message):
        if not self.ensure_connection():
            print("⚠️ Cannot send alert event: Not connected to Kafka")
            return False
            
        try:
            self.producer.send('alert_events', {
                'message': message
            })
            return True
        except Exception as e:
            print(f"Failed to send alert event: {e}")
            return False
        
    def send_test_event(self, message):
        if not self.ensure_connection():
            print("⚠️ Cannot send test event: Not connected to Kafka")
            return False
            
        try:
            self.producer.send('test_events', {
                'message': message
            })
            print("✅ Test event sent successfully")
            return True
        except Exception as e:
            print(f"Failed to send test event: {e}")
            return False