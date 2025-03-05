from kafka import KafkaProducer
import json
import subprocess
import time

class KafkaService:
    def __init__(self):
        self.producer = None
        self.initialize()
        
    def initialize(self):
        self.start_docker_containers()
        self.producer = KafkaProducer(
            bootstrap_servers='localhost:9092',
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
    def start_docker_containers(self):
        """Start Kafka and Zookeeper containers if they are not already running."""
        try:
            # Check if Kafka container is running
            kafka_running = subprocess.run(["sudo", "docker", "ps", "-q", "-f", "name=kafka"], capture_output=True, text=True).stdout.strip()
            zookeeper_running = subprocess.run(["sudo", "docker", "ps", "-q", "-f", "name=zookeeper"], capture_output=True, text=True).stdout.strip()

            if not kafka_running or not zookeeper_running:
                print("Kafka or Zookeeper container not running. Starting Kafka and Zookeeper containers...")
                # Run Docker Compose to start the containers
                subprocess.run(["sudo", "docker", "compose", "-f", "docker-compose.yml", "up", "-d"], check=True)

                # Wait for Kafka and Zookeeper to become available
                time.sleep(10) # Wait a few seconds to allow Kafka and Zookeeper to initialize
                print("Kafka and Zookeeper containers started successfully.")
            else:
                print("Kafka and Zookeeper containers are already running.")
        except Exception as e:
            print(f"Error while starting Docker containers: {e}")
        
    def send_posture_event(self, posture, message):
        self.producer.send('posture_events', {
            'posture': posture,
            'message': message
        })
        
    def send_alert_event(self, message):
        self.producer.send('alert_events', {
            'message': message
        })