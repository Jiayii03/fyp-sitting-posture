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
            # Check if Kafka and Zookeeper containers are running
            kafka_running = subprocess.run(["sudo", "docker", "ps", "-q", "-f", "name=kafka"], capture_output=True, text=True).stdout.strip()
            zookeeper_running = subprocess.run(["sudo", "docker", "ps", "-q", "-f", "name=zookeeper"], capture_output=True, text=True).stdout.strip()

            if not kafka_running or not zookeeper_running:
                print("Kafka or Zookeeper container not running. Starting Kafka and Zookeeper containers...")

                # Determine the correct Docker Compose command (V1 or V2)
                compose_command = ["sudo", "docker-compose"]  # Default to V1
                if subprocess.run(["docker", "compose", "version"], capture_output=True, text=True).returncode == 0:
                    compose_command = ["sudo", "docker", "compose"]  # Use V2 format

                # Run Docker Compose to start the containers
                subprocess.run(compose_command + ["-f", "pubSub/docker-compose.yml", "up", "-d"], check=True)

                # Wait for Kafka and Zookeeper to initialize
                print("Waiting for Kafka and Zookeeper to fully start...")
                time.sleep(15) 
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
        
    def send_test_event(self, message):
        self.producer.send('test_events', {
            'message': message
        })