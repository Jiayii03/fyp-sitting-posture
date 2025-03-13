## Commands

1. IP address:
`hostname -I`

2. CPU temperature:
`vcgencmd measure_temp`

3. Turn on the fan at full speed:
`nohup ~/activate_fan.sh &`
`ps aux | grep activate_fan.sh`

5. Check latency between laptop and raspberry pi:
`ping pi -n 10`

## Setup

1. Create python virtual environment, run

2. If `import libcamera` fails, run
```bash
sudo apt-get update
sudo apt-get install libcamera-apps python3-libcamera
```

3. To bind system-wide-packages, run
```bash
sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' myenv/pyvenv.cfg
```
## Install Docker

Docker is needed to run `backend`, for `kafkaService`. Run
```bash
sudo apt-get update
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```
Run `docker --version` to verify installation.

## Run Docker Kafka

Make sure Docker is installed. Then run from the root project directory
```bash
cd backend/pubSub
sudo docker compose -f docker-compose.yml up -d
```

Run `sudo docker ps` to see running status.

>If Kafka Pub-Sub is not working, make sure to stop all Kafka services running on laptop.


