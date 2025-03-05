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

If `import libcamera` fails, run
```bash
sudo apt-get update
sudo apt-get install libcamera-apps python3-libcamera
```

To bind system-wide-packages, run
```bash
sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' myenv/pyvenv.cfg
```
