## Commands

1. IP address:
`hostname -I`

2. CPU temperature:
`vcgencmd measure_temp`

3. Turn on the fan at full speed:
`nohup ~/force_fan_on.sh &`
`ps aux | grep force_fan_on.sh`

4. Check fan speed:
`cat /sys/class/hwmon/hwmon*/pcd fawm1`

5. Check latency between laptop and raspberry pi:
`ping pi -n 10`
