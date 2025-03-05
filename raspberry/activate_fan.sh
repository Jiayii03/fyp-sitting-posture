#!/bin/bash
# This script ensures the fan stays on continuously by setting the cooling state to 2 every few seconds

while true; do
  echo 3 | sudo tee /sys/class/thermal/cooling_device0/cur_state
  sleep 3  # Adjust the sleep duration as needed
done
