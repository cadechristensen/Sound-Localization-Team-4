#!/bin/bash
# Script to copy deployment files to Raspberry Pi
# Usage: ./COPY_TO_PI.sh <pi_ip_address>

PI_IP="${1:-10.125.202.212}"  # Default IP or use first argument

echo "======================================="
echo "Copying to Raspberry Pi at: $PI_IP"
echo "======================================="

# Create directories on Pi
echo "Creating directories on Pi..."
ssh bthz@$PI_IP "mkdir -p ~/baby_cry_detection/raspberry_pi ~/baby_cry_detection/models"

# Copy deployment files
echo "Copying deployment files..."
scp config_pi.py bthz@$PI_IP:~/baby_cry_detection/raspberry_pi/
scp audio_filtering.py bthz@$PI_IP:~/baby_cry_detection/raspberry_pi/
scp test_pi_filtering.py bthz@$PI_IP:~/baby_cry_detection/raspberry_pi/
scp pi_setup.py bthz@$PI_IP:~/baby_cry_detection/raspberry_pi/
scp README_FILTERING.md bthz@$PI_IP:~/baby_cry_detection/raspberry_pi/
scp PI_DEPLOYMENT_STEPS.md bthz@$PI_IP:~/baby_cry_detection/raspberry_pi/

# Copy quantized model (latest)
echo "Copying latest quantized model..."
scp ../../results/train_2025-10-05_01-09-46/model_quantized.pth bthz@$PI_IP:~/baby_cry_detection/models/

echo ""
echo "Files copied successfully!"
echo ""
echo "Next steps on your Pi:"
echo "1. SSH into Pi: ssh bthz@$PI_IP"
echo "2. cd ~/baby_cry_detection/raspberry_pi"
echo "3. python test_pi_filtering.py"
echo ""
