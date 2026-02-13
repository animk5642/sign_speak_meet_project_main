#!/bin/bash
# Auto-reconnecting SSH tunnel to HPC server
# Keep this running in a terminal while the app is hosted

echo "Starting SSH tunnel to HPC (localhost:9105 -> HPC:9105)..."
echo "Press Ctrl+C to stop"

while true; do
    ssh -o ServerAliveInterval=60 \
        -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        -L 9105:localhost:9105 \
        wyd22ec042@192.168.200.75
    
    # If SSH exits, wait 5 seconds before reconnecting
    echo "Tunnel disconnected. Reconnecting in 5 seconds..."
    sleep 5
done
