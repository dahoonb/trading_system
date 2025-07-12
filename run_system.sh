#!/bin/bash

# A script to launch the complete trading system, including the main
# application and the independent risk monitor.

echo "--- Starting Equity Trading System ---"

# Ensure no previous kill flag exists
echo "1. Cleaning up old kill flags..."
rm -f kill.flag

# Launch the main trading application in the background
# It will connect with the default Client ID (e.g., 1)
echo "2. Launching main trading application (main.py)..."
python main.py > main.log 2>&1 &
MAIN_PID=$!
echo "   - Main application started with PID: $MAIN_PID"

# Wait a moment to ensure the main app has connected and to avoid
# potential race conditions when both processes connect to TWS/Gateway.
sleep 5

# Launch the independent risk monitor in the background
# It will connect with a different Client ID (e.g., 2)
echo "3. Launching independent risk monitor (risk_monitor.py)..."
python risk_monitor.py > risk_monitor.log 2>&1 &
MONITOR_PID=$!
echo "   - Risk Monitor started with PID: $MONITOR_PID"

echo "--------------------------------------"
echo "System is now running."
echo " - Main app log: main.log"
echo " - Monitor log: risk_monitor.log"
echo "To stop the system, press Ctrl+C in this terminal or run:"
echo "kill $MAIN_PID $MONITOR_PID"
echo "--------------------------------------"

# The 'wait' command will cause the script to block here until the
# background processes are terminated. This makes it easy to stop
# everything with a single Ctrl+C in the terminal.
wait $MAIN_PID $MONITOR_PID

echo "--- Equity Trading System has been shut down. ---"