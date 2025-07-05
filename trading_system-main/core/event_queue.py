# core/event_queue.py

import queue

# This is the global, thread-safe event queue.
# All parts of the trading system will import this single instance
# to communicate with each other, ensuring a decoupled architecture.
main_queue = queue.Queue()