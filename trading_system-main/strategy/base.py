# strategy/base.py

import os
import pandas as pd
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """
    An abstract base class for trading strategies, providing a common
    framework for data handling and state management.
    
    This class handles the saving and loading of historical data and
    signal states to ensure persistence between trading sessions.
    """
    def __init__(self, event_queue, symbols, state_file):
        self.event_queue = event_queue
        self.symbols = symbols
        self.state_file = state_file
        
        # Load initial state if it exists, otherwise initialize
        self.data = self.load_state()
        if self.data is None:
            self.data = {s: pd.DataFrame() for s in self.symbols}
        
        # This dictionary tracks the last signal sent for each symbol
        # to prevent sending duplicate signals on consecutive market events.
        self.signals = {s: None for s in self.symbols}

    @abstractmethod
    def calculate_signals(self, event):
        """
        The core logic of the strategy. It should process a market event
        and generate SignalEvents if conditions are met.
        
        This method must be implemented by all subclasses.
        """
        raise NotImplementedError("Should implement calculate_signals()")

    def save_state(self):
        """Saves the internal data DataFrame to a pickle file."""
        if not self.state_file:
            return
        print(f"Saving state for {self.__class__.__name__} to {self.state_file}...")
        try:
            # Ensure the directory for the state file exists
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            pd.to_pickle(self.data, self.state_file)
            print("State saved successfully.")
        except Exception as e:
            print(f"Error saving state for {self.__class__.__name__}: {e}")

    def load_state(self):
        """
        Loads the internal data DataFrame from a pickle file if it exists.
        Returns the data dictionary if successful, otherwise None.
        """
        if self.state_file and os.path.exists(self.state_file):
            print(f"Loading previous state for {self.__class__.__name__} from {self.state_file}...")
            try:
                return pd.read_pickle(self.state_file)
            except Exception as e:
                print(f"Error loading state from {self.state_file}: {e}")
                # If file is corrupt, remove it to start fresh on next run
                os.remove(self.state_file)
                print(f"Removed corrupt state file: {self.state_file}")
        return None