# portfolio/monthly_counter.py
import json
import os
import datetime
import logging
import tempfile
import threading
from typing import Any, Dict, Optional

# --- Import the global config loader ---
from core.config_loader import config

logger = logging.getLogger("TradingSystem")

class CounterStateException(Exception):
    """Custom exception for critical errors in counter state management."""
    pass

class CounterPersistenceError(CounterStateException):
    """Specific exception for errors during state persistence."""
    pass

class MonthlyCounter:
    """
    Manages and persists a monthly event counter (e.g., for fills).

    Handles loading state, incrementing, checking budget, resetting monthly,
    and persisting state atomically. Raises exceptions on critical errors.
    Designed to be thread-safe.
    """

    def __init__(self, persistence_filepath: str):
        """
        Initializes the counter. The budget is now loaded from the global config.
        
        Args:
            persistence_filepath (str): Path to the file for persisting state.
        """
        # --- MODIFICATION: Load budget from config ---
        self._budget = config['portfolio'].get('max_fills_per_month', 60) # Default to 60 if not specified
        
        if not isinstance(self._budget, int) or self._budget <= 0:
            raise ValueError("Budget must be a positive integer.")

        self._filepath = persistence_filepath
        self._lock = threading.Lock()  # Protects access to state attributes and file ops

        self._current_year_month: Optional[str] = None # Format "YYYY-MM"
        self._current_count: int = 0

        logger.info(f"Initializing MonthlyCounter: Budget={self._budget}, File='{self._filepath}'")
        # Load initial state, potentially raising CounterStateException
        self._load_state()

    def _get_current_timestamp_utc(self) -> datetime.datetime:
        """Gets the current time in UTC."""
        return datetime.datetime.now(datetime.timezone.utc)

    def _load_state(self):
        """
        Loads state, ensuring lock is held. Raises CounterStateException on critical errors.
        """
        with self._lock:
            try:
                if not os.path.exists(self._filepath):
                    logger.warning(f"Persistence file '{self._filepath}' not found. Initializing state.")
                    current_ts = self._get_current_timestamp_utc()
                    self._current_year_month = current_ts.strftime('%Y-%m')
                    self._current_count = 0
                    self._save_state_logic()
                    return

                with open(self._filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                loaded_month = data.get('year_month')
                loaded_count = data.get('count')

                if not isinstance(loaded_month, str) or len(loaded_month) != 7 or loaded_month[4] != '-':
                    raise CounterStateException(f"Invalid 'year_month' format in state file: {loaded_month}")
                if not isinstance(loaded_count, int) or loaded_count < 0:
                    raise CounterStateException(f"Invalid 'count' value in state file: {loaded_count}")

                self._current_year_month = loaded_month
                self._current_count = loaded_count
                logger.info(f"Loaded counter state: Month={self._current_year_month}, Count={self._current_count}")

                self._reset_if_new_month_logic(self._get_current_timestamp_utc())

            except json.JSONDecodeError as e:
                raise CounterStateException(f"State file '{self._filepath}' corrupted (JSON decode error).") from e
            except CounterPersistenceError as e:
                 raise CounterStateException("Failed to save initial state after file not found.") from e
            except (OSError, IOError) as e:
                raise CounterStateException(f"Could not read state file '{self._filepath}'.") from e
            except Exception as e:
                logger.exception(f"Unexpected error loading counter state from '{self._filepath}': {e}")
                raise CounterStateException(f"Unexpected error loading state from '{self._filepath}'.") from e

    def _save_state_logic(self):
        """
        Internal logic to save state atomically. Assumes lock is held.
        Raises CounterPersistenceError on critical write/rename failure.
        """
        state_data = {
            'year_month': self._current_year_month,
            'count': self._current_count
        }
        temp_file_path = None
        try:
            file_dir = os.path.dirname(self._filepath) or '.'
            os.makedirs(file_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile('w', delete=False, dir=file_dir, encoding='utf-8', suffix='.tmp') as temp_f:
                temp_file_path = temp_f.name
                json.dump(state_data, temp_f, indent=4)
            os.replace(temp_file_path, self._filepath)
        except (OSError, IOError) as e:
            if temp_file_path and os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except OSError: pass
            raise CounterPersistenceError(f"Failed to save state to '{self._filepath}': {e}") from e
        except Exception as e:
             if temp_file_path and os.path.exists(temp_file_path):
                 try: os.remove(temp_file_path)
                 except OSError: pass
             logger.exception(f"Unexpected error saving state to '{self._filepath}': {e}")
             raise CounterPersistenceError(f"Unexpected error saving state: {e}") from e

    def _reset_if_new_month_logic(self, current_timestamp_utc: datetime.datetime):
        """
        Internal logic to check for month rollover and reset. Assumes lock is held.
        Raises CounterPersistenceError if saving the reset state fails.
        """
        current_month_str = current_timestamp_utc.strftime('%Y-%m')
        if self._current_year_month != current_month_str:
            logger.info(f"New month detected: {self._current_year_month} -> {current_month_str}. Resetting counter.")
            self._current_year_month = current_month_str
            self._current_count = 0
            self._save_state_logic()

    def increment(self, current_timestamp_utc: Optional[datetime.datetime] = None):
        """
        Increments the counter, checking for month rollover first. Persists state.
        """
        with self._lock:
            ts = current_timestamp_utc or self._get_current_timestamp_utc()
            self._reset_if_new_month_logic(ts)
            self._current_count += 1
            logger.info(f"Monthly fill counter incremented: {self._current_count} / {self._budget}")
            self._save_state_logic()

    def is_budget_exceeded(self, current_timestamp_utc: Optional[datetime.datetime] = None) -> bool:
        """Checks if the budget is met/exceeded, ensuring state reflects the correct month."""
        with self._lock:
            ts = current_timestamp_utc or self._get_current_timestamp_utc()
            try:
                self._reset_if_new_month_logic(ts)
            except CounterPersistenceError as e:
                logger.critical(f"Failed to ensure consistent state in is_budget_exceeded: {e}. Assuming budget exceeded.")
                return True

            exceeded = self._current_count >= self._budget
            if exceeded:
                logger.warning(f"Monthly fill budget of {self._budget} has been exceeded. No new orders will be generated.")
            return exceeded