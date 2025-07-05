import logging
import logging.handlers
import os
import sys
import datetime
from typing import Optional, List, Any

# Attempt to import python-json-logger
try:
    from pythonjsonlogger import jsonlogger
    PYTHON_JSON_LOGGER_AVAILABLE = True
except ImportError:
    PYTHON_JSON_LOGGER_AVAILABLE = False
    # print("Warning: python-json-logger library not found. Falling back to standard text logger.", file=sys.stderr)


# --- Custom Formatter (Original - kept for reference but not used by default in setup_logger anymore) ---
class CustomFormatter(logging.Formatter):
    """
    Custom logging formatter that adds color to log levels for console output.
    """
    # ANSI escape codes for colors
    GREY = "\x1b[38;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"
    GREEN = "\x1b[32;20m"
    CYAN = "\x1b[36;20m"

    # Define format for each level, without color for file logging
    # Updated format string to include module and line number consistently
    # %(name)s is the logger name (e.g., MainTradingEngine, RiskMonitorProcess)
    # %(module)s is the module name (e.g., main, risk_monitor)
    # %(funcName)s is the function name
    # %(lineno)d is the line number
    log_format_base = "%(asctime)s - %(name)s - [%(levelname)-8s] - %(module)s:%(funcName)s:%(lineno)d - %(message)s"

    FORMATS_NO_COLOR = {
        logging.DEBUG: log_format_base,
        logging.INFO: log_format_base,
        logging.WARNING: log_format_base,
        logging.ERROR: log_format_base,
        logging.CRITICAL: log_format_base,
    }

    # Define format for each level, with color for console logging
    FORMATS_WITH_COLOR = {
        logging.DEBUG: GREY + log_format_base + RESET,
        logging.INFO: GREEN + log_format_base + RESET, # Changed INFO to GREEN
        logging.WARNING: YELLOW + log_format_base + RESET,
        logging.ERROR: RED + log_format_base + RESET,
        logging.CRITICAL: BOLD_RED + log_format_base + RESET,
    }

    def __init__(self, use_color: bool = False, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        self.use_color = use_color
        # If a specific format string is provided, use it for all levels (no color)
        if fmt:
            super().__init__(fmt, datefmt)
        else: # Use level-specific formats
            super().__init__(datefmt=datefmt) # Base init, format applied in format()

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(self, '_style') or self._style is None: # Ensure _style is initialized if fmt was None
             # Default to basic style if super().__init__ didn't set it (e.g. if fmt was None)
             # This part is tricky as Formatter.__init__ handles _style.
             # The main point is that log_fmt will be used below.
             pass


        if self.use_color and not self._fmt: # Only apply color if no overriding fmt string and use_color is True
            log_fmt = self.FORMATS_WITH_COLOR.get(record.levelno, self.log_format_base)
        elif not self._fmt : # No color, no overriding fmt
            log_fmt = self.FORMATS_NO_COLOR.get(record.levelno, self.log_format_base)
        else: # An overriding fmt string was provided, use that (implies no dynamic color)
            log_fmt = self._fmt

        # Temporarily set the formatter for this record
        # This is a common pattern for dynamic formatting based on record level
        # Need to be careful with Formatter internals
        original_fmt = self._style._fmt
        self._style._fmt = log_fmt
        
        # Call the original Formatter.format method with the temporarily set format
        formatted_message = super().format(record)
        
        # Restore the original format string in the style object
        self._style._fmt = original_fmt
        
        return formatted_message

# --- Global Logger Setup (Original - kept for reference) ---
_global_logger_instance: Optional[logging.Logger] = None
_global_log_handlers: List[logging.Handler] = []

def setup_global_logger_for_later_use(log_directory: str = "logs",
                                      log_level: int = logging.INFO,
                                      logger_name: str = "GlobalTradingSystemLogger",
                                      use_json_formatting: bool = True): # Added flag
    """
    Sets up a global logger instance but doesn't return it directly.
    Useful if you need a logger configured early and then retrieved by name.
    MODIFIED to potentially use JSON formatting.
    """
    global _global_logger_instance, _global_log_handlers, PYTHON_JSON_LOGGER_AVAILABLE

    if _global_logger_instance is not None:
        # If logger already exists, just ensure level is appropriate, don't add handlers again
        _global_logger_instance.setLevel(log_level)
        return _global_logger_instance # Return the existing instance

    logger_to_setup = logging.getLogger(logger_name)
    logger_to_setup.setLevel(log_level)
    logger_to_setup.propagate = False  # Prevent duplicate logs in parent loggers if any

    # Create log directory if it doesn't exist
    os.makedirs(log_directory, exist_ok=True)

    # Determine formatter
    if use_json_formatting and PYTHON_JSON_LOGGER_AVAILABLE:
        # Example JSON format, customize as needed
        # This format string includes standard LogRecord attributes.
        # python-json-logger will automatically pick up 'extra' dictionary.
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(levelname)s %(name)s %(module)s %(funcName)s %(lineno)d %(message)s',
            rename_fields={'levelname': 'level', 'asctime': 'timestamp'} # Optional: rename for common JSON log schema
        )
        formatter_console = formatter # Use JSON for console too, or a text one if preferred
    else:
        if use_json_formatting and not PYTHON_JSON_LOGGER_AVAILABLE:
            # Using print as logger might not be fully set up for this warning path.
            print(f"WARNING: JSON logging requested for '{logger_name}' but python-json-logger is not available. Falling back to text.", file=sys.stderr)
        
        # Fallback to original CustomFormatter for text logging
        formatter = CustomFormatter(use_color=False) # File handler without color
        formatter_console = CustomFormatter(use_color=True) # Console handler with color


    # File Handler (TimedRotatingFileHandler)
    # Use a unique filename for the global logger if it's meant to be distinct
    log_file_path = os.path.join(log_directory, f"{logger_name.replace('.', '_')}_{datetime.date.today().strftime('%Y%m%d')}.log")
    
    # Check if this specific file handler is already added
    file_handler_exists = any(
        isinstance(h, logging.handlers.TimedRotatingFileHandler) and h.baseFilename == os.path.abspath(log_file_path)
        for h in logger_to_setup.handlers
    )
    if not file_handler_exists:
        fh = logging.handlers.TimedRotatingFileHandler(
            log_file_path, when="midnight", interval=1, backupCount=7, encoding='utf-8'
        )
        fh.setFormatter(formatter)
        logger_to_setup.addHandler(fh)
        _global_log_handlers.append(fh) # Track handler

    # Console Handler (StreamHandler)
    console_handler_exists = any(isinstance(h, logging.StreamHandler) for h in logger_to_setup.handlers)
    if not console_handler_exists:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter_console) # Use color formatter for console
        logger_to_setup.addHandler(ch)
        _global_log_handlers.append(ch) # Track handler

    _global_logger_instance = logger_to_setup
    return _global_logger_instance


# --- setup_logger FUNCTION (Primary function to be used by modules) ---
def setup_logger(log_directory: str = "logs",
                 log_level: int = logging.INFO,
                 logger_name: str = "TradingSystem", # Default logger name
                 use_json_formatting: bool = True, # Default to JSON
                 also_log_to_console: bool = True
                 ) -> logging.Logger:
    """
    Sets up and returns a logger instance with specified configurations.
    MODIFIED to prioritize JSON logging using python-json-logger.
    """
    global PYTHON_JSON_LOGGER_AVAILABLE # Access the global availability flag

    logger_instance = logging.getLogger(logger_name)
    
    # Avoid adding handlers multiple times if logger already configured (e.g., by name)
    if logger_instance.hasHandlers():
        # logger_instance.warning(f"Logger '{logger_name}' already has handlers. Re-applying level but not handlers.", extra={"note": "logger_reconfig_level_only"})
        logger_instance.setLevel(log_level) # Ensure level is set
        return logger_instance

    logger_instance.setLevel(log_level)
    logger_instance.propagate = False  # Important to prevent duplicate messages if root logger is also configured

    # Create log directory if it doesn't exist
    try:
        os.makedirs(log_directory, exist_ok=True)
    except OSError as e:
        # Using print as this is a setup phase, logger might not be fully ready for critical errors
        print(f"Error creating log directory '{log_directory}': {e}. Logs might not be written to file.", file=sys.stderr)
        # Continue to setup console handler at least

    # Determine formatter
    file_formatter: logging.Formatter
    console_formatter: logging.Formatter

    if use_json_formatting and PYTHON_JSON_LOGGER_AVAILABLE:
        # This format string includes standard LogRecord attributes.
        # python-json-logger will automatically pick up fields from the 'extra' dictionary.
        # Customize the format string as needed, e.g., to match specific JSON schema for log aggregators.
        # Standard fields available: %(asctime)s, %(created)f, %(filename)s, %(funcName)s, %(levelname)s,
        # %(levelno)d, %(lineno)d, %(message)s, %(module)s, %(msecs)d, %(name)s, %(pathname)s,
        # %(process)d, %(processName)s, %(relativeCreated)d, %(thread)d, %(threadName)s
        json_format_str = '%(asctime)s %(levelname)s %(name)s %(module)s %(funcName)s %(lineno)d %(message)s'
        file_formatter = jsonlogger.JsonFormatter(
            json_format_str,
            rename_fields={'levelname': 'level', 'asctime': 'timestamp'} # Optional: rename for common JSON log schema
        )
        # For console, you might still prefer text, or also JSON. Let's use JSON for console too for consistency.
        console_formatter = json_format_str # Use the same for console for now or define a new one
        if also_log_to_console:
            console_formatter = jsonlogger.JsonFormatter( # Re-init for console, can be different
                 json_format_str,
                 rename_fields={'levelname': 'level', 'asctime': 'timestamp'}
            )

    else: # Fallback to text logging
        if use_json_formatting and not PYTHON_JSON_LOGGER_AVAILABLE:
            # This warning will now go through the basicConfig logger if called before setup_logger fully returns
            logger_instance.warning("JSON logging requested but python-json-logger is not available. Falling back to standard text logging.", extra={"fallback_reason": "python-json-logger_missing"})
        
        # Use the original CustomFormatter for text-based logging
        file_formatter = CustomFormatter(use_color=False) 
        if also_log_to_console:
            console_formatter = CustomFormatter(use_color=True)

    # File Handler (TimedRotatingFileHandler)
    # Construct a unique log file name including date, ensuring it's within the log_directory
    # Replace potential problematic characters in logger_name for filename
    safe_logger_name_part = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in logger_name)
    log_file_base_name = f"{safe_logger_name_part}_{datetime.date.today().strftime('%Y%m%d')}.log"
    log_file_path = os.path.join(log_directory, log_file_base_name)

    try:
        fh = logging.handlers.TimedRotatingFileHandler(
            log_file_path, when="midnight", interval=1, backupCount=14, encoding='utf-8' # Keep 14 days of logs
        )
        fh.setFormatter(file_formatter)
        logger_instance.addHandler(fh)
    except Exception as e_fh:
        print(f"Error setting up file handler for logger '{logger_name}' at '{log_file_path}': {e_fh}", file=sys.stderr)


    # Console Handler (StreamHandler)
    if also_log_to_console:
        try:
            ch = logging.StreamHandler(sys.stdout) # Log to standard output
            ch.setFormatter(console_formatter)
            logger_instance.addHandler(ch)
        except Exception as e_ch:
            print(f"Error setting up console handler for logger '{logger_name}': {e_ch}", file=sys.stderr)
            
    return logger_instance

# Example of how to clean up global handlers if needed (e.g., for testing or re-init)
def cleanup_global_log_handlers():
    global _global_logger_instance, _global_log_handlers
    if _global_logger_instance:
        for handler in list(_global_logger_instance.handlers): # Iterate copy
            try:
                handler.close()
                _global_logger_instance.removeHandler(handler)
            except Exception:
                pass # Ignore errors during cleanup
    _global_log_handlers.clear()
    _global_logger_instance = None


if __name__ == '__main__':
    # Example Usage (demonstrates both JSON and fallback text logging)
    print("--- Demonstrating Logger Setup ---")

    # Test 1: JSON logging (if library available)
    print("\n--- Test 1: JSON Logger (if python-json-logger available) ---")
    json_test_logger = setup_logger(logger_name="JSONTestLogger", log_directory="logs_test", use_json_formatting=True)
    json_test_logger.debug("This is a JSON debug message.", extra={"custom_field": "debug_value", "user_id": 123})
    json_test_logger.info("This is a JSON info message.", extra={"transaction_id": "txn_abc_123", "items_count": 5})
    json_test_logger.warning("This is a JSON warning message.", extra={"warning_code": "W001", "system_load": 0.75})
    json_test_logger.error("This is a JSON error message.", extra={"error_id": "err_xyz_789", "is_critical": False, "service_name": "payment_api"})
    try:
        1/0
    except ZeroDivisionError:
        json_test_logger.exception("This is a JSON exception message (auto with exc_info).", extra={"exception_type": "ZeroDivisionError", "recovery_attempted": True})

    # Test 2: Text logging (forcing fallback or if library not available)
    print("\n--- Test 2: Text Logger (forcing no JSON or if library unavailable) ---")
    text_test_logger = setup_logger(logger_name="TextTestLogger", log_directory="logs_test", use_json_formatting=False)
    text_test_logger.debug("This is a Text debug message (custom format).", extra={"custom_field": "debug_value_text"}) # Extra won't appear structured in text unless formatter handles it
    text_test_logger.info("This is a Text info message (custom format).", extra={"transaction_id": "txn_def_456"})
    text_test_logger.warning("This is a Text warning message (custom format).")
    text_test_logger.error("This is a Text error message (custom format).", extra={"error_id": "err_uvw_000"})
    try:
        x = {}
        y = x['non_existent_key']
    except KeyError:
         text_test_logger.exception("This is a Text exception message (auto with exc_info via custom formatter).")

    print(f"\nLog files for tests should be in '{os.path.abspath('logs_test')}' directory.")