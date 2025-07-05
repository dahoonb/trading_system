# filename: core/ib_wrapper.py
# core/ib_wrapper.py
import logging
import threading
import time
import queue
import datetime # Retained for use in the new parse_ib_datetime
from typing import Optional, Callable, Dict, List # Retained for use in the new parse_ib_datetime
from concurrent.futures import ThreadPoolExecutor

try:                     # Python 3.9+: stdlib zoneinfo
    from zoneinfo import ZoneInfo # Retained for use in the new parse_ib_datetime
except ImportError:      # <3.9 – will treat tz as naive
    ZoneInfo = None      # type: ignore

from ibapi.wrapper import EWrapper
from ibapi.client import EClient
from ibapi.common import OrderId
from ibapi.connection import Connection # Needed for disconnect check

logger = logging.getLogger("TradingSystem")


class IBWrapper(EWrapper, EClient):
    def __init__(self, inbound_queue: Optional[queue.Queue] = None) -> None:
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)

        self._lock = threading.RLock()
        self._client_thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None

        # Create a thread pool for running listener callbacks asynchronously
        self._listener_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix='IBWrapperListener')
        logger.info(f"Initialized listener executor with max_workers={self._listener_executor._max_workers}")

        self._inside_error_handler = False
        self._reconnect_enabled = True
        self._reconnect_attempt = 0
        self._max_reconnect_delay = 300  # 5 min cap

        self._host: str = "127.0.0.1"
        self._port: int = 7497
        self._client_id: int = 1

        self.connection_active = False
        self.connected_event = threading.Event()

        self.next_valid_order_id: Optional[OrderId] = None
        self._next_req_id: int = 1

        self._req_id_lock = threading.Lock()
        self.req_id_to_symbol: Dict[int, str] = {}

        self.inbound_queue = inbound_queue
        self._callback_chains: Dict[str, List[Callable]] = {}
        self._req_map_lock = threading.RLock()

    def _run_listener_safely(self, func: Callable, *args, **kwargs):
        """Executes a listener function in a try-except block to catch errors."""
        try:
            func(*args, **kwargs)
        except Exception:
            logger.exception(f"Exception caught in background listener thread executing {func.__name__}")

    # ------------------------------------------------------------------ #
    # compatibility wrapper around add_callback                          # <<<
    # ------------------------------------------------------------------ #
    def register_listener(self, name: str, func: Callable) -> None:      # <<<
        """Alias kept for older components (maps to add_callback)."""     # <<<
        self.add_callback(name, func)                                     # <<<
                                                                          # <<<
    def unregister_listener(self, name: str, func: Callable) -> None:     # <<<
        """Remove *func* from the callback chain of *name* (noop if absent)."""  # <<<
        lst = self._callback_chains.get(name)                             # <<<
        if lst and func in lst:                                           # <<<
            try:                                                          # <<< FIX: Add try-except
                lst.remove(func)                                          # <<<
            except ValueError:                                            # <<< FIX: Handle case where func might already be removed
                logger.debug(f"Listener {name} func {func} already removed, skipping unregister.") # <<<
            if not lst:                                                   # <<<
                try:                                                      # <<< FIX: Add try-except
                    del self._callback_chains[name]                       # <<<
                except KeyError:                                          # <<< FIX: Handle case where key might already be removed
                    logger.debug(f"Callback chain {name} already removed, skipping delete.") # <<<

    # --------------------------------------------------
    # public helpers
    # --------------------------------------------------
    def add_callback(self, name: str, func: Callable) -> None:
        """Register *func* to be called after the built‑in callback *name*."""
        with self._lock: # FIX: Ensure thread-safe modification
            if func not in self._callback_chains.setdefault(name, []):
                 self._callback_chains.setdefault(name, []).append(func)
            else:
                 logger.debug(f"Listener {name} func {func} already registered.")

    # ------------------------------------------------------------------ #
    # back‑compat helper – monotonic request‑id generator                #
    # ------------------------------------------------------------------ #
    def get_next_req_id(self) -> int:
        """Return a new, thread‑safe incremental request ID."""
        with self._req_id_lock:
            req_id = self._next_req_id
            self._next_req_id += 1
            return req_id

    # --------------------------------------------------
    # connection management
    # --------------------------------------------------
    def start_connection(self,
                         host: str = "127.0.0.1",
                         port: int = 7497,
                         client_id: int = 1) -> None:
        """Open the socket and start the network + watchdog threads."""
        with self._lock:
            if self.connection_active:
                logger.info("start_connection() – already connected.")
                return

            self._host, self._port, self._client_id = host, port, client_id
            logger.info(f"Connecting to IB Gateway/TWS {host}:{port} (client‑id={client_id}) …")

            # --- ADD THIS BLOCK TO FIX THE RUNTIMEERROR ---
            # Check if the executor has been shut down and re-create it if necessary.
            if self._listener_executor._shutdown:
                logger.info("Listener executor was previously shut down. Re-initializing it for new connection.")
                self._listener_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix='IBWrapperListener')
            # --- END OF ADDED BLOCK ---

            self.connection_active = False
            self.connected_event.clear()

            try:
                self.connect(host, port, client_id)
            except Exception as e:
                 logger.exception(f"Connection attempt failed during EClient.connect: {e}")
                 self.stop_connection() # Ensure cleanup if connect fails
                 raise ConnectionError(f"Failed to connect to IB: {e}")

            if self._client_thread is None or not self._client_thread.is_alive():
                self._client_thread = threading.Thread(
                    target=self.run,
                    name="IBAPI‑NetworkThread",
                    daemon=True
                )
                self._client_thread.start()
            else:
                 logger.warning("start_connection: Network thread already running?")

            if not self.connected_event.wait(timeout=15):
                logger.error("Timed‑out waiting for nextValidId – aborting connect.")
                self.stop_connection()
                raise ConnectionError("Could not establish IB connection (nextValidId timeout).")

            if self._watchdog_thread is None or not self._watchdog_thread.is_alive():
                self._reconnect_enabled = True # Ensure reconnect is enabled on new connection start
                self._watchdog_thread = threading.Thread(
                    target=self._watchdog_loop,
                    name="IBAPI‑Watchdog",
                    daemon=True
                )
                self._watchdog_thread.start()
            else:
                 logger.warning("start_connection: Watchdog thread already running?")

            if self.isConnected():
                self.connection_active = True
                logger.info("IB connection established.")
            else:
                logger.error("Connection check failed *after* nextValidId received/timeout.")
                self.stop_connection()
                raise ConnectionError("IB Connection failed post-nextValidId.")

    def stop_connection(self) -> None:
        """Gracefully close the socket, shutdown executor, and join helper threads."""
        with self._lock:
            self._reconnect_enabled = False

            if not self.isConnected():
                logger.info("stop_connection: Already disconnected.")
                self.connection_active = False
                self.connected_event.clear()
            else:
                logger.info("Disconnecting from IB (stop_connection)...")
                try:
                    self.disconnect()
                except Exception as exc:
                    logger.warning(f"Exception during disconnect() call in stop_connection: {exc}", exc_info=True)
                finally:
                    self.connection_active = False
                    self.connected_event.clear()

        logger.debug("stop_connection: Shutting down listener executor...")
        self._listener_executor.shutdown(wait=False, cancel_futures=True) 
        logger.debug("stop_connection: Listener executor shutdown initiated.")

        logger.debug("stop_connection: Joining client thread...")
        if self._client_thread and self._client_thread.is_alive():
            self._client_thread.join(timeout=5)
            if self._client_thread.is_alive():
                 logger.warning("Client thread did not join within timeout.")
        logger.debug("stop_connection: Joining watchdog thread...")
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=1)
            if self._watchdog_thread.is_alive():
                 logger.warning("Watchdog thread did not join within timeout.")

        logger.info("IB connection shutdown sequence complete.")

    def disconnect(self):
        """Disconnects from TWS/Gateway. Safe to call multiple times."""
        is_really_connected = isinstance(self.conn, Connection) and self.conn.isConnected()

        if not is_really_connected:
            self.connection_active = False
            self.connected_event.clear()
            return

        logger.info("Disconnecting from IB socket...")
        try:
            if hasattr(super(), 'disconnect'):
                 super().disconnect()
            else:
                 if isinstance(self.conn, Connection):
                     self.conn.disconnect()
                 logger.info("Manual EClient disconnect logic executed.")

        except Exception as e:
            logger.exception(f"Exception during EClient disconnect logic: {e}")
        finally:
             self.connection_active = False
             self.connected_event.clear()


    # --------------------------------------------------
    # watchdog logic
    # --------------------------------------------------
    def _watchdog_loop(self) -> None:
        """Reconnect automatically if the socket drops unexpectedly."""
        logger.info("IBAPI Watchdog thread started.")
        while self._reconnect_enabled:
            if not self.isConnected():
                 if self.connection_active: 
                      logger.warning("Watchdog: Lost connection to IB – attempting reconnect …")
                      self.connection_active = False
                      self.connected_event.clear()
                      self._attempt_reconnect()
            # Sleep in short intervals to allow for a timely exit if reconnect is disabled.
            for _ in range(5):
                if not self._reconnect_enabled:
                    break
                time.sleep(1)
        logger.info("IBAPI Watchdog thread stopped.")

    def _attempt_reconnect(self) -> None:
        if not self._reconnect_enabled:
            logger.info("Reconnect aborted as reconnect is now disabled.")
            return

        delay = min(2 ** self._reconnect_attempt, self._max_reconnect_delay)
        logger.info(f"Waiting {delay}s before reconnect attempt #{self._reconnect_attempt + 1} …")
        time.sleep(delay)

        if not self._reconnect_enabled:
            logger.info("Reconnect aborted as reconnect is now disabled (post-wait).")
            return
        if self.isConnected():
             logger.info("Reconnect aborted, connection re-established during wait.")
             self.connection_active = True
             self.connected_event.set()
             self._reconnect_attempt = 0
             return

        try:
            logger.info(f"Attempting reconnect #{self._reconnect_attempt + 1}...")
            self.start_connection(self._host, self._port, self._client_id)
            logger.info(f"Reconnect attempt #{self._reconnect_attempt + 1} successful.")
            self._reconnect_attempt = 0
        except Exception as exc:
            logger.error(f"Reconnect attempt #{self._reconnect_attempt + 1} failed: {exc}")
            self._reconnect_attempt += 1

    # --------------------------------------------------
    # required IB callbacks
    # --------------------------------------------------
    def nextValidId(self, orderId: OrderId):
        """Callback receiving the next valid order ID at connection time."""
        logger.info(f"nextValidId received ⇒ {orderId}")
        self.next_valid_order_id = orderId
        # Only set connected_event if we are not already marked as active?
        # Or just always set it, start_connection waits for it anyway.
        self.connected_event.set()

        # Dispatch to listeners
        listeners = self._callback_chains.get("nextValidId", [])
        for fn in listeners[:]: # Iterate copy in case listener modifies list
            try:
                fn(orderId)
            except Exception as exc:
                logger.exception(f"User nextValidId hook {fn} raised: {exc}")

    def connectionClosed(self):
        """Callback invoked when TWS/Gateway closes the socket connection."""
        logger.warning("IB connectionClosed() callback fired.")
        self.connection_active = False
        self.connected_event.clear()

        listeners = self._callback_chains.get("connectionClosed", [])
        for fn in listeners[:]:
            try:
                fn()
            except Exception as exc:
                logger.exception(f"User connectionClosed hook {fn} raised: {exc}")


    def error(self, *args):
        """Callback receiving error messages from TWS/Gateway."""
        if hasattr(self, '_inside_error_handler') and self._inside_error_handler:
            return
        if not hasattr(self, '_inside_error_handler'):
            self._inside_error_handler = False

        try:
            self._inside_error_handler = True
            
            reqId = errorCode = -1
            errorString = advancedJson = ""
            errorTime = None
            extra_args = None
            arg_len = len(args)

            try:
                if arg_len >= 6:
                    reqId, errorTime, errorCode, errorString, advancedJson = args[:5]
                    extra_args = args[5:]
                elif arg_len == 5:
                    reqId, errorTime, errorCode, errorString, advancedJson = args
                elif arg_len == 4:
                    reqId, errorCode, errorString, advancedJson = args
                elif arg_len == 3:
                    reqId, errorCode, errorString = args
                else:
                    logger.warning(f"Unhandled error tuple length ({arg_len}). Attempting fallback parsing: {args}")
                    ints_found = [a for a in args if isinstance(a, int)]
                    strs_found = [a for a in args if isinstance(a, str)]
                    plausible_codes = [i for i in ints_found if 100 <= i <= 6000]
                    if plausible_codes:
                        errorCode = plausible_codes[0]
                        remaining_ints = [i for i in ints_found if i != errorCode]
                        if -1 in remaining_ints: reqId = -1
                        elif remaining_ints: reqId = remaining_ints[0]
                        else: reqId = -1
                    elif ints_found:
                        if -1 in ints_found: reqId = -1
                        else: reqId = ints_found[0]
                        errorCode = -1
                    else: reqId = errorCode = -1
                    if len(strs_found) >= 1: errorString = strs_found[0]
                    if len(strs_found) >= 2: advancedJson = " ".join(strs_found[1:])

            except Exception as parse_exc:
                logger.exception(f"Exception during argument parsing in error callback: {args} -> {parse_exc}")
                # --- Safer Fallback Parsing on Exception ---
                reqId = errorCode = -1; errorString = advancedJson = ""; errorTime = None; extra_args = None
                try:
                    ints_found = [a for a in args if isinstance(a, int)]
                    strs_found = [a for a in args if isinstance(a, str)]
                    plausible_codes = [i for i in ints_found if 100 <= i <= 6000]
                    if plausible_codes: errorCode = plausible_codes[0]
                    remaining_ints = [i for i in ints_found if i != errorCode]
                    if -1 in remaining_ints: reqId = -1
                    elif remaining_ints: reqId = remaining_ints[0]
                    if len(strs_found) >= 1: errorString = strs_found[0]
                    if len(strs_found) >= 2: advancedJson = " ".join(strs_found[1:])
                except Exception as fallback_exc: logger.error(f"Exception during fallback parsing: {fallback_exc}")
            # --- END OF REVISED PARSING LOGIC ---

            # --- Classification Logic (Uses correctly parsed errorCode) ---
            INFO_CODES = { 2104, 2106, 2108, 2158, 2103, 2105, 2107, 2119, 2100, 2150, 2109 }
            WARNING_CODES = { 366, 1101, 2110, 10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009, 10010, 10011, 10012, 10013, 10014, 10015, 10016, 10017 } # 366: Expected after cancel/timeout?
            CRITICAL_CODES = { 502, 1300 }
            default_level = logging.ERROR
            level = default_level

            if errorCode in INFO_CODES: level = logging.INFO
            elif errorCode in WARNING_CODES: level = logging.WARNING
            elif errorCode in CRITICAL_CODES: level = logging.CRITICAL
            elif 1100 <= errorCode <= 1102:
                if errorCode == 1100:
                    level = logging.ERROR
                    if hasattr(self, 'connection_active'): self.connection_active = False
                    if hasattr(self, 'connected_event'): self.connected_event.clear()
                elif errorCode == 1101: level = logging.WARNING
                elif errorCode == 1102:
                    level = logging.INFO
                    if hasattr(self, 'connection_active'): self.connection_active = True
            elif errorCode == 504:
                level = logging.ERROR
                if hasattr(self, 'connection_active'): self.connection_active = False
                if hasattr(self, 'connected_event'): self.connected_event.clear()

            # --- Construct Log Message ---
            level_name = logging.getLevelName(level)
            log_prefix = f"IB-{level_name:<8}"
            time_str = f" Time: {errorTime}" if errorTime else ""
            log_msg = f"{log_prefix}{time_str} Code={errorCode} | ReqId={reqId}: {errorString}"
            if advancedJson: log_msg += f" | RejectInfo: {advancedJson}"
            if extra_args: log_msg += f" | ExtraArgs: {extra_args}"

            # --- Log the message ---
            logger.log(level, log_msg)

            # --- Dispatch to Listeners (Use ThreadPoolExecutor) ---
            listeners = self._callback_chains.get("error", [])
            if listeners:
                for fn in listeners[:]:
                    try:
                        self._listener_executor.submit(
                            self._run_listener_safely,
                            fn,
                            reqId,
                            errorCode,
                            errorString,
                            advancedJson,
                            errorTime=errorTime
                        )
                    except Exception as submit_exc:
                        logger.exception(f"Failed to submit listener {fn.__name__} to executor: {submit_exc}")
       
        finally:
            self._inside_error_handler = False

    # ==================================================
    # Utility – parse IB date/time strings (Unchanged)
    # ==================================================
    # _IB_DT_FORMATS list removed from here (was a class attribute)

# ------------------------------------------------------------------ #
# Chain every IB callback to user‑registered listeners.              #
# This wraps the original method (which might be a no‑op stub) so    #
# both the original behaviour *and* listeners run.                   #
# ------------------------------------------------------------------ #

_IB_CALLBACK_NAMES = [
    # Add historical data callbacks here
    "historicalData", "historicalDataEnd", "historicalDataUpdate",
    # contract / market‑data
    "contractDetails", "contractDetailsEnd", "realtimeBar", # Added realtimeBar
    "tickPrice", "tickSize", "tickString", "tickGeneric", # Added tick data
    "marketDataType", # Added market data type

    # orders / executions
    "openOrder", "openOrderEnd", "orderStatus",
    "execDetails", "execDetailsEnd", "commissionReport", # Added execDetailsEnd

    # account / portfolio
    "updateAccountValue", "updatePortfolio", "updateAccountTime",
    "accountDownloadEnd", "position", "positionEnd",
    "accountSummary", "accountSummaryEnd", # Added account summary

    # Others often used
    "managedAccounts", "scannerParameters", "scannerData", "scannerDataEnd",
    "verifyMessageAPI", "verifyCompleted", "verifyAndAuthMessageAPI", "verifyAndAuthCompleted",
    "displayGroupList", "displayGroupUpdated",
    "positionMulti", "positionMultiEnd", "accountUpdateMulti", "accountUpdateMultiEnd",
    "securityDefinitionOptionalParameter", "securityDefinitionOptionalParameterEnd",
    "softDollarTiers",
    "familyCodes", "symbolSamples",
    "mktDepth", "mktDepthL2", "updateMktDepth", "updateMktDepthL2",
    "tickOptionComputation",
    "tickSnapshotEnd", "marketRule",
    "pnl", "pnlSingle",
    "historicalTicks", "historicalTicksBidAsk", "historicalTicksLast",
    "tickByTickAllLast", "tickByTickBidAsk", "tickByTickMidPoint",
    "orderBound", "completedOrder", "completedOrdersEnd",
    "replaceFAEnd",
    "wshMetaData", "wshEventData",
    "historicalSchedule",
    "userInfo",
]

# Use introspection to find all EWrapper methods to avoid manual list
import inspect
_WRAPPER_METHODS = [name for name, func in inspect.getmembers(EWrapper, inspect.isfunction)
                    if not name.startswith("_") and name != 'python_do_handshake'] # Add any other internal methods to exclude

def _install_chain_method(cls, name):
    original = getattr(cls, name, None)
    if original is None and name in _WRAPPER_METHODS: # If it's a known EWrapper method but maybe not implemented in EClient
        # Provide a default stub that just calls listeners
        def _stub_wrapper(self, *args, **kwargs):
             # forward to listeners
             listeners = self._callback_chains.get(name, [])
             # logger.debug(f"Stub wrapper called for {name} with {len(listeners)} listeners")
             for fn in listeners[:]: # Iterate copy
                try:
                    fn(*args, **kwargs)
                except Exception:
                    logger.exception(f"Listener {fn} for {name} raised")
        _stub_wrapper.__name__ = name
        setattr(cls, name, _stub_wrapper)

    elif callable(original): # If method exists in base class (EClient likely)
        # Wrap the original method
        def _wrapper(self, *args, **kwargs):
            # call the original (if any)
            try:
                original(self, *args, **kwargs)
            except Exception as e:
                 logger.exception(f"Original EWrapper callback {name} raised: {e}")


            # forward to listeners
            listeners = self._callback_chains.get(name, [])
            # logger.debug(f"Chain wrapper called for {name} with {len(listeners)} listeners")
            for fn in listeners[:]: # Iterate copy
                try:
                    fn(*args, **kwargs)
                except Exception:
                    logger.exception(f"Listener {fn} for {name} raised")

        _wrapper.__name__ = name
        setattr(cls, name, _wrapper)
    # else: logger.debug(f"Skipping method chaining for {name}, not callable or not found.")

# Install wrappers dynamically for all known EWrapper methods
for _cb_name in _WRAPPER_METHODS:
    _install_chain_method(IBWrapper, _cb_name)
# Manually add any methods missed by introspection if necessary
# _install_chain_method(IBWrapper, 'some_missed_callback')

# Old parse_ib_datetime function removed from here.

# --- Define known formats (can be part of IBWrapper class or module level) ---
_IB_DT_FORMATS = [
    "%Y%m%d %H:%M:%S %Z",     # 20250103 14:30:00 EST
    "%Y%m%d %H:%M:%S",        # 20250103 14:30:00
    "%Y%m%d",                 # 20250103
    "%Y-%m-%d %H:%M:%S.%f",   # 2025-01-03 14:30:00.000123
    "%Y-%m-%d %H:%M:%S",      # 2025-01-03 14:30:00
    "%Y%m%d-%H:%M:%S",        # 20250103-14:30:00
    "%Y%m%d %H:%M:%S%z",      # 20250103 14:30:00+0000 # UTC offset format
]

def parse_ib_datetime(dt_str: str) -> Optional[datetime.datetime]:
    """
    Convert the many different date/time strings emitted by IB into a
    timezone-aware `datetime` object (UTC).

    Handles common IB date/time formats and epoch timestamps.

    Args:
        dt_str: The date/time string received from IB API.

    Returns:
        A timezone-aware datetime object (UTC), or None if parsing fails.
    """
    if not dt_str:
        return None

    # Handle Unix timestamp first (often used for historical data)
    # Ensure it's a string representation of an integer
    if dt_str.isdigit():
        try:
            ts = int(dt_str)
            # Check for a reasonable range (e.g., after year 2000) to avoid misinterpreting other numbers
            if ts > 946684800: # Timestamp for 2000-01-01 00:00:00 UTC
                return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        except (OverflowError, ValueError):
            # Not a valid timestamp, fall through to string formats
            pass

    # Then try the known explicit string formats
    for fmt in _IB_DT_FORMATS:
        try:
            dt = datetime.datetime.strptime(dt_str, fmt)
            # If the format included timezone info (%Z or %z), make it UTC
            if dt.tzinfo is not None:
                 # If timezone is something other than UTC, convert it
                 if dt.tzinfo != datetime.timezone.utc:
                      return dt.astimezone(datetime.timezone.utc)
                 else:
                      # Already UTC
                      return dt
            else:
                 # No timezone info in the string. Assume UTC for consistency.
                 # While IB often uses EST/EDT, relying on that assumption is risky.
                 # Enforcing UTC avoids ambiguity.
                 return dt.replace(tzinfo=datetime.timezone.utc)

        except ValueError:
            # Format did not match, try the next one
            continue

    logger.warning(f"parse_ib_datetime() could not parse '{dt_str}' with known formats.")
    return None
# --- End of function ---