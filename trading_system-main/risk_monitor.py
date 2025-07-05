# risk_monitor.py

import os
import threading
import time
from datetime import datetime, timezone
import logging

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.wrapper import EWrapper

# These imports are necessary for the standalone execution of this script
from core.config_loader import config
from utils.logger import setup_logger

# Setup a dedicated logger for the risk monitor
setup_logger(logger_name="RiskMonitor", log_level=logging.INFO)
logger = logging.getLogger("RiskMonitor")

class RiskMonitor(EWrapper, EClient):
    """
    An independent "Two-Line Defense" risk management process that monitors
    financial (drawdown), operational (heartbeat), and order rate risk.
    """
    def __init__(self, kill_switch_drawdown_pct, initial_peak_equity, op_risk_config):
        EClient.__init__(self, self)
        self.kill_switch_drawdown_pct = kill_switch_drawdown_pct
        self.peak_equity = initial_peak_equity
        self.current_equity = initial_peak_equity
        self.next_order_id = None
        self.kill_switch_active = False
        self.is_connected_event = threading.Event()
        
        # Heartbeat monitoring setup
        self.heartbeat_path = op_risk_config.get('heartbeat_flag_path', 'state/heartbeat.flag')
        self.heartbeat_timeout = op_risk_config.get('heartbeat_timeout_seconds', 180)
        self.heartbeat_thread = None

        # Order rate monitoring setup
        self.order_rate_counter_path = op_risk_config.get('order_rate_counter_path', 'state/order_rate.counter')
        self.order_rate_limit = op_risk_config.get('order_rate_limit_per_minute', 100)
        self.order_rate_thread = None

    def start(self, host, port, client_id):
        """Connects to IB and starts all monitoring loops."""
        logger.info(f"Connecting to IB on {host}:{port} with Client ID {client_id}...")
        self.connect(host, port, clientId=client_id)

        eclient_thread = threading.Thread(target=self.run, daemon=True, name="RiskMonitorEClient")
        eclient_thread.start()

        if not self.is_connected_event.wait(timeout=10):
            self.disconnect()
            raise ConnectionError("Risk Monitor failed to connect to IB.")

        logger.info("Risk Monitor connected. Subscribing to account updates...")
        self.reqAccountSummary(9001, "All", "NetLiquidation")

        # Start heartbeat monitoring thread
        self.heartbeat_thread = threading.Thread(target=self._check_heartbeat, daemon=True, name="HeartbeatChecker")
        self.heartbeat_thread.start()
        logger.info(f"Heartbeat monitor started. Watching '{self.heartbeat_path}' with a {self.heartbeat_timeout}s timeout.")

        # Start order rate monitoring thread
        self.order_rate_thread = threading.Thread(target=self._check_order_rate, daemon=True, name="OrderRateChecker")
        self.order_rate_thread.start()
        logger.info(f"Order rate monitor started. Limit: {self.order_rate_limit}/minute.")

    def _check_heartbeat(self):
        """Periodically checks the heartbeat file from the main application."""
        time.sleep(self.heartbeat_timeout) # Initial grace period for main app to start
        while not self.kill_switch_active:
            try:
                if not os.path.exists(self.heartbeat_path):
                    self.trigger_kill_switch(reason="OPERATIONAL RISK: Heartbeat file not found.")
                    break

                with open(self.heartbeat_path, 'r') as f:
                    last_heartbeat_str = f.read()
                
                last_heartbeat_ts = datetime.fromisoformat(last_heartbeat_str)
                age = (datetime.now(timezone.utc) - last_heartbeat_ts).total_seconds()

                if age > self.heartbeat_timeout:
                    self.trigger_kill_switch(reason=f"OPERATIONAL RISK: Heartbeat is stale ({age:.0f}s old). Main app may be frozen.")
                    break
            except Exception as e:
                logger.error(f"Error in heartbeat check: {e}")
                self.trigger_kill_switch(reason=f"OPERATIONAL RISK: Critical error while checking heartbeat ({e}).")
                break
            
            time.sleep(30) # Check every 30 seconds

    def _check_order_rate(self):
        """Periodically checks the order submission rate."""
        last_count = 0
        if os.path.exists(self.order_rate_counter_path):
            try:
                with open(self.order_rate_counter_path, "r") as f:
                    last_count = int(f.read() or 0)
            except (IOError, ValueError):
                last_count = 0
        
        while not self.kill_switch_active:
            time.sleep(60) # Check every minute
            try:
                if not os.path.exists(self.order_rate_counter_path):
                    continue
                
                with open(self.order_rate_counter_path, "r") as f:
                    current_count = int(f.read() or 0)
                
                rate_this_minute = current_count - last_count
                if rate_this_minute > self.order_rate_limit:
                    reason = f"OPERATIONAL RISK: Order rate of {rate_this_minute}/min exceeded limit of {self.order_rate_limit}. Possible runaway algorithm."
                    self.trigger_kill_switch(reason=reason)
                    break
                
                logger.debug(f"Order rate check: {rate_this_minute} orders in the last minute (Limit: {self.order_rate_limit}).")
                last_count = current_count
            except Exception as e:
                logger.error(f"Error in order rate check: {e}")

    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        """Callback for account summary data."""
        if tag == "NetLiquidation":
            self.current_equity = float(value)
            if self.current_equity > self.peak_equity:
                self.peak_equity = self.current_equity
            self.check_drawdown()

    def check_drawdown(self):
        """Checks if the drawdown threshold has been breached."""
        if self.kill_switch_active or self.peak_equity <= 0:
            return

        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        print(f"\rRisk Monitor: Equity=${self.current_equity:,.2f}, Peak=${self.peak_equity:,.2f}, Drawdown={drawdown:.2%}", end="", flush=True)

        if drawdown >= self.kill_switch_drawdown_pct:
            reason = f"FINANCIAL RISK: Drawdown of {drawdown:.2%} exceeded threshold of {self.kill_switch_drawdown_pct:.2%}"
            self.trigger_kill_switch(reason=reason)

    def trigger_kill_switch(self, reason: str = "Unknown"):
        """Initiates a system-wide shutdown for a specific reason."""
        if self.kill_switch_active:
            return
        self.kill_switch_active = True
        self.reqCancelAccountSummary(9001)

        print("\n" + "!" * 60)
        logger.critical("CRITICAL: KILL-SWITCH TRIGGERED!")
        logger.critical(f"  REASON: {reason}")
        print("!" * 60)

        with open("kill.flag", "w") as f:
            f.write(f"Kill switch triggered at {datetime.now(timezone.utc).isoformat()}.\nReason: {reason}\n")

        logger.info("ACTION: Cancelling all open orders...")
        self.reqGlobalCancel()
        time.sleep(1)
        logger.info("ACTION: Requesting current positions to flatten...")
        self.reqPositions()

    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        """Callback for receiving position data. Flattens each position."""
        if position != 0:
            logger.info(f"  - Flattening position: {position} shares of {contract.symbol}")
            order = Order()
            order.action = "SELL" if position > 0 else "BUY"
            order.orderType = "MKT"
            order.totalQuantity = abs(position)
            self.placeOrder(self.next_order_id, contract, order)
            self.next_order_id += 1

    def nextValidId(self, orderId: int):
        """Receives the next valid order ID."""
        self.next_order_id = orderId
        self.is_connected_event.set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """Handles errors from the TWS, ignoring informational messages."""
        if errorCode not in [2104, 2106, 2158, 2108, 2100, 2150]:
            logger.error(f"Risk Monitor Error (Req ID: {reqId}, Code: {errorCode}): {errorString}")

if __name__ == "__main__":
    # Load configuration from the global config object
    risk_config = config.get('risk_monitor', {})
    portfolio_config = config.get('portfolio', {})
    ib_conn_config = config.get('ib_connection', {})
    op_risk_config = config.get('operational_risk', {})

    # Instantiate the monitor with parameters from the config
    monitor = RiskMonitor(
        kill_switch_drawdown_pct=risk_config.get('kill_switch_drawdown_pct', 0.15),
        initial_peak_equity=portfolio_config.get('initial_capital', 100000.0),
        op_risk_config=op_risk_config
    )
    try:
        # Start the monitor with connection details from the config
        monitor.start(
            host=ib_conn_config.get('host', '127.0.0.1'),
            port=ib_conn_config.get('port', 7497),
            client_id=ib_conn_config.get('risk_monitor_client_id', 2)
        )
        # Keep the main thread alive to allow background threads to run
        while True:
            time.sleep(1)
    except ConnectionError as e:
        logger.critical(f"Failed to run Risk Monitor: {e}")
    except (KeyboardInterrupt, SystemExit):
        logger.info("\nRisk Monitor shut down by user.")
    finally:
        monitor.disconnect()