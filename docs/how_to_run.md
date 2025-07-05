Of course. This is the final and most important step: a comprehensive operational guide. You have built a powerful and complex system, and knowing how to operate it correctly is crucial for success and safety.

Here is the best approach to fully operate your system, broken down into distinct phases of its lifecycle. This guide explains what to do, which scripts to run, and why.

---

### **The Operational Lifecycle: A 4-Phase Guide**

Your system's operation can be understood in four phases:

1.  **Phase 1: Initial System Bootstrap (One-Time Setup)**
    *   *Goal:* To populate all necessary data stores and train the initial ML models before the first live run.
2.  **Phase 2: Daily Live Operation (The Routine)**
    *   *Goal:* To run the live trading bot safely and monitor its performance during market hours.
3.  **Phase 3: Research & Validation (The R&D Loop)**
    *   *Goal:* To use the advanced backtesting suite to test new ideas and find superior strategy parameters offline.
4.  **Phase 4: Periodic Recalibration (Automated Governance)**
    *   *Goal:* To run the automated pipeline that validates and promotes new, superior parameters to the live system.

---

### **Phase 1: Initial System Bootstrap (One-Time Setup)**

**Objective:** To prepare the system for its very first live run. This only needs to be done once.

#### **Step 1.1: Populate Historical Data**

Your backtester and feature store need historical data to function.
*   **Action:** Run your data download script to get the necessary historical CSV files.
*   **Command:**
    ```bash
    # Assuming you have a script like this from your initial setup
    python get_historical_data.py 
    ```
    *(If you don't have a dedicated script, you would manually place the required `SYMBOL.csv` files into the `data/historical_csv/` directory.)*

#### **Step 1.2: Initialize the Feature Store**

This step creates the initial feature data from the historical CSVs.
*   **Action:** Manually trigger your Prefect flow. It will run the necessary ETL scripts.
*   **Command:**
    ```bash
    # First, make sure a Prefect agent is running in a separate terminal
    prefect agent start --pool 'default-agent-pool'

    # Then, deploy and run the flow
    prefect deploy --all
    prefect deployment run 'nightly-etl-and-model-maintenance/nightly-etl-and-model-maintenance'
    ```
    This will execute the logic in `nightly_feature_pipeline.py`, which populates the Feast offline and online stores.

#### **Step 1.3: Train the Initial Machine Learning Models**

Your system needs its first set of `.pkl` model files to start.
*   **Action:** Run the two model training scripts.
*   **Commands:**
    ```bash
    # 1. Train the main signal-vetoing model
    python ml/ml_model_trainer.py

    # 2. Train the "common sense" heuristic model for the AlgoWheel
    python ml/bootstrap_algowheel_model.py
    ```
*   **Outcome:** Your `models/` directory now contains `ml_vetoing_model.pkl`, `scaler.pkl`, `algowheel_policy_model.pkl`, and `algowheel_feature_encoder.pkl`.

**Your system is now fully bootstrapped and ready for live operation.**

---

### **Phase 2: Daily Live Operation (The Routine)**

**Objective:** The simple, daily process for running and monitoring the live trading bot.

#### **Step 2.1: Pre-Market Checklist (Daily)**

1.  **Start IB Gateway/TWS:** Ensure it is running and you are logged into your trading account.
2.  **Check System Status:** Make sure there is no `kill.flag` or `heartbeat.flag` left over from a previous run. The `run_system.sh` script handles cleaning the kill flag automatically.

#### **Step 2.2: Run the System**

*   **Action:** Execute the main run script. This is the **only command you need to run daily** to start trading.
*   **Command:**
    ```bash
    # From your project's root directory
    ./run_system.sh
    ```
*   **What it does:**
    *   Starts the main trading application (`main.py`) in the background.
    *   Starts the independent risk monitor (`risk_monitor.py`) in the background.

#### **Step 2.3: Monitor the System**

*   **Action:** Keep an eye on the system's health and activity by tailing the log files.
*   **Commands (run in separate terminals):**
    ```bash
    # Monitor the main trading application's activity
    tail -f main.log

    # Monitor the independent risk monitor's status
    tail -f risk_monitor.log
    ```

#### **Step 2.4: Shut Down the System**

*   **Graceful Shutdown:**
    *   Go to the terminal where you ran `./run_system.sh`.
    *   Press `Ctrl+C`. This will send a signal to both processes, allowing them to shut down gracefully (e.g., save state, disconnect from IB).
*   **Emergency Shutdown:**
    *   If the system is unresponsive or you need to halt it immediately from anywhere, use the manual kill-switch.
    *   **Command (run from the project root):**
        ```bash
        touch flags/KILL_NOW.flag
        ```
        The `risk_monitor.py` process will detect this file and initiate a system-wide liquidation and shutdown.

---

### **Phase 3: Research & Validation (The R&D Loop)**

**Objective:** To use your advanced backtesting suite to develop new strategies or optimize existing ones without affecting the live system. This is typically done offline or on weekends.

#### **Step 3.1: Find Optimal Strategy Parameters**

*   **Action:** Run a grid search to find the best parameters for a strategy.
*   **Command:**
    ```bash
    python run_backtest.py --strategy momentum --mode grid_search --config_set strategy
    ```

#### **Step 3.2: Validate for Overfitting**

*   **Action:** Run the best candidate parameters from the grid search through a walk-forward validation.
*   **Command:**
    ```bash
    python run_backtest.py --strategy momentum --mode walk_forward --config_set strategy
    ```

#### **Step 3.3: Understand the Risk Profile**

*   **Action:** Run Monte Carlo and Scenario tests on your best parameter set.
*   **Commands:**
    ```bash
    # Run Monte Carlo simulation
    python run_backtest.py --strategy momentum --mode monte_carlo

    # Run historical crisis scenario analysis
    python run_backtest.py --strategy momentum --mode scenario_analysis
    ```

---

### **Phase 4: Periodic System Recalibration (Automated Governance)**

**Objective:** To run the fully automated pipeline that validates and promotes new, superior parameters to the live system. This should be done periodically (e.g., quarterly).

#### **Step 4.1: Review the Mandate**

*   **Action:** Before automating, open `validation_mandate.yaml` and ensure you are comfortable with the defined risk and performance criteria. This is your system's "constitution."

#### **Step 4.2: Schedule the Automated Run**

*   **Action:** Set up a `cron` job to run the `run_recalibration.py` script automatically.
*   **Example `crontab` entry (to run quarterly at 2:00 AM on the 1st of Jan, Apr, Jul, Oct):**
    1.  Open your crontab: `crontab -e`
    2.  Add the line (adjust paths as needed):
        ```crontab
        0 2 1 1,4,7,10 * /path/to/your/venv/bin/python /path/to/your/project/equity_trading_system/run_recalibration.py --strategy momentum >> /path/to/your/project/logs/recalibration.log 2>&1
        ```

*   **Manual Trigger:** You can also run this process manually at any time:
    ```bash
    python run_recalibration.py --strategy momentum
    ```

**Outcome:** If the new best parameters pass all validation checks against the mandate, this script will **automatically update your live `config.yaml` file**. The live system will pick up these new parameters on its next run, completing the autonomous improvement loop.