# Equity Trading System

![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-Proprietary-red)
![Status](https://img.shields.io/badge/status-Active%20Development-green)

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture Diagram](#architecture-diagram)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [IB API Installation](#ib-api-installation)
  - [Dependencies](#dependencies)
- [Configuration (`config.yaml`)](#configuration-configyaml)
- [Usage](#usage)
  - [Running Live Trading](#running-live-trading-mainpy)
  - [Running the Risk Monitor](#risk-monitor-risk_monitorpy)
  - [Running Backtests](#backtesting-backtestmain_looppy)
  - [Other Key Scripts](#other-key-scripts)
- [Risk Management & Fail-Safes](#risk-management--fail-safes)
- [Compliance and Auditing](#compliance-and-auditing)
- [License & Disclaimer](#license--disclaimer)

## Overview

This project implements a modular, event-driven algorithmic trading system in Python, designed to interface with the Interactive Brokers (IBKR) Trader Workstation (TWS) or IB Gateway. The system provides a comprehensive framework for developing, backtesting, and deploying automated equity trading strategies. It emphasizes safety, compliance, and robust risk management, making it suitable for various trading approaches.

A core strength of the system is its **`risk_monitor.py`** component. This independent process acts as a "second line of defense," continuously monitoring the main trading application for financial and operational risks. It can trigger a system-wide kill-switch if predefined critical conditions are breached, ensuring a robust safety net.

The system supports live trading, advanced backtesting (including Walk-Forward Analysis), feature management with Feast, adaptive execution with an AlgoWheel, ML-based trade vetoing, and detailed Transaction Cost Analysis (TCA).

## Key Features

-   **Event-Driven Architecture**: Decoupled components communicate via a central event queue (`core/event_queue.py`), facilitating modular development and testing.
-   **Robust IBKR Integration**: A resilient wrapper (`core/ib_wrapper.py`) manages the `ibapi` connection with automatic reconnect logic and a watchdog thread.
-   **Live Portfolio Management**: Real-time tracking of holdings, cash (including T+N settlement logic), P&L, and dynamic, volatility-adjusted position sizing.
-   **Independent Risk Monitor**: A separate OS process (`risk_monitor.py`) provides a fail-safe by monitoring:
    -   Main engine liveness via ZMQ/CSV heartbeats.
    -   Runaway order submission rates.
    -   Critical portfolio risk metrics like drawdown and fill budgets.
    -   Manual kill-switch flags (`KILL_NOW.flag`).
-   **Advanced Backtesting Suite**: The `backtest/` module supports historical simulations, including **Walk-Forward Analysis** to combat overfitting.
-   **Feast Feature Store**: Centralized feature management using local Feast stores (DuckDB/SQLite) for consistency between research, backtesting, and live trading.
-   **Adaptive AlgoWheel**: Dynamically selects the optimal execution method (MOC, LOC, ADAPTIVE, TWAP) based on learned performance from historical TCA data.
-   **Transaction Cost Analysis (TCA)**: Logs detailed execution data to a DuckDB database for performance analysis and to feed the AlgoWheel.
-   **ML Model Integration**: Supports loading trained models (e.g., from `ml/ml_model_trainer.py`) to apply an ML-based "veto" on low-quality trading signals.
-   **Comprehensive Configuration**: All system parameters are managed centrally in `config.yaml`, including risk thresholds, strategy parameters, and operational flags.

## Architecture Diagram

```mermaid
flowchart LR
    %% === Main Application ===
    subgraph Main_Trading_Application["Main Trading Application"]
        direction TB
        
        subgraph Event_Bus["Event Bus"]
            direction TB
            EQ["core/event_queue.py"]
        end

        subgraph Data_Execution_Flow["Data & Execution Flow"]
            direction TB
            IBAPI["IBKR TWS/Gateway"] -- "Market Data/Fills" --> IBW["core/ib_wrapper.py"]
            IBW -- "Connects" --> DH["data/ib_handler.py"]
            DH -- "MarketEvent" --> EQ
            EH["execution/ib_executor.py"] -- "Places Order" --> IBAPI
        end

        subgraph Strategy_Portfolio["Strategy & Portfolio Logic"]
            direction TB
            STRAT["strategy/"] -- "SignalEvent" --> EQ
            PM["portfolio/live_manager.py"]
            EQ -- "MarketEvent" --> STRAT
            EQ -- "SignalEvent" --> PM
            EQ -- "OrderEvent" --> EH
            EQ -- "FillEvent/OrderFailedEvent" --> PM
        end

        subgraph App_Services["Application Services & Clients"]
            direction TB
            FEAST_CLIENT["Feast Client"]
            AW_CLIENT["AlgoWheel Client"]
            ML_MODEL_CLIENT["ML Client"]
            FEAST_CLIENT -- "Uses" --> PM
            AW_CLIENT -- "Uses" --> PM
            ML_MODEL_CLIENT -- "Uses" --> PM
        end
    end

    %% === Independent Services & Monitors ===
    subgraph Independent_Processes["Independent Processes"]
        direction TB
        
        subgraph Risk_Monitor["Risk Monitor"]
            direction TB
            RM_PROCESS["risk_monitor.py Process"]
            MAIN_IPC_REP["ZMQ REP Socket"] -- "Commands" --> RM_PROCESS
        end

        subgraph System_Flags["System Flags & Signals"]
            direction TB
            KNF["flags/KILL_NOW.flag"]
            KF["flags/KILL.flag"]
        end
    end

    %% === External Dependencies ===
    subgraph External_Services_Data["External Services & Data"]
        direction TB
        
        subgraph Data_Stores["Data Stores"]
            direction TB
            FEAST_FS["Feast Feature Store<br>(DuckDB/SQLite)"]
            TCA_DB["TCA DuckDB"]
            ML_MODELS_STORE["ML Models"]
        end

        subgraph Log_Files["Log Files"]
            direction TB
            RLOG["logs/risk_log.csv"]
            OSLOG["logs/order_submissions.csv"]
            RM_HB["logs/risk_monitor_heartbeat.json"]
        end

        subgraph External_Connections["External Connections"]
            direction TB
            IBAPI_RM["IBKR TWS/Gateway (RM)"]
            HB_MAIN["Main Engine<br>Heartbeat"]
        end
    end

    %% === Connections Between Subgraphs ===
    PM -- "Logs Trades for TCA" --> TCA_DB
    PM -- "Writes" --> RLOG
    PM -- "Writes Logs & Heartbeat" --> OSLOG
    PM -- "Publishes" --> HB_MAIN
    KF -- "Signals Main App to Halt" --> PM

    FEAST_CLIENT -- "Fetches Features" --> FEAST_FS
    AW_CLIENT    -- "Reads TCA Data"   --> TCA_DB
    ML_MODEL_CLIENT -- "Loads Model"   --> ML_MODELS_STORE

    RM_PROCESS -- "Subscribes" --> HB_MAIN
    RM_PROCESS -- "Tails"      --> OSLOG
    RM_PROCESS -- "Reads"      --> RLOG
    RM_PROCESS -- "Checks for" --> KNF
    RM_PROCESS -- "Creates"    --> KF
    RM_PROCESS -- "Publishes"  --> RM_HB
    RM_PROCESS -- "Connects"   --> IBAPI_RM

    %% === Styling ===
    classDef coreComponent     fill:#cde4ff,stroke:#4a69bd;
    classDef externalInterface fill:#ffc0cb,stroke:#8b0000;
    classDef dataStore         fill:#b2dfdb,stroke:#004d40;
    classDef logFile           fill:#fff59d,stroke:#f57f17;
    classDef process           fill:#d1c4e9,stroke:#4527a0;
    classDef criticalSignal    fill:#ffab91,stroke:#bf360c;
    classDef client            fill:#e6e6fa,stroke:#333;
    classDef subText           color:#000;

    %% assign style classes
    class EQ,IBAPI,IBW,DH,EH,STRAT,PM,FEAST_CLIENT,AW_CLIENT,ML_MODEL_CLIENT,RM_PROCESS,KNF,KF,FEAST_FS,TCA_DB,ML_MODELS_STORE,RLOG,OSLOG,RM_HB,IBAPI_RM,HB_MAIN subText;

    class PM,STRAT,DH,IBW,EH,EQ coreComponent;
    class IBAPI,IBAPI_RM externalInterface;
    class FEAST_FS,TCA_DB,ML_MODELS_STORE dataStore;
    class RLOG,OSLOG,RM_HB,HB_MAIN logFile;
    class RM_PROCESS process;
    class KNF,KF criticalSignal;
    class FEAST_CLIENT,AW_CLIENT,ML_MODEL_CLIENT client;
```

## Project Structure

```text
equity_trading_system/
├── backtester/
│   ├── distributed_runner.py
│   ├── engine.py
│   ├── performance.py
│   └── validation.py
├── core/
│   ├── config_loader.py
│   ├── event_queue.py
│   ├── events.py
│   └── ib_wrapper.py
├── data/
│   └── ib_handler.py
├── docs/
│   ├── README.md
│   └── requirements.txt
├── etl/
│   ├── fetch_daily_bars.py
│   └── tca_feature_etl.py
├── execution/
│   └── ib_executor.py
├── feature_repo/
│   ├── data/
│   ├── data_sources.py
│   ├── entity.py
│   ├── feature_store.yaml
│   └── feature_view.py
├── flags/
├── logs/
├── ml/
│   ├── algowheel_model_trainer.py
│   ├── bootstrap_algowheel_model.py
│   └── ml_model_trainer.py
├── mlruns/
├── models/
├── performance/
│   └── tracker.py
├── portfolio/
│   ├── algo_wheel.py
│   ├── live_manager.py
│   ├── monthly_counter.py
│   ├── optimizer.py
│   ├── risk_manager.py
│   └── shared_sizing_logic.py
├── prefect_flows/
│   └── nightly_feature_pipeline.py
├── results/
├── state/
├── strategy/
│   ├── base.py
│   ├── mean_reversion.py
│   └── momentum.py
├── tca/
│   ├── calculator.py
│   └── logger.py
├── utils/
│   └── logger.py
├── venv/
├── .gitignore
├── .prefectignore
├── .python-version
├── config.yaml
├── main.py
├── prefect.yaml
├── risk_monitor.py
├── run_backtest.py
├── run_recalibration.py
├── run_system.sh
└── validation_mandate.yaml
```

## Installation

### Prerequisites

1.  **Python:** Version 3.9+ is recommended.
2.  **Interactive Brokers Account:** A live or paper trading account.
3.  **IB Gateway or TWS:** Must be installed and running.
4.  **TA-Lib C Library:** This must be installed on your system before installing the Python wrapper.
    -   **macOS:** `brew install ta-lib`
    -   **Linux/Windows:** Refer to the [official TA-Lib documentation](https://github.com/TA-Lib/ta-lib-python).

### IB API Installation

The official `ibapi` package requires manual installation:
1.  Download and install the **TWS API** software from the [IBKR Website](https://interactivebrokers.github.io/).
2.  Navigate to the installation directory (e.g., `~/IBJts/source/pythonclient/ibapi`).
3.  Activate your virtual environment and run:
    ```bash
    python setup.py install
    ```

### Dependencies

1.  **Clone the Repository:**
    ```bash
    git clone <your_repository_url>
    cd equity_trading_system
    ```
2.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install Python Packages:**
    The core dependencies are listed in `docs/requirements.txt`.
    ```bash
    pip install -r docs/requirements.txt
    ```

## Configuration (`config.yaml`)

The `config.yaml` file is the central hub for all system settings. Before running, you must review and customize this file, especially the following sections:

-   **`ib_connection`**: Set your `host`, `port`, and a unique `main_client_id`.
-   **`risk_monitor`**: Set the `risk_monitor_client_id` to be different from the main client ID.
-   **`portfolio`**: Define your `initial_capital` and default `risk_per_trade_pct`.
-   **`symbols`**: List the equity symbols the system is authorized to trade.
-   **`operational_risk`**: Ensure the paths for `heartbeat_flag_path` and `order_rate_counter_path` are correct.
-   **`system`**: Verify the paths for `kill_flag_path` and `kill_now_flag_path`.

## Usage

### Running Live Trading

The `run_system.sh` script is the recommended way to start the live system, as it launches both the main application and the independent risk monitor.

1.  Ensure IB Gateway or TWS is running and you are logged in.
2.  In your terminal, from the project root, make the script executable and run it:
    ```bash
    chmod +x run_system.sh
    ./run_system.sh
    ```
    This will start both processes in the background and direct their output to `main.log` and `risk_monitor.log`.

### Running Backtests and Validation

The `run_backtest.py` script is the entry point for all research and validation.

-   **Run a simple parameter grid search:**
    ```bash
    python run_backtest.py --strategy momentum --mode grid_search
    ```
-   **Run a full Walk-Forward Analysis:**
    ```bash
    python run_backtest.py --strategy momentum --mode walk_forward
    ```
-   **Run a Monte Carlo Simulation on the best parameters:**
    ```bash
    python run_backtest.py --strategy momentum --mode monte_carlo
    ```
-   **Run Scenario and Stress Tests:**
    ```bash
    python run_backtest.py --strategy momentum --mode scenario_analysis
    ```

### Automated Recalibration

The `run_recalibration.py` script automates the entire validation and promotion pipeline. It should be run periodically (e.g., quarterly via a cron job).

```bash
python run_recalibration.py --strategy momentum
```

## Risk Management & Fail-Safes

The system employs a multi-layered "Two-Line Defense" approach to risk:

1.  **First Line (Inside `main.py`)**:
    -   **Dynamic Position Sizing**: Adjusts trade size based on volatility and risk parameters.
    -   **Portfolio Constraints**: Enforces hard caps on exposure and position count.
    -   **Drawdown Throttling**: The `RiskManager` reduces trade size automatically as portfolio drawdown increases.
    -   **Volatility Targeting**: The `PortfolioOptimizer` scales overall leverage to maintain a target portfolio volatility.

2.  **Second Line (Independent `risk_monitor.py`)**:
    -   **Financial Monitoring**: Independently checks portfolio drawdown against a critical kill-switch threshold.
    -   **Operational Monitoring**:
        -   **Heartbeat Check**: Ensures the main application has not frozen or crashed.
        -   **Order Rate Check**: Protects against runaway algorithms flooding the market with orders.
    -   **Kill-Switch**: If a critical, confirmed risk breach occurs, it creates a `kill.flag` to trigger a system-wide, graceful shutdown and liquidation of all positions.
    -   **Manual Override**: An operator can create a `KILL_NOW.flag` file to trigger an immediate shutdown.

## Compliance and Auditing

The system is designed with compliance and auditing in mind. The `tca/logger.py` creates a permanent, detailed record of all executions. The `run_recalibration.py` script provides an objective, repeatable process for deploying strategy changes. For users operating under specific constraints (e.g., "passive investor" status), the system's configurable limits on trade frequency and its batch-processing capabilities can be used to align with regulatory safe harbors.

## License

Proprietary.

## Disclaimer

> This software is for educational and informational purposes ONLY. Trading financial instruments involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results.
>
> The authors and contributors provide this software *as is* without warranty of any kind, express or implied. They are not responsible for any trading losses incurred using this software or based on its output.
>
> Always conduct thorough backtesting and simulation in a Paper Trading account before considering any live deployment with real capital. This software does not constitute financial, investment, or trading advice. Consult with a qualified financial professional before making any investment decisions. Use this software entirely at your own risk.
