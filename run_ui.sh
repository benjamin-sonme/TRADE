#!/bin/bash
# Launcher for U.S. Equity Trading Backtesting System Web UI

cd "$(dirname "$0")"

echo "========================================================"
echo "🚀 启动美股量化交易与资产配置回测系统 Web UI..."
echo "========================================================"

if [ -f ".venv/bin/streamlit" ]; then
    .venv/bin/streamlit run app.py
elif [ -f "$HOME/.venv/bin/streamlit" ]; then
    $HOME/.venv/bin/streamlit run app.py
else
    streamlit run app.py
fi
