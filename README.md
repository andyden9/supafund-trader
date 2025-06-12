# Supafund Trader Agent

This agent predicts the funding outcomes of startup projects listed on Supafund and places bets on corresponding prediction markets on Olas.

## Key Features

- **Data-Driven Predictions**: Leverages exclusive project data from the Supafund platform, including AI evaluation scores, application details, and program requirements.
- **Real-Time Signal Enhancement**: Augments predictions with real-time data from sources like GitHub to gauge project momentum.
- **Customizable Strategy**: Allows users to specify weights for different feature dimensions (e.g., founder strength, market potential, technical execution) to tailor the prediction and trading strategy.
- **Automated Trading**: Identifies relevant prediction markets on Olas and automatically places bets based on the model's output.

## Core Logic

The agent's logic is built using LangGraph to create a robust and modular pipeline for:
1.  Fetching and processing data from Supafund and external APIs.
2.  Assembling a comprehensive feature set for each project.
3.  Generating predictions with confidence scores and detailed reasoning.
4.  Executing trades on prediction markets. 

# 进入原型脚本所在目录
cd scripts/supafund_trader

# 创建一个新的虚拟环境 (例如 venv)
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 安装独立的依赖
pip install -r requirements.txt

# run
python -m streamlit run scripts/supafund_trader/app.py