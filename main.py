from fastapi import FastAPI
from typing import List, Dict, Any, Optional

app = FastAPI()

# This is a placeholder for the agent's state.
# In the future, this will be dynamically populated by the Open Autonomy agent's FSM.
mock_health_data = {
    "seconds_since_last_transition": 15.0,
    "is_tm_healthy": True,
    "period": 2,
    "reset_pause_duration": 30.0,
    "rounds": [
        "RegistrationRound",
        "FetchApplicationsRound",
        "AnalyzeApplicationRound",
    ],
    "is_transitioning_fast": True,
    "agent_health": {
        "is_making_on_chain_transactions": False, # Not applicable yet
        "is_staking_kpi_met": True, # Not applicable yet
        "has_required_funds": True, # Not applicable yet
        "staking_status": "NOT_STAKED" # Staking is not used
    },
    "rounds_info": {
        "RegistrationRound": {
            "name": "Registration Round",
            "description": "Initial state of the agent.",
            "transitions": {"DONE": "FetchApplicationsRound"}
        },
        "FetchApplicationsRound": {
            "name": "Fetch Applications Round",
            "description": "Fetching new applications to analyze.",
            "transitions": {"DONE": "AnalyzeApplicationRound", "NO_APPLICATIONS": "ResetAndPauseRound"}
        },
        "AnalyzeApplicationRound": {
            "name": "Analyze Application Round",
            "description": "Analyzing a single application.",
            "transitions": {"DONE": "AnalyzeApplicationRound", "NO_MORE_APPLICATIONS": "ResetAndPauseRound", "ERROR": "ResetAndPauseRound"}
        },
        "ResetAndPauseRound": {
            "name": "Reset and Pause Round",
            "description": "Waiting for a configured period before checking for new applications again.",
            "transitions": {"DONE": "FetchApplicationsRound"}
        }
    },
    "env_var_status": {
        "needs_update": False,
        "env_vars": {}
    }
}


@app.get("/healthcheck")
def healthcheck() -> Dict[str, Any]:
    """
    Provides a healthcheck endpoint compliant with the Pearl platform requirements.
    
    Currently returns mock data. This will be integrated with the agent's
    Finite State Machine (FSM) to provide real-time status.
    """
    return mock_health_data

# Optional: Add a root endpoint for basic verification
@app.get("/")
def read_root():
    return {"message": "Supafund Trader Agent is running. Visit /healthcheck for status."}

# Placeholder for the main agent loop
def run_agent_loop():
    """
    This function will contain the main logic loop for the agent.
    It will be responsible for driving the FSM and executing business logic.
    """
    print("Agent loop would run here...")
    # In a real scenario, this would be a long-running process,
    # likely started in a separate thread or using asyncio.

if __name__ == "__main__":
    import uvicorn
    print("Starting Supafund Trader Agent server...")
    print("Access healthcheck at http://127.0.0.1:8716/healthcheck")
    
    # In a real integration, the agent loop would be started here.
    # For now, we just run the server.
    uvicorn.run(app, host="0.0.0.0", port=8716) 