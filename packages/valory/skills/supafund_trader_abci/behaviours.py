import json
from typing import Set, Type, Generator, Optional, Dict, Any

from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour,
    BaseBehaviour,
)
from packages.valory.skills.supafund_trader_abci.models import SharedState
from packages.valory.skills.supafund_trader_abci.payloads import FetchApplicationsPayload, AnalyzeApplicationPayload
from packages.valory.skills.supafund_trader_abci.rounds import (
    SupafundTraderAbciApp,
    AnalyzeApplicationRound,
    FetchApplicationsRound,
    ResetAndPauseRound,
)
# We need to import the logic from the root of the project.
# The Open Autonomy runtime will handle the python path.
from supafund_trader.agent_logic import SupafundClient, get_pipeline_logic

class FetchApplicationsBehaviour(BaseBehaviour):
    """Fetches applications to be processed."""

    matching_round = FetchApplicationsRound

    def async_act(self) -> Generator:
        """Do the action."""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            try:
                # Read configuration for Supabase
                supabase_url = self.context.params.supabase_url
                supabase_key = self.context.params.supabase_key
                
                # Instantiate the client and fetch data
                client = SupafundClient(url=supabase_url, key=supabase_key)
                applications = client.get_all_applications()
                
                # Serialize the list of applications to store in the shared state
                applications_json = json.dumps(applications)
                
                payload = FetchApplicationsPayload(
                    self.context.agent_address, applications_json
                )

            except Exception as e:
                self.context.logger.error(f"Failed to fetch applications: {e}")
                # For simplicity, we'll just create an empty payload on error
                payload = FetchApplicationsPayload(self.context.agent_address, "[]")

        with self.context.benchmark_tool.measure(self.behaviour_id).a2a_transaction():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()


class AnalyzeApplicationBehaviour(BaseBehaviour):
    """Analyzes a single application."""
    
    matching_round = AnalyzeApplicationRound

    def _get_next_application(self) -> Optional[Dict[str, Any]]:
        """Get the next application to analyze."""
        processed_apps = set(d.get("id") for d in self.synchronized_data.processed_applications)
        apps_to_process = self.synchronized_data.applications_to_process
        
        for app in apps_to_process:
            if app.get("id") not in processed_apps:
                return app
        return None

    def async_act(self) -> Generator:
        """Do the action."""
        
        next_app = self._get_next_application()

        if next_app is None:
            self.context.logger.info("No more applications to analyze.")
            # This case should be handled by the round logic, but as a fallback:
            yield from self.sleep(1.0)
            self.set_done()
            return

        app_id = next_app["id"]
        self.context.logger.info(f"Analyzing application: {app_id}")

        prediction_result = "ERROR" # Default to error

        try:
            # In a real agent, the pipeline would be initialized once and reused.
            # For simplicity here, we re-initialize it.
            pipeline = get_pipeline_logic()
            
            # The pipeline expects a fixed structure for weights and question.
            # We'll use defaults for now.
            # In a real agent, these could come from configuration.
            default_inputs = {
                "application_id": app_id,
                "market_question": "Will this project get accepted into the program?",
                "user_weights": {"founder": 3, "market": 3, "technical": 3, "social": 3, "tokenomics": 3},
            }

            final_state = pipeline.invoke(default_inputs)
            
            if final_state.get("error_message"):
                self.context.logger.error(f"Error analyzing application {app_id}: {final_state['error_message']}")
            else:
                prediction = final_state.get("prediction", {})
                prediction_result = prediction.get("prediction", "ERROR")
                self.context.logger.info(f"Prediction for {app_id}: {prediction_result}")
        
        except Exception as e:
            self.context.logger.error(f"A critical error occurred during analysis of {app_id}: {e}")

        payload = AnalyzeApplicationPayload(
            self.context.agent_address,
            application_id=app_id,
            prediction=prediction_result,
        )

        with self.context.benchmark_tool.measure(self.behaviour_id).a2a_transaction():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()
        
        self.set_done()


class ResetAndPauseBehaviour(BaseBehaviour):
    """Resets and pauses the agent."""

    matching_round = ResetAndPauseRound

    def async_act(self) -> Generator:
        """Do the action."""
        self.context.logger.info("ResetAndPauseBehaviour: Pausing for a moment.")
        yield from self.sleep(10.0) # Pause for 10 seconds
        self.set_done()


class SupafundTraderRoundBehaviour(AbstractRoundBehaviour):
    """Class to define the behaviours this AbciApp has."""

    initial_behaviour_cls = FetchApplicationsBehaviour
    abci_app_cls = SupafundTraderAbciApp
    behaviours: Set[Type[BaseBehaviour]] = {
        FetchApplicationsBehaviour,
        AnalyzeApplicationBehaviour,
        ResetAndPauseBehaviour,
    } 