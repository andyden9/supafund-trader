import json
from enum import Enum
from typing import Dict, Optional, Tuple, List, Any, cast

from packages.valory.skills.abstract_round_abci.base import (
    AbciApp,
    AbciAppTransitionFunction,
    AbstractRound,
    BaseSynchronizedData,
    DegenerateRound,
)
from packages.valory.skills.supafund_trader_abci.payloads import FetchApplicationsPayload, AnalyzeApplicationPayload

class Event(Enum):
    """SupafundTraderAbciApp Events"""
    DONE = "done"
    ROUND_TIMEOUT = "round_timeout"
    NO_MAJORITY = "no_majority"
    ERROR = "error"
    NO_APPLICATIONS = "no_applications"
    NO_MORE_APPLICATIONS = "no_more_applications"

class SynchronizedData(BaseSynchronizedData):
    """
    Class to represent the synchronized data.

    This data is replicated across all agents.
    """
    @property
    def applications_to_process(self) -> list:
        """Get the applications to process."""
        return cast(list, self.db.get_strict("applications_to_process"))

    @property
    def processed_applications(self) -> list:
        """Get the processed applications."""
        return cast(list, self.db.get("processed_applications", []))

    @property
    def analysis_results(self) -> list:
        """Get the analysis results."""
        return cast(list, self.db.get("analysis_results", []))

class RegistrationRound(DegenerateRound):
    """A round in which the agents registration takes place"""
    
    payload_class = None # No payload for this round
    synchronized_data_class = SynchronizedData

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        return self.synchronized_data, Event.DONE

    def check_payload(self, payload: Any) -> None:
        """Check payload."""

    def process_payload(self, payload: Any) -> None:
        """Process payload."""

class FetchApplicationsRound(AbstractRound):
    """A round in which agents fetch applications"""

    payload_class = FetchApplicationsPayload
    synchronized_data_class = SynchronizedData

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        if self.threshold_reached:
            # For now, we'll just use the payload from the first agent.
            # In a real implementation, we would need a more robust consensus mechanism.
            first_payload = self.most_voted_payload_class(self.collection, self.payload_class)
            
            apps_json = first_payload.applications_json
            apps = json.loads(apps_json)

            if not apps:
                return self.synchronized_data, Event.NO_APPLICATIONS

            synchronized_data = self.synchronized_data.update(
                synchronized_data_class=self.synchronized_data_class,
                **{
                    "applications_to_process": apps,
                    "processed_applications": [], # Reset processed list
                }
            )
            return synchronized_data, Event.DONE
        
        return self.synchronized_data, Event.NO_MAJORITY

    def check_payload(self, payload: FetchApplicationsPayload) -> None:
        """Check payload."""
        try:
            # Ensure the content is a valid JSON list.
            json.loads(payload.applications_json)
        except (json.JSONDecodeError, TypeError):
            # To be implemented: proper validation
            pass

    def process_payload(self, payload: FetchApplicationsPayload) -> None:
        """Process payload."""
        super().process_payload(payload)

class AnalyzeApplicationRound(AbstractRound):
    """A round in which agents analyze an application"""

    payload_class = AnalyzeApplicationPayload
    synchronized_data_class = SynchronizedData
    
    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        if self.threshold_reached:
            # Consensus reached on the analysis of one application
            payload = self.most_voted_payload_class(self.collection, self.payload_class)
            
            # Update processed applications and results
            processed_apps = self.synchronized_data.processed_applications
            analysis_results = self.synchronized_data.analysis_results
            
            updated_processed = processed_apps + [{"id": payload.application_id}]
            updated_results = analysis_results + [{payload.application_id: payload.prediction}]

            synchronized_data = self.synchronized_data.update(
                synchronized_data_class=self.synchronized_data_class,
                **{
                    "processed_applications": updated_processed,
                    "analysis_results": updated_results,
                }
            )

            # Check if there are more applications to process
            apps_to_process = self.synchronized_data.applications_to_process
            if len(updated_processed) >= len(apps_to_process):
                return synchronized_data, Event.NO_MORE_APPLICATIONS

            return synchronized_data, Event.DONE

        return self.synchronized_data, Event.NO_MAJORITY

    def check_payload(self, payload: AnalyzeApplicationPayload) -> None:
        """Check payload."""
        # Add validation for prediction format if necessary
        is_valid_prediction = payload.prediction in {"YES", "NO", "ERROR"}
        if not is_valid_prediction:
            # handle invalid payload
            pass

    def process_payload(self, payload: AnalyzeApplicationPayload) -> None:
        """Process payload."""
        super().process_payload(payload)

class ResetAndPauseRound(DegenerateRound):
    """A round that represents a pause before restarting the cycle."""


class SupafundTraderAbciApp(AbciApp[Event]):
    """SupafundTraderAbciApp"""

    initial_round_cls: AbstractRound = RegistrationRound
    transition_function: AbciAppTransitionFunction = {
        RegistrationRound: {
            Event.DONE: FetchApplicationsRound,
            Event.ROUND_TIMEOUT: RegistrationRound,
            Event.NO_MAJORITY: RegistrationRound,
        },
        FetchApplicationsRound: {
            Event.DONE: AnalyzeApplicationRound,
            Event.NO_APPLICATIONS: ResetAndPauseRound,
            Event.ERROR: ResetAndPauseRound,
            Event.ROUND_TIMEOUT: ResetAndPauseRound,
            Event.NO_MAJORITY: ResetAndPauseRound,
        },
        AnalyzeApplicationRound: {
            Event.DONE: AnalyzeApplicationRound, # Loop to process all apps
            Event.NO_MORE_APPLICATIONS: ResetAndPauseRound,
            Event.ERROR: ResetAndPauseRound,
            Event.ROUND_TIMEOUT: ResetAndPauseRound,
            Event.NO_MAJORITY: ResetAndPauseRound,
        },
        ResetAndPauseRound: {
            Event.DONE: FetchApplicationsRound,
            Event.ROUND_TIMEOUT: ResetAndPauseRound,
        },
    }
    final_states = frozenset()
    db_pre_conditions = {RegistrationRound: frozenset()}
    db_post_conditions = {
        ResetAndPauseRound: frozenset(),
    }
    event_to_timeout: Dict[Event, float] = {
        Event.ROUND_TIMEOUT: 30.0,
    } 