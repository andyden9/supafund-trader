from packages.valory.skills.abstract_round_abci.base import BaseTxPayload
from typing import Any, Dict

# In the future, we can define specific payload classes for each round
# to transfer data between agents. For example:
#
# class FetchApplicationsPayload(BaseTxPayload):
#     ...
#
# class AnalyzeApplicationPayload(BaseTxPayload):
#     ...

class FetchApplicationsPayload(BaseTxPayload):
    """Represents a transaction payload for the FetchApplicationsRound."""

    transaction_type = "fetch_applications"

    def __init__(self, sender: str, applications_json: str, **kwargs: Any) -> None:
        """Initialize a 'fetch_applications' transaction payload.

        :param sender: the sender's address.
        :param applications_json: the json-serialized list of applications.
        :param kwargs: the keyword arguments.
        """
        super().__init__(sender, **kwargs)
        self._applications_json = applications_json

    @property
    def applications_json(self) -> str:
        """Get the json-serialized applications."""
        return self._applications_json

    @property
    def data(self) -> Dict:
        """Get the data."""
        return dict(applications_json=self.applications_json)

class AnalyzeApplicationPayload(BaseTxPayload):
    """Represents a transaction payload for the AnalyzeApplicationRound."""

    transaction_type = "analyze_application"

    def __init__(self, sender: str, application_id: str, prediction: str, **kwargs: Any) -> None:
        """Initialize a 'analyze_application' transaction payload.

        :param sender: the sender's address.
        :param application_id: the id of the application that was analyzed.
        :param prediction: the prediction result ("YES", "NO", "ERROR").
        :param kwargs: the keyword arguments.
        """
        super().__init__(sender, **kwargs)
        self._application_id = application_id
        self._prediction = prediction

    @property
    def application_id(self) -> str:
        """Get the application id."""
        return self._application_id

    @property
    def prediction(self) -> str:
        """Get the prediction."""
        return self._prediction

    @property
    def data(self) -> Dict:
        """Get the data."""
        return dict(
            application_id=self.application_id,
            prediction=self.prediction,
        )

# For now, we don't have any custom payloads. 