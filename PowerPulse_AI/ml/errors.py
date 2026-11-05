"""Application errors that are safe to show in the interface."""


class PowerPulseError(Exception):
    """Base error for predictable PowerPulse failures."""


class ModelNotFoundError(PowerPulseError):
    """Raised when a required model artifact is missing or unreadable."""


class InvalidMeasurementError(PowerPulseError):
    """Raised when measurements are incomplete or physically implausible."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))
