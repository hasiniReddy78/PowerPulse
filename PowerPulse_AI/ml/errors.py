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


class WarrantyError(PowerPulseError):
    """Base error for battery warranty issues."""


class WarrantyValidationError(WarrantyError):
    """Raised when warranty registration inputs are invalid."""


class WarrantyExpiredError(WarrantyError):
    """Raised when an ML assessment is attempted on an expired battery warranty."""


class WarrantyNotRegisteredError(WarrantyError):
    """Raised when an ML assessment is attempted on an unregistered battery."""
