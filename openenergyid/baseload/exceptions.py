"""Custom exceptions for baseload analysis."""


class InsufficientDataError(Exception):
    """Raised when input data doesn't meet minimum requirements."""


class InvalidDataError(Exception):
    """Raised when input data is invalid or corrupt."""
