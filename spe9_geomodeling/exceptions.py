"""
Custom exceptions for PyGeomodeling toolkit.

Provides descriptive error messages and helpful suggestions for common issues.
"""


class PyGeoModelingError(Exception):
    """Base exception for all PyGeomodeling errors."""

    def __init__(self, message: str, suggestion: str = None):
        self.message = message
        self.suggestion = suggestion
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with optional suggestion."""
        msg = f"PyGeomodeling Error: {self.message}"
        if self.suggestion:
            msg += f"\n\nSuggestion: {self.suggestion}"
        return msg


class DataLoadError(PyGeoModelingError):
    """Raised when data loading fails."""

    pass


class DataValidationError(PyGeoModelingError):
    """Raised when data validation fails."""

    pass


class FileFormatError(PyGeoModelingError):
    """Raised when file format is invalid or unsupported."""

    pass


class GridDimensionError(PyGeoModelingError):
    """Raised when grid dimensions are invalid or inconsistent."""

    pass


class PropertyNotFoundError(PyGeoModelingError):
    """Raised when a required property is not found in the dataset."""

    pass


class ModelNotTrainedError(PyGeoModelingError):
    """Raised when attempting to use an untrained model."""

    pass


class ModelNotFoundError(PyGeoModelingError):
    """Raised when a requested model is not found."""

    pass


class BackendNotAvailableError(PyGeoModelingError):
    """Raised when a required backend is not available."""

    pass


class InvalidParameterError(PyGeoModelingError):
    """Raised when invalid parameters are provided."""

    pass


class SerializationError(PyGeoModelingError):
    """Raised when model serialization/deserialization fails."""

    pass


class CrossValidationError(PyGeoModelingError):
    """Raised when cross-validation fails."""

    pass


# Convenience functions for common error scenarios
def raise_file_not_found(filepath: str, file_type: str = "data"):
    """Raise a descriptive file not found error."""
    raise DataLoadError(
        f"The {file_type} file was not found: {filepath}",
        suggestion=(
            f"Please check that:\n"
            f"  1. The file path is correct\n"
            f"  2. The file exists at the specified location\n"
            f"  3. You have read permissions for the file"
        ),
    )


def raise_invalid_format(filepath: str, expected_format: str, details: str = ""):
    """Raise a descriptive invalid format error."""
    msg = f"Invalid file format for: {filepath}\nExpected format: {expected_format}"
    if details:
        msg += f"\nDetails: {details}"

    raise FileFormatError(
        msg,
        suggestion=(
            f"Please ensure the file is a valid {expected_format} file. "
            f"Check the file contents and structure."
        ),
    )


def raise_property_not_found(property_name: str, available_properties: list = None):
    """Raise a descriptive property not found error."""
    msg = f"Property '{property_name}' not found in dataset"
    if available_properties:
        msg += f"\nAvailable properties: {', '.join(available_properties)}"

    raise PropertyNotFoundError(
        msg,
        suggestion=(
            f"Check that:\n"
            f"  1. The property name is spelled correctly\n"
            f"  2. The property exists in your GRDECL file\n"
            f"  3. The property was successfully parsed during data loading"
        ),
    )


def raise_dimension_mismatch(
    expected: tuple, actual: tuple, context: str = "data array"
):
    """Raise a descriptive dimension mismatch error."""
    raise GridDimensionError(
        f"Dimension mismatch in {context}\n"
        f"Expected: {expected}\n"
        f"Actual: {actual}",
        suggestion=(
            "Ensure that:\n"
            "  1. The data array matches the grid dimensions from SPECGRID\n"
            "  2. All property arrays have the same dimensions\n"
            "  3. The data is properly reshaped (Fortran order for reservoir data)"
        ),
    )


def raise_model_not_trained(model_name: str):
    """Raise a descriptive model not trained error."""
    raise ModelNotTrainedError(
        f"Model '{model_name}' has not been trained yet",
        suggestion=(
            f"Train the model first using:\n"
            f"  toolkit.train_sklearn_model(model, '{model_name}')\n"
            f"or:\n"
            f"  toolkit.train_gpytorch_model(model, '{model_name}')"
        ),
    )


def raise_backend_not_available(backend: str, package: str):
    """Raise a descriptive backend not available error."""
    raise BackendNotAvailableError(
        f"Backend '{backend}' is not available because {package} is not installed",
        suggestion=(
            f"Install the required package:\n"
            f"  pip install {package}\n"
            f"or install with all dependencies:\n"
            f"  pip install pygeomodeling[all]"
        ),
    )


def raise_invalid_parameter(param_name: str, param_value, valid_values: list = None):
    """Raise a descriptive invalid parameter error."""
    msg = f"Invalid value for parameter '{param_name}': {param_value}"
    if valid_values:
        msg += f"\nValid values: {', '.join(map(str, valid_values))}"

    raise InvalidParameterError(
        msg,
        suggestion=f"Please provide a valid value for '{param_name}'",
    )
