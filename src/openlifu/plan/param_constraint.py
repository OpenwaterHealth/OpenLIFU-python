from dataclasses import dataclass
from typing import Literal, Tuple

from more_itertools import is_sorted

from openlifu.util.dict_conversion import DictMixin

PARAM_STATUS_SYMBOLS = {
    "ok": "✅",
    "warning": "⚠️",
    "error": "❌"
}

@dataclass
class ParameterConstraint(DictMixin):
    operator: Literal['<','<=','>','>=','within','inside','outside','outside_inclusive']
    warning_value: float|int|Tuple[float|int, float|int]|None = None
    error_value: float|int|Tuple[float|int, float|int]|None = None

    def __post_init__(self):
        if self.warning_value is None and self.error_value is None:
            raise ValueError("At least one of warning_value or error_value must be set")
        if self.operator in ['within','inside','outside','outside_inclusive']:
            if self.warning_value is not None:
                if len(self.warning_value) != 2 or not is_sorted(self.warning_value):
                    raise ValueError("Warning value must be a tuple of two values")
                self.warning_value = tuple(sorted(self.warning_value))
            if self.error_value is not None:
                if len(self.error_value) != 2 or not is_sorted(self.error_value):
                    raise ValueError("Error value must be a tuple of two values")
                self.error_value = tuple(sorted(self.error_value))
        elif self.operator in ['<','<=','>','>=']:
            if self.warning_value is not None and not isinstance(self.warning_value, (int, float)):
                raise ValueError("Warning value must be a single value")
            if self.error_value is not None and not isinstance(self.error_value, (int, float)):
                raise ValueError("Error value must be a single value")

    @staticmethod
    def compare(value, operator, threshold) -> bool:
        if operator == '<':
            return value < threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '>':
            return value > threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == 'within':
            return threshold[0] < value < threshold[1]
        elif operator == 'inside':
            return threshold[0] <= value <= threshold[1]
        elif operator == 'outside':
            return value < threshold[0] or value > threshold[1]
        elif operator == 'outside_inclusive':
            return value <= threshold[0] or value >= threshold[1]

    def is_warning(self, value:float|int) -> bool:
        if self.warning_value is not None:
            return not self.compare(value, self.operator, self.warning_value)
        return False

    def is_error(self, value:float|int) -> bool:
        if self.error_value is not None:
            return not self.compare(value, self.operator, self.error_value)

    def get_status(self, value:float) -> str:
        if self.is_error(value):
            return "error"
        elif self.is_warning(value):
            return "warning"
        else:
            return "ok"

    def get_status_symbol(self, value:float) -> str:
        return PARAM_STATUS_SYMBOLS[self.get_status(value)]
