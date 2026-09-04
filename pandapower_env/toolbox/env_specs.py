from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class Output:
    """
    NamedTuple for storing the output of the environment.

    These are compatible with "normal" tuples, but provide more context.
    The order of the defined variables is important.
    Hence, they can be used in rllib, etc.
    """

    observation: dict = field(default_factory=dict)
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize prev_actions and index_profile always for observation."""
        # Ensure observation is initialized correctly
        self.info["prev_actions"] = []
        self.info["index_profile"] = 0

    def unpack(self) -> tuple:
        """
        Unpacks the tuple into its individual components.

        :return: a tuple of (observation, reward, terminated, truncated, info).
        :rtype: tuple
        """
        return self.observation, self.reward, self.terminated, self.truncated, self.info

    def __iter__(self) -> Iterator:
        """Allow the object to be unpacked like a tuple."""
        return iter(self.unpack())


class LoggedArray:
    """
    A fixed-capacity logging buffer backed by a NumPy array, initialized empty.

    Attributes
    ----------
    data : np object array; for ints + lists
        The underlying NumPy array holding logged values (with unlogged slots as np.nan).
    capacity : int
        Maximum number of values that can be logged.
    _log_idx : int
        Current number of logged values; next insertion index.
    """

    __slots__ = ("data", "_log_idx")

    def __init__(self, capacity: np.integer | int) -> None:
        """
        Initialize the logging buffer with NaNs.

        Parameters
        ----------
        capacity : np.integer | int
            Maximum number of values to store.
        """
        self.data = np.empty(capacity, dtype=object)
        self._log_idx = 0

    @property
    def capacity(self) -> int:
        return len(self.data)

    def append(self, value: int | np.integer | Iterator) -> None:
        """
        Log a new value into the buffer at the next available position.

        Parameters
        ----------
        value : int | np.integer
            The value to store.

        Raises
        ------
        IndexError
            If the buffer is already full.
        TypeError
            If something else than a list or an integer is appended.
        """
        if self._log_idx >= self.capacity:
            msg = "Exceeded capacity for logging"
            raise IndexError(msg)
        if isinstance(value, (int, np.integer)):
            self.data[self._log_idx] = int(value)
        elif isinstance(value, (list, np.ndarray)):
            self.data[self._log_idx] = np.array(value, dtype=int)
        else:
            msg = f"Unsupported input type: {type(value)}"
            raise TypeError(msg)
        self._log_idx += 1

    def to_numpy(self) -> np.ndarray:
        """Return the data as a NumPy array."""
        return self.data[:self._log_idx].copy()

    def __len__(self) -> int:
        """
        Return the number of logged values.

        Returns
        -------
        int
            Count of values logged so far.
        """
        return self._log_idx

    def reset(self) -> None:
        """Clear the buffer, resetting all entries to np.nan and the counter to zero."""
        self.data[:self._log_idx] = [None] * self._log_idx
        self._log_idx = 0

    def __iter__(self) -> Iterator[int]:
        """Allow iteration only over logged (non-NaN) values."""
        return iter(self.data[:self._log_idx])

    def __getitem__(self, key: int | slice) -> float |np.ndarray:
        """
        Access logged data using indexing or slicing.

        Parameters
        ----------
        key : int or slice
            The index or range of indices to access.

        Raises
        ------
        IndexError
            If a value is called that is not present.

        Returns
        -------
        float or np.ndarray
            The logged value(s) at the specified position(s).
        """
        if isinstance(key, int) and (key >= self._log_idx or key < -self._log_idx):
            msg = "Index out of bounds for logged values"
            raise IndexError(msg)
        return self.data[: self._log_idx][key]

    def __deepcopy__(self, memo: dict) -> LoggedArray:
        new_obj = LoggedArray(self.capacity)
        new_obj._log_idx = self._log_idx # noqa: SLF001
        new_obj.data[:self._log_idx] = self.data[:self._log_idx]
        new_obj.data[self._log_idx:] = np.nan
        return new_obj

    def __repr__(self) -> str:
        return f"LoggedArray({self.data})"

    def __str__(self) -> str:
        """User-friendly string representation without None values."""
        return str(self.data[:self._log_idx])
