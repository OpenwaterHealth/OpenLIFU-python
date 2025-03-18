from __future__ import annotations


class LIFUSignal:
    def __init__(self):
        # Initialize a list to store connected slots (callback functions)
        self._slots = []

    def connect(self, slot):
        """
        Connect a slot (callback function) to the signal.

        Args:
            slot (callable): A callable to be invoked when the signal is emitted.
        """
        if callable(slot) and slot not in self._slots:
            self._slots.append(slot)

    def disconnect(self, slot):
        """
        Disconnect a slot (callback function) from the signal.

        Args:
            slot (callable): The callable to disconnect.
        """
        if slot in self._slots:
            self._slots.remove(slot)

    def emit(self, *args, **kwargs):
        """
        Emit the signal, invoking all connected slots.

        Args:
            *args: Positional arguments to pass to the connected slots.
            **kwargs: Keyword arguments to pass to the connected slots.
        """
        for slot in self._slots:
            slot(*args, **kwargs)
