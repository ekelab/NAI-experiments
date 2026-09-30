"""
Keep the machine awake while a benchmark runs. On a laptop, Windows' Modern
Standby can suspend a run mid-search while NAI's clock keeps going, so that a
search "spends its budget" asleep and stops early. SetThreadExecutionState
asks the system not to sleep for idleness while this process lives; nothing
in the user's power settings is changed, and closing the lid still sleeps.
"""
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake():
    """Returns True if the request was accepted (Windows only)."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        return ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) != 0
    except (AttributeError, OSError):
        return False
