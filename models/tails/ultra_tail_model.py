import config

class UltraTailModel:
    """Specialized ultra tail model for HUNTER (<5c)."""
    def __init__(self):
        pass

    def is_valid(self, price: float) -> bool:
        if price >= 0.05:
            return True # Let regular models handle it
        return config.HUNTER_ULTRA_TAIL_VALIDATION_ENABLED
