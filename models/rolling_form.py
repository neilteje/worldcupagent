class RollingFormBuilder:
    """Builds short and long window exponentially weighted xG form."""
    def __init__(self, short_half_life_days: float = 75.0, long_half_life_days: float = 240.0):
        self.short_half_life = short_half_life_days
        self.long_half_life = long_half_life_days
        # ... logic for building form ...
