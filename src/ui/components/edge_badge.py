"""Edge badge component."""


def edge_badge(edge: float) -> str:
    """Return colored badge for edge value.

    Args:
        edge: Edge value as decimal (e.g., 0.10 for 10%)

    Returns:
        Formatted string with emoji and edge percentage
    """
    if edge >= 0.10:
        return f"🟢 **{edge * 100:.1f}%** (Strong)"
    elif edge >= 0.07:
        return f"🟡 **{edge * 100:.1f}%** (Moderate)"
    elif edge >= 0.05:
        return f"🟠 **{edge * 100:.1f}%** (Marginal)"
    else:
        return f"⚪ **{edge * 100:.1f}%** (Below threshold)"
