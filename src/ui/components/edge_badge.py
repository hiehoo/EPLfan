"""Edge badge component for fintech-style UI."""

# Color palette matching the fintech dark theme
EDGE_COLORS = {
    "strong": "#10B981",    # Green - Strong edge (>=10%)
    "moderate": "#F59E0B",  # Gold - Moderate edge (>=7%)
    "marginal": "#FB923C",  # Orange - Marginal edge (>=5%)
    "below": "#64748B",     # Gray - Below threshold
}


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


def edge_badge_html(edge: float) -> str:
    """Return HTML styled badge for edge value.

    Args:
        edge: Edge value as decimal (e.g., 0.10 for 10%)

    Returns:
        HTML string with styled badge
    """
    if edge >= 0.10:
        color = EDGE_COLORS["strong"]
        label = "Strong"
    elif edge >= 0.07:
        color = EDGE_COLORS["moderate"]
        label = "Moderate"
    elif edge >= 0.05:
        color = EDGE_COLORS["marginal"]
        label = "Marginal"
    else:
        color = EDGE_COLORS["below"]
        label = "Below"

    return f"""
    <span style="
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: {color}20;
        color: {color};
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
    ">
        <span style="
            width: 8px;
            height: 8px;
            background: {color};
            border-radius: 50%;
        "></span>
        {edge * 100:.1f}% ({label})
    </span>
    """


def get_edge_color(edge: float) -> str:
    """Get hex color for edge value.

    Args:
        edge: Edge value as decimal

    Returns:
        Hex color string
    """
    if edge >= 0.10:
        return EDGE_COLORS["strong"]
    elif edge >= 0.07:
        return EDGE_COLORS["moderate"]
    elif edge >= 0.05:
        return EDGE_COLORS["marginal"]
    return EDGE_COLORS["below"]


def get_edge_tier(edge: float) -> dict:
    """Get complete edge tier info.

    Args:
        edge: Edge value as decimal

    Returns:
        Dict with color, label, and tier name
    """
    if edge >= 0.10:
        return {"color": EDGE_COLORS["strong"], "label": "Strong", "tier": "strong"}
    elif edge >= 0.07:
        return {"color": EDGE_COLORS["moderate"], "label": "Moderate", "tier": "moderate"}
    elif edge >= 0.05:
        return {"color": EDGE_COLORS["marginal"], "label": "Marginal", "tier": "marginal"}
    return {"color": EDGE_COLORS["below"], "label": "Below", "tier": "below"}
