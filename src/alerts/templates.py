"""Alert message templates."""
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class SignalAlert:
    """Signal alert data structure."""

    match_id: int
    home_team: str
    away_team: str
    kickoff: str
    market: str
    selection: str
    fair_prob: float
    market_prob: float
    edge: float
    weight_profile: str
    confidence: str = "medium"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


# Alert message templates (HTML format for Telegram)
TEMPLATES = {
    "signal_single": """
{edge_emoji} <b>EPL Signal</b> {market_emoji}

<b>{home_team} vs {away_team}</b>
📅 {kickoff}

<b>Market:</b> {market}
<b>Selection:</b> {selection}

<b>Fair Price:</b> {fair_prob:.1%}
<b>Polymarket:</b> {market_prob:.1%}
<b>Edge:</b> <code>{edge:.1%}</code>

<b>Profile:</b> {weight_profile}
<b>Confidence:</b> {confidence}

⚠️ <i>Indicator only - verify before trading</i>
""",
    "signal_batch_header": """🚨 <b>EPL Signals Update</b>
📊 {count} signal(s) found

""",
    "signal_batch_line": "{edge_emoji} <b>{home_abbr} v {away_abbr}</b> | {market} {selection} | {edge:.1%}",
    "daily_summary": """📊 <b>Daily Summary</b>

<b>Signals Generated:</b> {total_signals}
<b>Markets:</b>
  • 1X2: {count_1x2}
  • O/U: {count_ou}
  • BTTS: {count_btts}

<b>Average Edge:</b> {avg_edge:.1%}

<b>Matches Tomorrow:</b> {matches_tomorrow}
""",
    "error": """⚠️ <b>System Alert</b>

{error_type}: {error_message}

<i>Timestamp: {timestamp}</i>
""",
    "startup": """🟢 <b>EPL Bet Indicator Started</b>

Scan interval: {scan_interval} min
Thresholds:
  • 1X2: {threshold_1x2:.0f}%
  • O/U: {threshold_ou:.0f}%
  • BTTS: {threshold_btts:.0f}%

Monitoring active.
""",
    "shutdown": """🔴 <b>EPL Bet Indicator Stopped</b>

Shutdown time: {timestamp}
""",
    "no_signals": """📭 <b>Scan Complete</b>

No signals above threshold.
Next scan in {next_scan_minutes} minutes.
""",
}


def get_edge_emoji(edge: float) -> str:
    """Return emoji based on edge strength."""
    if edge >= 0.10:
        return "🟢"
    elif edge >= 0.07:
        return "🟡"
    return "🟠"


def get_market_emoji(market: str) -> str:
    """Return emoji based on market type."""
    market_lower = market.lower()
    if "1x2" in market_lower:
        return "⚽"
    elif "o/u" in market_lower or "ou" in market_lower or "over" in market_lower or "under" in market_lower:
        return "📈"
    elif "btts" in market_lower:
        return "🎯"
    return "📊"


def format_template(template_name: str, **kwargs) -> str:
    """Format a template with provided values.

    Args:
        template_name: Name of template in TEMPLATES dict
        **kwargs: Values to substitute into template

    Returns:
        Formatted message string
    """
    template = TEMPLATES.get(template_name, "")
    if not template:
        return f"Unknown template: {template_name}"

    try:
        return template.format(**kwargs).strip()
    except KeyError as e:
        return f"Missing template key: {e}"


def format_signal(signal: SignalAlert) -> str:
    """Format a single signal alert.

    Args:
        signal: SignalAlert object

    Returns:
        Formatted HTML message
    """
    return format_template(
        "signal_single",
        edge_emoji=get_edge_emoji(signal.edge),
        market_emoji=get_market_emoji(signal.market),
        home_team=signal.home_team,
        away_team=signal.away_team,
        kickoff=signal.kickoff,
        market=signal.market,
        selection=signal.selection,
        fair_prob=signal.fair_prob,
        market_prob=signal.market_prob,
        edge=signal.edge,
        weight_profile=signal.weight_profile,
        confidence=signal.confidence,
    )


def format_batch_signals(signals: list[SignalAlert]) -> str:
    """Format multiple signals as a summary message.

    Args:
        signals: List of SignalAlert objects

    Returns:
        Formatted HTML message
    """
    if not signals:
        return "No signals to display"

    header = format_template("signal_batch_header", count=len(signals))

    lines = []
    for s in sorted(signals, key=lambda x: x.edge, reverse=True):
        line = format_template(
            "signal_batch_line",
            edge_emoji=get_edge_emoji(s.edge),
            home_abbr=s.home_team[:3].upper(),
            away_abbr=s.away_team[:3].upper(),
            market=s.market,
            selection=s.selection,
            edge=s.edge,
        )
        lines.append(line)

    return header + "\n".join(lines)
