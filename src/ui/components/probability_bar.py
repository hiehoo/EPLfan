"""Probability bar visualization."""
import html
import streamlit as st


def _escape(text: str) -> str:
    """Escape HTML special characters for safe rendering."""
    return html.escape(str(text))


def probability_bar(probs: dict[str, float], labels: list[str]) -> None:
    """Render horizontal stacked probability bar.

    Args:
        probs: Dictionary of probabilities keyed by lowercase label
        labels: List of labels to display (e.g., ["Home", "Draw", "Away"])
    """
    colors = ["#28a745", "#6c757d", "#dc3545"]  # Green, Gray, Red

    html_out = '<div style="display: flex; width: 100%; height: 30px; border-radius: 5px; overflow: hidden;">'
    for idx, label in enumerate(labels):
        prob = probs.get(label.lower(), 0)
        color = colors[idx % len(colors)]
        safe_label = _escape(label)
        html_out += f'<div style="width: {prob * 100}%; background: {color}; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px;">'
        if prob > 0.1:
            html_out += f"{safe_label}: {prob * 100:.0f}%"
        html_out += "</div>"
    html_out += "</div>"

    st.markdown(html_out, unsafe_allow_html=True)


def probability_bar_dual(probs: dict[str, float], labels: list[str]) -> None:
    """Render dual probability bar (for binary markets like O/U, BTTS).

    Args:
        probs: Dictionary with two probabilities
        labels: List of two labels (e.g., ["Over", "Under"])
    """
    colors = ["#28a745", "#dc3545"]  # Green, Red

    html_out = '<div style="display: flex; width: 100%; height: 30px; border-radius: 5px; overflow: hidden;">'
    for idx, label in enumerate(labels[:2]):
        prob = probs.get(label.lower(), 0)
        color = colors[idx]
        safe_label = _escape(label)
        html_out += f'<div style="width: {prob * 100}%; background: {color}; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px;">'
        html_out += f"{safe_label}: {prob * 100:.0f}%"
        html_out += "</div>"
    html_out += "</div>"

    st.markdown(html_out, unsafe_allow_html=True)
