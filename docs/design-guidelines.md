# EPL Bet Indicator - Design Guidelines

**Version:** 1.0
**Updated:** January 8, 2026

## Design System Overview

Professional dark mode fintech dashboard optimized for sports betting analytics.

## Color Palette

| Role | Hex | CSS Variable | Usage |
|------|-----|--------------|-------|
| Primary | `#F59E0B` | `--primary` | CTAs, accents, highlights |
| Secondary | `#FBBF24` | `--secondary` | Hover states, gradients |
| CTA | `#8B5CF6` | `--cta` | Action buttons |
| Background | `#0F172A` | `--bg` | Main background |
| Surface | `#1E293B` | `--surface` | Cards, containers |
| Text | `#F8FAFC` | `--text` | Primary text |
| Text Muted | `#94A3B8` | `--text-muted` | Labels, captions |
| Border | `#334155` | `--border` | Dividers, borders |

### Edge Tier Colors

| Tier | Threshold | Hex | Usage |
|------|-----------|-----|-------|
| Strong | >= 10% | `#10B981` | High-value signals |
| Moderate | >= 7% | `#F59E0B` | Medium-value signals |
| Marginal | >= 5% | `#FB923C` | Low-value signals |
| Below | < 5% | `#64748B` | Below threshold |

## Typography

**Font Pairing:** Modern Professional
- **Headings:** Poppins (500-700 weight)
- **Body:** Open Sans (400-600 weight)

```css
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;500;600&family=Poppins:wght@500;600;700&display=swap');
```

## Component Styling

### Metric Cards
- Background: Gradient `#1E293B` to `#0F172A`
- Border: `1px solid #334155`
- Border radius: `10px`
- Hover: Gold border + shadow

### Signal Cards
- Left border: Color-coded by edge tier
- Background: Surface gradient
- Hover: Gold border glow

### Buttons
- Background: Gold gradient (`#F59E0B` to `#D97706`)
- Text: Dark `#0F172A`
- Hover: Lighter gradient + shadow

## Streamlit Config

```toml
[theme]
primaryColor = "#F59E0B"
backgroundColor = "#0F172A"
secondaryBackgroundColor = "#1E293B"
textColor = "#F8FAFC"
font = "sans serif"
```

## File Structure

```
src/ui/
├── dashboard.py          # Main entry + CSS injection
├── components/
│   ├── edge_badge.py     # Edge tier badges (EDGE_COLORS)
│   └── probability_bar.py
└── pages/
    ├── live_signals.py   # Signal cards + metrics
    ├── match_analysis.py
    ├── historical.py
    └── settings_page.py
```

## Best Practices

1. Use `st.markdown(CUSTOM_CSS, unsafe_allow_html=True)` for CSS injection
2. Reference `EDGE_COLORS` from components/edge_badge.py for consistency
3. Apply left border color-coding for edge tiers in cards
4. Use gold accents sparingly for emphasis
5. Maintain WCAG AA contrast ratios
