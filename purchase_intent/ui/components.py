from html import escape

import streamlit as st


class InterfaceComponents:
    ICON_PATHS = {
        "brand": '<path d="M4 18V6m0 12h16"/><path d="m8 13 4-4 4 3 4-7"/><path d="M16 5h4v4"/>',
        "profile": '<circle cx="12" cy="8" r="3.5"/><path d="M5 21v-2a7 7 0 0 1 14 0v2"/>',
    }

    @classmethod
    def icon(cls, name):
        return (
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true">{cls.ICON_PATHS[name]}</svg>'
        )

    @staticmethod
    def render(markup):
        html = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
        st.markdown(html, unsafe_allow_html=True)

    @classmethod
    def render_header(cls):
        cls.render(f"""
            <header class="topbar">
                <div class="brand">
                    <span class="brand-mark">{cls.icon('brand')}</span>
                    <span class="brand-name">intent<span class="brand-period">.</span></span>
                    <span class="brand-divider"></span>
                    <span class="brand-description">Purchase intelligence</span>
                </div>
                <span class="connection-badge"><span class="connection-dot"></span>Model ready</span>
            </header>
            <section class="workspace-intro">
                <div class="eyebrow">YOUR WORKSPACE<span class="eyebrow-divider">/</span>PREDICTIVE ANALYTICS</div>
                <h1>A clearer view of <span class="title-accent">purchase intent.</span></h1>
                <p>Explore purchase likelihood with customer, product, and purchase details.</p>
            </section>
        """)

    @classmethod
    def render_panel_heading(cls, title, description, label=None):
        eyebrow = f'<div class="panel-eyebrow">{escape(label)}</div>' if label else ""
        cls.render(f"""
            <div class="panel-heading">
                {eyebrow}
                <h2>{escape(title)}</h2>
                <p>{escape(description)}</p>
            </div>
        """)

    @classmethod
    def render_form_section(cls, number, title):
        cls.render(f"""
            <div class="form-section">
                <span class="section-number">{escape(number)}</span>
                <h3>{escape(title)}</h3>
                <span class="section-rule"></span>
            </div>
        """)

    @classmethod
    def render_verdict(cls, result):
        if result is None:
            badge = "Awaiting input"
            status = "neutral"
            title = "No prediction yet"
            description = "Complete the customer profile and run a prediction to see the likelihood of purchase."
        else:
            status = "positive" if result.is_likely else "low"
            badge = "Likely to purchase" if result.is_likely else "Less likely to purchase"
            title = "Higher purchase intent" if result.is_likely else "Lower purchase intent"
            position = "above" if result.is_likely else "at or below"
            description = f"This profile falls {position} the decision threshold for a likely purchase."

        cls.render(f"""
            <div class="verdict">
                <span class="decision-badge decision-{status}"><span></span>{badge}</span>
                <h3>{title}</h3>
                <p>{description}</p>
                <div class="threshold-detail"><span>Decision threshold</span><strong>50%</strong></div>
            </div>
        """)

    @classmethod
    def render_profile(cls, result):
        if result is None:
            cls.render(f"""
                <div class="profile-empty">
                    <span class="empty-icon">{cls.icon('profile')}</span>
                    <div><h3>Your profile will appear here</h3><p>Run a prediction to review the details behind your result.</p></div>
                </div>
            """)
            return

        inputs = result.inputs
        fields = (
            ("Customer age", f"{inputs.age} years"),
            ("Gender", inputs.gender),
            ("Category", inputs.category),
            ("Brand", inputs.brand),
            ("Product price", f"${inputs.price:,.2f}"),
            ("Purchase frequency", f"{inputs.frequency:,} purchases"),
        )
        cards = "".join(
            f'<div class="profile-field"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'
            for label, value in fields
        )
        cls.render(f'<div class="profile-grid">{cards}</div>')
        cls.render('<p class="profile-note">Showing the last submitted profile.</p>')

    @classmethod
    def render_feature_legend(cls):
        cls.render("""
            <div class="feature-legend">
                <span><i class="legend-positive"></i>Increases likelihood</span>
                <span><i class="legend-negative"></i>Decreases likelihood</span>
            </div>
        """)

    @classmethod
    def render_footer(cls):
        cls.render("""
            <footer class="workspace-footer">
                <span><strong>intent.</strong> Purchase intelligence workspace</span>
                <span>All prices in USD</span>
            </footer>
        """)
