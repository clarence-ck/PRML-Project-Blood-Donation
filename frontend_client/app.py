"""Blood Donor Outreach Console - Gradio Frontend.

A modern, user-friendly interface for outreach notification centre staff to score
donor return likelihood using the deployed ML model.
"""

from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import httpx
import gradio as gr
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup for process_data imports
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process_data import REFERENCE_DATE, EDUCATION_MAPPING

# ---------------------------------------------------------------------------
# Feature schema (must match app/main.py EXPECTED_FEATURES)
# ---------------------------------------------------------------------------
EXPECTED_FEATURES = [
    "age",
    "days_since_last_donation",
    "donor_tenure_days",
    "mean_donation_interval_days",
    "last_donation_volume_ml",
    "avg_donation_volume",
    "donation_count_outram",
    "donation_count_dhoby_ghaut",
    "donation_count_woodlands",
    "donation_count_jurong_east",
    "donation_count_punggol",
    "recency_ratio",
    "volume_consistency",
    "primary_location_pct",
    "donations_per_year",
    "risk_recency",
    "education_level_encoded",
    "gender_Male",
    "had_adverse_reaction_ever",
    "last_blood_quality_screen",
    "blood_group_A-",
    "blood_group_AB+",
    "blood_group_AB-",
    "blood_group_B+",
    "blood_group_B-",
    "blood_group_O+",
    "blood_group_O-",
]

# Blood group one-hot columns (drop_first=True means A+ is baseline)
BLOOD_GROUP_DUMMIES = [
    "blood_group_A-",
    "blood_group_AB+",
    "blood_group_AB-",
    "blood_group_B+",
    "blood_group_B-",
    "blood_group_O+",
    "blood_group_O-",
]


def _default_form_values() -> Dict[str, object]:
    """Return the default values for all form inputs."""

    return {
        # Default age (years)
        "age": 30.0,
        "gender": "Female",
        "education_level": "University",
        "blood_group": "O+",
        "first_donation_date": datetime.datetime(2020, 2, 10),
        "last_donation_date": datetime.datetime(2025, 5, 1),
        "last_donation_volume_ml": 420,
        "donation_count_outram": 1,
        "donation_count_dhoby_ghaut": 0,
        "donation_count_woodlands": 0,
        "donation_count_jurong_east": 0,
        "donation_count_punggol": 0,
        "had_adverse_reaction_ever": False,
        "last_blood_quality_screen": "Pass",
    }


def reset_form() -> Tuple[object, ...]:
    """Provide default values for all inputs and clear outputs."""

    defaults = _default_form_values()
    return (
        defaults["age"],
        defaults["gender"],
        defaults["education_level"],
        defaults["blood_group"],
        defaults["first_donation_date"],
        defaults["last_donation_date"],
        defaults["last_donation_volume_ml"],
        defaults["donation_count_outram"],
        defaults["donation_count_dhoby_ghaut"],
        defaults["donation_count_woodlands"],
        defaults["donation_count_jurong_east"],
        defaults["donation_count_punggol"],
        defaults["had_adverse_reaction_ever"],
        defaults["last_blood_quality_screen"],
        "",
        "",
        "",
        "",
        "",
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _normalize_base_url(value: str) -> str:
    return value.strip().rstrip("/")


def _get_api_base_url() -> str:
    base = _normalize_base_url(os.getenv("API_BASE_URL", ""))
    if not base:
        raise gr.Error(
            "API_BASE_URL environment variable is not set. "
            "Configure it with the Terraform output 'http_api_invoke_url'."
        )
    return base


def _build_features_directly(
    age: float,
    gender: str,
    education_level: str,
    blood_group: str,
    first_donation_date: str,
    last_donation_date: str,
    last_donation_volume_ml: float,
    donation_count_outram: float,
    donation_count_dhoby_ghaut: float,
    donation_count_woodlands: float,
    donation_count_jurong_east: float,
    donation_count_punggol: float,
    had_adverse_reaction_ever: bool,
    last_blood_quality_screen: str,
) -> Dict[str, float]:
    """Build the 27 model features directly from form inputs.
    
    This avoids pandas get_dummies issues with single-row inputs by
    computing each feature explicitly.
    """
    # Parse dates
    first_date = pd.to_datetime(first_donation_date)
    last_date = pd.to_datetime(last_donation_date)
    ref_date = REFERENCE_DATE

    # Basic temporal validation: first <= last <= REFERENCE_DATE
    if first_date > last_date:
        raise gr.Error(
            "First donation date must be on or before the last donation date. "
            "Please adjust the dates in the 'Donation Timeline' section."
        )
    if last_date > ref_date:
        raise gr.Error(
            f"Last donation date cannot be after the modelling reference date {ref_date.date().isoformat()}. "
            "Please adjust the last donation date."
        )

    # Temporal features
    age = float(age)
    if age < 18:
        raise gr.Error(
            "Donor must be at least 18 years old as of 2025-11-30. "
            "Please enter an age of at least 18 years.",
        )
    days_since_last_donation = (ref_date - last_date).days
    donor_tenure_days = (last_date - first_date).days

    # Location features
    location_counts = [
        donation_count_outram,
        donation_count_dhoby_ghaut,
        donation_count_woodlands,
        donation_count_jurong_east,
        donation_count_punggol,
    ]
    total_location = sum(location_counts)

    # Derive total number of donations from sum of location counts
    if total_location <= 0:
        raise gr.Error(
            "Total number of donations across all locations must be at least 1. "
            "Please set the per-location counts to sum to one or more donations."
        )

    n_donations = int(total_location)

    # If there is exactly one donation overall, enforce first == last date
    if n_donations == 1 and first_date.date() != last_date.date():
        raise gr.Error(
            "For a single donation, the first and last donation dates must be the same. "
            "Either adjust the dates to match or update the location counts."
        )

    # Interval and frequency
    interval_denom = max(1, n_donations - 1)
    mean_donation_interval_days = (
        donor_tenure_days / interval_denom if donor_tenure_days > 0 else 0.0
    )

    # Volume features
    # In Dataset.py, total volume is generated from an average per-donation
    # volume between 350 and 450 ml. We mirror that behaviour by estimating
    # an average close to the last donation volume but anchored around 400 ml.
    if n_donations == 1:
        avg_donation_volume = float(last_donation_volume_ml)
    else:
        approx_avg = (last_donation_volume_ml + 400.0) / 2.0
        avg_donation_volume = float(approx_avg)

    volume_consistency = (
        last_donation_volume_ml / avg_donation_volume if avg_donation_volume > 0 else 0.0
    )

    # Recency ratio
    recency_ratio = (
        days_since_last_donation / donor_tenure_days if donor_tenure_days > 0 else 0.0
    )
    primary_location_pct = (max(location_counts) / total_location) if total_location > 0 else 0.0

    # Donations per year
    tenure_years = donor_tenure_days / 365.25
    donations_per_year = (n_donations / tenure_years) if tenure_years > 0 else 0.0

    # Risk recency
    adverse_int = 1 if had_adverse_reaction_ever else 0
    risk_recency = adverse_int * days_since_last_donation

    # Encodings
    try:
        education_encoded = EDUCATION_MAPPING[education_level]
    except KeyError as exc:
        raise gr.Error(
            f"Unsupported education level '{education_level}'. Please choose from the dropdown options."
        ) from exc
    gender_male = 1.0 if gender == "Male" else 0.0
    quality_screen = 1.0 if last_blood_quality_screen == "Pass" else 0.0

    # Blood group one-hot (A+ is baseline, so all zeros if A+)
    blood_dummies = {col: 0.0 for col in BLOOD_GROUP_DUMMIES}
    bg_col = f"blood_group_{blood_group}"
    if bg_col in blood_dummies:
        blood_dummies[bg_col] = 1.0

    # Build final feature dict
    features: Dict[str, float] = {
        "age": float(age),
        "days_since_last_donation": float(days_since_last_donation),
        "donor_tenure_days": float(donor_tenure_days),
        "mean_donation_interval_days": float(mean_donation_interval_days),
        "last_donation_volume_ml": float(last_donation_volume_ml),
        "avg_donation_volume": float(avg_donation_volume),
        "donation_count_outram": float(donation_count_outram),
        "donation_count_dhoby_ghaut": float(donation_count_dhoby_ghaut),
        "donation_count_woodlands": float(donation_count_woodlands),
        "donation_count_jurong_east": float(donation_count_jurong_east),
        "donation_count_punggol": float(donation_count_punggol),
        "recency_ratio": float(recency_ratio),
        "volume_consistency": float(volume_consistency),
        "primary_location_pct": float(primary_location_pct),
        "donations_per_year": float(donations_per_year),
        "risk_recency": float(risk_recency),
        "education_level_encoded": float(education_encoded),
        "gender_Male": gender_male,
        "had_adverse_reaction_ever": float(adverse_int),
        "last_blood_quality_screen": quality_screen,
        **blood_dummies,
    }

    return features


def _priority_from_probability(prob: float) -> Tuple[str, str, str]:
    """Return (priority_label, guidance_text, css_class) based on probability."""
    if prob >= 0.7:
        return (
            "🟢 HIGH PRIORITY",
            "Call this donor early in the session. High likelihood of returning. "
            "Confirm preferred location and timing.",
            "prob-high",
        )
    if prob >= 0.4:
        return (
            "🟡 MEDIUM PRIORITY",
            "Consider a reminder call or SMS. Ask about any concerns or "
            "scheduling constraints.",
            "prob-medium",
        )
    return (
        "🔴 LOW PRIORITY",
        "Lower immediate priority. Batch outreach with other low-priority donors. "
        "Consider re-engagement campaigns.",
        "prob-low",
    )


# ---------------------------------------------------------------------------
# API interaction
# ---------------------------------------------------------------------------
def check_health() -> str:
    """Check API health endpoint."""
    base = _get_api_base_url()
    url = f"{base}/health"

    try:
        response = httpx.get(url, timeout=5.0)
    except httpx.RequestError as exc:
        return f"❌ Connection error: {exc}"

    if response.status_code != 200:
        return f"❌ Health check failed (status {response.status_code})"

    return f"✅ API is healthy: {response.json()}"


def score_donor_form(
    age: float,
    gender: str,
    education_level: str,
    blood_group: str,
    first_donation_date: str,
    last_donation_date: str,
    last_donation_volume_ml: float,
    donation_count_outram: float,
    donation_count_dhoby_ghaut: float,
    donation_count_woodlands: float,
    donation_count_jurong_east: float,
    donation_count_punggol: float,
    had_adverse_reaction_ever: bool,
    last_blood_quality_screen: str,
) -> Tuple[str, str, str, str]:
    """Score a donor and return formatted results."""
    
    # Quick validation on locations + dates before feature engineering
    first_date = pd.to_datetime(first_donation_date)
    last_date = pd.to_datetime(last_donation_date)
    location_counts = [
        donation_count_outram,
        donation_count_dhoby_ghaut,
        donation_count_woodlands,
        donation_count_jurong_east,
        donation_count_punggol,
    ]
    total_location = sum(location_counts)

    if total_location <= 0:
        raise gr.Error(
            "Total number of donations across all locations must be at least 1. "
            "Please set the per-location counts to sum to one or more donations."
        )

    if int(total_location) == 1 and first_date.date() != last_date.date():
        raise gr.Error(
            "For a single donation, the first and last donation dates must be the same. "
            "Either adjust the dates to match or update the location counts."
        )

    # Build features directly (no pandas dummies issues)
    features = _build_features_directly(
        age,
        gender,
        education_level,
        blood_group,
        first_donation_date,
        last_donation_date,
        last_donation_volume_ml,
        donation_count_outram,
        donation_count_dhoby_ghaut,
        donation_count_woodlands,
        donation_count_jurong_east,
        donation_count_punggol,
        had_adverse_reaction_ever,
        last_blood_quality_screen,
    )

    # Call API
    base = _get_api_base_url()
    url = f"{base}/predict"

    try:
        response = httpx.post(url, json={"features": features}, timeout=15.0)
    except httpx.RequestError as exc:
        raise gr.Error(f"Error contacting API: {exc}") from exc

    if response.status_code != 200:
        raise gr.Error(f"API error {response.status_code}: {response.text}")

    data = response.json()
    prob = float(data.get("probability", 0.0))
    pred = int(data.get("prediction", 0))

    # Format results
    label = "✅ Likely to return" if pred == 1 else "⚠️ Unlikely to return soon"
    priority, guidance, _ = _priority_from_probability(prob)
    prob_display = f"{prob * 100:.1f}%"

    return prob_display, label, priority, guidance


# ---------------------------------------------------------------------------
# UI Builder
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    """Build the Gradio Blocks app with elevated aesthetics."""
    
    defaults = _default_form_values()

    theme = gr.themes.Soft(
        primary_hue="red",      # primary CTAs in brand red
        secondary_hue="amber",  # warm amber for secondary elements
        neutral_hue="amber",    # warm base background like the banner
    ).set(
        body_background_fill="#FFEFD5",   # warm papaya tone instead of white
        block_background_fill="#FFF3E0",  # soft cream for panels
    )

    with gr.Blocks(
        title="Blood Donor Outreach Console",
        theme=theme,
    ) as demo:

        # Header image
        gr.Image(
            value="frontend_client/images/Blood_bank_SG_banner.png",
            show_label=False,
            interactive=False,
            width=960,
            show_download_button=False,
        )

        # Title and subtitle below the image
        gr.Markdown(
            """
            <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
                <h1 style="color: #8b0000; margin-bottom: 0.25rem;">🩸 Blood Donor Outreach Console</h1>
                <p style="color: #004b8d; font-size: 1.05rem;">
                    Score donor return likelihood and get outreach notification centre guidance
                </p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            # Left column: Input form
            with gr.Column(scale=3):
                gr.Markdown(
                    """<div class="section-header">📋 Donor Information</div>""",
                    elem_classes=["section-header"],
                )

                with gr.Accordion("👤 Demographics", open=True):
                    with gr.Row():
                        age = gr.Slider(
                            minimum=18,
                            maximum=80,
                            value=defaults["age"],
                            step=1,
                            label="Age (years)",
                        )
                        gender = gr.Radio(
                            choices=["Female", "Male"],
                            value=defaults["gender"],
                            label="Gender",
                        )
                    with gr.Row():
                        education_level = gr.Dropdown(
                            choices=["Primary", "Secondary", "Diploma", "University", "Postgraduate"],
                            value=defaults["education_level"],
                            label="Education Level",
                        )
                        blood_group = gr.Dropdown(
                            choices=["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"],
                            value=defaults["blood_group"],
                            label="Blood Group",
                        )

                with gr.Accordion("📅 Donation Timeline", open=True):
                    with gr.Row():
                        first_donation_date = gr.DateTime(
                            label="First Donation",
                            value=defaults["first_donation_date"],
                            include_time=False,
                            info="Date of first ever donation",
                        )
                        last_donation_date = gr.DateTime(
                            label="Last Donation",
                            value=defaults["last_donation_date"],
                            include_time=False,
                            info="Most recent donation date",
                        )

                with gr.Accordion("📊 Donation Statistics", open=True):
                    last_donation_volume_ml = gr.Slider(
                        minimum=350,
                        maximum=450,
                        value=defaults["last_donation_volume_ml"],
                        step=5,
                        label="Last Donation Volume (ml)",
                        info="Typical single-donation volume (ml)",
                    )

                with gr.Accordion("📍 Donation Locations", open=False):
                    gr.Markdown(
                        "*Enter the number of donations at each centre; their sum will be treated as the total number of donations.*"
                    )
                    with gr.Row():
                        donation_count_outram = gr.Number(
                            label="Outram",
                            value=defaults["donation_count_outram"],
                            minimum=0,
                            maximum=20,
                        )
                        donation_count_dhoby_ghaut = gr.Number(
                            label="Dhoby Ghaut",
                            value=defaults["donation_count_dhoby_ghaut"],
                            minimum=0,
                            maximum=20,
                        )
                        donation_count_woodlands = gr.Number(
                            label="Woodlands",
                            value=defaults["donation_count_woodlands"],
                            minimum=0,
                            maximum=20,
                        )
                    with gr.Row():
                        donation_count_jurong_east = gr.Number(
                            label="Jurong East",
                            value=defaults["donation_count_jurong_east"],
                            minimum=0,
                            maximum=20,
                        )
                        donation_count_punggol = gr.Number(
                            label="Punggol",
                            value=defaults["donation_count_punggol"],
                            minimum=0,
                            maximum=20,
                        )

                with gr.Accordion("⚠️ Health & Safety", open=True):
                    with gr.Row():
                        had_adverse_reaction_ever = gr.Checkbox(
                            label="Has had adverse reaction?",
                            value=defaults["had_adverse_reaction_ever"],
                            info="Check if donor ever experienced an adverse reaction",
                        )
                        last_blood_quality_screen = gr.Radio(
                            choices=["Pass", "Fail"],
                            value=defaults["last_blood_quality_screen"],
                            label="Last Quality Screen",
                        )

                with gr.Row():
                    submit_btn = gr.Button(
                        "🔍 Analyze Donor",
                        variant="primary",
                        size="lg",
                    )
                    reset_btn = gr.Button(
                        "↺ Reset Form",
                        variant="secondary",
                    )

            # Right column: Results
            with gr.Column(scale=2):
                gr.Markdown(
                    """<div class="section-header">📈 Prediction Results</div>""",
                    elem_classes=["section-header"],
                )

                prob_out = gr.Textbox(
                    label="Return Probability",
                    interactive=False,
                    elem_classes=["result-display"],
                )
                label_out = gr.Textbox(
                    label="Model Decision",
                    interactive=False,
                )
                priority_out = gr.Textbox(
                    label="Call Priority",
                    interactive=False,
                )
                guidance_out = gr.Textbox(
                    label="Recommended Action",
                    lines=4,
                    interactive=False,
                )

                gr.Markdown("---")
                gr.Markdown("### 🔧 System Status")
                with gr.Row():
                    health_btn = gr.Button("Check API Health", size="sm")
                health_out = gr.Textbox(
                    label="API Status",
                    interactive=False,
                    show_label=False,
                )

        # Event handlers
        submit_btn.click(
            fn=score_donor_form,
            inputs=[
                age,
                gender,
                education_level,
                blood_group,
                first_donation_date,
                last_donation_date,
                last_donation_volume_ml,
                donation_count_outram,
                donation_count_dhoby_ghaut,
                donation_count_woodlands,
                donation_count_jurong_east,
                donation_count_punggol,
                had_adverse_reaction_ever,
                last_blood_quality_screen,
            ],
            outputs=[prob_out, label_out, priority_out, guidance_out],
        )

        reset_btn.click(
            fn=reset_form,
            inputs=[],
            outputs=[
                age,
                gender,
                education_level,
                blood_group,
                first_donation_date,
                last_donation_date,
                last_donation_volume_ml,
                donation_count_outram,
                donation_count_dhoby_ghaut,
                donation_count_woodlands,
                donation_count_jurong_east,
                donation_count_punggol,
                had_adverse_reaction_ever,
                last_blood_quality_screen,
                prob_out,
                label_out,
                priority_out,
                guidance_out,
                health_out,
            ],
        )

        health_btn.click(
            fn=check_health,
            inputs=[],
            outputs=[health_out],
        )

        # Footer
        gr.Markdown(
            """
            <div style="text-align: center; padding: 1rem; color: #6c757d; font-size: 0.9rem;">
                Blood Donor Return Prediction Model · Powered by FastAPI + AWS Lambda
            </div>
            """
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
