"""Synthetic blood donor dataset generator.

This module packages the entire donor-synthesis pipeline—demographic sampling,
donation timeline construction, volume estimation, availability scoring, and
dataset export—into reusable helpers plus a dead-simple entry point. Running it
produces a CSV simulating a donor program that launched in January 2020 and is
observed through October 2025, including a COVID-era dip in 2020–2021. This creates
realistic cohorts (new vs established donors), respects the minimum 12-week interval
between donations, includes seasonal/pandemic variation, and enforces realistic
QC/reaction rates. Import it to call ``generate_dataset`` in notebooks or scripts,
or execute ``python Dataset.py`` to refresh ``blood_donor_dataset.csv`` using the
defaults defined below.
"""

from __future__ import annotations

import random
import string
from datetime import date, timedelta
from pathlib import Path
import math
import pandas as pd

PANDEMIC_START = date(2020, 3, 1)
PANDEMIC_END = date(2022, 12, 31)

# ---------------------------------------------------------------------------
# Script defaults (adjust here when running ``python Dataset.py`` directly).
# ---------------------------------------------------------------------------
DEFAULT_NUM_DONORS = 75978
DEFAULT_OUTPUT_PATH = Path("blood_donor_dataset.csv")
DEFAULT_RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Global configuration bounds / vocabularies
# ---------------------------------------------------------------------------

MIN_AGE = 18  # Legal minimum eligibility
MAX_AGE = 60  # Upper bound legal eligibility

# Donor program timeline (launch in 2020, snapshot Oct 2025)
MIN_FIRST_DONATION_DATE = date(2020, 1, 1)      # Earliest possible first donation
CUTOFF_DONATION_DATE = date(2025, 10, 31)       # Latest allowable timeline reference

MIN_INTERVAL_DAYS = 84  # Enforce 12-week gap between donations

LOCATIONS = ["Outram", "Dhoby_Ghaut", "Woodlands", "Jurong_East", "Punggol"]
BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
BLOOD_GROUP_DISTRIBUTION = {
    "O+": 0.36,
    "O-": 0.14,
    "A+": 0.28,
    "A-": 0.08,
    "B+": 0.08,
    "B-": 0.03,
    "AB+": 0.02,
    "AB-": 0.01,
}
EDUCATION_LEVELS = ["Primary", "Secondary", "Diploma", "University", "Postgraduate"]
GENDERS = ["Male", "Female"]

# Quality-control parameters (never revealed to donors)
BLOOD_QUALITY_VALUES = ["Pass", "Fail"]
MIN_LAST_VOL = 350
MAX_LAST_VOL = 450

ADVERSE_REACTION_BASE_PROB = 0.08  # Base chance that any donor has ever reacted
BLOOD_QUALITY_FAIL_PROB = 0.015   # Share of QC screens that fail


# ---------------------------------------------------------------------------
# Random sampling helpers (names, demographics, etc.)
# ---------------------------------------------------------------------------

def random_name(gender):
    
    first_names_m = [
        # English / Western
        "James", "John", "Michael", "William", "David", "Richard", "Joseph",
        "Daniel", "Matthew", "Andrew", "Joshua", "Ryan", "Nicholas", "Christopher",
        "Ethan", "Logan", "Lucas", "Noah", "Liam", "Owen", "Connor", "Blake",
        "Gavin", "Caleb", "Mason", "Hunter", "Cooper", "Finn", "Wyatt", "Parker",
        # Chinese
        "Wei", "Jian", "Hao", "Jun", "Qi", "Tian", "Ming", "Jie", "Chenwei",
        "Guang", "Zhi", "Han", "Lei", "Dong", "Yong", "Xiang", "Chao", "Peng",
        # Malay / Indonesian
        "Firdaus", "Hafiz", "Rahman", "Irfan", "Hakim", "Syafiq", "Azman", "Imran",
        "Ridzuan", "Saiful", "Zulkifli", "Khairul",
        # Indian / South Asian
        "Arjun", "Ravi", "Sanjay", "Karthik", "Vikram", "Rahul", "Pranav", "Dev",
        "Amit", "Suresh", "Rohan", "Naveen", "Anand", "Vishal", "Harish",
        # Others
        "Mateo", "Diego", "Thiago", "Miguel", "Marcus", "Jamal", "Omar", "Samir",
        "Youssef", "Hassan", "Jean", "Luca", "Victor", "Andrei"
    ]
    first_names_f = [
        # English / Western
        "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan",
        "Jessica", "Sarah", "Emily", "Samantha", "Natalie", "Isabella", "Sophia",
        "Olivia", "Charlotte", "Amelia", "Grace", "Chloe", "Madison", "Abigail",
        "Avery", "Hannah", "Mia", "Lily", "Harper", "Ella", "Scarlett", "Victoria",
        # Chinese
        "Li Na", "Mei", "Xinyi", "Yue", "Jing", "Hui", "Qiao", "Fen", "Ling",
        "Yan", "Liling", "Suyin", "Pei", "Jiayi", "Xin", "Ruoyi", "Yuqing",
        # Malay / Indonesian
        "Aisyah", "Nurul", "Farah", "Siti", "Amira", "Hidayah", "Putri", "Dewi",
        "Nadia", "Syazwani", "Amina", "Izzah",
        # Indian / South Asian
        "Priya", "Ananya", "Lakshmi", "Kavya", "Divya", "Neha", "Riya", "Ishita",
        "Pooja", "Sneha", "Meera", "Aditi", "Shreya", "Tanvi", "Sai",
        # Others
        "Fatima", "Zara", "Layla", "Aaliyah", "Yara", "Inaya", "Sofia", "Maya",
        "Carmen", "Lucia", "Ana", "Elena", "Gabriela", "Bianca"
    ]
    last_names = [
        # Chinese surnames
        "Chen", "Li", "Zhang", "Liu", "Wang", "Yang", "Huang", "Zhao", "Wu",
        "Zhou", "Tan", "Lim", "Lee", "Ng", "Wong", "Lin", "Xu", "Sun", "Ma",
        "Guo", "He", "Gao", "Luo", "Zheng", "Liang", "Feng", "Qiu",
        # Malay / Indonesian surnames / patronymics
        "Iskandar", "Rahman", "Hakim", "Aziz", "Ridwan", "Salleh", "Rosli",
        "Hassan", "Ibrahim", "Mahmud", "Yusof",
        # Indian / South Asian surnames
        "Iyer", "Menon", "Nair", "Pillai", "Singh", "Kumar", "Sharma", "Patel",
        "Reddy", "Chandra", "Das", "Varma",
        # Western / Others
        "Smith", "Johnson", "Brown", "Taylor", "Anderson", "Clark", "Young",
        "Scott", "Bennett", "Mitchell", "Murphy", "Reyes", "Gonzalez", "Silva",
        "Fernandez", "Martinez", "Lopez", "Garcia", "Rodriguez", "Carter"
    ]

    if gender == "Male": first_names = first_names_m
    else: first_names = first_names_f

    return f"{random.choice(first_names)} {random.choice(last_names)}"


def email_for_name(name):
    """Create a synthetic email from name + random digits, using gmail/hotmail."""
    base = "".join(c for c in name.lower() if c.isalpha())
    suffix = random.randint(10, 9999)
    domain = random.choice(["gmail.com", "hotmail.com"])
    return f"{base}{suffix}@{domain}"


def random_phone():
    """Generate a simple Singapore-style contact number."""
    start = random.choice(["8", "9"])
    rest = "".join(random.choices(string.digits, k=7))
    return f"+65 {start}{rest}"


def random_password(length=10):
    chars = string.ascii_letters + string.digits + "!@#$%^&*?"
    return "".join(random.choices(chars, k=length))


def random_age():
    """Sample a realistic donor age between MIN_AGE and MAX_AGE."""
    r = random.random()
    if r < 0.2:
        return random.randint(MIN_AGE, 25)
    elif r < 0.7:
        return random.randint(26, 45)
    elif r < 0.95:
        return random.randint(46, min(60, MAX_AGE))
    else:
        senior_lower = 61
        if MAX_AGE < senior_lower:
            senior_lower = min(60, MAX_AGE)
        return random.randint(senior_lower, MAX_AGE)


def date_from_age(age, ref=CUTOFF_DONATION_DATE):
    """Generate a date_of_birth given age, approximating years as 365 days."""
    days_old = age * 365 + random.randint(0, 364)
    return ref - timedelta(days=days_old)


def random_education():
    """Sample an education level with a plausible distribution."""
    r = random.random()
    if r < 0.1:
        return "Primary"
    elif r < 0.45:
        return "Secondary"
    elif r < 0.7:
        return "Diploma"
    elif r < 0.9:
        return "University"
    else:
        return "Postgraduate"


def sample_location_probabilities(concentration: float = 2.5) -> dict[str, float]:
    """Draw per-center weights so donors are not split perfectly evenly."""

    draws = [random.gammavariate(concentration, 1.0) for _ in LOCATIONS]
    total = sum(draws)
    return {loc: draw / total for loc, draw in zip(LOCATIONS, draws)}


def random_home_location(location_probs: dict[str, float] | None = None):
    """Choose a 'home' location where the donor donates most often."""

    if not location_probs:
        return random.choice(LOCATIONS)
    locations, weights = zip(*location_probs.items())
    return random.choices(locations, weights=weights, k=1)[0]


def random_blood_group():
    """Sample blood group using empirically informed population percentages."""

    weights = [BLOOD_GROUP_DISTRIBUTION[group] for group in BLOOD_GROUPS]
    return random.choices(BLOOD_GROUPS, weights=weights, k=1)[0]


def random_donation_counts(age_years):
    """
    Generate a plausible number_of_donation based on age,
    but cap it so the 12-week minimum interval is always feasible
    between 2023-01-01 and 2025-10-31.
    """
    # Base by age
    if age_years < 25:
        sampled = random.randint(1, 5)
    elif age_years < 40:
        sampled = random.randint(1, 15)
    elif age_years < 55:
        sampled = random.randint(1, 20)
    else:
        sampled = random.randint(1, 15)

    # Global maximum feasible donations given timeline and min interval.
    total_days = (CUTOFF_DONATION_DATE - MIN_FIRST_DONATION_DATE).days
    global_max_donations = total_days // MIN_INTERVAL_DAYS + 1  # (span / 84) + 1

    return min(sampled, global_max_donations)


# ---------------------------------------------------------------------------
# Donation timeline + volume generators
# ---------------------------------------------------------------------------

def apply_seasonal_factor(candidate_date):
    """Apply seasonal weight to reduce donations in Dec/Jan and June/July (holidays)."""
    month = candidate_date.month
    # Lower probability for holiday months
    if month in [12, 1]:  # Dec-Jan (Christmas/New Year)
        return 0.6
    elif month in [6, 7]:  # Jun-Jul (Summer holidays)
        return 0.7
    elif month in [2, 8, 9]:  # Post-holiday recovery months (higher activity)
        return 1.2
    else:
        return 1.0


def pandemic_weight(candidate_date):
    """Simulate extended COVID-era suppression (2020-2022) with gradual recovery."""

    if candidate_date < PANDEMIC_START or candidate_date > PANDEMIC_END:
        return 1.0

    total_days = (PANDEMIC_END - PANDEMIC_START).days or 1
    progress = (candidate_date - PANDEMIC_START).days / total_days
    progress = min(max(progress, 0.0), 1.0)

    # Very low activity early, slow recovery that accelerates late 2022
    recovery = 1.0 / (1.0 + math.exp(-9 * (progress - 0.85)))
    weight = 0.12 + 0.88 * recovery
    return max(0.1, min(1.0, weight))


def program_activity_weight(candidate_date, launch_date, total_days):
    """Program awareness curve: slow ramp-up, mid-program peak, gentle taper."""

    if total_days <= 0:
        return 1.0

    progress = (candidate_date - launch_date).days / total_days
    progress = min(max(progress, 0.0), 1.0)

    # Strong ramp-up begins once pandemic impact fades (progress ~0.55)
    ramp = 1.0 / (1.0 + math.exp(-10 * (progress - 0.6)))
    # Bell-shaped intensity peaks toward late program (progress ~0.85)
    gaussian = math.exp(-((progress - 0.85) ** 2) / 0.03)

    weight = 0.03 + 1.0 * ramp * gaussian
    return min(1.1, max(0.03, weight))


def generate_donation_dates(dob, num_donations, cutoff=CUTOFF_DONATION_DATE):
    """
    Generate first_donation_date and last_donation_date with realistic temporal patterns:

    - first_donation_date >= 2020-01-01 (program launch through 2025)
    - first_donation_date weighted by seasonality, COVID dip, and program ramp
    - first_donation_date >= dob + 18 years (eligibility constraint)
    - last_donation_date <= 2025-10-31 (current snapshot date)
    - last_donation_date weighted exponentially (more recent donations more likely)
    - Minimum 12-week (84 days) interval between donations
    - Seasonal variation (fewer donations in Dec/Jan and Jun/Jul)
    """

    # Per-donor minimum first date: must be at least global min & 18 years old
    earliest_eligible = dob + timedelta(days=MIN_AGE * 365)
    min_first_date = max(MIN_FIRST_DONATION_DATE, earliest_eligible)
    end_date = cutoff

    # If eligibility is after end date, clamp (should be rare)
    if min_first_date > end_date:
        min_first_date = end_date

    # Minimum total span required to respect 84-day intervals
    min_span_days = (num_donations - 1) * MIN_INTERVAL_DAYS

    # Max first date such that first_date + min_span_days <= end_date
    max_first_date = end_date - timedelta(days=min_span_days)

    # If max_first_date is before min_first_date, we must reduce num_donations
    if max_first_date < min_first_date:
        available_span = (end_date - min_first_date).days
        max_don_here = max(1, available_span // MIN_INTERVAL_DAYS + 1)
        if num_donations > max_don_here:
            num_donations = max_don_here
            min_span_days = (num_donations - 1) * MIN_INTERVAL_DAYS
            max_first_date = end_date - timedelta(days=min_span_days)
        else:
            max_first_date = min_first_date

    # Use rejection sampling with program + seasonal weights for first_donation_date
    if max_first_date < min_first_date:
        first_donation_date = min_first_date
    else:
        total_days = (max_first_date - min_first_date).days
        attempts = 0
        first_donation_date = min_first_date
        while attempts < 25:
            attempts += 1
            first_offset = random.randint(0, total_days)
            candidate_first = min_first_date + timedelta(days=first_offset)
            seasonal_weight = apply_seasonal_factor(candidate_first)
            program_weight = program_activity_weight(candidate_first, min_first_date, total_days)
            covid_weight = pandemic_weight(candidate_first)
            acceptance_prob = min(1.0, seasonal_weight * program_weight * covid_weight)
            if random.random() <= acceptance_prob:
                first_donation_date = candidate_first
                break
        else:
            # Fallback to last sampled candidate
            first_donation_date = candidate_first

    # Compute minimum last_donation_date respecting required span
    min_last_date = first_donation_date + timedelta(days=min_span_days)
    if min_last_date > end_date:
        min_last_date = end_date

    # Realistic last_donation_date: not all clustered at cutoff
    # Use exponential distribution to favor more recent dates but with spread
    if num_donations == 1:
        last_donation_date = first_donation_date
    else:
        remaining_slack_days = (end_date - min_last_date).days
        if remaining_slack_days <= 0:
            last_donation_date = min_last_date
        else:
            # Exponential distribution (lambda=1.0) favors recent donations while keeping realistic spread
            exp_sample = min(1.0, random.expovariate(1.0))  # Clamp to [0, 1]
            # Invert so 1.0 = end_date (recent), 0.0 = min_last_date (old)
            extra = int((1.0 - exp_sample) * remaining_slack_days)
            
            candidate_last = min_last_date + timedelta(days=extra)
            seasonal_weight = apply_seasonal_factor(candidate_last)
            
            # Apply seasonal adjustment
            if random.random() > seasonal_weight:
                extra = random.randint(0, remaining_slack_days)
            
            last_donation_date = min_last_date + timedelta(days=extra)

    return first_donation_date, last_donation_date, num_donations


def generate_volumes(num_donations):
    """Generate last_donation_volume_ml and total_donation_volume_ml."""
    last_vol = random.randint(MIN_LAST_VOL, MAX_LAST_VOL)
    if num_donations == 1:
        total_volume = last_vol
    else:
        avg_volume = random.uniform(350, 450)  # ml per donation
        total_volume = int(round(avg_volume * num_donations))
    return last_vol, total_volume


# ---------------------------------------------------------------------------
# Availability model (probabilistic scoring)
# ---------------------------------------------------------------------------

def compute_age(ref, dob):
    years = ref.year - dob.year
    if (ref.month, ref.day) < (dob.month, dob.day):
        years -= 1
    return years


def age_group(age):
    if age <= 25:
        return "young"
    if age <= 40:
        return "prime"
    if age <= 55:
        return "mid"
    return "senior"


def recency_group(ref, last_date):
    days = (ref - last_date).days
    if days <= 120:
        return "very_recent"
    if days <= 365:
        return "recent"
    if days <= 730:
        return "stale"
    return "lapsed"


def donation_intensity(num):
    if num <= 2:
        return "new"
    if num <= 10:
        return "regular"
    return "veteran"


def compute_availability_probability(
    dob,
    last_date,
    num_don,
    edu,
    donation_count_outram,
    donation_count_dhoby_ghaut,
    had_reaction,
):
    """
    Compute p(availability = Yes) from behavioural features + noise.

    Location effect: based on share of donations at central sites (Outram + Dhoby_Ghaut).
    """
    age = compute_age(CUTOFF_DONATION_DATE, dob)
    ag = age_group(age)
    rg = recency_group(CUTOFF_DONATION_DATE, last_date)
    di = donation_intensity(num_don)

    base = {"young": 0.30, "prime": 0.45, "mid": 0.50, "senior": 0.35}[ag]
    rec = {"very_recent": 0.25, "recent": 0.15, "stale": 0.00, "lapsed": -0.15}[rg]
    don = {"new": 0.00, "regular": 0.10, "veteran": 0.20}[di]

    edu_delta = {
        "Primary": 0.00,
        "Secondary": 0.00,
        "Diploma": 0.05,
        "University": 0.08,
        "Postgraduate": 0.10,
    }.get(edu, 0.0)

    # Central location share
    central_donations = donation_count_outram + donation_count_dhoby_ghaut
    if num_don > 0:
        central_share = central_donations / num_don
    else:
        central_share = 0.0
    # Max +0.05 if all donations are in central locations
    loc_delta = 0.05 * central_share

    reaction_delta = -0.20 if had_reaction else 0.00

    score = base + rec + don + edu_delta + loc_delta + reaction_delta
    score = max(0.0, min(1.0, score))

    noise = random.gauss(0.0, 0.10)
    score_noisy = max(0.0, min(1.0, score + noise))

    return score_noisy


# ---------------------------------------------------------------------------
# Location counts generator
# ---------------------------------------------------------------------------

def generate_location_counts(num_donations, home_location):
    """
    Generate donation counts per location.

    - Sum of all 5 counts = num_donations
    - Donor has a 'home' location with higher weight (more donations there).
    """
    counts = {
        "Outram": 0,
        "Dhoby_Ghaut": 0,
        "Woodlands": 0,
        "Jurong_East": 0,
        "Punggol": 0,
    }

    base_weights = {loc: 1.0 for loc in LOCATIONS}
    base_weights[home_location] = 4.0  # bias towards home location
    total_w = sum(base_weights.values())
    probs = {loc: w / total_w for loc, w in base_weights.items()}

    for _ in range(num_donations):
        r = random.random()
        cum = 0.0
        chosen = None
        for loc, p in probs.items():
            cum += p
            if r <= cum:
                chosen = loc
                break
        if chosen is None:
            chosen = home_location
        counts[chosen] += 1

    return counts


# ---------------------------------------------------------------------------
# Generate one donor row
# ---------------------------------------------------------------------------

def generate_single_donor(location_probs: dict[str, float] | None = None) -> dict[str, object]:
    """Generate one synthetic donor row following the schema and logic."""
    # Demographics
    age = random_age()
    dob = date_from_age(age)
    gender = random.choice(GENDERS)
    education = random_education()
    blood_group = random_blood_group()

    # Donation history
    num_don = random_donation_counts(age)
    first_date, last_date, num_don = generate_donation_dates(dob, num_don)
    last_vol, total_vol = generate_volumes(num_don)

    # Location donation counts
    home_loc = random_home_location(location_probs)
    loc_counts = generate_location_counts(num_don, home_loc)

    # Safety / reactions
    if num_don <= 2:
        adv_prob = ADVERSE_REACTION_BASE_PROB + 0.03
    else:
        adv_prob = ADVERSE_REACTION_BASE_PROB
    had_reaction = random.random() < adv_prob

    # QC (donors are NOT informed, no effect on availability)
    blood_quality = "Fail" if random.random() < BLOOD_QUALITY_FAIL_PROB else "Pass"

    # Behavioural availability
    p_avail = compute_availability_probability(
        dob,
        last_date,
        num_don,
        education,
        loc_counts["Outram"],
        loc_counts["Dhoby_Ghaut"],
        had_reaction,
    )
    availability = "Yes" if random.random() < p_avail else "No"

    # PII
    name = random_name(gender)
    email = email_for_name(name)
    phone = random_phone()
    pwd = random_password()
    donor_id = "".join(random.choices(string.hexdigits.lower(), k=10))

    # Build row
    row = {
        "donor_id": donor_id,
        "name": name,
        "email": email,
        "contact_number": phone,
        "password": pwd,
        "date_of_birth": dob.isoformat(),
        "gender": gender,
        "education_level": education,
        "blood_group": blood_group,
        "first_donation_date": first_date.isoformat(),
        "last_donation_date": last_date.isoformat(),
        "number_of_donation": num_don,
        "total_donation_volume_ml": total_vol,
        "last_donation_volume_ml": last_vol,
        "donation_count_outram": loc_counts["Outram"],
        "donation_count_dhoby_ghaut": loc_counts["Dhoby_Ghaut"],
        "donation_count_woodlands": loc_counts["Woodlands"],
        "donation_count_jurong_east": loc_counts["Jurong_East"],
        "donation_count_punggol": loc_counts["Punggol"],
        "had_adverse_reaction_ever": had_reaction,
        "last_blood_quality_screen": blood_quality,
        "availability": availability,
    }
    return row


def generate_dataset(
    num_donors: int = DEFAULT_NUM_DONORS,
    random_seed: int | None = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a synthetic donor dataset with optional deterministic seeding."""

    if num_donors <= 0:
        raise ValueError("num_donors must be positive")

    if random_seed is not None:
        random.seed(random_seed)

    location_probs = sample_location_probabilities()
    rows = [generate_single_donor(location_probs) for _ in range(num_donors)]
    return pd.DataFrame(rows)


def main() -> None:
    """Entry point for `python Dataset.py` using module-level defaults."""
    df = generate_dataset(DEFAULT_NUM_DONORS, DEFAULT_RANDOM_SEED)
    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEFAULT_OUTPUT_PATH, index=False)
    print(f"Generated dataset with {len(df)} donors at {DEFAULT_OUTPUT_PATH.resolve()}")
    print("NEXT STEP: Run 'python eda.py' to perform exploratory data analysis.")


if __name__ == "__main__":
    main()
