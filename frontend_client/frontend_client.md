# Frontend Client Guide

This document explains how the Gradio-based donor outreach console works, how it is configured, and how to run or package it independently from the rest of the project.

---

## 1. Purpose & High-level Architecture

The frontend client provides call-centre / outreach notification staff with a guided form to collect donor information, call the hosted prediction API, and surface prioritisation guidance. It is built with **Gradio Blocks** (`frontend_client/app.py`) and interacts with the FastAPI backend (`app/main.py`) that is deployed behind the AWS API Gateway + Lambda stack.

Key responsibilities:

- **Feature engineering**: collects donor form inputs and converts them into the 27 engineered features expected by the model (mirrors `process_data.py`).
- **API integration**: sends `/predict` requests to the backend and displays probability, label, and recommended action; provides a `/health` check button.
- **UX polish**: branded with Blood Bank Singapore imagery, warm theme, meta tags (Open Graph, Twitter) and favicon for rich previews, plus validation on donation counts/dates.
- **Deployment options**: runnable locally via Python or containerised via `frontend_client/Dockerfile` for platforms such as Koyeb / ECS / local Docker.

---

## 2. Repository Layout (frontend_client/)

| File | Description |
| --- | --- |
| `app.py` | Main Gradio Blocks definition, feature-engineering helpers, API calls, validations, meta tags, and UI layout. |
| `requirements.txt` | Dependencies for the frontend (numpy, pandas, gradio, httpx). |
| `Dockerfile` | Minimal Python 3.11 container that installs `frontend_client/requirements.txt`, copies `frontend_client/` + `process_data.py`, and runs `python frontend_client/app.py`. |
| `images/` | Branding assets. `Blood_bank_SG_banner.png` (header), `Blood_bank_SG_logo.png` (social-preview image), `Blood_bank_SG_favicon.png` (favicon). |

`process_data.py` is copied alongside the app so `_build_features_directly` can reuse constants such as `EDUCATION_MAPPING` and `REFERENCE_DATE`.

---

## 3. Dependencies

`frontend_client/requirements.txt` installs a minimal stack:

```
numpy==1.26.4
pandas==2.2.3
gradio==5.49.1
httpx
```

- **Gradio** powers the UI and theming.
- **Pandas/Numpy** are used for lightweight feature engineering.
- **httpx** handles async-safe HTTP calls to the backend API.

> Note: The app expects Python 3.11 (matching the Docker image and the repo's Conda environment).

---

## 4. Configuration & Environment Variables

1. **Retrieve the API Gateway invoke URL via Terraform** (after provisioning the backend):

   ```powershell
   cd infra
   terraform init            # first time only
   terraform apply           # provisions Lambda + API Gateway
   terraform output http_api_invoke_url
   ```

2. **Export the value as `API_BASE_URL` before launching Gradio**:

   ```powershell
   export API_BASE_URL="https://<api-id>.execute-api.<region>.amazonaws.com"
   ```

- This variable is required; it must point at the deployed FastAPI Lambda (normally the Terraform `http_api_invoke_url`).
- No trailing slash; the code normalises the value internally.
- If unset, the app raises a Gradio error explaining how to configure it.

No other environment variables are needed for the client.

---

## 5. Feature Engineering & Validation Highlights

- `_build_features_directly` recreates the inference feature vector without relying on pandas `get_dummies`. It validates timeline ordering, location counts, and ensures single-donation scenarios have matching first/last dates.
- Location counts are the sole source of the total donation count; consistency checks ensure counts sum correctly and do not exceed overall donations.
- Education, gender, blood group, adverse reaction, and quality screen inputs are encoded exactly like training-time transformations in `process_data.py`.
- Donation timelines implicitly follow the training rule of **minimum 84 days between donations** (as enforced in `process_data.py`); ensure your first/last dates plus total donation count are historically plausible for that spacing.
- Meta tags (`HEAD_META`) include Open Graph and Twitter metadata plus a favicon link so shared links show Blood Bank branding.

---

## 6. Running Locally (Python)

1. Activate the existing Conda environment defined in `environment.yml` (or any Python 3.11 env that already has the frontend deps installed):

   ```powershell
   conda activate prml-project
   ```

2. Set the backend URL:

   ```powershell
   $env:API_BASE_URL = "https://<api-id>.execute-api.ap-southeast-1.amazonaws.com"
   ```

3. Launch the frontend (FastAPI + mounted Gradio) via Uvicorn:

   ```powershell
   python -m frontend_client.app
   # or: uvicorn frontend_client.app:app --host 0.0.0.0 --port 7860
   ```

4. Open <http://127.0.0.1:7860> to interact with the UI.

---

## 7. Running via Docker

From the repo root:

```powershell
docker build -f frontend_client/Dockerfile -t donor-frontend .
docker run --rm -p 7860:7860 `
  -e API_BASE_URL="https://<api-id>.execute-api.ap-southeast-1.amazonaws.com" `
  donor-frontend
```

The Docker image includes only what the client needs (`frontend_client/` files + `process_data.py`) to keep it lean. Gradio still serves on port 7860 inside the container.

---

## 8. UI Overview & Key Components

- **Branding**: header uses `Blood_bank_SG_banner.png`; favicon + meta tags refer to `images/Blood_bank_SG_favicon.png` and `Blood_bank_SG_logo.png`.
- **Sections**:
  - *Demographics*: Age slider (18–65), gender, education, blood group.
  - *Donation Timeline*: Date pickers defaulting to 20 Mar 2024 for both first and last donation.
  - *Donation Statistics*: Last donation volume slider (350–450 ml).
  - *Locations*: Numeric inputs per centre; sums determine `n_donations` with built-in validation.
  - *Health & Safety*: Adverse reaction checkbox, quality screen radio buttons.
  - *Outputs*: Probability, label, priority, and recommended guidance cards.
  - *System Status*: Button to call `/health` and display API connectivity.

- **Events**:
  - Submit triggers `score_donor_form`, which validates inputs, builds features, and posts `predict` to the backend.
  - Reset sets all inputs back to `_default_form_values()` (Age 35 / Male / O+ / etc.).

---

## 9. API Calls from the Client

`frontend_client/app.py` uses two helper functions:

- `check_health()` → GET `${API_BASE_URL}/health` via httpx; status is displayed in the "System Status" card.
- `score_donor_form()` → POST `${API_BASE_URL}/predict` with `{"features": {...}}` (27 features) and renders the result cards.

Both functions raise `gr.Error` with descriptive messages if httpx encounters network or non-200 responses.

---

## 10. Extending or Customising the UI

- Add new inputs by modifying `_default_form_values`, `reset_form`, and the Gradio component block; ensure `_build_features_directly` emits the updated feature.
- Styling tweaks should use Gradio themes (as the current code does) rather than custom CSS, per user requirements.
- To change branding assets, swap files under `frontend_client/images/` and adjust the paths in `app.py` if filenames differ.
- Meta tags can be edited by changing the `HEAD_META` string near the top of `app.py`.

---

This guide should give you everything needed to understand, run, and extend the Gradio frontend. For deployment details of the backend/API, see `MLOps_Deployment_Guide.md` and the Terraform files under `infra/`.
