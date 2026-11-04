# Deployment Guide

This guide details deployment options for the **Bank Statement Processing & Classification System**:
1. **Vercel Serverless Function** (cloud serverless target)
2. **Docker Container** (self-hosted / container target with Tesseract)
3. **Local Development Server** (uvicorn)

---

## 1. Vercel Serverless Deployment

The project is structured to deploy directly to Vercel via the Python Runtime (`vercel.json` + `api/index.py`).

### Vercel Architectural Adaptations

Because Vercel serverless functions operate within strict resource boundaries, the codebase implements specific architectural features to ensure smooth execution:

| Vercel Constraint | Platform Limit | Project Adaptation |
|---|---|---|
| **Max Request Body** | 4.5 MB | 4.0 MB upload limit enforced by `file_validator.py` and client-side HTML form validation. |
| **No Tesseract Binary** | No apt-get / binary install | Default OCR engine set to **RapidOCR (ONNX Runtime)** via `OCR_ENGINE=rapid`. |
| **Serverless Bundle Size** | 250 MB uncompressed | Trained model exported to **NumPy-only inference** format (`models/v1/`). Excludes scikit-learn, scipy, and pandas from runtime dependencies. |
| **Stateless Instances** | No shared local filesystem | **Stateless Backend** (`core/storage/backend.py`) compresses and signs processed statement states into client-persisted HMAC-SHA256 zlib tokens. |
| **Execution Timeout** | 300s (Hobby) / 800s (Pro) | Native text processing parses 200+ page statements in under 150 seconds. OCR runs on-demand per page. |

### Configuration Files

- **`vercel.json`**:
  ```json
  {
    "version": 2,
    "builds": [
      {
        "src": "api/index.py",
        "use": "@vercel/python"
      }
    ],
    "routes": [
      {
        "src": "/public/(.*)",
        "dest": "/public/$1"
      },
      {
        "src": "/(.*)",
        "dest": "api/index.py"
      }
    ]
  }
  ```

- **`api/index.py`**:
  Entrypoint that initializes the FastAPI application instance:
  ```python
  from app.main import app
  ```

- **`.vercelignore`**:
  Excludes test suites, synthetic training datasets, benchmark reports, and development caches to minimize bundle size.

### Deploying to Vercel via CLI

```bash
# 1. Install Vercel CLI (if not already installed)
npm install -g vercel

# 2. Login to Vercel
vercel login

# 3. Deploy Preview
vercel

# 4. Deploy Production
vercel --prod
```

### Required Environment Variables on Vercel Dashboard

Navigate to **Project Settings → Environment Variables**:
- `OCR_ENGINE`: `rapid`
- `SECRET_KEY`: `<secure-random-32-char-string>`
- `DEPLOY_TARGET`: `vercel`
- `LOG_LEVEL`: `INFO`

---

## 2. Docker Deployment (Local / VM / Cloud Run)

For self-hosted deployments or Kubernetes/Cloud Run, a multi-stage `Dockerfile` is provided with **Tesseract OCR** and full system dependencies pre-installed.

### Building & Running the Image

```bash
# 1. Build the Docker image
docker build -t bank-statement-processor .

# 2. Run container on port 8000
docker run -d \
  -p 8000:8000 \
  -e OCR_ENGINE=tesseract \
  -e SECRET_KEY=prod-secret-change-me \
  --name bank-processor \
  bank-statement-processor
```

The application will be live at `http://localhost:8000`.

### Health Check

```bash
curl http://localhost:8000/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "ocr_engine": "tesseract",
  "bank_profiles_loaded": 11,
  "ml_model_loaded": true,
  "categories_count": 23
}
```

---

## 3. Local Development Setup

### Prerequisites
- Python 3.10+ (Python 3.11 or 3.12 recommended)
- Optional: Tesseract OCR (`tesseract` in system PATH)

### Installation Steps

```bash
# 1. Clone repository & enter directory
cd "Bank Statement Processing & Classification System"

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Vendor frontend libraries (HTMX & Chart.js)
python scripts/download_vendors.py

# 5. Start development server
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000` in your web browser.

---

## 4. Make Automation Commands

A convenience `Makefile` is provided for standard operations:

| Command | Action |
|---|---|
| `make install` | Installs all production dependencies |
| `make test` | Runs unit, integration, and dataset test suites |
| `make train` | Generates synthetic data, trains ML classifier, and exports NumPy artifacts |
| `make evaluate` | Evaluates classifier on holdout dataset and generates reports |
| `make benchmark` | Runs end-to-end benchmark on full PDF dataset |
| `make run` | Starts the FastAPI server locally |
| `make docker-build` | Builds the Docker image with Tesseract |
| `make docker-run` | Runs the Docker container |

