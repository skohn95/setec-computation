# Setec Computation Service

Statistical computation service for Setec AI Hub. This FastAPI service provides deterministic Python calculations using scipy and pandas.

## Architecture

This service is the **single source of truth** for all statistical calculations in the Setec AI Hub platform:
- **MSA (Measurement System Analysis)** - Gauge R&R calculations (Story 4.2)
- **Control Charts (SPC)** - Statistical Process Control analysis (Epic 5)

TypeScript/JavaScript in the Next.js application **never** performs mathematical computations. Claude AI **only** interprets results from this service.

## Project Structure

```
setec-computation/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI entry point
│   ├── config.py         # Environment config (pydantic-settings)
│   ├── routers/
│   │   ├── health.py     # Health check endpoint
│   │   ├── msa.py        # MSA computation (Story 4.2)
│   │   └── control_charts.py  # Control charts (Epic 5)
│   ├── services/         # Calculation services
│   ├── validators/       # Input validation
│   └── models/           # Pydantic models
├── tests/
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── railway.toml
└── .env.example
```

## Local Development

### Prerequisites

- Python 3.11+
- pip

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env from example
cp .env.example .env
```

### Run Development Server

```bash
uvicorn app.main:app --reload --port 8000
```

### Run Tests

```bash
python -m pytest -v
```

## API Endpoints

| Endpoint | Method | Description | Status |
|----------|--------|-------------|--------|
| `/health` | GET | Health check for Railway monitoring | ✅ Implemented |
| `/api/msa/compute` | POST | MSA (Gauge R&R) calculation | ✅ Implemented |
| `/api/control-charts/compute` | POST | Control charts calculation | 🚧 Epic 5 |

### Health Check Response

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "service": "setec-computation"
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `8000` |
| `PYTHON_ENV` | Environment (development/production) | `development` |
| `ALLOWED_ORIGINS` | Comma-separated list of CORS origins | `http://localhost:3000` |

### Railway Dashboard Configuration

Set these in the Railway dashboard:
- `PORT`: Provided automatically by Railway
- `PYTHON_ENV`: `production`
- `ALLOWED_ORIGINS`: `https://setec-ai-hub.vercel.app,https://*.vercel.app`

## Deployment

### Railway

This service is configured for Railway deployment using Docker:

1. Connect Railway to the GitHub repository
2. Configure environment variables in Railway dashboard
3. Railway will auto-deploy on push to `main`

The `railway.toml` configuration includes:
- Docker-based builds
- Health check monitoring at `/health`
- Automatic restart on failure

## Communication with Next.js

The Next.js application communicates with this service via REST API:

```typescript
// lib/python-service/client.ts
const result = await callPythonService<MSAResult>('/api/msa/compute', {
  method: 'POST',
  body: JSON.stringify(data)
});
```

## License

MIT
