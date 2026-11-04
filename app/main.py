"""FastAPI application factory, lifespan management, and router composition."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.routers import pages, api, exports
from core.banks.registry import BankRegistry
from core.classification.rules_engine import RulesEngine
from core.classification.ml_model import NumpyClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("bank_statement_system")

PUBLIC_DIR = PROJECT_ROOT / "public"
TEMPLATES_DIR = PROJECT_ROOT / "app" / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context: preloads registries, rules, and ML weights."""
    try:
        logger.info("Initializing Bank Statement Processing System...")
        # Pre-warm bank registry
        registry = BankRegistry()
        registry.load_all()
        logger.info("Bank profiles loaded: %s", ", ".join(registry.list_banks()))

        # Pre-warm classification engine
        rules = RulesEngine()
        model = NumpyClassifier()
        logger.info("Classification engines initialized (Rules: %d, ML: %s)", len(rules.rules), "Loaded" if model.is_loaded else "Untrained")
    except Exception as e:
        logger.warning("Startup pre-warming warning (non-fatal): %s", e)

    yield
    logger.info("Shutting down Bank Statement Processing System...")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title="Bank Statement Processing & Classification System",
        description="Production prototype for multi-bank statement parsing, ledger reconciliation, and non-LLM machine learning expense categorization.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Middleware to normalize Vercel rewritten paths
    @app.middleware("http")
    async def fix_vercel_path(request: Request, call_next):
        scope = request.scope
        raw_query = scope.get("query_string", b"").decode("latin-1")
        target_path = None

        # 1. Check if rewrite forwarded __vercel_path__ in query params
        if "__vercel_path__=" in raw_query:
            import urllib.parse
            parsed = urllib.parse.parse_qs(raw_query, keep_blank_values=True)
            if "__vercel_path__" in parsed:
                val = parsed.pop("__vercel_path__")[0]
                if val:
                    target_path = urllib.parse.unquote(val).split("?")[0]
                scope["query_string"] = urllib.parse.urlencode(parsed, doseq=True).encode("latin-1")

        # 2. Check x-matched-path header from Vercel edge router
        if not target_path:
            x_matched = request.headers.get("x-matched-path")
            if x_matched and not x_matched.startswith("/api/index") and not x_matched.startswith("/static"):
                target_path = x_matched.split("?")[0]

        if target_path:
            if not target_path.startswith("/"):
                target_path = "/" + target_path
            scope["path"] = target_path
        else:
            path = scope.get("path", "")
            if path.startswith("/api/index.py"):
                new_path = path[len("/api/index.py"):] or "/"
                if not new_path.startswith("/"):
                    new_path = "/" + new_path
                scope["path"] = new_path
            elif path.startswith("/api/index"):
                new_path = path[len("/api/index"):] or "/"
                if not new_path.startswith("/"):
                    new_path = "/" + new_path
                scope["path"] = new_path

        return await call_next(request)

    # Static assets (gracefully handle read-only filesystems in serverless)
    try:
        PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    if PUBLIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="static")

    # Include routers
    app.include_router(pages.router)
    app.include_router(api.router)
    app.include_router(exports.router)

    # Custom Error Handlers
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        if request.url.path.startswith("/api/"):
            return JSONResponse(status_code=404, content={"error": "NOT_FOUND", "message": "Endpoint not found"})
        return templates.TemplateResponse(
            request=request,
            name="partials/error.html",
            context={
                "error_title": "404 - Not Found",
                "error_message": "The requested statement session or page does not exist.",
                "error_code": "NOT_FOUND",
            },
            status_code=404,
        )

    @app.exception_handler(500)
    async def server_error_handler(request: Request, exc):
        logger.exception("Internal server error: %s", exc)
        if request.url.path.startswith("/api/"):
            return JSONResponse(status_code=500, content={"error": "SERVER_ERROR", "message": str(exc)})
        return templates.TemplateResponse(
            request=request,
            name="partials/error.html",
            context={
                "error_title": "500 - Processing Error",
                "error_message": "An unexpected error occurred during statement processing.",
                "error_code": "INTERNAL_ERROR",
            },
            status_code=500,
        )

    return app


# Root app instance for Vercel / Uvicorn
app = create_app()
