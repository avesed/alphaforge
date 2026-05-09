"""AlphaForge -- ML stock prediction compute engine."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.core.request_id import RequestIdMiddleware

logger = logging.getLogger(__name__)


async def _ensure_default_admin() -> None:
    """Create a default admin user if no users exist in the database."""
    from sqlalchemy import select, exists

    from app.core.orm import get_session_factory
    from app.core.auth import hash_password
    from app.models.user import User

    factory = get_session_factory()
    async with factory() as session:
        has_any_user = (
            await session.execute(select(exists(select(User.id))))
        ).scalar()
        if has_any_user:
            return

        user = User(
            email="admin@alphaforge.dev",
            password_hash=hash_password("Admin123"),
            role="admin",
        )
        session.add(user)
        await session.commit()
        logger.info("Default admin user created: admin@alphaforge.dev")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # Startup
    from app.core.database import init_db_pool, close_db_pool
    from app.core.redis import close_redis
    from app.core.secrets import bootstrap_jwt_secret
    from app.core.scheduler import start_scheduler, stop_scheduler

    await init_db_pool()
    await bootstrap_jwt_secret()
    await _ensure_default_admin()
    await start_scheduler()

    logger.info("AlphaForge started on port %d", settings.PORT)

    yield

    # Shutdown
    from app.executor import shutdown_executors
    from app.api.public.indicators import shutdown_indicator_executor
    from app.services.stockpulse_client import close_stockpulse_async_client

    shutdown_executors()
    shutdown_indicator_executor()
    await stop_scheduler()
    await close_stockpulse_async_client()
    await close_db_pool()
    await close_redis()
    logger.info("AlphaForge shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="AlphaForge",
        description="ML stock prediction compute engine",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    from app.api.health import router as health_router
    from app.api.admin.auth import router as admin_auth_router
    from app.api.admin.users import router as admin_users_router
    from app.api.admin.consumers import router as admin_consumers_router
    from app.api.admin.settings import router as admin_settings_router
    from app.api.admin.scheduler import router as admin_scheduler_router
    from app.api.admin.stats import router as admin_stats_router

    app.include_router(health_router)
    app.include_router(admin_auth_router, prefix="/api/v1/admin")
    app.include_router(admin_users_router, prefix="/api/v1/admin")
    app.include_router(admin_consumers_router, prefix="/api/v1/admin")
    app.include_router(admin_settings_router, prefix="/api/v1/admin")
    app.include_router(admin_scheduler_router, prefix="/api/v1/admin")
    app.include_router(admin_stats_router, prefix="/api/v1/admin")

    # Public API routers (X-API-Key auth)
    # Qlib routes (no heavy ML deps)
    from app.api.public.factors import router as factors_router
    from app.api.public.expression import router as expression_router
    from app.api.public.backtests import router as backtests_router
    from app.api.public.data import router as data_router
    from app.api.public.indicators import router as indicators_router

    app.include_router(factors_router, prefix="/api/v1")
    app.include_router(expression_router, prefix="/api/v1")
    app.include_router(backtests_router, prefix="/api/v1")
    app.include_router(data_router, prefix="/api/v1")
    app.include_router(indicators_router, prefix="/api/v1")

    # ML routes (require lightgbm, scikit-learn, etc.)
    _ml_routes = [
        ("app.api.public.predictions", "predictions"),
        ("app.api.public.rdagent", "rdagent"),
        ("app.api.public.ml_tools", "ml-tools"),
        ("app.api.public.ml_backtests", "ml-backtests"),
        ("app.api.public.llm_proxy", "llm-proxy"),
    ]
    for module_path, name in _ml_routes:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            prefix = "/api/v1" if name != "llm-proxy" else ""
            app.include_router(mod.router, prefix=prefix)
        except ImportError as e:
            logger.warning("Skipping %s routes (missing dependency: %s)", name, e)

    return app


app = create_app()
