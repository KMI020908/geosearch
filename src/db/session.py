from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import settings

engine = create_async_engine(
    url=settings.database_url
    or f"postgresql+asyncpg://{settings.postgres_user}:{settings.postgres_password}@localhost:5432/{settings.postgres_db}",
    echo=False,
)

AsyncSessionFactory = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session (for use as a FastAPI dependency)."""
    async with AsyncSessionFactory() as session:
        yield session
