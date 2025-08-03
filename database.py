"""
Database models and management for the advanced bot.
"""
import os
import asyncio
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, Float, JSON, select, update, delete
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
import logging

logger = logging.getLogger(__name__)

# Database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgresql://", "postgresql+asyncpg://", 1)

# Create async engine
async_engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(async_engine, class_=AsyncSession)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    telegram_id = Column(Integer, unique=True, nullable=False)
    username = Column(String(100))
    first_name = Column(String(100))
    last_name = Column(String(100))
    preferred_ai = Column(String(50), default="gemini")
    language = Column(String(10), default="en")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_active = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)
    settings = Column(JSON, default=dict)


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    message = Column(Text, nullable=False)
    response = Column(Text)
    ai_model = Column(String(50))
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    message_type = Column(String(20), default="text")
    extra_metadata = Column(JSON, default=dict)


class UserMemory(Base):
    __tablename__ = "user_memory"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    key = Column(String(100), nullable=False)
    value = Column(Text, nullable=False)
    category = Column(String(50), default="general")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Analytics(Base):
    __tablename__ = "analytics"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    action = Column(String(50), nullable=False)
    details = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ai_model_used = Column(String(50))
    processing_time = Column(Float)


class FileUpload(Base):
    __tablename__ = "file_uploads"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    file_id = Column(String(200), nullable=False)
    file_name = Column(String(200))
    file_type = Column(String(50))
    file_size = Column(Integer)
    processed = Column(Boolean, default=False)
    processing_result = Column(Text)
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


async def init_database():
    try:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def get_session():
    return async_session()


async def get_or_create_user(telegram_id: int, username: str = None,
                             first_name: str = None, last_name: str = None):
    async with async_session() as session:
        try:
            stmt = select(User).where(User.telegram_id == telegram_id)
            result = await session.execute(stmt)
            user = result.scalar_one_or_none()

            if user:
                await session.execute(
                    update(User).where(User.telegram_id == telegram_id).values(
                        last_active=datetime.now(timezone.utc)
                    )
                )
                await session.commit()
                return user
            else:
                new_user = User(
                    telegram_id=telegram_id,
                    username=username or "",
                    first_name=first_name or "",
                    last_name=last_name or ""
                )
                session.add(new_user)
                await session.commit()
                await session.refresh(new_user)
                return new_user
        except Exception as e:
            logger.error(f"Error in get_or_create_user: {e}")
            await session.rollback()
            raise


async def save_conversation(user_id: int, message: str, response: str = None,
                            ai_model: str = None, message_type: str = "text",
                            metadata: dict = None):
    async with async_session() as session:
        try:
            conversation = Conversation(
                user_id=user_id,
                message=message,
                response=response,
                ai_model=ai_model,
                message_type=message_type,
                extra_metadata=metadata or {}
            )
            session.add(conversation)
            await session.commit()
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            await session.rollback()


async def get_user_memory(user_id: int, key: str = None) -> list:
    async with async_session() as session:
        try:
            if key:
                stmt = select(UserMemory).where(
                    UserMemory.user_id == user_id,
                    UserMemory.key == key).order_by(
                    UserMemory.updated_at.desc())
            else:
                stmt = select(UserMemory).where(
                    UserMemory.user_id == user_id).order_by(
                    UserMemory.updated_at.desc()).limit(50)
            result = await session.execute(stmt)
            return [row._asdict() for row in result.scalars().all()]
        except Exception as e:
            logger.error(f"Error getting user memory: {e}")
            return []


async def save_user_memory(
        user_id: int,
        key: str,
        value: str,
        category: str = "general"):
    async with async_session() as session:
        try:
            stmt = select(
                UserMemory.id).where(
                UserMemory.user_id == user_id,
                UserMemory.key == key)
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                await session.execute(
                    update(UserMemory).where(UserMemory.id == existing).values(
                        value=value,
                        updated_at=datetime.now(timezone.utc)
                    )
                )
            else:
                memory = UserMemory(
                    user_id=user_id,
                    key=key,
                    value=value,
                    category=category
                )
                session.add(memory)

            await session.commit()
        except Exception as e:
            logger.error(f"Error saving user memory: {e}")
            await session.rollback()


async def delete_user_memory(user_id: int, key: str = None):
    async with async_session() as session:
        try:
            if key:
                stmt = delete(UserMemory).where(
                    UserMemory.user_id == user_id, UserMemory.key == key)
            else:
                stmt = delete(UserMemory).where(UserMemory.user_id == user_id)
            await session.execute(stmt)
            await session.commit()
        except Exception as e:
            logger.error(f"Error deleting user memory: {e}")
            await session.rollback()
