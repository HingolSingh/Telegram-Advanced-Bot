"""
Analytics and tracking system for the advanced bot.
"""
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import json
from collections import defaultdict

from database import async_session, Analytics, User, Conversation, UserMemory

logger = logging.getLogger(__name__)


class AnalyticsManager:
    """Comprehensive analytics and tracking system."""

    def __init__(self):
        self.session_data = defaultdict(dict)

    async def track_event(self,
                          user_id: int,
                          action: str,
                          details: Dict[str,
                                        Any],
                          ai_model: str = None,
                          processing_time: float = None):
        """Track user events and actions."""
        try:
            async with async_session() as session:
                analytics_entry = Analytics(
                    user_id=user_id,
                    action=action,
                    details=details,
                    ai_model_used=ai_model,
                    processing_time=processing_time
                )
                session.add(analytics_entry)
                await session.commit()

        except Exception as e:
            logger.error(f"Error tracking event: {e}")

    async def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Get basic user statistics."""
        try:
            async with async_session() as session:
                # Total messages
                result = await session.execute(
                    "SELECT COUNT(*) FROM conversations WHERE user_id = :user_id",
                    {"user_id": user_id}
                )
                total_messages = result.scalar() or 0

                # AI requests
                result = await session.execute(
                    "SELECT COUNT(*) FROM analytics WHERE user_id = :user_id AND action = 'text_message'",
                    {"user_id": user_id}
                )
                ai_requests = result.scalar() or 0

                # Images processed
                result = await session.execute(
                    "SELECT COUNT(*) FROM analytics WHERE user_id = :user_id AND action = 'image_analysis'",
                    {"user_id": user_id}
                )
                images_processed = result.scalar() or 0

                # Files processed
                result = await session.execute(
                    "SELECT COUNT(*) FROM analytics WHERE user_id = :user_id AND action = 'file_processing'",
                    {"user_id": user_id}
                )
                files_processed = result.scalar() or 0

                # Voice messages
                result = await session.execute(
                    "SELECT COUNT(*) FROM analytics WHERE user_id = :user_id AND action = 'voice_message'",
                    {"user_id": user_id}
                )
                voice_messages = result.scalar() or 0

                return {
                    "total_messages": total_messages,
                    "ai_requests": ai_requests,
                    "images_processed": images_processed,
                    "files_processed": files_processed,
                    "voice_messages": voice_messages
                }

        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
            return {}

    async def get_detailed_stats(self, user_id: int) -> Dict[str, Any]:
        """Get detailed analytics for a user."""
        try:
            async with async_session() as session:
                # Week statistics
                week_ago = datetime.now(timezone.utc) - timedelta(days=7)

                result = await session.execute(
                    "SELECT COUNT(*) FROM conversations WHERE user_id = :user_id AND timestamp >= :week_ago",
                    {"user_id": user_id, "week_ago": week_ago}
                )
                week_messages = result.scalar() or 0

                result = await session.execute(
                    "SELECT COUNT(*) FROM analytics WHERE user_id = :user_id AND timestamp >= :week_ago",
                    {"user_id": user_id, "week_ago": week_ago}
                )
                week_ai_requests = result.scalar() or 0

                # Average response time
                result = await session.execute(
                    "SELECT AVG(processing_time) FROM analytics WHERE user_id = :user_id AND processing_time IS NOT NULL",
                    {"user_id": user_id}
                )
                avg_response_time = result.scalar() or 0

                # Feature usage percentages
                result = await session.execute(
                    "SELECT action, COUNT(*) as count FROM analytics WHERE user_id = :user_id GROUP BY action",
                    {"user_id": user_id}
                )

                action_counts = {
                    row.action: row.count for row in result.fetchall()}
                total_actions = sum(action_counts.values()) or 1

                # AI model usage
                result = await session.execute(
                    "SELECT ai_model_used, COUNT(*) as count FROM analytics WHERE user_id = :user_id AND ai_model_used IS NOT NULL GROUP BY ai_model_used",
                    {"user_id": user_id}
                )

                model_counts = {
                    row.ai_model_used: row.count for row in result.fetchall()}
                total_ai_usage = sum(model_counts.values()) or 1

                return {
                    "week_messages": week_messages,
                    "week_ai_requests": week_ai_requests,
                    "avg_response_time": avg_response_time,
                    "text_usage": round((action_counts.get("text_message", 0) / total_actions) * 100, 1),
                    "image_usage": round((action_counts.get("image_analysis", 0) / total_actions) * 100, 1),
                    "voice_usage": round((action_counts.get("voice_message", 0) / total_actions) * 100, 1),
                    "file_usage": round((action_counts.get("file_processing", 0) / total_actions) * 100, 1),
                    "gemini_usage": round((model_counts.get("gemini", 0) / total_ai_usage) * 100, 1),
                    "openai_usage": round((model_counts.get("openai", 0) / total_ai_usage) * 100, 1),
                    "anthropic_usage": round((model_counts.get("anthropic", 0) / total_ai_usage) * 100, 1),
                    "learning_sessions": action_counts.get("learning_session", 0),
                    # Simplified
                    "topics_count": len(set([details.get('topic', '') for details in []])),
                    "questions_count": action_counts.get("text_message", 0)
                }

        except Exception as e:
            logger.error(f"Error getting detailed stats: {e}")
            return {}

    async def generate_usage_chart(self, user_id: int) -> Dict[str, Any]:
        """Generate usage chart data for the last 7 days."""
        try:
            async with async_session() as session:
                chart_data = []

                for i in range(7):
                    date = datetime.now(timezone.utc) - timedelta(days=i)
                    start_date = date.replace(
                        hour=0, minute=0, second=0, microsecond=0)
                    end_date = start_date + timedelta(days=1)

                    result = await session.execute(
                        "SELECT COUNT(*) FROM analytics WHERE user_id = :user_id AND timestamp >= :start_date AND timestamp < :end_date",
                        {"user_id": user_id, "start_date": start_date, "end_date": end_date}
                    )

                    count = result.scalar() or 0
                    chart_data.append({
                        "date": start_date.strftime("%Y-%m-%d"),
                        "count": count
                    })

                return {"chart_data": list(reversed(chart_data))}

        except Exception as e:
            logger.error(f"Error generating chart data: {e}")
            return {"chart_data": []}

    async def export_user_data(self, user_id: int) -> Dict[str, Any]:
        """Export all user data for GDPR compliance."""
        try:
            async with async_session() as session:
                # User profile
                result = await session.execute(
                    "SELECT * FROM users WHERE telegram_id = :user_id",
                    {"user_id": user_id}
                )
                user_data = result.fetchone()

                # Conversations
                result = await session.execute(
                    "SELECT * FROM conversations WHERE user_id = :user_id ORDER BY timestamp",
                    {"user_id": user_id}
                )
                conversations = [dict(row._mapping)
                                 for row in result.fetchall()]

                # Memories
                result = await session.execute(
                    "SELECT * FROM user_memory WHERE user_id = :user_id ORDER BY created_at",
                    {"user_id": user_id}
                )
                memories = [dict(row._mapping) for row in result.fetchall()]

                # Analytics
                result = await session.execute(
                    "SELECT * FROM analytics WHERE user_id = :user_id ORDER BY timestamp",
                    {"user_id": user_id}
                )
                analytics = [dict(row._mapping) for row in result.fetchall()]

                export_data = {
                    "export_date": datetime.now(
                        timezone.utc).isoformat(),
                    "user_profile": dict(
                        user_data._mapping) if user_data else {},
                    "conversations": conversations,
                    "personal_memories": memories,
                    "analytics": analytics,
                    "summary": {
                        "total_conversations": len(conversations),
                        "total_memories": len(memories),
                        "total_analytics_events": len(analytics),
                        "account_created": user_data.created_at.isoformat() if user_data else None}}

                return export_data

        except Exception as e:
            logger.error(f"Error exporting user data: {e}")
            return {"error": "Failed to export data"}

    async def get_global_stats(self) -> Dict[str, Any]:
        """Get global bot statistics (for admin use)."""
        try:
            async with async_session() as session:
                # Total users
                result = await session.execute("SELECT COUNT(*) FROM users")
                total_users = result.scalar() or 0

                # Active users (last 7 days)
                week_ago = datetime.now(timezone.utc) - timedelta(days=7)
                result = await session.execute(
                    "SELECT COUNT(DISTINCT user_id) FROM analytics WHERE timestamp >= :week_ago",
                    {"week_ago": week_ago}
                )
                active_users = result.scalar() or 0

                # Total conversations
                result = await session.execute("SELECT COUNT(*) FROM conversations")
                total_conversations = result.scalar() or 0

                # Most popular AI model
                result = await session.execute(
                    "SELECT ai_model_used, COUNT(*) as count FROM analytics WHERE ai_model_used IS NOT NULL GROUP BY ai_model_used ORDER BY count DESC LIMIT 1"
                )
                popular_model = result.fetchone()

                return {
                    "total_users": total_users,
                    "active_users_week": active_users,
                    "total_conversations": total_conversations,
                    "most_popular_ai": popular_model.ai_model_used if popular_model else "N/A"}

        except Exception as e:
            logger.error(f"Error getting global stats: {e}")
            return {}

    async def cleanup_old_data(self, days: int = 90):
        """Clean up old analytics data (retention policy)."""
        try:
            async with async_session() as session:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

                # Delete old analytics entries
                await session.execute(
                    "DELETE FROM analytics WHERE timestamp < :cutoff_date",
                    {"cutoff_date": cutoff_date}
                )

                await session.commit()
                logger.info(
                    f"Cleaned up analytics data older than {days} days")

        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
