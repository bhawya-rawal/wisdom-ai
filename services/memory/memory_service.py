import logging
from typing import List, Dict, Any, Tuple
from sqlmodel import Session, select
from models.db_models import ChatMessage, ChatSummary
from services.llm.llm_service import llm_service

logger = logging.getLogger(__name__)

class MemoryService:
    """Manages multi-turn conversation memory, chat log history, and rolling summaries"""
    
    def __init__(self, message_history_limit: int = 10, summary_trigger_count: int = 6):
        self.history_limit = message_history_limit
        self.summary_trigger_count = summary_trigger_count

    def get_conversation_context(self, user_id: int, session: Session) -> Tuple[str, List[Dict[str, str]]]:
        """
        Retrieve accumulated summary and recent messages for LLM context
        Returns:
            Tuple containing:
            - history_summary: cumulative context summary string
            - recent_messages: list of recent message dicts e.g., [{"role": "user", "content": "..."}]
        """
        # 1. Fetch cumulative summary (from the latest ChatSummary entry)
        summary_stmt = select(ChatSummary).where(ChatSummary.user_id == user_id).order_by(ChatSummary.date.desc())
        latest_summary_record = session.exec(summary_stmt).first()
        history_summary = latest_summary_record.summary if latest_summary_record else ""

        # 2. Fetch recent chat messages
        msg_stmt = select(ChatMessage).where(ChatMessage.user_id == user_id).order_by(ChatMessage.timestamp.desc()).limit(self.history_limit)
        db_messages = session.exec(msg_stmt).all()
        
        # Sort messages back to chronological order
        db_messages.reverse()
        
        recent_messages = [{"role": msg.role, "content": msg.content} for msg in db_messages]
        return history_summary, recent_messages

    def add_message(self, user_id: int, role: str, content: str, session: Session) -> ChatMessage:
        """Persist a single chat turn (user query or assistant response)"""
        new_msg = ChatMessage(user_id=user_id, role=role, content=content)
        session.add(new_msg)
        session.commit()
        session.refresh(new_msg)
        return new_msg

    def update_summary_if_needed(self, user_id: int, session: Session) -> str:
        """
        Check if unsummarized message logs exceed thresholds. 
        If yes, execute LLM summarization and write a new ChatSummary record.
        """
        # Check total messages count
        count_stmt = select(ChatMessage).where(ChatMessage.user_id == user_id)
        total_messages = len(session.exec(count_stmt).all())
        
        # Get latest summary details
        summary_stmt = select(ChatSummary).where(ChatSummary.user_id == user_id).order_by(ChatSummary.date.desc())
        latest_summary_record = session.exec(summary_stmt).first()
        
        current_summary = latest_summary_record.summary if latest_summary_record else ""
        
        # If we have enough messages to warrant updating the summary
        if total_messages > 0 and (total_messages % self.summary_trigger_count == 0 or not latest_summary_record):
            # Fetch last batch of messages
            limit = self.summary_trigger_count if latest_summary_record else 10
            msg_stmt = select(ChatMessage).where(ChatMessage.user_id == user_id).order_by(ChatMessage.timestamp.desc()).limit(limit)
            new_msgs_db = session.exec(msg_stmt).all()
            new_msgs_db.reverse()
            
            new_msgs = [{"role": m.role, "content": m.content} for m in new_msgs_db]
            
            # Generate summary via LLM
            logger.info("Triggering rolling conversation summary update for user_id=%d", user_id)
            updated_summary = llm_service.summarize_history(current_summary, new_msgs)
            
            # Save new ChatSummary row
            new_summary_record = ChatSummary(
                user_id=user_id,
                summary=updated_summary,
                mood=None,  # Will be updated by chat route
                verse_id=None # Will be updated by chat route
            )
            session.add(new_summary_record)
            session.commit()
            return updated_summary
            
        return current_summary

# Global memory service instance
memory_service = MemoryService()
