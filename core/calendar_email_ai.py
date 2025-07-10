#!/usr/bin/env python3
"""
JARVIS Calendar & Email AI Management
Intelligent email summaries, calendar management, and meeting assistance.
"""

import os
import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta, time
import logging
from pathlib import Path
import imaplib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import email
from email.header import decode_header
import caldav
from icalendar import Calendar, Event
import pytz

# Try to import AI libraries for summarization
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger("jarvis.calendar_email")


@dataclass
class EmailMessage:
    """Represents an email message"""
    uid: str
    subject: str
    sender: str
    recipients: List[str]
    date: datetime
    body: str
    importance: str = "normal"  # low, normal, high
    category: str = "general"  # work, personal, newsletter, spam
    summary: Optional[str] = None
    action_required: bool = False
    

@dataclass
class CalendarEvent:
    """Represents a calendar event"""
    event_id: str
    title: str
    start: datetime
    end: datetime
    location: Optional[str] = None
    description: Optional[str] = None
    attendees: List[str] = None
    reminders: List[int] = None  # Minutes before event
    recurring: bool = False
    meeting_url: Optional[str] = None
    

@dataclass
class MeetingPrep:
    """Meeting preparation information"""
    event: CalendarEvent
    agenda_items: List[str]
    relevant_emails: List[EmailMessage]
    action_items: List[str]
    suggested_talking_points: List[str]
    

class EmailAI:
    """AI-powered email management"""
    
    def __init__(self, email_config: Dict[str, Any]):
        self.config = email_config
        self.imap = None
        self.smtp = None
        self.connected = False
        
        # AI providers
        self.ai_provider = self._setup_ai_provider()
        
    def _setup_ai_provider(self):
        """Setup AI provider for summarization"""
        api_key = os.getenv("OPENAI_API_KEY")
        if OPENAI_AVAILABLE and api_key:
            openai.api_key = api_key
            return "openai"
            
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if ANTHROPIC_AVAILABLE and api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=api_key)
            return "anthropic"
            
        return None
        
    def connect(self) -> bool:
        """Connect to email server"""
        try:
            # IMAP connection
            self.imap = imaplib.IMAP4_SSL(
                self.config.get("imap_server", "imap.gmail.com"),
                self.config.get("imap_port", 993)
            )
            self.imap.login(
                self.config["email"],
                self.config["password"]
            )
            
            # SMTP connection
            self.smtp = smtplib.SMTP(
                self.config.get("smtp_server", "smtp.gmail.com"),
                self.config.get("smtp_port", 587)
            )
            self.smtp.starttls()
            self.smtp.login(
                self.config["email"],
                self.config["password"]
            )
            
            self.connected = True
            logger.info("Email connection established")
            return True
            
        except Exception as e:
            logger.error(f"Email connection failed: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from email server"""
        if self.imap:
            self.imap.close()
            self.imap.logout()
        if self.smtp:
            self.smtp.quit()
        self.connected = False
        
    async def fetch_recent_emails(self, folder: str = "INBOX", limit: int = 20) -> List[EmailMessage]:
        """Fetch recent emails"""
        if not self.connected:
            return []
            
        emails = []
        
        try:
            self.imap.select(folder)
            
            # Search for recent emails
            _, message_ids = self.imap.search(None, "ALL")
            message_ids = message_ids[0].split()[-limit:]  # Get last N emails
            
            for msg_id in reversed(message_ids):
                _, msg_data = self.imap.fetch(msg_id, "(RFC822)")
                
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        
                        # Parse email
                        email_msg = self._parse_email(msg_id.decode(), msg)
                        emails.append(email_msg)
                        
        except Exception as e:
            logger.error(f"Failed to fetch emails: {e}")
            
        return emails
        
    def _parse_email(self, uid: str, msg) -> EmailMessage:
        """Parse email message"""
        # Decode subject
        subject, encoding = decode_header(msg["Subject"])[0]
        if isinstance(subject, bytes):
            subject = subject.decode(encoding or "utf-8")
            
        # Get sender
        sender = msg.get("From", "")
        
        # Get recipients
        recipients = []
        for field in ["To", "Cc"]:
            if msg.get(field):
                recipients.extend([addr.strip() for addr in msg[field].split(",")])
                
        # Get date
        date_str = msg.get("Date", "")
        try:
            date = email.utils.parsedate_to_datetime(date_str)
        except:
            date = datetime.now()
            
        # Get body
        body = self._get_email_body(msg)
        
        # Determine importance
        importance = "normal"
        if msg.get("Importance", "").lower() == "high":
            importance = "high"
        elif msg.get("X-Priority", "3") in ["1", "2"]:
            importance = "high"
            
        return EmailMessage(
            uid=uid,
            subject=subject,
            sender=sender,
            recipients=recipients,
            date=date,
            body=body,
            importance=importance
        )
        
    def _get_email_body(self, msg) -> str:
        """Extract email body"""
        body = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        body = part.get_payload(decode=True).decode()
                        break
                    except:
                        pass
        else:
            try:
                body = msg.get_payload(decode=True).decode()
            except:
                body = str(msg.get_payload())
                
        return body
        
    async def summarize_email(self, email_msg: EmailMessage) -> str:
        """Generate AI summary of email"""
        if not self.ai_provider or not email_msg.body:
            return "No summary available"
            
        prompt = f"""Summarize this email in 2-3 sentences:
        Subject: {email_msg.subject}
        From: {email_msg.sender}
        Body: {email_msg.body[:1000]}..."""
        
        try:
            if self.ai_provider == "openai":
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100
                )
                return response.choices[0].message.content
                
            elif self.ai_provider == "anthropic":
                response = self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
        except Exception as e:
            logger.error(f"Email summarization failed: {e}")
            
        # Fallback to simple summary
        return f"Email from {email_msg.sender} about {email_msg.subject}"
        
    async def categorize_emails(self, emails: List[EmailMessage]) -> List[EmailMessage]:
        """Categorize emails using AI"""
        for email_msg in emails:
            # Simple rule-based categorization (could be enhanced with AI)
            if any(word in email_msg.sender.lower() for word in ["noreply", "newsletter", "updates"]):
                email_msg.category = "newsletter"
            elif any(word in email_msg.subject.lower() for word in ["meeting", "call", "sync"]):
                email_msg.category = "work"
                email_msg.action_required = True
            elif any(word in email_msg.subject.lower() for word in ["urgent", "asap", "important"]):
                email_msg.importance = "high"
                email_msg.action_required = True
                
            # Generate summary
            email_msg.summary = await self.summarize_email(email_msg)
            
        return emails
        
    async def get_email_insights(self, emails: List[EmailMessage]) -> Dict[str, Any]:
        """Get insights from emails"""
        insights = {
            "total": len(emails),
            "unread": 0,  # Would need to track read status
            "high_priority": sum(1 for e in emails if e.importance == "high"),
            "action_required": sum(1 for e in emails if e.action_required),
            "by_category": {},
            "top_senders": {},
            "summary": ""
        }
        
        # Count by category
        for email_msg in emails:
            category = email_msg.category
            insights["by_category"][category] = insights["by_category"].get(category, 0) + 1
            
            # Track top senders
            sender = email_msg.sender.split("<")[0].strip()
            insights["top_senders"][sender] = insights["top_senders"].get(sender, 0) + 1
            
        # Sort top senders
        insights["top_senders"] = dict(sorted(
            insights["top_senders"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5])
        
        # Generate overall summary
        if insights["high_priority"] > 0:
            insights["summary"] = f"You have {insights['high_priority']} high-priority emails. "
        if insights["action_required"] > 0:
            insights["summary"] += f"{insights['action_required']} emails need your attention."
            
        return insights
        
    async def draft_reply(self, email_msg: EmailMessage, instruction: str) -> str:
        """Draft an email reply using AI"""
        if not self.ai_provider:
            return "AI reply generation not available"
            
        prompt = f"""Draft a professional email reply based on this instruction: {instruction}
        
        Original email:
        From: {email_msg.sender}
        Subject: {email_msg.subject}
        Body: {email_msg.body[:500]}...
        
        Keep the reply concise and professional."""
        
        try:
            if self.ai_provider == "openai":
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200
                )
                return response.choices[0].message.content
                
            elif self.ai_provider == "anthropic":
                response = self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
        except Exception as e:
            logger.error(f"Reply generation failed: {e}")
            
        return "Could not generate reply"


class CalendarAI:
    """AI-powered calendar management"""
    
    def __init__(self, calendar_config: Dict[str, Any]):
        self.config = calendar_config
        self.client = None
        self.calendar = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to calendar service"""
        try:
            # CalDAV connection (works with iCloud, Google, etc.)
            self.client = caldav.DAVClient(
                url=self.config["caldav_url"],
                username=self.config["username"],
                password=self.config["password"]
            )
            
            principal = self.client.principal()
            calendars = principal.calendars()
            
            if calendars:
                self.calendar = calendars[0]  # Use first calendar
                self.connected = True
                logger.info("Calendar connection established")
                return True
                
        except Exception as e:
            logger.error(f"Calendar connection failed: {e}")
            
        return False
        
    async def get_upcoming_events(self, days: int = 7) -> List[CalendarEvent]:
        """Get upcoming calendar events"""
        if not self.connected:
            return []
            
        events = []
        start_date = datetime.now()
        end_date = start_date + timedelta(days=days)
        
        try:
            # Search for events
            results = self.calendar.date_search(
                start=start_date,
                end=end_date,
                expand=True
            )
            
            for event in results:
                cal_event = self._parse_calendar_event(event)
                if cal_event:
                    events.append(cal_event)
                    
            # Sort by start time
            events.sort(key=lambda x: x.start)
            
        except Exception as e:
            logger.error(f"Failed to fetch calendar events: {e}")
            
        return events
        
    def _parse_calendar_event(self, event) -> Optional[CalendarEvent]:
        """Parse calendar event"""
        try:
            cal = Calendar.from_ical(event.data)
            
            for component in cal.walk():
                if component.name == "VEVENT":
                    return CalendarEvent(
                        event_id=str(component.get('uid', '')),
                        title=str(component.get('summary', '')),
                        start=component.get('dtstart').dt,
                        end=component.get('dtend').dt,
                        location=str(component.get('location', '')),
                        description=str(component.get('description', '')),
                        attendees=self._get_attendees(component),
                        meeting_url=self._extract_meeting_url(component)
                    )
                    
        except Exception as e:
            logger.error(f"Failed to parse calendar event: {e}")
            
        return None
        
    def _get_attendees(self, component) -> List[str]:
        """Extract attendees from event"""
        attendees = []
        
        for attendee in component.get('attendee', []):
            if hasattr(attendee, 'to_ical'):
                email = attendee.to_ical().decode().replace('mailto:', '')
                attendees.append(email)
                
        return attendees
        
    def _extract_meeting_url(self, component) -> Optional[str]:
        """Extract meeting URL from event"""
        description = str(component.get('description', ''))
        location = str(component.get('location', ''))
        
        # Look for common meeting URLs
        url_patterns = [
            r'https://[^\s]*zoom\.us/[^\s]+',
            r'https://[^\s]*meet\.google\.com/[^\s]+',
            r'https://[^\s]*teams\.microsoft\.com/[^\s]+',
            r'https://[^\s]*webex\.com/[^\s]+'
        ]
        
        for pattern in url_patterns:
            # Check description
            match = re.search(pattern, description)
            if match:
                return match.group(0)
                
            # Check location
            match = re.search(pattern, location)
            if match:
                return match.group(0)
                
        return None
        
    async def check_conflicts(self, start: datetime, end: datetime) -> List[CalendarEvent]:
        """Check for calendar conflicts"""
        events = await self.get_upcoming_events(days=30)
        
        conflicts = []
        for event in events:
            # Check if times overlap
            if (event.start < end and event.end > start):
                conflicts.append(event)
                
        return conflicts
        
    async def create_event(self, title: str, start: datetime, end: datetime,
                          location: str = "", description: str = "") -> bool:
        """Create a calendar event"""
        if not self.connected:
            return False
            
        try:
            cal = Calendar()
            event = Event()
            
            event.add('summary', title)
            event.add('dtstart', start)
            event.add('dtend', end)
            
            if location:
                event.add('location', location)
            if description:
                event.add('description', description)
                
            cal.add_component(event)
            
            self.calendar.save_event(cal.to_ical())
            return True
            
        except Exception as e:
            logger.error(f"Failed to create calendar event: {e}")
            return False
            
    async def prepare_for_meeting(self, event: CalendarEvent, email_ai: EmailAI) -> MeetingPrep:
        """Prepare for an upcoming meeting"""
        prep = MeetingPrep(
            event=event,
            agenda_items=[],
            relevant_emails=[],
            action_items=[],
            suggested_talking_points=[]
        )
        
        # Find relevant emails
        if email_ai.connected:
            all_emails = await email_ai.fetch_recent_emails(limit=50)
            
            # Filter emails related to meeting
            for email_msg in all_emails:
                # Check if email is from attendees or mentions meeting topic
                if any(attendee in email_msg.sender for attendee in event.attendees):
                    prep.relevant_emails.append(email_msg)
                elif event.title.lower() in email_msg.subject.lower():
                    prep.relevant_emails.append(email_msg)
                    
        # Extract action items from emails
        for email_msg in prep.relevant_emails:
            if "action" in email_msg.body.lower() or "todo" in email_msg.body.lower():
                # Simple extraction (could be enhanced with AI)
                lines = email_msg.body.split('\n')
                for line in lines:
                    if any(marker in line.lower() for marker in ["action:", "todo:", "- [ ]"]):
                        prep.action_items.append(line.strip())
                        
        # Generate talking points
        if event.description:
            # Extract from description
            prep.suggested_talking_points.append(f"Review: {event.description[:100]}...")
            
        if prep.action_items:
            prep.suggested_talking_points.append(f"Discuss {len(prep.action_items)} action items")
            
        return prep


class CalendarEmailCommandProcessor:
    """Process calendar and email voice commands"""
    
    def __init__(self, email_ai: EmailAI, calendar_ai: CalendarAI):
        self.email_ai = email_ai
        self.calendar_ai = calendar_ai
        
    async def process_command(self, command: str) -> Tuple[bool, str]:
        """Process calendar/email command"""
        command_lower = command.lower().strip()
        
        # Email commands
        if "email" in command_lower or "mail" in command_lower:
            if "check" in command_lower or "new" in command_lower:
                return await self._check_emails()
            elif "summarize" in command_lower:
                return await self._summarize_emails()
            elif "reply" in command_lower:
                return await self._handle_reply(command)
                
        # Calendar commands
        if "calendar" in command_lower or "schedule" in command_lower or "meeting" in command_lower:
            if "today" in command_lower:
                return await self._get_todays_events()
            elif "tomorrow" in command_lower:
                return await self._get_tomorrows_events()
            elif "week" in command_lower:
                return await self._get_week_events()
            elif "create" in command_lower or "schedule" in command_lower:
                return await self._create_event(command)
            elif "prepare" in command_lower:
                return await self._prepare_meeting()
                
        return False, "I didn't understand that email/calendar command"
        
    async def _check_emails(self) -> Tuple[bool, str]:
        """Check for new emails"""
        if not self.email_ai.connected:
            return False, "Email not connected"
            
        emails = await self.email_ai.fetch_recent_emails(limit=10)
        emails = await self.email_ai.categorize_emails(emails)
        insights = await self.email_ai.get_email_insights(emails)
        
        response = f"You have {insights['total']} recent emails. "
        
        if insights['high_priority'] > 0:
            response += f"{insights['high_priority']} are high priority. "
            
        if insights['action_required'] > 0:
            response += f"{insights['action_required']} require action. "
            
        # Mention top sender
        if insights['top_senders']:
            top_sender = list(insights['top_senders'].keys())[0]
            count = insights['top_senders'][top_sender]
            response += f"Most emails are from {top_sender} ({count}). "
            
        return True, response
        
    async def _summarize_emails(self) -> Tuple[bool, str]:
        """Summarize recent emails"""
        if not self.email_ai.connected:
            return False, "Email not connected"
            
        emails = await self.email_ai.fetch_recent_emails(limit=5)
        emails = await self.email_ai.categorize_emails(emails)
        
        response = "Here are your recent email summaries: "
        
        for i, email_msg in enumerate(emails[:3], 1):
            response += f"\n{i}. {email_msg.summary}"
            
        return True, response
        
    async def _get_todays_events(self) -> Tuple[bool, str]:
        """Get today's calendar events"""
        if not self.calendar_ai.connected:
            return False, "Calendar not connected"
            
        events = await self.calendar_ai.get_upcoming_events(days=1)
        
        today = datetime.now().date()
        todays_events = [e for e in events if e.start.date() == today]
        
        if not todays_events:
            return True, "You have no events scheduled for today."
            
        response = f"You have {len(todays_events)} events today: "
        
        for event in todays_events:
            time_str = event.start.strftime("%I:%M %p")
            response += f"\n• {time_str}: {event.title}"
            if event.location:
                response += f" at {event.location}"
                
        return True, response
        
    async def _get_tomorrows_events(self) -> Tuple[bool, str]:
        """Get tomorrow's calendar events"""
        if not self.calendar_ai.connected:
            return False, "Calendar not connected"
            
        events = await self.calendar_ai.get_upcoming_events(days=2)
        
        tomorrow = (datetime.now() + timedelta(days=1)).date()
        tomorrows_events = [e for e in events if e.start.date() == tomorrow]
        
        if not tomorrows_events:
            return True, "You have no events scheduled for tomorrow."
            
        response = f"You have {len(tomorrows_events)} events tomorrow: "
        
        for event in tomorrows_events:
            time_str = event.start.strftime("%I:%M %p")
            response += f"\n• {time_str}: {event.title}"
            
        return True, response
        
    async def _get_week_events(self) -> Tuple[bool, str]:
        """Get this week's events"""
        if not self.calendar_ai.connected:
            return False, "Calendar not connected"
            
        events = await self.calendar_ai.get_upcoming_events(days=7)
        
        if not events:
            return True, "You have no events scheduled this week."
            
        response = f"You have {len(events)} events this week:"
        
        current_date = None
        for event in events[:10]:  # Limit to 10 events
            event_date = event.start.date()
            
            if event_date != current_date:
                current_date = event_date
                response += f"\n\n{event_date.strftime('%A, %B %d')}:"
                
            time_str = event.start.strftime("%I:%M %p")
            response += f"\n• {time_str}: {event.title}"
            
        return True, response
        
    async def _create_event(self, command: str) -> Tuple[bool, str]:
        """Create a calendar event from natural language"""
        # This is a simplified version - could be enhanced with better NLP
        
        # Extract meeting details
        if "tomorrow" in command:
            date = datetime.now() + timedelta(days=1)
        else:
            date = datetime.now()
            
        # Default to 1 hour meeting at 2 PM
        start = date.replace(hour=14, minute=0, second=0, microsecond=0)
        end = start + timedelta(hours=1)
        
        # Extract title (everything after "meeting about" or "event for")
        title = "New Meeting"
        if "about" in command:
            title = command.split("about")[-1].strip()
        elif "for" in command:
            title = command.split("for")[-1].strip()
            
        # Check for conflicts
        conflicts = await self.calendar_ai.check_conflicts(start, end)
        
        if conflicts:
            return False, f"There's a conflict with '{conflicts[0].title}' at that time"
            
        # Create event
        success = await self.calendar_ai.create_event(
            title=title,
            start=start,
            end=end
        )
        
        if success:
            return True, f"Created '{title}' for {start.strftime('%A at %I:%M %p')}"
        else:
            return False, "Failed to create calendar event"
            
    async def _prepare_meeting(self) -> Tuple[bool, str]:
        """Prepare for next meeting"""
        if not self.calendar_ai.connected:
            return False, "Calendar not connected"
            
        events = await self.calendar_ai.get_upcoming_events(days=1)
        
        if not events:
            return True, "No upcoming meetings to prepare for"
            
        next_event = events[0]
        prep = await self.calendar_ai.prepare_for_meeting(next_event, self.email_ai)
        
        response = f"Preparing for '{next_event.title}' at {next_event.start.strftime('%I:%M %p')}. "
        
        if prep.relevant_emails:
            response += f"Found {len(prep.relevant_emails)} related emails. "
            
        if prep.action_items:
            response += f"You have {len(prep.action_items)} action items to discuss. "
            
        if next_event.meeting_url:
            response += f"Meeting URL is ready. "
            
        return True, response
        
    async def _handle_reply(self, command: str) -> Tuple[bool, str]:
        """Handle email reply request"""
        # This would need more context about which email to reply to
        return True, "To reply to an email, please specify which email and what you'd like to say"


# Integration function for JARVIS
def integrate_calendar_email_with_jarvis(jarvis_instance, email_config: Dict, calendar_config: Dict):
    """Integrate calendar and email capabilities with JARVIS"""
    
    # Initialize AI components
    email_ai = EmailAI(email_config)
    calendar_ai = CalendarAI(calendar_config)
    
    # Try to connect
    email_connected = email_ai.connect()
    calendar_connected = calendar_ai.connect()
    
    # Create processor
    processor = CalendarEmailCommandProcessor(email_ai, calendar_ai)
    
    # Add to JARVIS
    jarvis_instance.email_ai = email_ai
    jarvis_instance.calendar_ai = calendar_ai
    jarvis_instance.calendar_email_processor = processor
    
    logger.info(f"Calendar/Email integration initialized - Email: {email_connected}, Calendar: {calendar_connected}")
    
    return processor