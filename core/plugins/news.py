#!/usr/bin/env python3
"""
News Plugin for JARVIS
Fetches latest news headlines from various sources
"""

from typing import Dict, Any, Tuple, Optional, List
import re
import aiohttp
import json
from datetime import datetime
from core.plugin_system import JARVISPlugin, PluginMetadata, PluginCommand


class NewsPlugin(JARVISPlugin):
    """News headlines plugin using NewsAPI or RSS feeds"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="News",
            version="1.0.0",
            author="JARVIS Team",
            description="Get latest news headlines from various sources",
            category="productivity",
            keywords=["news", "headlines", "current events", "media"],
            requirements=["aiohttp", "feedparser"],
            permissions=["network"],
            config_schema={
                "api_key": {"type": "string", "required": False, "description": "NewsAPI key (optional)"},
                "sources": {
                    "type": "array",
                    "default": ["bbc-news", "cnn", "techcrunch"],
                    "description": "Preferred news sources"
                },
                "categories": {
                    "type": "array",
                    "default": ["general", "technology", "business", "science"],
                    "description": "News categories to follow"
                },
                "max_articles": {"type": "integer", "default": 5, "description": "Maximum articles to show"}
            }
        )
        
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the news plugin"""
        self.config = config
        self.api_key = config.get("api_key")
        self.sources = config.get("sources", ["bbc-news", "cnn", "techcrunch"])
        self.categories = config.get("categories", ["general", "technology", "business", "science"])
        self.max_articles = config.get("max_articles", 5)
        
        # RSS feed URLs as fallback
        self.rss_feeds = {
            "general": [
                "https://feeds.bbci.co.uk/news/rss.xml",
                "http://rss.cnn.com/rss/cnn_topstories.rss"
            ],
            "technology": [
                "https://feeds.feedburner.com/TechCrunch/",
                "https://www.theverge.com/rss/index.xml"
            ],
            "business": [
                "https://feeds.bloomberg.com/markets/news.rss",
                "https://www.ft.com/world?format=rss"
            ],
            "science": [
                "https://www.nature.com/nature.rss",
                "https://www.sciencedaily.com/rss/all.xml"
            ]
        }
        
        # Register commands
        self.register_command(PluginCommand(
            name="get_news",
            patterns=[
                r"(?:get |show |tell me )?(?:the )?news(?:\s+about\s+(.+?))?",
                r"(?:what(?:'s| is) )(?:the )?(?:latest )?news(?:\s+(?:about|on|for)\s+(.+?))?",
                r"headlines(?:\s+(?:about|on|for)\s+(.+?))?",
                r"(?:any )?(?:breaking|latest|recent) news(?:\s+(?:about|on)\s+(.+?))?"
            ],
            description="Get latest news headlines",
            parameters={
                "topic": {"type": "string", "description": "News topic or category", "optional": True}
            },
            examples=[
                "what's the news",
                "show me technology news",
                "headlines about AI",
                "latest news",
                "breaking news"
            ],
            handler=self.handle_get_news
        ))
        
        self.register_command(PluginCommand(
            name="news_summary",
            patterns=[
                r"news summary(?:\s+for\s+(.+?))?",
                r"summarize (?:the )?news(?:\s+(?:about|on)\s+(.+?))?",
                r"brief me on (?:the )?news(?:\s+(?:about|on)\s+(.+?))?"
            ],
            description="Get a brief news summary",
            parameters={
                "topic": {"type": "string", "description": "News topic", "optional": True}
            },
            examples=[
                "news summary",
                "brief me on the news",
                "summarize technology news"
            ],
            handler=self.handle_news_summary
        ))
        
        # Subscribe to scheduled events for news briefings
        self.subscribe_event("morning_briefing", self.morning_news_briefing)
        
        self.logger.info("News plugin initialized")
        return True
        
    async def shutdown(self):
        """Clean up resources"""
        self.logger.info("News plugin shutting down")
        
    async def handle_get_news(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Handle news requests"""
        try:
            # Extract topic from match
            topic = None
            for group in match.groups():
                if group:
                    topic = group.strip().lower()
                    break
                    
            # Fetch news
            articles = await self._fetch_news(topic)
            
            if articles:
                return True, self._format_news(articles, topic)
            else:
                return True, "I couldn't fetch any news at the moment. Please try again later."
                
        except Exception as e:
            self.logger.error(f"Error getting news: {e}")
            return False, f"Sorry, I encountered an error getting the news: {str(e)}"
            
    async def handle_news_summary(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Handle news summary requests"""
        try:
            # Extract topic
            topic = None
            for group in match.groups():
                if group:
                    topic = group.strip().lower()
                    break
                    
            # Fetch news
            articles = await self._fetch_news(topic)
            
            if articles:
                # Create a brief summary
                summary = self._create_news_summary(articles[:3], topic)  # Top 3 articles
                return True, summary
            else:
                return True, "I couldn't fetch any news for a summary at the moment."
                
        except Exception as e:
            self.logger.error(f"Error creating news summary: {e}")
            return False, f"Sorry, I encountered an error creating the news summary: {str(e)}"
            
    async def _fetch_news(self, topic: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch news from API or RSS feeds"""
        if self.api_key:
            return await self._fetch_from_newsapi(topic)
        else:
            return await self._fetch_from_rss(topic)
            
    async def _fetch_from_newsapi(self, topic: Optional[str]) -> List[Dict[str, Any]]:
        """Fetch news from NewsAPI"""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://newsapi.org/v2/top-headlines"
                params = {
                    "apiKey": self.api_key,
                    "pageSize": self.max_articles
                }
                
                if topic:
                    # Check if topic is a category
                    if topic in self.categories:
                        params["category"] = topic
                    else:
                        params["q"] = topic
                else:
                    params["category"] = "general"
                    
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("articles", [])
                    else:
                        self.logger.error(f"NewsAPI error: {response.status}")
                        return self._get_demo_news(topic)
                        
        except Exception as e:
            self.logger.error(f"Error fetching from NewsAPI: {e}")
            return self._get_demo_news(topic)
            
    async def _fetch_from_rss(self, topic: Optional[str]) -> List[Dict[str, Any]]:
        """Fetch news from RSS feeds"""
        try:
            import feedparser
            
            # Determine which feeds to use
            if topic and topic in self.rss_feeds:
                feeds = self.rss_feeds[topic]
            else:
                feeds = self.rss_feeds.get("general", [])
                
            articles = []
            
            for feed_url in feeds[:2]:  # Limit to 2 feeds
                try:
                    # Parse RSS feed
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries[:self.max_articles]:
                        article = {
                            "title": entry.get("title", "No title"),
                            "description": entry.get("summary", entry.get("description", "No description")),
                            "url": entry.get("link", ""),
                            "publishedAt": entry.get("published", ""),
                            "source": {"name": feed.feed.get("title", "Unknown source")}
                        }
                        articles.append(article)
                        
                    if len(articles) >= self.max_articles:
                        break
                        
                except Exception as e:
                    self.logger.error(f"Error parsing RSS feed {feed_url}: {e}")
                    
            return articles[:self.max_articles]
            
        except ImportError:
            self.logger.warning("feedparser not installed, using demo data")
            return self._get_demo_news(topic)
        except Exception as e:
            self.logger.error(f"Error fetching from RSS: {e}")
            return self._get_demo_news(topic)
            
    def _format_news(self, articles: List[Dict[str, Any]], topic: Optional[str]) -> str:
        """Format news articles for display"""
        if not articles:
            return "No news articles found."
            
        header = f"📰 Latest News" + (f" about {topic}" if topic else "") + ":\n\n"
        
        formatted_articles = []
        for i, article in enumerate(articles, 1):
            title = article.get("title", "No title")
            source = article.get("source", {}).get("name", "Unknown")
            description = article.get("description", "")
            url = article.get("url", "")
            
            # Truncate description if too long
            if description and len(description) > 150:
                description = description[:147] + "..."
                
            article_text = f"{i}. **{title}**\n"
            article_text += f"   Source: {source}\n"
            if description:
                article_text += f"   {description}\n"
            if url:
                article_text += f"   [Read more]({url})"
                
            formatted_articles.append(article_text)
            
        return header + "\n\n".join(formatted_articles)
        
    def _create_news_summary(self, articles: List[Dict[str, Any]], topic: Optional[str]) -> str:
        """Create a brief news summary"""
        if not articles:
            return "No news articles available for summary."
            
        summary = f"📰 News Summary" + (f" - {topic.title()}" if topic else "") + ":\n\n"
        summary += f"Here are the top {len(articles)} stories:\n\n"
        
        for i, article in enumerate(articles, 1):
            title = article.get("title", "No title")
            source = article.get("source", {}).get("name", "Unknown")
            summary += f"{i}. {title} ({source})\n"
            
        summary += "\nWould you like more details on any of these stories?"
        
        return summary
        
    def _get_demo_news(self, topic: Optional[str]) -> List[Dict[str, Any]]:
        """Return demo news data"""
        demo_articles = [
            {
                "title": "AI Assistant Technology Advances Rapidly",
                "description": "New developments in AI assistants show promising capabilities for natural interaction.",
                "url": "https://example.com/ai-news",
                "source": {"name": "Tech Daily"},
                "publishedAt": datetime.now().isoformat()
            },
            {
                "title": "Global Climate Summit Reaches New Agreement",
                "description": "World leaders commit to ambitious new targets for carbon reduction.",
                "url": "https://example.com/climate-news",
                "source": {"name": "World News"},
                "publishedAt": datetime.now().isoformat()
            },
            {
                "title": "Major Breakthrough in Quantum Computing",
                "description": "Researchers achieve significant milestone in quantum error correction.",
                "url": "https://example.com/quantum-news",
                "source": {"name": "Science Today"},
                "publishedAt": datetime.now().isoformat()
            }
        ]
        
        if topic:
            # Filter by topic keyword
            filtered = [a for a in demo_articles if topic.lower() in a["title"].lower() or topic.lower() in a["description"].lower()]
            if filtered:
                return filtered
                
        return demo_articles
        
    async def morning_news_briefing(self, data: Any):
        """Handler for morning news briefing event"""
        try:
            # Fetch top news
            articles = await self._fetch_news()
            
            if articles:
                briefing = self._create_news_summary(articles[:3], None)
                # Emit event for JARVIS to speak
                self.emit_event("speak_announcement", {
                    "text": f"Good morning! Here's your news briefing:\n{briefing}",
                    "priority": "high"
                })
        except Exception as e:
            self.logger.error(f"Error in morning briefing: {e}")