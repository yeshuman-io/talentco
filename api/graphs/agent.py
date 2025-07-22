"""
TalentCo Career Agent - Talent matching and career guidance agent.

Clean architecture with role-based system prompting and career-focused tools.
"""
import os
import django
import logging
import time
from django.conf import settings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Django setup - must be done before importing Django models
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'talentco.settings')
if not settings.configured:
    django.setup()

# Set up logger
logger = logging.getLogger('graphs.agent')

# Now we can safely import Django models and LangGraph
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langgraph.prebuilt import create_react_agent
from apps.memories.backends import DjangoMemoryBackend
from pydantic import BaseModel, Field
from typing import Optional, Type

# Initialize our custom Django memory backend
memory_client = DjangoMemoryBackend()
logger.info("Initialized TalentCo Career Agent with Django memory backend")


class StoreMemoryInput(BaseModel):
    """Input schema for storing career memories."""
    content: str = Field(
        description="The career information to store",
        min_length=1,
        max_length=10000,
        examples=[
            "User is interested in sustainability roles",
            "Has 5 years experience in renewable energy",
            "Looking for remote work opportunities"
        ]
    )
    session_id: Optional[str] = Field(
        default=None, 
        description="Optional session identifier (defaults to 'default')",
        pattern=r"^[a-zA-Z0-9_-]*$",
        max_length=100
    )


class SearchMemoryInput(BaseModel):
    """Input schema for searching career memories."""
    query: str = Field(
        description="Search query for career information",
        min_length=1,
        max_length=1000,
        examples=[
            "What are my career interests?",
            "What experience do I have?",
            "What type of work am I looking for?"
        ]
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier (defaults to 'default')",
        pattern=r"^[a-zA-Z0-9_-]*$",
        max_length=100
    )


class StoreMemoryTool(BaseTool):
    """Tool for storing career-related memories."""
    
    name: str = "store_career_memory"
    description: str = """
    Store important career information, preferences, and experiences.
    Use this for:
    - Career interests and goals
    - Work experience and skills
    - Job preferences (location, salary, type)
    - Professional achievements
    - Learning goals and development areas
    """
    args_schema: Type[BaseModel] = StoreMemoryInput

    def _run(
        self, 
        content: str, 
        session_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Store career memory synchronously."""
        try:
            session_id = session_id or "default"
            
            # Categorize based on content keywords
            category = "career"
            subcategory = "general"
            memory_type = "factual"
            
            # Simple keyword-based categorization
            content_lower = content.lower()
            if any(word in content_lower for word in ["experience", "worked", "job", "role", "position"]):
                subcategory = "experience"
                memory_type = "episodic"
            elif any(word in content_lower for word in ["skill", "know", "proficient", "expert"]):
                subcategory = "skills"
            elif any(word in content_lower for word in ["want", "looking", "interested", "prefer"]):
                subcategory = "preferences"
            elif any(word in content_lower for word in ["goal", "aim", "objective", "plan"]):
                subcategory = "goals"
            
            # Prepare memory data
            messages = [{"content": content}]
            metadata = {
                "category": category,
                "subcategory": subcategory,
                "memory_type": memory_type,
                "session_id": session_id,
                "source": "career_agent",
                "importance": "medium"
            }
            
            # Store memory using Django backend (sync version)
            result = memory_client._add_impl(messages, session_id, metadata)
            
            if "error" in result:
                logger.error("Failed to store career memory: %s", result["error"])
                return f"Failed to store memory: {result['error']}"
            
            logger.info("Successfully stored career memory for session %s", session_id)
            return "Career information stored successfully in memory."
            
        except Exception as e:
            logger.error("Error in store_career_memory tool: %s", str(e))
            return f"Error storing memory: {str(e)}"

    async def _arun(
        self, 
        content: str, 
        session_id: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Store career memory asynchronously."""
        try:
            session_id = session_id or "default"
            
            # Categorize based on content keywords
            category = "career"
            subcategory = "general"
            memory_type = "factual"
            
            # Simple keyword-based categorization
            content_lower = content.lower()
            if any(word in content_lower for word in ["experience", "worked", "job", "role", "position"]):
                subcategory = "experience"
                memory_type = "episodic"
            elif any(word in content_lower for word in ["skill", "know", "proficient", "expert"]):
                subcategory = "skills"
            elif any(word in content_lower for word in ["want", "looking", "interested", "prefer"]):
                subcategory = "preferences"
            elif any(word in content_lower for word in ["goal", "aim", "objective", "plan"]):
                subcategory = "goals"
            
            # Prepare memory data
            messages = [{"content": content}]
            metadata = {
                "category": category,
                "subcategory": subcategory,
                "memory_type": memory_type,
                "session_id": session_id,
                "source": "career_agent",
                "importance": "medium"
            }
            
            # Store memory using Django backend
            result = await memory_client.add(messages, session_id, metadata)
            
            if "error" in result:
                logger.error("Failed to store career memory: %s", result["error"])
                return f"Failed to store memory: {result['error']}"
            
            logger.info("Successfully stored career memory for session %s", session_id)
            return "Career information stored successfully in memory."
            
        except Exception as e:
            logger.error("Error in store_career_memory tool: %s", str(e))
            return f"Error storing memory: {str(e)}"


class SearchMemoryTool(BaseTool):
    """Tool for searching career-related memories."""
    
    name: str = "search_career_memory"
    description: str = """
    Search for relevant career information from memory.
    Use this to:
    - Find user's career interests and goals
    - Look up work experience and skills
    - Check job preferences and requirements
    - Retrieve professional background information
    """
    args_schema: Type[BaseModel] = SearchMemoryInput

    def _run(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Search career memories synchronously."""
        try:
            session_id = session_id or "default"
            
            # Search memories using Django backend (sync version)
            result = memory_client._search_impl(query, session_id, limit=5)
            
            if "error" in result:
                logger.error("Failed to search career memories: %s", result["error"])
                return f"Memory search failed: {result['error']}"
            
            memories = result.get("results", [])
            
            if not memories:
                logger.info("No relevant career memories found for query: %s", query)
                return "No relevant career information found in memory."
            
            # Format memories for response
            formatted_memories = []
            for memory in memories:
                content = memory["memory"]
                metadata = memory["metadata"]
                similarity = metadata.get("similarity", 0)
                subcategory = metadata.get("subcategory", "general")
                
                formatted_memories.append(
                    f"[{subcategory.title()}] {content} (relevance: {similarity:.2f})"
                )
            
            logger.info("Found %d relevant career memories for session %s", len(memories), session_id)
            return "Relevant career information:\n" + "\n".join(formatted_memories)
            
        except Exception as e:
            logger.error("Error in search_career_memory tool: %s", str(e))
            return f"Error searching memory: {str(e)}"

    async def _arun(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Search career memories asynchronously."""
        try:
            session_id = session_id or "default"
            
            # Search memories using Django backend
            result = await memory_client.search(query, session_id, limit=5)
            
            if "error" in result:
                logger.error("Failed to search career memories: %s", result["error"])
                return f"Memory search failed: {result['error']}"
            
            memories = result.get("results", [])
            
            if not memories:
                logger.info("No relevant career memories found for query: %s", query)
                return "No relevant career information found in memory."
            
            # Format memories for response
            formatted_memories = []
            for memory in memories:
                content = memory["memory"]
                metadata = memory["metadata"]
                similarity = metadata.get("similarity", 0)
                subcategory = metadata.get("subcategory", "general")
                
                formatted_memories.append(
                    f"[{subcategory.title()}] {content} (relevance: {similarity:.2f})"
                )
            
            logger.info("Found %d relevant career memories for session %s", len(memories), session_id)
            return "Relevant career information:\n" + "\n".join(formatted_memories)
            
        except Exception as e:
            logger.error("Error in search_career_memory tool: %s", str(e))
            return f"Error searching memory: {str(e)}"


# Define tools
tools = [StoreMemoryTool(), SearchMemoryTool()]

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    timeout=30
)

# System prompt for TalentCo career agent
SYSTEM_PROMPT = """You are TalentCo's AI Career Assistant, specializing in sustainability and ESG career guidance.

Your expertise includes:
- Sustainability careers and green jobs
- ESG (Environmental, Social, Governance) roles
- Renewable energy and clean technology positions
- Environmental consulting and impact roles
- Corporate sustainability and CSR positions

Your capabilities:
- Career guidance and job matching
- Skills assessment and development recommendations
- Industry insights and trend analysis
- Professional networking advice
- Interview preparation and resume guidance

Instructions:
1. Use memory tools to remember user preferences, experience, and goals
2. Provide personalized career advice based on stored information
3. Focus on sustainability and ESG career opportunities
4. Be encouraging and provide actionable guidance
5. Ask clarifying questions to better understand user needs

Remember: Always store important career information using the store_career_memory tool and search for relevant context using search_career_memory before providing advice.
"""

# Create the agent using LangGraph's prebuilt create_react_agent
# Pass system prompt via prompt parameter
from langchain_core.messages import SystemMessage

graph = create_react_agent(
    llm,
    tools,
    prompt=SystemMessage(content=SYSTEM_PROMPT)
)

logger.info("TalentCo Career Agent initialized successfully")

# Export for LangGraph
if __name__ == "__main__":
    logger.info("TalentCo Career Agent graph is ready for deployment") 