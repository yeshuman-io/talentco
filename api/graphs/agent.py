"""
TalentCo Career Agent - Talent matching and career guidance agent.

Clean architecture with role-based system prompting and career-focused tools.
"""
import os
import django
import logging
from django.conf import settings
from dotenv import load_dotenv
from enum import Enum
from typing import Optional, Type, Union

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
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from apps.memories.backends import DjangoMemoryBackend
from pydantic import BaseModel, Field
from graphs.agent_tools import EMPLOYER_TOOLS, CANDIDATE_TOOLS, ALL_AGENT_TOOLS

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


# ================================================================
# CONFIGURABLE AGENT CLASS - Role-based agent configuration  
# ================================================================




class AgentRole(Enum):
    """Define available agent roles."""
    EMPLOYER = "employer"
    CANDIDATE = "candidate"
    GENERAL = "general"


class TalentCoAgent:
    """
    Configurable TalentCo agent that adapts based on role.
    Supports employer, candidate, and general configurations.
    """
    
    def __init__(
        self, 
        role: Union[AgentRole, str] = AgentRole.GENERAL,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3
    ):
        """
        Initialize a role-specific TalentCo agent.
        
        Args:
            role: The agent role (employer, candidate, or general)
            model: OpenAI model to use
            temperature: Model temperature for responses
        """
        # Convert string to enum if needed
        if isinstance(role, str):
            role = AgentRole(role.lower())
        
        self.role = role
        self.model = model
        self.temperature = temperature
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            timeout=30
        )
        
        # Configure role-specific settings
        self.tools = self._get_role_tools()
        self.system_prompt = self._get_role_system_prompt()
        self.memory_prefix = self._get_memory_prefix()
        
        # Create the agent graph
        self.graph = create_react_agent(
            self.llm,
            self.tools,
            prompt=SystemMessage(content=self.system_prompt)
        )
        
        logger.info(f"TalentCo {role.value.title()} Agent initialized successfully")
    
    def _get_role_tools(self) -> list:
        """Get tools based on agent role."""
        # Always include memory tools with role-based configuration
        memory_tools = [
            RoleSpecificStoreMemoryTool(role=self.role),
            RoleSpecificSearchMemoryTool(role=self.role)
        ]
        
        if self.role == AgentRole.EMPLOYER:
            return memory_tools + EMPLOYER_TOOLS
        elif self.role == AgentRole.CANDIDATE:
            return memory_tools + CANDIDATE_TOOLS
        else:  # GENERAL
            return memory_tools + ALL_AGENT_TOOLS
    
    def _get_role_system_prompt(self) -> str:
        """Get system prompt based on agent role."""
        base_expertise = """
Your expertise includes:
- Sustainability careers and green jobs
- ESG (Environmental, Social, Governance) roles  
- Renewable energy and clean technology positions
- Environmental consulting and impact roles
- Corporate sustainability and CSR positions
"""
        
        if self.role == AgentRole.EMPLOYER:
            return f"""You are TalentCo's AI Hiring Assistant, specializing in finding and evaluating talent for sustainability and ESG roles.

{base_expertise}

Your employer-focused capabilities:
- Finding qualified candidates for open positions
- Detailed candidate evaluation and assessment
- Talent pool analysis and market insights
- Hiring strategy recommendations
- Skills gap analysis for roles

Instructions:
1. Use find_candidates_for_opportunity to discover top talent
2. Use evaluate_candidate_profile for detailed candidate assessment
3. Use analyze_talent_pool to understand market dynamics
4. Always store hiring preferences and requirements using memory tools
5. Provide data-driven hiring recommendations with supporting evidence
6. Focus on ESG and sustainability expertise when evaluating candidates

Remember: You help employers make informed hiring decisions through comprehensive candidate analysis and market insights."""

        elif self.role == AgentRole.CANDIDATE:
            return f"""You are TalentCo's AI Career Advisor, specializing in career development and opportunity discovery for sustainability and ESG professionals.

{base_expertise}

Your career-focused capabilities:
- Finding personalized job opportunities
- Detailed opportunity fit analysis
- Skill development recommendations
- Career progression guidance
- Learning pathway suggestions

Instructions:
1. Use find_opportunities_for_profile to discover relevant roles
2. Use analyze_opportunity_fit to assess role compatibility
3. Use get_learning_recommendations to identify skill development opportunities
4. Always store career goals, preferences, and experience using memory tools
5. Provide personalized career guidance with actionable next steps
6. Focus on sustainability and ESG career advancement

Remember: You help professionals advance their careers in the sustainability and ESG sectors through personalized guidance and opportunity discovery."""

        else:  # GENERAL
            return f"""You are TalentCo's AI Career Assistant, providing comprehensive career and hiring guidance for sustainability and ESG professionals.

{base_expertise}

Your comprehensive capabilities:
- Career guidance and job matching (for both candidates and employers)
- Skills assessment and development recommendations
- Industry insights and trend analysis
- Talent evaluation and hiring support
- Professional networking advice
- Interview preparation and resume guidance

Instructions:
1. Identify whether the user is a candidate seeking opportunities or an employer seeking talent
2. Use appropriate tools based on the user's needs:
   - For candidates: focus on opportunity discovery and career development
   - For employers: focus on candidate finding and evaluation
3. Store relevant information using memory tools
4. Provide personalized, actionable guidance
5. Focus on sustainability and ESG career opportunities

Remember: You serve both sides of the talent marketplace with specialized tools and expertise."""
    
    def _get_memory_prefix(self) -> str:
        """Get memory session prefix for role isolation."""
        return f"talentco_{self.role.value}"
    
    def invoke(self, message: str, session_id: str = "default") -> dict:
        """
        Invoke the agent with a message.
        
        Args:
            message: The user message
            session_id: Session identifier (will be prefixed with role)
            
        Returns:
            Agent response
        """
        # Add role prefix to session for memory isolation
        role_session_id = f"{self.memory_prefix}_{session_id}"
        
        return self.graph.invoke({
            "messages": [{"role": "user", "content": message}],
            "session_id": role_session_id
        })


# ================================================================
# ROLE-SPECIFIC MEMORY TOOLS - For memory isolation
# ================================================================

class RoleSpecificStoreMemoryTool(StoreMemoryTool):
    """Memory tool with role-specific session prefixing."""
    
    def __init__(self, role: AgentRole):
        super().__init__()
        # Set role as an instance variable (not a Pydantic field)
        object.__setattr__(self, 'role', role)
        object.__setattr__(self, 'name', f"store_{role.value}_memory")
        object.__setattr__(self, 'description', f"Store {role.value}-specific information in memory.")
    
    def _run(self, content: str, session_id: Optional[str] = None, 
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        # Add role prefix to session
        prefixed_session_id = f"talentco_{self.role.value}_{session_id or 'default'}"
        return super()._run(content, prefixed_session_id, run_manager)
    
    async def _arun(self, content: str, session_id: Optional[str] = None,
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        # Add role prefix to session
        prefixed_session_id = f"talentco_{self.role.value}_{session_id or 'default'}"
        return await super()._arun(content, prefixed_session_id, run_manager)


class RoleSpecificSearchMemoryTool(SearchMemoryTool):
    """Memory search tool with role-specific session prefixing."""
    
    def __init__(self, role: AgentRole):
        super().__init__()
        # Set role as an instance variable (not a Pydantic field)
        object.__setattr__(self, 'role', role)
        object.__setattr__(self, 'name', f"search_{role.value}_memory") 
        object.__setattr__(self, 'description', f"Search {role.value}-specific information from memory.")
    
    def _run(self, query: str, session_id: Optional[str] = None,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        # Add role prefix to session
        prefixed_session_id = f"talentco_{self.role.value}_{session_id or 'default'}"
        return super()._run(query, prefixed_session_id, run_manager)
    
    async def _arun(self, query: str, session_id: Optional[str] = None,
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        # Add role prefix to session
        prefixed_session_id = f"talentco_{self.role.value}_{session_id or 'default'}"
        return await super()._arun(query, prefixed_session_id, run_manager)


# ================================================================
# AGENT FACTORY FUNCTIONS - Convenience constructors
# ================================================================

def create_employer_agent(**kwargs) -> TalentCoAgent:
    """Create an employer-focused agent."""
    return TalentCoAgent(role=AgentRole.EMPLOYER, **kwargs)


def create_candidate_agent(**kwargs) -> TalentCoAgent:
    """Create a candidate-focused agent."""
    return TalentCoAgent(role=AgentRole.CANDIDATE, **kwargs)


def create_general_agent(**kwargs) -> TalentCoAgent:
    """Create a general-purpose agent."""
    return TalentCoAgent(role=AgentRole.GENERAL, **kwargs)


# ================================================================
# DEFAULT AGENTS - For backward compatibility and testing
# ================================================================

# Create default instances for each role
employer_agent = create_employer_agent()
candidate_agent = create_candidate_agent()
general_agent = create_general_agent()

# Backward compatibility - default graph is the general agent
graph = general_agent.graph

logger.info("TalentCo configurable agents initialized successfully")
logger.info("Available agents: employer_agent, candidate_agent, general_agent")

# Export for LangGraph  
if __name__ == "__main__":
    logger.info("TalentCo configurable agent system is ready for deployment") 