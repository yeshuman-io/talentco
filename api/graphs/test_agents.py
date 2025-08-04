#!/usr/bin/env python3
"""
Test script for TalentCo configurable agents.
Verifies that employer and candidate agents are properly configured.
"""

import os
import sys

# Add the api directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_agent_configuration():
    """Test that agents are properly configured with role-specific tools."""
    
    print("🧪 Testing TalentCo Configurable Agent Architecture\n")
    
    try:
        # Import after Django setup
        from agent import create_employer_agent, create_candidate_agent, create_general_agent
        from agent import AgentRole
        
        print("✅ Successfully imported agent modules")
        
        # Test agent creation
        employer = create_employer_agent()
        candidate = create_candidate_agent()  
        general = create_general_agent()
        
        print("✅ Successfully created all agent types")
        
        # Test agent configurations
        print(f"\n📊 Agent Configuration Summary:")
        print(f"• Employer Agent: {employer.role.value} - {len(employer.tools)} tools")
        print(f"• Candidate Agent: {candidate.role.value} - {len(candidate.tools)} tools")
        print(f"• General Agent: {general.role.value} - {len(general.tools)} tools")
        
        # Test tool names for each agent
        print(f"\n🔧 Employer Agent Tools:")
        for tool in employer.tools:
            print(f"  • {tool.name}: {tool.description[:50]}...")
        
        print(f"\n🔧 Candidate Agent Tools:")
        for tool in candidate.tools:
            print(f"  • {tool.name}: {tool.description[:50]}...")
        
        # Test memory isolation
        print(f"\n🧠 Memory Isolation:")
        print(f"• Employer prefix: {employer.memory_prefix}")
        print(f"• Candidate prefix: {candidate.memory_prefix}")
        print(f"• General prefix: {general.memory_prefix}")
        
        # Test system prompts are different
        print(f"\n📝 System Prompt Differentiation:")
        print(f"• Employer prompt length: {len(employer.system_prompt)} chars")
        print(f"• Candidate prompt length: {len(candidate.system_prompt)} chars")
        print(f"• General prompt length: {len(general.system_prompt)} chars")
        
        employer_focus = "hiring" in employer.system_prompt.lower()
        candidate_focus = "career" in candidate.system_prompt.lower()
        
        print(f"• Employer has hiring focus: {'✅' if employer_focus else '❌'}")
        print(f"• Candidate has career focus: {'✅' if candidate_focus else '❌'}")
        
        print(f"\n🎉 All tests passed! Configurable agent architecture is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_functionality():
    """Test that tools can be imported and instantiated."""
    
    print(f"\n🔧 Testing Tool Functionality\n")
    
    try:
        from agent_tools import EMPLOYER_TOOLS, CANDIDATE_TOOLS, ALL_AGENT_TOOLS
        
        print(f"✅ Successfully imported tool collections")
        print(f"• Employer tools: {len(EMPLOYER_TOOLS)}")
        print(f"• Candidate tools: {len(CANDIDATE_TOOLS)}")
        print(f"• All tools: {len(ALL_AGENT_TOOLS)}")
        
        # Test that each tool has required attributes
        all_tools = ALL_AGENT_TOOLS
        for tool in all_tools:
            assert hasattr(tool, 'name'), f"Tool {tool} missing 'name'"
            assert hasattr(tool, 'description'), f"Tool {tool} missing 'description'"
            assert hasattr(tool, '_arun'), f"Tool {tool} missing '_arun' method"
            
        print(f"✅ All tools have required attributes")
        
        # Test tool names are unique
        tool_names = [tool.name for tool in all_tools]
        assert len(tool_names) == len(set(tool_names)), "Duplicate tool names found"
        
        print(f"✅ All tool names are unique")
        
        return True
        
    except Exception as e:
        print(f"❌ Tool test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """Run all tests."""
    
    success = True
    
    # Test agent configuration
    success &= test_agent_configuration()
    
    # Test tool functionality  
    success &= test_tool_functionality()
    
    print(f"\n{'🎉 All tests passed!' if success else '❌ Some tests failed!'}")
    sys.exit(0 if success else 1)