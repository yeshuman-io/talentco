# TalentCo Agentic Platform: Proof of Concept Proposal

**Prepared for:** Neil Salisbury, Earth Ventures  
**Prepared by:** Daryl Antony, Yes Human  
**Date:** July 2025  
**Project:** TalentCo AI Agent Proof of Concept

---

## Executive Summary

**Yes Human** proposes to develop a **proof of concept** for TalentCo that demonstrates how AI agents can enhance the platform's capabilities for sustainability and ESG professionals. This PoC will provide Neil and stakeholders with hands-on experience exploring agent-driven interactions and capabilities.

### Key PoC Objectives
- Demonstrate AI agent capabilities for job matching and career guidance
- Provide interactive tools for stakeholders to explore agent functionality
- Establish technical foundation for future platform development
- Validate agent-driven approaches for TalentCo's specific use cases

---

## Current State Analysis

### Platform Context
Based on the existing [TalentCo platform](https://talentco-dev.netlify.app) and [Earth Ventures portfolio](https://www.earthventures.io/), TalentCo serves as a specialized job platform connecting sustainability professionals with relevant opportunities.

### Identified Opportunities
- **Enhanced matching** through AI-powered candidate analysis
- **Personalized career guidance** using agent-driven recommendations
- **Automated job discovery** from external sources
- **Skills gap analysis** and learning recommendations

---

## Proof of Concept Approach

### Core Focus: ReAct Agent with Custom Tools

The PoC will center on building a **ReAct (Reasoning + Acting) agent** using **LangGraph** that can:
- Reason about user queries related to careers and job opportunities
- Take actions using custom tools to gather information
- Provide personalized recommendations based on user profiles
- Learn and adapt through conversation

### Technical Architecture

#### **LangGraph ReAct Framework**
- **Agent Core:** Multi-step reasoning with action execution
- **State Management:** Persistent conversation context
- **Tool Integration:** Custom tools for job market data and user analysis
- **Memory System:** Personalized user interaction history

#### **LangChain/Graph UI Integration**
- **Interactive Interface:** Real-time agent conversations in the existing `/ui` directory
- **Tool Visualization:** Show agent reasoning and tool usage
- **Feedback Collection:** Gather stakeholder input on agent behavior

#### **Observability & Deployment**
- **LangSmith Integration:** Agent tracing and observability for debugging and performance monitoring
- **Railway Deployment:** Cloud deployment platform for hosting the PoC application

### Initial Tool Development

#### **Indeed API Integration**
Following a review of [Indeed's API documentation](https://docs.indeed.com/api), we've identified several endpoints that might be valuable for agent tool calls:

- **Job Search Tool:** Query Indeed's job database for sustainability roles (`/jobs/search`)
- **Job Details Tool:** Get comprehensive job information for agent analysis (`/jobs/{job_id}`)
- **Company Research Tool:** Access company information and ESG practices (`/companies/{company_id}`)
- **Salary Insights Tool:** Provide compensation guidance for sustainability roles

#### **mem0 Integration (Potential)**
- **Personal Memory:** Store and recall user preferences and career history
- **Learning Adaptation:** Improve recommendations based on user interactions
- **Context Preservation:** Maintain conversation continuity across sessions

#### **Additional Tools (Subject to Neil's Input)**
- **Skills Assessment Tool:** Evaluate candidate capabilities
- **Learning Recommendation Tool:** Suggest relevant courses and training
- **Company Research Tool:** Provide insights on potential employers
- **ESG Compliance Tool:** Check sustainability credentials and requirements

---

## Implementation Plan

### Phase 1: Foundation Setup
- Configure LangGraph environment with ReAct agent
- Integrate LangChain UI in existing `/ui` directory
- Set up Indeed API integration and basic job search tools
- Implement basic conversation interface

### Phase 2: Tool Development
- Build custom tools based on Neil's API preferences
- Integrate mem0 for personalization (if approved)
- Develop additional tools identified through stakeholder discussions
- Test agent reasoning and tool coordination

### Phase 3: Stakeholder Exploration
- Deploy interactive demo for Neil and team
- Gather feedback on agent capabilities and tool utility
- Iterate on tool functionality based on user experience
- Document insights and recommendations for future development

---

## API Integration Strategy

### Confirmed APIs
- **Indeed API:** Job market data and search capabilities
- **mem0 (Potential):** Agent memory and personalization

### APIs for Discussion with Neil
- **LinkedIn API:** Professional network integration
- **Educational Platform APIs:** Course and training content
- **ESG Data Sources:** Sustainability-specific job requirements
- **Carbon Footprint APIs:** Environmental impact assessment

### Model Context Protocol (MCP) Servers
- **Linear MCP:** Project management integration (already configured)
- **Browser Tools MCP:** Web scraping capabilities
- **Custom MCP Servers:** For specific TalentCo requirements

---

## Investment Requirements

### Proof of Concept Budget
**Total Investment:** $15,000 AUD (plus GST)

**Includes:**
- Agent development and tool creation
- API integration and testing
- UI implementation and deployment
- Documentation and handover

---

## Stakeholder Exploration Process

### Interactive Demo Sessions
- **Agent Conversations:** Experience AI-driven career guidance
- **Tool Demonstrations:** See how agents use external APIs
- **Feedback Collection:** Gather insights on functionality and user experience
- **Capability Assessment:** Evaluate potential for full platform integration

### Demonstration Platform
The resulting agent will be made available within a TalentCo website or widget with cut-down functionality for early demonstration purposes. This approach allows stakeholders to:
- Experience the agent in a realistic TalentCo-branded environment
- Test core functionality without full platform complexity
- Provide feedback on user experience and interface design
- Assess integration potential with existing TalentCo systems

### Key Questions for Exploration
- How effectively can agents understand career goals and preferences?
- What tool combinations provide the most value to users?
- How can personalization enhance the job matching process?
- What additional capabilities, aka tools, would be most valuable?

---

## Next Steps

### Immediate Actions
1. **Stakeholder Alignment:** Confirm PoC scope and objectives
2. **API Selection:** Finalize which APIs to integrate initially
3. **Technical Setup:** Begin LangGraph and UI configuration
4. **Development Sprint:** Build core agent and initial tools

### Collaboration with Neil
- **Regular Check-ins:** Work in cycles with Neil (facilitated with Linear)
- **API Prioritization:** Identify most valuable integrations
- **Use Case Refinement:** Focus on specific TalentCo scenarios
- **Future Planning:** Discuss potential full platform development

---

## Conclusion

This proof of concept will provide Neil and the Earth Ventures team with hands-on experience of how AI agents can enhance TalentCo's capabilities. By focusing on practical demonstrations rather than theoretical concepts, we can validate the approach and identify the most valuable applications for the platform.

The PoC will establish a foundation for understanding how agent-driven interactions can improve user experience while providing concrete insights for future development decisions.

---

## Contact Information

**Daryl Antony**  
Founder & Technical Lead  
Yes Human  
Email: daryl@yeshuman.io  
Web: https://yeshuman.io