# Cursor Rules Migration Guide

## 🎯 Overview: From Single `.cursorrules` to Organized `.mdc` Files

Cursor has evolved beyond the legacy `.cursorrules` file to support a sophisticated multi-level rules system using `.mdc` files. This provides:

- **Better Token Efficiency**: Only relevant rules load in context
- **Team Collaboration**: Version-controlled, organized rules
- **Context-Aware Activation**: Rules activate based on file types and situations
- **Reduced Noise**: AI gets just the guidance it needs, when it needs it

## 📊 Rule Types Overview

| Rule Type | File Naming | Frontmatter | When It Activates |
|-----------|-------------|-------------|-------------------|
| **Always** | `*-always.mdc` | `alwaysApply: true` | Every conversation |
| **Auto-Attached** | `*-auto.mdc` | `globs: pattern` | When matching files are referenced |
| **Agent-Requested** | `*-agent.mdc` | `description: detailed` | When AI determines it's relevant |
| **Manual** | `*-manual.mdc` | All fields blank | Only when explicitly mentioned |

## 🏗️ New Directory Structure

```
.cursor/rules/
├── deployment/
│   ├── docker-deployment-always.mdc     # Critical deployment rules (always active)
│   ├── dockerfile-auto.mdc              # Dockerfile-specific (auto-attach)
│   └── railway-auto.mdc                 # Railway config (auto-attach)
├── troubleshooting/
│   └── systematic-debugging-agent.mdc   # Debugging guidance (agent-requested)
├── python/
│   └── package-management-auto.mdc      # Python package rules (auto-attach)
└── testing/
    └── rollback-strategy-manual.mdc     # Rollback procedures (manual)
```

## 📝 Frontmatter Structure

Each `.mdc` file must start with YAML frontmatter:

```yaml
---
description: "Detailed description for agent-requested rules (when to apply this rule)"
globs: "file1.ext, **/*.pattern, path/**/*"  # Comma-separated, no quotes around individual patterns
alwaysApply: true  # or false
---

# Rule Title

Your rule content here...

## Examples

<example>
Good example of following the rule
</example>

<example type="invalid">
Bad example showing what not to do
</example>
```

## 🎮 Rule Type Details

### **Always Rules** (`alwaysApply: true`)
- **Purpose**: Critical rules that should influence every AI interaction
- **Token Cost**: HIGH (always loaded)
- **Best For**: Fundamental principles, safety rules
- **Example**: Docker deployment safety, coding standards

```yaml
---
description: Critical Docker deployment safety practices
globs: 
alwaysApply: true
---
```

### **Auto-Attached Rules** (`globs` defined)
- **Purpose**: Context-specific guidance for file types
- **Token Cost**: MEDIUM (loads when relevant files are open)
- **Best For**: Language-specific rules, tool configurations
- **Example**: Dockerfile best practices, Python package management

```yaml
---
description: 
globs: Dockerfile, **/*.dockerfile, pyproject.toml
alwaysApply: false
---
```

### **Agent-Requested Rules** (`description` detailed)
- **Purpose**: Specialized knowledge the AI can choose to use
- **Token Cost**: LOW (loads only when AI determines relevance)
- **Best For**: Debugging procedures, architectural guidance
- **Example**: Systematic debugging, performance optimization

```yaml
---
description: Apply systematic debugging approach when troubleshooting deployment failures, environment mismatches, or circular import errors. Essential for comparing local vs deployed environments.
globs: 
alwaysApply: false
---
```

### **Manual Rules** (all fields minimal)
- **Purpose**: Reference documentation for specific procedures
- **Token Cost**: ZERO (loads only when explicitly mentioned)
- **Best For**: Rollback procedures, emergency protocols
- **Usage**: Reference with `@rollback-strategy-manual` in chat

```yaml
---
description: 
globs: 
alwaysApply: false
---
```

## 🚀 Migration Strategy

### Phase 1: Create Directory Structure
```bash
mkdir -p .cursor/rules/{deployment,troubleshooting,python,testing}
```

### Phase 2: Break Down Your `.cursorrules`
1. **Identify Always Rules**: Critical safety principles → `*-always.mdc`
2. **Extract File-Specific Rules**: Language/tool specific → `*-auto.mdc`
3. **Convert Debugging Guides**: Troubleshooting procedures → `*-agent.mdc`
4. **Archive Procedures**: Emergency/reference docs → `*-manual.mdc`

### Phase 3: Test and Optimize
1. Test each rule type works as expected
2. Monitor token usage and AI behavior
3. Consolidate or split rules based on effectiveness
4. Remove legacy `.cursorrules` file

## 🎯 Best Practices

### ✅ Do:
- Keep rules under 50 lines for optimal token efficiency
- Use clear, descriptive filenames
- Include both valid and invalid examples
- Test rule activation with relevant files
- Start with fewer, more focused rules

### ❌ Don't:
- Quote glob patterns individually (`"*.py"` → `*.py`)
- Create rules that duplicate existing functionality
- Use overly broad always rules
- Mix unrelated concerns in one rule file
- Forget to test rule behavior

## 🔧 Advanced Configuration

### Custom Rule Priority
Higher specificity rules override general ones:
```
.cursor/rules/python/django-specific-auto.mdc  # More specific
.cursor/rules/python/general-python-auto.mdc   # Less specific
```

### Multi-Pattern Globs
```yaml
globs: "*.py, **/*.pyx, **/pyproject.toml, requirements.txt"
```

### Complex Agent Descriptions
```yaml
description: "Apply when troubleshooting circular imports, environment mismatches, or deployment failures where local development works but deployment fails. Critical for comparing Python versions, package installations, and container configurations."
```

## 🎉 Benefits of the New System

1. **Token Efficiency**: 60-80% reduction in unnecessary context
2. **Team Collaboration**: Rules are version-controlled and shareable
3. **Context Awareness**: AI gets the right guidance at the right time
4. **Maintainability**: Organized, focused rules are easier to update
5. **Scalability**: System grows with your project complexity

## 📚 References

- [Official Cursor Rules Documentation](https://docs.cursor.com/context/rules)
- [Community Best Practices](https://medium.com/django-unleashed/django-ninja-unlocking-its-full-potential-part-2-a4e0b5a6ad1b)
- [Advanced Rule Framework](https://solosalon.clinamenic.com/cursor-rules-framework) 