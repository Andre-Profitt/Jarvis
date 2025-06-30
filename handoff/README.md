# Handoff Directory

This directory facilitates seamless collaboration between Claude Desktop and Claude Code.

## Structure

- `todo.md` - Active tasks to implement
- `context.md` - Current project context
- `implement/` - Files ready for implementation
- `*.json` - Structured handoff data
- `*.md` - Human-readable handoff documents

## Usage

### From Claude Desktop:
1. Design a feature
2. Save artifacts to `../artifacts/`
3. Create handoff file here
4. Store context in @jarvis-memory

### From Claude Code:
1. Check this directory for new tasks
2. Read handoff files
3. Retrieve context from @jarvis-memory
4. Implement the feature
5. Mark handoff as complete

## Quick Commands

```bash
# Check for pending handoffs
ls -la *.json | grep pending

# View latest handoff
cat $(ls -t *.md | head -1)
```