---
title: For Developers - Integration Path
description: Add learning to your existing agent without training models
---

# Developer Path: Add Learning in 15 Minutes

You have an agent. It works. But it makes the same mistakes repeatedly. Let's fix that without changing your architecture or training models.

## Your Current Agent

```python
# What you have today
class YourAgent:
    def execute(self, task):
        # Your existing logic
        return result
```

## Step 1: Add Learning (2 minutes)

```python
from rcl import add_learning

# Your existing agent
agent = YourAgent()

# Add learning capability
agent = add_learning(
    agent,
    verifier="acceptance_test",  # Your existing tests become rewards
    shadow_mode=True             # Test safely without affecting production
)

# Use normally - it now learns
result = agent.execute(task)
```

## Step 2: Connect Your Metrics (5 minutes)

Turn your existing observability into learning signals:

```python
from rcl import RIMAdapter

# Use your SLOs
rim = RIMAdapter(
    success_metric="latency < 100ms AND status == 200",
    failure_penalty=2.0  # Failures matter more than successes
)

agent = add_learning(agent, verifier=rim)
```

Or use human feedback:

```python
from rcl import HumanFeedbackAdapter

# Simple thumbs up/down
rim = HumanFeedbackAdapter(
    feedback_endpoint="/api/feedback",
    aggregation="majority"  # or "unanimous", "weighted"
)
```

## Step 3: Test in Shadow Mode (5 minutes)

```python
# Enable shadow learning - no production impact
agent.configure(
    shadow_mode=True,
    shadow_traffic_percent=10,  # Start with 10% of traffic
    rollback_on_regression=True  # Auto-rollback if performance drops
)

# Monitor without risk
dashboard = agent.get_dashboard_url()
print(f"Monitor at: {dashboard}")
```

## What Happens Next?

### Hour 1: Baseline
- Agent observes patterns
- Builds performance baseline
- No changes to behavior

### Day 1: First Improvements
- Identifies recurring failures
- Proposes small prompt adjustments
- Shadow tests improvements

### Week 1: Measurable Gains
- 10-20% reduction in repeat errors
- Faster resolution of known issues
- Automatic rollback of bad changes

### Month 1: Compound Learning
- Skills transfer to similar tasks
- Team-wide improvements from individual learning
- Audit trail of what changed and why

## Common Integrations

### FastAPI/Flask
```python
from rcl import add_learning

@app.post("/api/agent")
async def agent_endpoint(request: Request):
    agent = get_or_create_agent(request.user_id)
    agent = add_learning(agent, verifier="http_status")
    return agent.execute(request.task)
```

### LangChain
```python
from langchain.agents import AgentExecutor
from rcl import add_learning

chain = AgentExecutor(...)
chain = add_learning(
    chain,
    verifier="langchain_callbacks",
    track_tool_usage=True
)
```

### OpenAI Assistants
```python
from openai import OpenAI
from rcl import add_learning

client = OpenAI()
assistant = client.beta.assistants.retrieve("asst_...")

# Wrap with learning
assistant = add_learning(
    assistant,
    verifier="assistant_feedback",
    preserve_thread_context=True
)
```

## Production Rollout

### Phase 1: Shadow Mode (Week 1)
```python
agent.configure(shadow_mode=True, shadow_traffic_percent=10)
```
- No production impact
- Collect baseline metrics
- Validate reward signals

### Phase 2: Canary Deployment (Week 2)
```python
agent.configure(shadow_mode=False, canary_percent=5)
```
- 5% of traffic gets improvements
- A/B test improvements
- Automatic rollback on regression

### Phase 3: Full Rollout (Week 3+)
```python
agent.configure(production_mode=True)
```
- All traffic benefits
- Continuous learning enabled
- Weekly PEFT updates for recurring patterns

## Monitoring & Observability

```python
# Built-in metrics
metrics = agent.get_metrics()
print(f"Learning Rate: {metrics.learning_rate}")
print(f"Success Rate: {metrics.success_rate}")
print(f"Regression Events: {metrics.regressions}")

# Prometheus export
agent.export_metrics(format="prometheus")

# Grafana dashboard
agent.generate_dashboard(output="grafana.json")
```

## Safety & Rollback

Every change is reversible:

```python
# List all policy versions
versions = agent.list_versions()

# Rollback to specific version
agent.rollback(version="v1.2.3")

# Or rollback to time
agent.rollback(to_timestamp="2024-01-15T10:00:00Z")

# Emergency stop
agent.disable_learning()
```

## FAQ

**Q: Do I need GPUs?**
A: No. Integration mode uses your existing compute. Only custom training needs GPUs.

**Q: Will this break my agent?**
A: Shadow mode ensures zero production impact. Automatic rollback protects against regressions.

**Q: How is this different from prompt engineering?**
A: Prompt engineering is manual and one-time. RCL learns continuously from actual outcomes.

**Q: Can I bring my own verifiers?**
A: Yes. Any function that returns a score can be a verifier.

**Q: What about sensitive data?**
A: Learning happens on rewards, not raw data. Configure `privacy_mode=True` for additional protection.

## Next Steps

### Ready to Deploy?
1. [Production Checklist](../platform/production-checklist.md)
2. [Monitoring Setup](../platform/monitoring.md)
3. [Team Rollout Guide](../platform/rollout.md)

### Want to Customize?
1. [Custom Verifiers](../integration/custom-verifiers.md)
2. [Advanced Configurations](../integration/advanced-config.md)
3. [GEPA Prompt Evolution](../integration/gepa.md)

### Need Help?
- [Common Issues](../troubleshooting/integration-issues.md)
- [Discord Community](https://discord.gg/rcl)
- [Office Hours](https://calendly.com/arc/office-hours)

---

*Start with shadow mode. See improvements in days, not months.*