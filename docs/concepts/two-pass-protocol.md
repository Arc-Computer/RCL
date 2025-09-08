---
title: Two-Pass Protocol
description: Diagnostic probe followed by adaptive teaching with GRPO training.
---

# Two-Pass Protocol

RCL uses a two-pass protocol per example:

1) Probe the student’s capability with a ≤50-token diagnostic.
2) Generate adaptive teaching conditioned on the probe.

```mermaid
sequenceDiagram
    participant U as User/Problem
    participant S as Student
    participant T as Teacher
    U->>S: Problem prompt
    S->>T: Diagnostic approach (≤50 tokens)
    T->>S: Adaptive teaching (targeted hints/scaffolding)
    S->>U: Final solution (with or without teaching)
```

In training, GRPO optimizes the teacher using rewards from `AdaptiveTeachingReward`, comparing student performance with and without the teacher’s intervention.

