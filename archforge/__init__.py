"""
ArchForge - prototype implementation of the Draft / Ground / Validate pipeline
described in the ArchForge framework overview.

This package is a working starting point, not the full production framework.
It shows the three-pillar runtime end to end:

    1. Draft    - the fine-tuned (or base) model writes a first answer
    2. Ground   - relevant snippets are retrieved from local reference docs
    3. Validate - the answer is checked against coding standards, with a
                   self-repair loop that fixes issues before the engineer
                   sees them

Everything here runs locally. No code, query, or answer is sent anywhere
except to the model endpoint you configure (which itself is expected to be
on-premise, e.g. an OpenAI-compatible vLLM server).
"""

__version__ = "0.1.0"
