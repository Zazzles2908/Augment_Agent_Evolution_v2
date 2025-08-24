"""ZhipuAI (GLM) provider module location after reorg.

This file relocates GLMModelProvider to providers/zhipu/provider.py to
make the provider layout consistent. Implementation is imported from the
legacy module to keep change minimal during reorg; a follow-up step can fully
move code and delete the old file once imports are updated everywhere.
"""

from ..glm import GLMModelProvider  # re-export for compatibility

