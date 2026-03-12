"""
Primary Aldosteronism (PA) Diagnosis Guidelines Module
=======================================================
This module implements a combined clinical AI assistant based on:
- 2016 Chinese Medical Association Expert Consensus on PA Diagnosis and Treatment
- 2025 Endocrine Society (ES) Clinical Practice Guidelines for PA

Key modifications incorporated:
1. AVS (Adrenal Vein Sampling) is optional under defined conditions.
2. CXCR4 imaging is recommended for patients where the lateralization is
   difficult to determine.
3. Confirmation tests may be omitted for patients with unambiguous clinical
   indications of PA.
"""

from .guidelines import COMBINED_PA_SYSTEM_PROMPT
from .assistant import PADiagnosisAssistant

__all__ = ["COMBINED_PA_SYSTEM_PROMPT", "PADiagnosisAssistant"]
