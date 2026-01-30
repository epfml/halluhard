from __future__ import annotations

from pathlib import Path

from ..core.domain_strategy import DomainStrategy
from .research_questions import ResearchQuestionsStrategy
from .medical_guidelines import MedicalGuidelinesStrategy
from .legal_cases import LegalCasesStrategy
from .coding import CodingStrategy


STRATEGY_MAP = {
    "research_questions": ResearchQuestionsStrategy,
    "medical_guidelines": MedicalGuidelinesStrategy,
    "legal_cases": LegalCasesStrategy,
    "coding": CodingStrategy,
}


def get_strategy(task_name: str, base_path: Path | None = None) -> DomainStrategy:
    """Factory to get the appropriate domain strategy.
    
    Args:
        task_name: The name of the task (e.g., "research_questions")
        base_path: Optional base path. If provided, usually points to repo root.
                   The function tries to locate the 'judging_pipeline' or 'prompts' directory.
    """
    if task_name not in STRATEGY_MAP:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(STRATEGY_MAP.keys())}")
    
    # Default pipeline root based on file structure: .../judging_pipeline/strategies/__init__.py
    default_root = Path(__file__).parent.parent
    
    if base_path:
        # 1. Check if base_path is the repo root (contains judging_pipeline/prompts)
        # This is the standard case when called from run_pipeline.py with default args
        repo_pipeline_path = base_path / "judging_pipeline"
        if (repo_pipeline_path / "prompts").exists():
            strategy_root = repo_pipeline_path
            
        # 2. Check if base_path is the pipeline root itself (contains prompts directly)
        elif (base_path / "prompts").exists():
            strategy_root = base_path
            
        # 3. Fallback: assume the user knows what they are doing with base_path
        else:
            strategy_root = base_path
    else:
        strategy_root = default_root
    
    return STRATEGY_MAP[task_name](strategy_root)
