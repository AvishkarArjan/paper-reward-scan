from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class PaperMetadata(BaseModel):
    paper_name: str
    paper_path: str
    text_preview: str
    num_chars: int
    processed_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class EvaluationResult(BaseModel):
    paper_name: str
    paper_path: str
    is_relevant_domain: bool
    has_reward_function: Optional[bool] = None
    reward_is_single_well_defined: Optional[bool] = None
    reward_quality_score: Optional[int] = None
    task_clearly_defined: Optional[bool] = None
    passes_quality: bool
    reasoning: str
    model_used: str
    evaluateed_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class ExtractionResult(BaseModel):
    paper_name: str
    paper_path: str
    task_description: str
    environment_context: str
    components: list[str]
    reward_function_code: str
    model_used: str
    extracted_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class SFTPair(BaseModel):
    instruction: str
    input: str
    output: str
