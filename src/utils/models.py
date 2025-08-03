"""
Core Data models

This module consists of all pydantic data models used throughout this application
for type safety and validation
"""

from pydantic import BaseModel, Field, EmailStr
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, List, Literal, Any, Dict, Union
from enum import Enum

#Enums
class DocType(str, Enum):
    """Enum for document types"""
    CSV = "csv"
    JSON = "json"
    XLSX = "xlsx"
    PARQUET = "parquet"

class ProcessStatus(str, Enum):
    """Enum for Process Status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskType(str, Enum):
    """Enum for Task Type"""
    UPLOAD = "upload"
    PROFILING = "profiling"
    VISUALIZATION = "visualization"
    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_RECOMMENDATION = "model_recommendation"

class QualityMetric(str, Enum):
    """Enum for Quality Metrics of the data"""
    MISSING_VALUES = "missing_values"
    DUPLICATES = "duplicates"
    OUTLIERS = "outliers"
    SKEWNESS = "skewness"
    DATA_TYPE_MISMATCH = "dtype_mismatch"
    ZERO_VARIANCE = "zero_variance"
    HIGH_CARDINALITY = "high_cardinality"
    CORRELATION = "correlation"
    IMBALANCED_CLASSES = "imbalanced_classes"

#Data Models
class RegisterUser(BaseModel):
    """Base model for New User Registration"""
    email: EmailStr
    password: str
    username: str

class LoginUser(BaseModel):
    """Base model for User Login"""
    email: EmailStr
    password: str

class AuthResponse(BaseModel):
    """Base model for Authentication Response"""
    user_id: UUID
    access_token: str

class DatasetMetadata(BaseModel):
    """Base Model for Dataset Metadata"""
    filename: str
    file_type: DocType
    upload_time: datetime
    num_rows: int
    num_columns: int
    column_names: list[str]

class ColumnSchema(BaseModel):
    """Base model for Column Schema"""
    name: str
    dtype: str
    num_missing: Optional[int] = 0
    num_unique: Optional[int] = None
    inferred_type: Optional[str] = None

class QualityCheck(BaseModel):
    """Base model for Quality Check"""
    metric: QualityMetric
    passed: bool
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0) 
    message: Optional[str] = None
    affected_columns: Optional[List[str]] = None

class DataProfile(BaseModel):
    """Base model for Data Profile"""
    data_id: UUID
    columns: List[ColumnSchema]
    summary: str
    recommendations: List[str] = Field(default_factory=list)
    quality_checks: List[QualityCheck] = Field(default_factory=list)

class TaskStatus(BaseModel):
    """Base model for Task Status"""
    task_id: UUID
    dataset_id: UUID
    task_type: TaskType
    status: ProcessStatus
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str] = None