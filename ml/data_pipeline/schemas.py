#!/usr/bin/env python3
"""
Pikkit Data Schema Validation
Pydantic schemas for validating incoming bet data
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class BetStatus(str, Enum):
    PENDING = "PENDING"
    SETTLED_WIN = "SETTLED_WIN"
    SETTLED_LOSS = "SETTLED_LOSS"
    SETTLED_PUSH = "SETTLED_PUSH"
    SETTLED_VOID = "SETTLED_VOID"
    SETTLED_CASHEDOUT = "SETTLED_CASHEDOUT"


class BetType(str, Enum):
    STRAIGHT = "straight"
    PARLAY = "parlay"
    ROUND_ROBIN_3 = "round_robin_3"
    ROUND_ROBIN_4 = "round_robin_4"
    ROUND_ROBIN_5 = "round_robin_5"


class BetSchema(BaseModel):
    """Schema for validating incoming bet data"""
    id: str = Field(..., min_length=1, max_length=100)
    bet_type: Optional[str] = None

    # Financial fields
    odds: Optional[float] = Field(None, ge=-10000, le=10000)
    american_odds: Optional[int] = Field(None, ge=-10000, le=10000)
    amount: Optional[float] = Field(None, ge=0, le=1000000)
    profit: Optional[float] = Field(None, ge=-1000000, le=1000000)
    roi: Optional[float] = Field(None, ge=-1000, le=10000)

    # CLV fields
    clv_percentage: Optional[float] = Field(None, ge=-100, le=100)
    clv_ev: Optional[float] = Field(None, ge=-100, le=100)
    clv_current_odds: Optional[float] = None

    # Categorization
    sport: Optional[str] = Field(None, max_length=100)
    league: Optional[str] = Field(None, max_length=100)
    market: Optional[str] = Field(None, max_length=200)
    institution_name: Optional[str] = Field(None, max_length=100)
    status: Optional[str] = None

    # Flags
    is_live: Optional[bool] = False
    is_settled: Optional[bool] = None
    is_win: Optional[bool] = None

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    time_placed: Optional[datetime] = None

    class Config:
        extra = 'allow'  # Allow additional fields

    @validator('roi', always=True)
    def validate_roi_consistency(cls, v, values):
        """Validate ROI is consistent with amount and profit"""
        amount = values.get('amount')
        profit = values.get('profit')

        if amount and profit and amount > 0:
            expected_roi = (profit / amount) * 100
            if v is not None and abs(v - expected_roi) > 1.0:  # 1% tolerance
                # Log warning but don't fail
                pass
        return v

    @root_validator
    def validate_settled_state(cls, values):
        """Validate consistency between status and is_settled/is_win"""
        status = values.get('status')
        is_settled = values.get('is_settled')
        is_win = values.get('is_win')

        if status:
            status_str = str(status)
            expected_settled = status_str.startswith('SETTLED_')
            expected_win = status_str == 'SETTLED_WIN'

            # Warn on inconsistency but don't fail
            if is_settled is not None and is_settled != expected_settled:
                values['_warnings'] = values.get('_warnings', [])
                values['_warnings'].append(
                    f"is_settled={is_settled} inconsistent with status={status}"
                )

        return values


class ValidationResult(BaseModel):
    """Result of schema validation"""
    valid: bool
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    record_count: int = 0
    valid_count: int = 0
    invalid_count: int = 0


class SchemaValidator:
    """Validate bet data against schema"""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def validate_record(self, record: Dict) -> tuple:
        """Validate a single record, returns (is_valid, errors, warnings)"""
        errors = []
        warnings = []

        try:
            validated = BetSchema(**record)
            # Check for warnings added during validation
            if hasattr(validated, '_warnings'):
                warnings.extend(validated._warnings)
            return True, errors, warnings
        except Exception as e:
            errors.append({
                'record_id': record.get('id', 'unknown'),
                'error': str(e)
            })
            return False, errors, warnings

    def validate_batch(self, records: List[Dict]) -> ValidationResult:
        """Validate a batch of records"""
        valid_count = 0
        invalid_count = 0
        all_errors = []
        all_warnings = []

        for record in records:
            is_valid, errors, warnings = self.validate_record(record)
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                all_errors.extend(errors)
            all_warnings.extend(warnings)

        return ValidationResult(
            valid=invalid_count == 0,
            errors=all_errors,
            warnings=all_warnings,
            record_count=len(records),
            valid_count=valid_count,
            invalid_count=invalid_count
        )


if __name__ == '__main__':
    # Test validation
    test_records = [
        {
            'id': 'test_001',
            'sport': 'Basketball',
            'market': 'Spread',
            'american_odds': -110,
            'amount': 100,
            'status': 'SETTLED_WIN'
        },
        {
            'id': 'test_002',
            'sport': 'Football',
            'american_odds': 50000,  # Invalid - too high
            'amount': -50,  # Invalid - negative
        }
    ]

    validator = SchemaValidator()
    result = validator.validate_batch(test_records)
    print(f"Validation result: {result.valid_count}/{result.record_count} valid")
    if result.errors:
        print(f"Errors: {result.errors}")
