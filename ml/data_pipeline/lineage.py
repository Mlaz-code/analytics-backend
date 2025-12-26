#!/usr/bin/env python3
"""
Pikkit Data Lineage Tracking
Track data transformations and provenance for auditability
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import hashlib
import os


@dataclass
class TransformationStep:
    """Represents a single data transformation"""
    step_id: str
    operation: str  # 'filter', 'transform', 'aggregate', 'join', 'validate'
    description: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    input_count: int = 0
    output_count: int = 0
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataLineage:
    """Track data lineage through the pipeline"""
    run_id: str
    source: str  # 'pikkit_api', 'supabase', 'manual'
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    transformations: List[TransformationStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_transformation(self, operation: str, description: str,
                          input_count: int = 0, output_count: int = 0,
                          parameters: Dict = None) -> TransformationStep:
        """Add a transformation step to lineage"""
        step = TransformationStep(
            step_id=f"{self.run_id}_{len(self.transformations)}",
            operation=operation,
            description=description,
            input_count=input_count,
            output_count=output_count,
            parameters=parameters or {}
        )
        self.transformations.append(step)
        return step

    def complete(self):
        """Mark pipeline run as complete"""
        self.completed_at = datetime.utcnow()

    def compute_hash(self, df) -> str:
        """Compute hash of dataframe for verification"""
        try:
            return hashlib.md5(
                df.to_json().encode()
            ).hexdigest()
        except Exception:
            return "hash_error"

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'run_id': self.run_id,
            'source': self.source,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'transformations': [
                {
                    'step_id': t.step_id,
                    'operation': t.operation,
                    'description': t.description,
                    'timestamp': t.timestamp.isoformat(),
                    'input_count': t.input_count,
                    'output_count': t.output_count,
                    'parameters': t.parameters,
                }
                for t in self.transformations
            ],
            'metadata': self.metadata,
        }

    def save(self, filepath: str):
        """Save lineage to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class LineageTracker:
    """Manage lineage across pipeline runs"""

    def __init__(self, storage_dir: str = '/root/pikkit/ml/data_pipeline/lineage'):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def start_run(self, source: str) -> DataLineage:
        """Start a new pipeline run"""
        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        return DataLineage(run_id=run_id, source=source)

    def save_run(self, lineage: DataLineage):
        """Save completed run"""
        filepath = f"{self.storage_dir}/{lineage.run_id}.json"
        lineage.save(filepath)
        print(f"Lineage saved to {filepath}")

    def get_recent_runs(self, limit: int = 10) -> List[Dict]:
        """Get recent pipeline runs"""
        import glob

        files = sorted(
            glob.glob(f"{self.storage_dir}/*.json"),
            key=os.path.getmtime,
            reverse=True
        )[:limit]

        runs = []
        for f in files:
            try:
                with open(f) as fp:
                    runs.append(json.load(fp))
            except Exception:
                continue
        return runs

    def get_run(self, run_id: str) -> Optional[Dict]:
        """Get a specific run by ID"""
        filepath = f"{self.storage_dir}/{run_id}.json"
        if os.path.exists(filepath):
            with open(filepath) as f:
                return json.load(f)
        return None


if __name__ == '__main__':
    # Test lineage tracking
    tracker = LineageTracker()

    # Start a new run
    lineage = tracker.start_run(source='supabase')

    # Add some transformations
    lineage.add_transformation(
        operation='fetch',
        description='Fetched raw data from Supabase',
        output_count=1000
    )

    lineage.add_transformation(
        operation='validate',
        description='Validated schema',
        input_count=1000,
        output_count=998,
        parameters={'invalid_records': 2}
    )

    lineage.add_transformation(
        operation='transform',
        description='Feature engineering',
        input_count=998,
        output_count=998,
        parameters={'features_added': 26}
    )

    lineage.metadata['model_version'] = '1.0.0'
    lineage.complete()

    # Save
    tracker.save_run(lineage)

    # Print
    print(json.dumps(lineage.to_dict(), indent=2))
