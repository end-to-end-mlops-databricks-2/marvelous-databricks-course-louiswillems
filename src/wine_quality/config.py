from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    num_features: List[str]
    cat_features: List[str]
    target: str
    catalog_name: str
    schema_name: str
    parameters: Dict[str, Any]  # Dictionary to hold model-related parameters
    # ab_test: Dict[str, Any]  # Dictionary to hold A/B test parameters
    # pipeline_id: str  # pipeline id for data live tables
    experiment_name_basic: str = "/Shared/wine-quality-basic"
    experiment_name_custom: str = "/Shared/wine-quality-custom"
    experiment_name_fe: Optional[str] = None

    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class Tags(BaseModel):
    git_sha: str
    branch: str