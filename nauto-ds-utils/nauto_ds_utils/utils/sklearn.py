import pandas as pd
import boto3
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from pathlib import Path
from typing import Union, List, Optional
from io import BytesIO

from nauto_datasets.core.serialization import (FileHandler, FileLocation,
                                               FileSource)
from nauto_datasets.utils.boto import path_to_bucket_and_key


def read_features(path: Path,
                  file_source: Optional[FileSource] = None,
                  sep=',') -> List[str]:
    """Read in features about a sklearn model from s3 or local"""
    if not isinstance(path, Path):
        path = Path(path.replace('s3://', ''))
    if file_source is None:
        if path.parts[0] == 's3:':
            path = path.parts[1:]
            file_source = FileSource(3)
        elif path.parts[0].startswith('nauto-'):
            file_source = FileSource(3)
        else:
            file_source = FileSource(1)
    loc = FileLocation(path, file_source)
    return FileHandler().read_data(loc).decode('utf-8').split(',')


def load_model(path: Path):
    """Loads sklearn models from s3 or local machine"""
    if isinstance(path, str):
        path = Path(path.replace('s3://', ''))

    if path.parts[0].startswith('s3:'):
        path = Path('/'.join(path.parts[1:]))

    if path.parts[0].startswith('nauto-'):
        bucket, key = path_to_bucket_and_key(path)
        with BytesIO() as f:
            boto3.client('s3').download_fileobj(
                Bucket=bucket, Key=key, Fileobj=f)
            f.seek(0)
            model = joblib.load(f)
    else:
        with open(path, 'rb') as f:
            model = joblib.load(f)
    return model


def display_feature_importance(model,
                               cols: List[str]) -> Union[pd.DataFrame, str]:
    """Returns Feature Importantance DataFrame for Logistic and Random Forest 
    Classifier"""
    if type(model) == RandomForestClassifier or type(model) == XGBClassifier:
        importance_coeff = getattr(model, 'feature_importances_')
    elif type(model) == LogisticRegression:
        importance_coeff = getattr(model, 'coef_')[0]
    else:
        return f"""Feature Importance isn't supported for {type(model)}"""
    result_df = pd.DataFrame(importance_coeff,
                             index=cols,
                             columns=['importance']) \
        .sort_values(by='importance', ascending=False)
    result_df['importance'] = round(result_df['importance'], 2) * 100
    return result_df
