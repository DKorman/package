"""
Machine learning pipeline for Clstrlobe purposes
==================================
INSERT
DETAILED
DESCRIPTION
HERE

See <LINK-TO-DETAILED-SPHINX-DOCUMENTATION> for complete documentation
"""

__version__ = "0.0.1"
__author__ = "davor_korman"


# from ml_pipeline import data_exploratory_analysis
# from ml_pipeline import data_preprocessing
# from ml_pipeline import data_quality_checking
# from ml_pipeline import feature_engineering
# from ml_pipeline import model_building
# from ml_pipeline import model_eval
# from ml_pipeline import utility
# from ml_pipeline.model_building import XgboostRegressor

from .data_exploratory_analysis import BaseExploratoryDataAnalysis
# from ml_pipeline import data_exploratory_analysis
# from ml_pipeline import data_preprocessing
# from ml_pipeline import data_quality_checking
# from ml_pipeline import feature_engineering
# from ml_pipeline import model_building
# from ml_pipeline import model_eval
# from ml_pipeline import utility
# from .model_building import XgboostRegressor

__all__ = [
    'BaseExploratoryDataAnalysis',
    # "data_exploratory_analysis",
    # "data_preprocessing",
    # "data_quality_checking",
    # "feature_engineering",
    # "model_building",
    # "model_eval",
    # "utility",
    # "XgboostRegressor"
    ]

