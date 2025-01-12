from __future__ import print_function

import os
import logging
import warnings
import numpy
import pandas
import torch
import torch.nn as nn

from .class1_presentation_predictor import Class1PresentationPredictor
from .percent_rank_transform import PercentRankTransform

class TorchPresentationPredictor(Class1PresentationPredictor):
    """PyTorch implementation of the presentation predictor"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._torch_models = {}

    def get_model(self, name=None):
        """
        Load or instantiate a new logistic regression model in PyTorch.
        
        Parameters
        ----------
        name : string
            Model variant name ('with_flanks' or 'without_flanks')
            
        Returns
        -------
        torch.nn.Module
        """
        if name is None:
            return nn.Linear(len(self.model_inputs), 1)
            
        if name not in self._torch_models:
            model = nn.Sequential(
                nn.Linear(len(self.model_inputs), 1)
            )
            row = self.weights_dataframe.loc[name]
            
            # Convert weights and bias to PyTorch tensors
            weights = torch.FloatTensor(row[self.model_inputs].values)
            bias = torch.FloatTensor([row.intercept])
            
            # Assign the weights to first layer
            with torch.no_grad():
                model[0].weight.copy_(weights.unsqueeze(0))
                model[0].bias.copy_(bias)
                
            model = model.to(self.device)
            self._torch_models[name] = model
            
        return self._torch_models[name]

    def get_model(self, name=None):
        """
        Load or instantiate a new logistic regression model in PyTorch.
        
        Parameters
        ----------
        name : string
            Model variant name ('with_flanks' or 'without_flanks')
            
        Returns
        -------
        torch.nn.Module
        """
        if name is None:
            return nn.Linear(len(self.model_inputs), 1)
            
        if name not in self._torch_models:
            model = nn.Linear(len(self.model_inputs), 1)
            row = self.weights_dataframe.loc[name]
            
            # Convert weights and bias to PyTorch tensors
            weights = torch.FloatTensor(row[self.model_inputs].values)
            bias = torch.FloatTensor([row.intercept])
            
            # Assign the weights
            with torch.no_grad():
                model.weight.copy_(weights.unsqueeze(0))
                model.bias.copy_(bias)
                
            model = model.to(self.device)
            self._torch_models[name] = model
            
        return self._torch_models[name]

    def predict(self, *args, **kwargs):
        """
        Override predict to use PyTorch models for the final presentation score calculation
        """
        df = super().predict(*args, **kwargs)
        
        if "processing_score" in df.columns and "affinity_score" in df.columns:
            if len(df) > 0:
                model_name = 'with_flanks' if 'n_flank' in df.columns else "without_flanks"
                model = self.get_model(model_name)
                
                input_matrix = df[self.model_inputs]
                null_mask = None
                if not kwargs.get("throw", True):
                    null_mask = input_matrix.isnull().any(axis=1)
                    input_matrix = input_matrix.fillna(0.0)
                
                # Convert to PyTorch tensor
                inputs = torch.FloatTensor(input_matrix.values).to(self.device)
                
                # Get predictions
                with torch.no_grad():
                    model.eval()
                    logits = model(inputs)
                    df["presentation_score"] = torch.sigmoid(logits).squeeze().cpu().numpy()
                
                if null_mask is not None:
                    df.loc[null_mask, "presentation_score"] = numpy.nan
                    
                df["presentation_percentile"] = self.percentile_ranks(
                    df["presentation_score"], throw=kwargs.get("throw", True))
            else:
                df["presentation_score"] = []
                df["presentation_percentile"] = []
        
        return df

    @classmethod
    def load(cls, models_dir=None, max_models=None):
        """Load a presentation predictor with PyTorch models"""
        predictor = super().load(models_dir, max_models)
        return TorchPresentationPredictor(
            affinity_predictor=predictor.affinity_predictor,
            processing_predictor_with_flanks=predictor.processing_predictor_with_flanks,
            processing_predictor_without_flanks=predictor.processing_predictor_without_flanks,
            weights_dataframe=predictor.weights_dataframe,
            metadata_dataframes=predictor.metadata_dataframes,
            percent_rank_transform=predictor.percent_rank_transform,
            provenance_string=predictor.provenance_string)
