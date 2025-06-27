# Project Update

## New Notebooks

- **1data_after_curation_firstview**  
  Comparison plots between dataset 3.0 and dataset 2.0.

- **1share_test_splitting**  
  Splitting target test set and share test set.

- **1regression_plot_and_test**  
  Prediction results in scatter plots for regression tasks on DC50 and Dmax.

- **1SASA_CalAndPlot**  
  Calculate SASA and plot the relationship between Lys SASA and Dmax value.

## Source Code

- **Regression_get_studies_datasets.py**  
  Used to get similarity and random datasets for regression tasks (DC50 and Dmax).

- **Regression_run_experiments_pytorch.py**  
  Used for training PyTorch models on regression.

## Model Changes

The main changes include:  
1. Input column 
2. Loss function  
3. Evaluation metrics, data statistic 
4. Cross-validation group methods
5. ## Model Changes

The main changes include:  
1. Input columns  
2. Loss function  
3. Evaluation metrics and data statistics  
4. Cross-validation group methods  
5. Addition of a model test function: `get_protac_active_regression_proba` in `protac_degradation_predictor.py`

   All changes in src and protac_degradation_predictor are marked with the comment `#When regression`.
