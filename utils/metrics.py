import torch
import torchmetrics

class MetricsLogger:
  """
  Metrics logging system for tracking model performance metrics
  across different stages of training and evaluation.
  """
  
  def __init__(self, config, device='cuda'):
    """
    Initialize metrics logger based on configuration.
    """
    self.device = device
    self.metrics = {}
    self.stages = ['train', 'val', 'test']
    
    # Create metrics for each stage based on config
    for stage in self.stages:
      self.metrics[stage] = self._create_metrics_for_stage(config, stage)
            
  def _create_metrics_for_stage(self, config):
    """
    Create metrics for a specific stage based on configuration.
    """
    task = config.get('task', 'binary')
    num_classes = config.get('num_classes', 2)
    average = config.get('average', 'macro')
    
    metrics_dict = {}
    
    # Basic metrics that apply to most tasks
    metrics_dict['accuracy'] = torchmetrics.Accuracy(task=task, num_classes=num_classes).to(self.device)
    
    # Add AUROC for classification tasks
    if task in ['binary', 'multiclass', 'multilabel']:
      metrics_dict['auroc'] = torchmetrics.AUROC(
        task=task, 
        num_classes=num_classes, 
        average=average
      ).to(self.device)
        
    # Add F1 score for classification
    if task in ['binary', 'multiclass', 'multilabel']:
      metrics_dict['f1'] = torchmetrics.F1Score(
        task=task,
        num_classes=num_classes,
        average=average
      ).to(self.device)
    
    # Add more metrics based on configuration
    if config.get('include_precision', False):
      metrics_dict['precision'] = torchmetrics.Precision(
        task=task,
        num_classes=num_classes,
        average=average
      ).to(self.device)
        
    if config.get('include_recall', False):
      metrics_dict['recall'] = torchmetrics.Recall(
        task=task,
        num_classes=num_classes,
        average=average
      ).to(self.device)
        
    return metrics_dict
    
  def update(self, stage, y_pred, y_true):
    """
    Update metrics for the given stage with predictions and ground truth.
    """
    if stage not in self.metrics:
      raise ValueError(f"Unknown stage: {stage}")
    
    for _, metric in self.metrics[stage].items():
      metric.update(y_pred, y_true)
            
  def compute_and_log(self, stage, logger, prefix=''):
    """
    Compute metrics for the stage and log them.
    """
    if stage not in self.metrics:
      raise ValueError(f"Unknown stage: {stage}")
    
    results = {}
    for name, metric in self.metrics[stage].items():
      value = metric.compute()
      metric_name = f"{prefix}{stage}_{name}"
      logger(metric_name, value, prog_bar=True)
      results[metric_name] = value
        
    return results
    
  def reset(self, stage=None):
    """
    Reset metrics for the given stage or all stages.
    """
    if stage is None:
      for s in self.stages:
        self._reset_stage(s)
    elif stage in self.metrics:
      self._reset_stage(stage)
    else:
      raise ValueError(f"Unknown stage: {stage}")
            
  def _reset_stage(self, stage):
    """
    Reset all metrics for a specific stage.
    """
    for metric in self.metrics[stage].values():
      metric.reset() 