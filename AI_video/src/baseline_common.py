"""Shared CPU configuration, integrity checks and small-sample metrics."""
import os
import hashlib
import numpy as np
from video_pipeline import ROOT

os.environ['KERAS_HOME'] = str(ROOT / '.tools/keras')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '4'


def weight_hash(model):
    digest = hashlib.sha256()
    for weight in model.get_weights():
        digest.update(weight.tobytes())
    return digest.hexdigest()


def metrics(y, scores):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores)
    predicted = scores >= 0.5
    tp = int(np.sum((y == 1) & predicted))
    tn = int(np.sum((y == 0) & ~predicted))
    fp = int(np.sum((y == 0) & predicted))
    fn = int(np.sum((y == 1) & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    positives, negatives = scores[y == 1], scores[y == 0]
    auc = float(np.mean((positives[:, None] > negatives[None, :]) +
                        0.5 * (positives[:, None] == negatives[None, :]))) if len(positives) and len(negatives) else None
    return {'video_count': len(y), 'real_count': int(np.sum(y == 0)), 'fake_count': int(np.sum(y == 1)),
            'accuracy': (tp + tn) / len(y), 'precision_fake': precision, 'recall_fake': recall,
            'f1_fake': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'roc_auc': auc, 'false_positive_rate': fp / (fp + tn) if fp + tn else None,
            'confusion_matrix_rows_real_fake': [[tn, fp], [fn, tp]], 'threshold': 0.5}
