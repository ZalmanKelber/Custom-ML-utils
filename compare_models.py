from collections.abc import Iterable
import numpy as np
import pandas as pd
from typing import TypeAlias
from itertools import chain, combinations
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import os

Prediction: TypeAlias = Iterable[tuple[str, Iterable[float]]]

class ModelComparer:
    def __init__(self, y_true: Iterable | None = None, 
                 predictions: Prediction | None = None, 
                 probs: Prediction | None = None) -> None:
        self.y_true = np.array(y_true)
        self.predictions = predictions if predictions is not None else []
        self.probs = probs if probs is not None else []
        self._clean_names(probs=True)
        self._clean_names(probs=False)
        self._scorer_map = {
            'accuracy': accuracy_score,
            'roc_auc': roc_auc_score,
            'f1': f1_score
        }

    def set_y_true(self, y_true: Iterable) -> None:
        self.y_true = np.array(y_true)

    def add_predictions(self, predictions: Prediction, replace: bool = False) -> None:
        if replace:
            self.predictions = predictions
        else:
            self.predictions = np.concatenate(self.predictions, predictions)
        self._clean_names(probs=False)

    def add_probs(self, probs: Prediction, replace: bool = False) -> None:
        if replace:
            self.probs = probs
        else:
            self.probs = np.concatenate(self.probs, probs)
        self.clean_names(probs=True)
    
    def evaluatate_hard_voting(self, splits: int = 1, tiebreak: int = 1, scoring: str = 'accuracy') -> pd.DataFrame:
        return self._get_scores(splits=splits, tiebreak=tiebreak, scoring=scoring, probs=False)
    
    def evaluatate_soft_voting(self, splits: int = 1, scoring: str = 'accuracy') -> pd.DataFrame:
        return self._get_scores(splits=splits, tiebreak=1, scoring=scoring, probs=True)

    def _get_scores(self, splits: int = 1, tiebreak: int = 1, scoring: str = 'accuracy', probs: bool = False) -> pd.DataFrame:
        if scoring not in ['accuracy', 'roc_auc', 'f1']:
            raise ValueError('scoring must either be "accuracy," "f1" or "roc_auc"')
        if tiebreak not in [0, 1]:
            raise ValueError('tiebreak must be equal to zero or one')
        if splits < 1:
            raise ValueError('number of splits must be greater than zero')
        self._scorer = self._scorer_map[scoring]
        # First check that the predictions and y_true all have the same length
        self._check_input_lengths(probs=probs)
        preds = self.probs if probs else self.predictions
        # Get all combinations of the different models
        s = list(range(len(preds)))
    
        combs = chain.from_iterable([combinations(s, r + 1) for r in range(len(s))])
        cols = ['Model(s) used', 'Full score']
        if splits > 1:
            cols += ['Fold ' + str(i) for i in range(splits)]
        score_df = pd.DataFrame(columns=cols)
        for comb in combs:
            names = ', '.join([preds[i][0] for i in comb])
            row = [names]
            comp_pred = []
            for j in range(len(self.y_true)):
                n = 2 if probs else 1
                score = np.sum([preds[i][1][j] * n for i in comb]) / len(comb)
                comp_pred.append(tiebreak if score == .5 else score // 1)
            row.append(self._scorer(self.y_true, comp_pred))
            if splits > 1:
                for i in range(splits):
                    split_length = len(self.y_true) // splits
                    if len(self.y_true) % splits > i:
                        split_length += 1
                    y_true_split = [self.y_true[splits * j + i] for j in range(split_length)]
                    y_comp_split = [comp_pred[splits * j + i] for j in range(split_length)]
                    row.append(self._scorer(y_true_split, y_comp_split))
            score_df.loc[len(score_df.index)] = row
        return score_df.sort_values(by=['Full score'], ascending=False)
            
                

    def _clean_names(self, probs: bool = False) -> None:
        name_counts = dict()
        preds = self.probs if probs else self.predictions
        for i, pred in enumerate(preds):
            name, pred = pred[0], pred[1]
            if name in name_counts:
                name_counts[name] += 1
                new_name = name + '_' + str(name_counts[name])
                preds[i] = (new_name, pred)
                name_counts[new_name] = 1
            else:
                name_counts[name] = 1

    def _check_input_lengths(self, probs: bool = False) -> None:
        if len(self.y_true) == 0:
            raise ValueError('"y_true" must have non zero length')
        preds = self.probs if probs else self.predictions
        if len(preds) == 0:
            raise ValueError('at least one set of predictions or prabilities is required')
        for name, pred in preds:
            if len(pred) != len(self.y_true):
                raise ValueError("""
                                y_true must have same length as all predictions or prabilities.
                                found model {name} with length {len}
                                """).format(name=name, len=len(self.y_true))
            for i in pred:
                if probs and (i < 0 or i > 1):
                    raise ValueError("""
                                     probabilities must be between 0.0 and 1.0.
                                     Found value of {val} in model {name} 
                                     """).format(val=i, name=name)
                if not probs and i not in [0, 1]:
                    raise ValueError("""
                                     predictions must be either zero or one.
                                     Found value of {val} in model {name} 
                                     """).format(val=i, name=name)

print(os. getcwd())