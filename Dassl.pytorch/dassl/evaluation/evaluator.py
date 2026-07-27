import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .build import EVALUATOR_REGISTRY


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        # Keep one-vs-rest probabilities so that multiclass AUC can be
        # reported together with the original accuracy metrics.
        self._y_score = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        self._y_score = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())
        probs = torch.softmax(mo.detach(), dim=1).cpu().numpy()
        self._y_score.extend(probs.tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )
        balanced_acc = 100.0 * balanced_accuracy_score(
            self._y_true, self._y_pred
        )

        labels = np.arange(len(self._y_score[0]))
        try:
            macro_auc = 100.0 * roc_auc_score(
                self._y_true,
                np.asarray(self._y_score),
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        except ValueError:
            # AUC is undefined if a split is missing a class. Keep the log
            # usable and make the condition explicit instead of crashing.
            macro_auc = float("nan")

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1
        results["balanced_accuracy"] = balanced_acc
        results["auc"] = macro_auc

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.2f}%\n"
            f"* error: {err:.2f}%\n"
            f"* macro_f1: {macro_f1:.2f}%\n"
            f"* balanced_accuracy: {balanced_acc:.2f}%\n"
            f"* auc: {macro_auc:.2f}%"
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            y_true = np.asarray(self._y_true)
            y_pred = np.asarray(self._y_pred)
            y_score = np.asarray(self._y_score)
            precision, recall, f1, _support = precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=labels,
                zero_division=0,
            )
            cmat = confusion_matrix(y_true, y_pred, labels=labels)
            print("=> per-class result")
            accs = []

            for idx, label in enumerate(labels):
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                true_positive = cmat[idx, idx]
                false_positive = cmat[:, idx].sum() - true_positive
                negative = cmat.sum() - cmat[idx, :].sum()
                specificity = (
                    100.0 * (negative - false_positive) / negative
                    if negative > 0 else float("nan")
                )
                try:
                    class_auc = 100.0 * roc_auc_score(
                        (y_true == label).astype(np.int32), y_score[:, idx]
                    )
                except ValueError:
                    class_auc = float("nan")
                print(
                    "* class: {} ({})\t"
                    "total: {:,}\t"
                    "correct: {:,}\t"
                    "acc: {:.2f}%\t"
                    "precision: {:.2f}%\t"
                    "recall: {:.2f}%\t"
                    "specificity: {:.2f}%\t"
                    "f1: {:.2f}%\t"
                    "auc: {:.2f}%".format(
                        label,
                        classname,
                        total,
                        correct,
                        acc,
                        100.0 * precision[idx],
                        100.0 * recall[idx],
                        specificity,
                        100.0 * f1[idx],
                        class_auc,
                    )
                )
                results["recall_class_{}".format(label)] = 100.0 * recall[idx]
            mean_acc = np.mean(accs)
            print("* average: {:.2f}%".format(mean_acc))

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print('Confusion matrix is saved to "{}"'.format(save_path))

        return results
