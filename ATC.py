import numpy as np


class ATC(object):
    def ATC_accuracy(
        self, source_probs, source_labels, target_probs, score_function="MC"
    ):
        """
        Calculate the accuracy of the ATC model.
        # Arguments
            source_probs: numpy array of shape (N, num_classes)
            source_labels: numpy array of shape (N, )
            target_probs: numpy array of shape (M, num_classes)
            score_function: string, either "MC" or "NE"
        # Returns
            ATC_acc: float
        """

        if score_function == "MC":
            source_score = np.max(source_probs, axis=-1)
            target_score = np.max(target_probs, axis=-1)

        elif score_function == "NE":
            source_score = np.sum(
                np.multiply(source_probs, np.log(source_probs + 1e-20)), axis=1
            )
            target_score = np.sum(
                np.multiply(target_probs, np.log(target_probs + 1e-20)), axis=1
            )

        source_preds = np.argmax(source_probs, axis=-1)

        _, ATC_threshold = self.find_threshold_balance(
            source_score, source_labels == source_preds
        )

        ATC_acc = np.mean(target_score >= ATC_threshold) * 100.0

        return ATC_acc

    def find_threshold_balance(self, score, labels):
        sorted_idx = np.argsort(score)

        sorted_score = score[sorted_idx]
        sorted_labels = labels[sorted_idx]

        fp = np.sum(labels == 0)
        fn = 0.0

        min_fp_fn = np.abs(fp - fn)
        thres = 0.0
        min_fp = fp
        min_fn = fn
        for i in range(len(labels)):
            if sorted_labels[i] == 0:
                fp -= 1
            else:
                fn += 1

            if np.abs(fp - fn) < min_fp_fn:
                min_fp = fp
                min_fn = fn
                min_fp_fn = np.abs(fp - fn)
                thres = sorted_score[i]

        return min_fp_fn, thres

    # def fit(self, source_probs, source_labels, score_function="MC"):
    def fit(self, X, Y, score_function="MC"):
        self.score_function = score_function
        if self.score_function == "MC":
            source_score = np.max(X, axis=-1)
        elif self.score_function == "NE":
            source_score = np.sum(np.multiply(X, np.log(X + 1e-20)), axis=1)

        source_preds = np.argmax(X, axis=-1)

        _, self.ATC_threshold = self.find_threshold_balance(
            source_score, Y == source_preds
        )

    # def predict(self, target_probs):
    def predict(self, X):
        if self.score_function == "MC":
            target_score = np.max(X, axis=-1)
        elif self.score_function == "NE":
            target_score = np.sum(np.multiply(X, np.log(X + 1e-20)), axis=1)
        return np.mean(target_score >= self.ATC_threshold) * 100.0
