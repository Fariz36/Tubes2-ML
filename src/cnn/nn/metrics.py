import numpy as np
import tensorflow as tf

def macro_f1_score(
    y_true,
    y_pred,
    num_classes=None,
):
    true_labels = np.asarray(y_true).reshape(-1)
    predicted = np.asarray(y_pred)
    if predicted.ndim > 1:
        predicted_labels = np.argmax(predicted, axis=-1).reshape(-1)
    else:
        predicted_labels = predicted.reshape(-1)

    if true_labels.shape != predicted_labels.shape:
        raise ValueError(
            f"Shape mismatch between y_true {true_labels.shape} and y_pred {predicted_labels.shape}"
        )

    if num_classes is None:
        num_classes = int(max(true_labels.max(initial=0), predicted_labels.max(initial=0)) + 1)

    scores = []
    for class_id in range(num_classes):
        true_positive = np.sum((true_labels == class_id) & (predicted_labels == class_id))
        false_positive = np.sum((true_labels != class_id) & (predicted_labels == class_id))
        false_negative = np.sum((true_labels == class_id) & (predicted_labels != class_id))

        precision = true_positive / (true_positive + false_positive + 1e-12)
        recall = true_positive / (true_positive + false_negative + 1e-12)
        score = 2.0 * precision * recall / (precision + recall + 1e-12)
        scores.append(score)

    return float(np.mean(scores)) if scores else 0.0

class SparseMacroF1(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="macro_f1", dtype=tf.float32):
        super().__init__(name=name, dtype=dtype)
        self.num_classes = int(num_classes)
        self.metric = tf.keras.metrics.F1Score(
            average="macro",
            name=f"{name}_delegate",
            dtype=dtype,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true = tf.one_hot(y_true, depth=self.num_classes, dtype=self.dtype)
        y_pred = tf.cast(tf.reshape(y_pred, [-1, self.num_classes]), self.dtype)
        self.metric.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return self.metric.result()

    def reset_state(self):
        self.metric.reset_state()

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config