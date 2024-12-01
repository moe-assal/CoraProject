import torch


def f1_accuracy(predictions, ground_truth, num_classes=7):
    # Initialize precision, recall, and F1 scores for each class
    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)
    f1 = torch.zeros(num_classes)

    for class_id in range(num_classes):
        # True positives (TP): correct predictions for the current class
        tp = ((predictions == class_id) & (ground_truth == class_id)).sum().item()
        # False positives (FP): incorrect predictions (predicted class is the current class but actual is different)
        fp = ((predictions == class_id) & (ground_truth != class_id)).sum().item()
        # False negatives (FN): actual class is the current class but predicted differently
        fn = ((predictions != class_id) & (ground_truth == class_id)).sum().item()

        # Calculate precision and recall for the current class
        precision[class_id] = tp / (tp + fp) if tp + fp > 0 else 0
        recall[class_id] = tp / (tp + fn) if tp + fn > 0 else 0

        # Calculate F1 score for the current class
        if precision[class_id] + recall[class_id] > 0:
            f1[class_id] = 2 * (precision[class_id] * recall[class_id]) / (precision[class_id] + recall[class_id])
        else:
            f1[class_id] = 0

    # Return the average F1 score across all classes
    return f1.mean()


def confusion_matrix(predictions, ground_truth, num_classes=7):
    # Initialize the confusion matrix
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    # Loop over each instance and update the confusion matrix
    for true_label, predicted_label in zip(ground_truth, predictions):
        cm[true_label.item(), predicted_label.item()] += 1

    return cm
