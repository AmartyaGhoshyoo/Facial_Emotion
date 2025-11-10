from Model_Save import model_saving
from Data_Preprocessing import X_test
from Common_Imports import *
model=model_saving()
X_test_3 = np.repeat(X_test, 3, axis=-1)

loss, accuracy =model.evaluate(X_test_3, y_test)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Get predictions on the balanced test set
y_pred_balanced = model.predict(X_test_3)
y_pred_classes_balanced = np.argmax(y_pred_balanced, axis=1)

# Generate classification report for the balanced test set
print("Classification Report on Balanced Test Set:")
print(classification_report(y_test, y_pred_classes_balanced))

# Define emotion labels
emotion_labels = {
    0: 'Anger',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happiness',
    4: 'Sadness',
    5: 'Surprise',
    6: 'Neutral',
    7: 'Contempt'
}

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred_classes_balanced)

# Plot confusion matrix with labels and colors
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=emotion_labels.values(),
            yticklabels=emotion_labels.values())
plt.xlabel('Predicted Emotion')
plt.ylabel('True Emotion')
plt.title('Confusion Matrix on Balanced Test Set')
plt.show()