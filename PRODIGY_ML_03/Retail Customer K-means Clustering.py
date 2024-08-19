import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define paths to the dataset
TRAIN_DIR = 'C:\\Users\\S.Bharathi\\Desktop\\dataset\\Cat'  # Path to the Cat directory
DOG_DIR = 'C:\\Users\\S.Bharathi\\Desktop\\dataset\\Dog'   # Path to the Dog directory
IMAGE_SIZE = (64, 64)  # Resize images to 64x64 pixels

def load_images_labels(directories, limit=None):
    images = []
    labels = []
    
    for directory, label in directories:
        for idx, filename in enumerate(os.listdir(directory)):
            if limit and idx >= limit:
                break
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE)  # Resize to a smaller size for faster processing
                images.append(img)
                labels.append(label)
    
    return np.array(images), np.array(labels)

# Load training data
directories = [(TRAIN_DIR, 0), (DOG_DIR, 1)]  # 0 for Cat, 1 for Dog
X, y = load_images_labels(directories, limit=2000)  # Limit to 2000 images for faster processing

# Check class distribution in the dataset
unique, counts = np.unique(y, return_counts=True)
class_distribution = dict(zip(unique, counts))
print("Full dataset class distribution:", class_distribution)

# EDA: Visualize the distribution of classes
sns.countplot(x=y)
plt.title('Distribution of Cats and Dogs')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Cat', 'Dog'])
plt.show()

# Show some sample images
def plot_samples(images, labels, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = images[i].astype('uint8')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Dog" if labels[i] == 1 else "Cat")
        plt.axis("off")
    plt.show()

plot_samples(X, y, n=10)

# Flatten images for SVM
X_flattened = X.reshape(X.shape[0], -1)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_flattened, y, test_size=0.2, random_state=42)

# Check class distribution in training and validation sets
train_unique, train_counts = np.unique(y_train, return_counts=True)
val_unique, val_counts = np.unique(y_val, return_counts=True)
train_class_distribution = dict(zip(train_unique, train_counts))
val_class_distribution = dict(zip(val_unique, val_counts))

print("Training set class distribution:", train_class_distribution)
print("Validation set class distribution:", val_class_distribution)

# Ensure both classes are present
if len(train_class_distribution) < 2:
    raise ValueError("The training set must contain at least two classes.")

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train SVM classifier
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_val, y_pred, target_names=['Cat', 'Dog']))

# Confusion Matrix
conf_matrix = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Visualize some predictions
plot_samples(X_val.reshape(-1, *IMAGE_SIZE, 3)[:10], y_val[:10], n=10)
plt.show()
