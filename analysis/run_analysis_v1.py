import matplotlib.pyplot as plt
import pandas as pd

# Data: Model, Dataset, Accuracy
data = [
    ('resnet18', 'BeeOrWasp', 0.9377777777777778),
    ('resnet18', 'InsectRecognition', 0.6696428571428571),
    ('resnet18', 'ImagesCV', 0.5017131669114048),
    ('mobilenet_v2', 'BeeOrWasp', 0.9422222222222222),
    ('mobilenet_v2', 'InsectRecognition', 0.6741071428571429),
    ('mobilenet_v2', 'ImagesCV', 0.5330396475770925),
    ('vgg16', 'BeeOrWasp', 0.9422222222222222),
    ('vgg16', 'InsectRecognition', 0.65625),
    ('vgg16', 'ImagesCV', 0.46500244738130203)
]

# Create a DataFrame
df = pd.DataFrame(data, columns=['Model', 'Dataset', 'Accuracy'])

# Plot accuracy per model and dataset
plt.figure(figsize=(10, 6))
plt.bar(df['Model'] + ' - ' + df['Dataset'], df['Accuracy'], color=['blue', 'green', 'red', 'blue', 'green', 'red', 'blue', 'green', 'red'])

# Adding titles and labels
plt.title('Model Accuracy for Different Datasets')
plt.xlabel('Model - Dataset')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.ylim(0, 1)

# Show the plot
plt.tight_layout()
plt.show()

# Calculate the average accuracy for each model
average_accuracy_per_model = df.groupby('Model')['Accuracy'].mean()

# Plot average accuracy for each model
plt.figure(figsize=(8, 5))
average_accuracy_per_model.plot(kind='bar', color=['blue', 'green', 'red'])
plt.title('Average Accuracy for Each Model')
plt.xlabel('Model')
plt.ylabel('Average Accuracy')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
