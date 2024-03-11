import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


epochs_list =  [15]
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
# Initialize lists to store results
results_list = []

for num_epochs in epochs_list:
    accuracy_list = []
    precision_list = []
    recall_list = []
    auc_list = []

    for fold, (train_index, test_index) in enumerate(skf.split(data_array_images, y_combined.argmax(axis=1))):
        X_train_images, X_test_images = data_array_images[train_index], data_array_images[test_index]
        y_train, y_test = y_combined[train_index], y_combined[test_index]

        base_model =  DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in base_model.layers:
            layer.trainable = False

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(128, activation='relu')(x)
        output = Dense(n_classes, activation='sigmoid')(x)  

        model = Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall', 'AUC'])
        print(f"Training Fold {fold + 1}/{n_splits} with {num_epochs} epochs...")
        model.fit(x=[X_train_images], y=y_train, validation_data=([X_test_images], y_test),
                  epochs=num_epochs, batch_size=64)

        y_pred = model.predict([X_test_images])
        accuracy = accuracy_score(y_test, (y_pred > 0.5).astype(int))
        precision = precision_score(y_test, (y_pred > 0.5).astype(int), average='micro')
        recall = recall_score(y_test, (y_pred > 0.5).astype(int), average='micro')
        auc = roc_auc_score(y_test, y_pred, average='micro')
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        auc_list.append(auc)


    average_accuracy = np.mean(accuracy_list)
    average_precision = np.mean(precision_list)
    average_recall = np.mean(recall_list)
    average_auc = np.mean(auc_list)


    results_list.append({
        'Epochs': num_epochs,
        'Average Accuracy': average_accuracy,
        'Average Precision': average_precision,
        'Average Recall': average_recall,
        'Average AUC': average_auc
    })


results_df = pd.DataFrame(results_list)
print(results_df)
