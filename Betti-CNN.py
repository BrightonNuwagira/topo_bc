import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import concatenate, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def create_MLP(dim, regress=False):
    model = Sequential()
    model.add(Dense(800, input_dim=dim, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    return model

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

# Define the epochs to be tested
epochs_list =  [15]

results_list = []  # Initialize a list to store results

for num_epochs in epochs_list:
    accuracy_list = []  
    precision_list = [] 
    recall_list = [] 
    auc_list = [] 

    for fold, (train_index, test_index) in enumerate(skf.split(X_tda_np, y_combined.argmax(axis=1))):
        X_train_tda, X_test_tda = X_tda_np[train_index], X_tda_np[test_index]
        X_train_images, X_test_images = data_array_images[train_index], data_array_images[test_index]
        y_train, y_test = y_combined[train_index], y_combined[test_index]

        mlp = create_MLP(X_tda_np.shape[1], regress=False)

        model_cnn = Sequential([
             EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        ])

        for layer in model_cnn.layers:
            layer.trainable = False

        model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
        model_cnn.add(MaxPooling2D(2, 2))
        model_cnn.add(Flatten())
        model_cnn.add(Dense(64, activation='relu'))

        combinedInput = concatenate([mlp.output, model_cnn.output])
        x = Dense(256, activation="relu")(combinedInput)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(n_classes, activation='sigmoid')(x)

        model = Model(inputs=[mlp.input, model_cnn.input], outputs=x)
        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

      
        print(f"Training Fold {fold + 1}/{n_splits} with {num_epochs} epochs...")
        model.fit(x=[X_train_tda, X_train_images], y=y_train,
                  validation_data=([X_test_tda, X_test_images], y_test),
                  epochs=num_epochs, batch_size=64)

        y_pred = model.predict([X_test_tda, X_test_images])
        accuracy = accuracy_score(y_test, (y_pred > 0.5).astype(int))
        precision = precision_score(y_test, (y_pred > 0.5).astype(int), average='micro')
        recall = recall_score(y_test, (y_pred > 0.5).astype(int), average='micro')
        auc = roc_auc_score(y_test, y_pred, average='micro')
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        auc_list.append(auc)
    avg_metrics = {
        'Epochs': num_epochs,
        'Average Accuracy': np.mean(accuracy_list),
        'Average Precision': np.mean(precision_list),
        'Average Recall': np.mean(recall_list),
        'Average AUC': np.mean(auc_list),
    }
    results_list.append(avg_metrics)
results_df = pd.DataFrame(results_list)
print(results_df)
