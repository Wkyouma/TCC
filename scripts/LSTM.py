import os
import numpy as np
import pandas as pd
import glob
from tensorflow.keras.models import Sequential
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "pose")

data = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

def get_input():
    NAMES = [
        "nose",
        "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear",
        "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_pinky", "right_pinky",
        "left_index", "right_index",
        "left_thumb", "right_thumb",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_heel", "right_heel",
        "left_foot_index", "right_foot_index",
    ]
    colunas = []
    for i in NAMES:
        colunas.append(i+"_wx")
        colunas.append(i+"_wy")
        colunas.append(i+"_wz")
    return colunas

def model(n_frames,n_features):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(n_frames, n_features)))
    model.add(Dropout(0.3))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(3, activation="softmax"))

    model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return model

def windowed_data(df, window_size=15):
    colunas = get_input()
    X = []
    y = []
    for i in range(len(df) - window_size):
        if pd.isna(df["classe"].iloc[i + window_size]):
            continue
        X.append(df[colunas].iloc[i:i + window_size].values)
        y.append(df["classe"].iloc[i + window_size])
    return np.array(X), np.array(y)

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df[get_input() + ["classe"]]



def main():
    x_list, y_list, group_list = [], [], []
    for p in data:
        df = load_data(p)
        X, y = windowed_data(df)
        x_list.append(X)
        y_list.append(y)
        group_list.extend([os.path.basename(p)] * len(y))
        print(os.path.basename(p), X.shape, y.shape)
    X = np.concatenate(x_list)
    y = np.concatenate(y_list).astype(int)
    grupos = np.array(group_list)
    separador = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    treino, validacao = next(separador.split(X, y, grupos))
    modelo = model(X.shape[1], X.shape[2])
    modelo.summary()

    modelo.fit(
        X[treino], y[treino],
        validation_data=(X[validacao], y[validacao]),
        epochs=50,
        batch_size=64,
        class_weight={0: 1.0, 1: 4.0, 2: 1.5}
    )

    pred = modelo.predict(X[validacao]).argmax(axis=1)
    print(confusion_matrix(y[validacao], pred))
    print(classification_report(y[validacao], pred,
        target_names=["estavel", "desequilibrio", "queda"]))
if __name__ == "__main__":
    main()



