import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "vocal_features.csv")
WAV_ROOT = r"C:\healthcare_model2\data"

SAMPLE_PER_CLASS = 1500
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 8
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 65)
print("AVHFM + EFFICIENTNETB0 SPECTROGRAM EXPERIMENT")
print("=" * 65)

# ============================================================
# LOAD CSV
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(CSV_PATH)

print("CSV recordings:", len(df))
print("Patients:", df["patient_id"].nunique())

# ============================================================
# INDEX WAV FILES
# ============================================================

print("\nIndexing WAV files...")

wav_map = {}

for root, dirs, files in os.walk(WAV_ROOT):
    for file in files:
        if file.lower().endswith(".wav"):
            wav_map[file.lower()] = os.path.join(root, file)

print("WAV files indexed:", len(wav_map))

# ============================================================
# MATCH WAV FILES
# ============================================================

df["wav_path"] = df["file_path"].apply(
    lambda x: wav_map.get(
        os.path.basename(x).lower()
    )
)

df = df.dropna(
    subset=["wav_path"]
).reset_index(drop=True)

print("Matched labeled WAVs:", len(df))

# ============================================================
# BALANCED SAMPLE
# ============================================================

healthy = df[df["label"] == 0]
disorder = df[df["label"] == 1]

n = min(
    SAMPLE_PER_CLASS,
    len(healthy),
    len(disorder)
)

healthy = healthy.sample(
    n=n,
    random_state=SEED
)

disorder = disorder.sample(
    n=n,
    random_state=SEED
)

sample = pd.concat(
    [healthy, disorder],
    ignore_index=True
)

sample = sample.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)

print("\nExperiment recordings:", len(sample))
print(
    "Healthy:",
    int((sample["label"] == 0).sum())
)
print(
    "Disorder:",
    int((sample["label"] == 1).sum())
)

# ============================================================
# RECORDING-LEVEL PILOT SPLIT
# ============================================================

# This is intentionally a recording-level pilot.
# It is NOT the final patient-independent benchmark.

train_df, val_df = train_test_split(
    sample,
    test_size=0.20,
    stratify=sample["label"],
    random_state=SEED
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print("\nTraining recordings:", len(train_df))
print("Validation recordings:", len(val_df))

# ============================================================
# AUDIO → RGB MEL SPECTROGRAM
# ============================================================

def audio_to_image(path, label):

    audio_binary = tf.io.read_file(path)

    audio, sample_rate = tf.audio.decode_wav(
        audio_binary,
        desired_channels=1
    )

    audio = tf.squeeze(
        audio,
        axis=-1
    )

    target_samples = 32000

    # Crop
    audio = audio[:target_samples]

    # Pad
    current_length = tf.shape(audio)[0]

    padding = tf.maximum(
        0,
        target_samples - current_length
    )

    audio = tf.pad(
        audio,
        [[0, padding]]
    )

    audio = audio[:target_samples]

    # --------------------------------------------------------
    # STFT
    # --------------------------------------------------------

    stft = tf.signal.stft(
        audio,
        frame_length=1024,
        frame_step=256,
        fft_length=1024
    )

    power = tf.square(
        tf.abs(stft)
    )

    # --------------------------------------------------------
    # MEL SPECTROGRAM
    # --------------------------------------------------------

    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=128,
        num_spectrogram_bins=513,
        sample_rate=16000,
        lower_edge_hertz=50.0,
        upper_edge_hertz=8000.0
    )

    mel = tf.matmul(
        power,
        mel_matrix
    )

    mel = tf.math.log(
        mel + 1e-6
    )

    # Normalize
    mean = tf.reduce_mean(mel)
    std = tf.math.reduce_std(mel)

    mel = (
        mel - mean
    ) / (
        std + 1e-6
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    mel = tf.expand_dims(
        mel,
        axis=-1
    )

    mel = tf.image.resize(
        mel,
        IMG_SIZE
    )

    # --------------------------------------------------------
    # Convert grayscale → RGB
    # --------------------------------------------------------

    mel = tf.image.grayscale_to_rgb(
        mel
    )

    # EfficientNet preprocessing expects float tensors.
    mel = tf.cast(
        mel,
        tf.float32
    )

    label = tf.cast(
        label,
        tf.float32
    )

    return mel, label


# ============================================================
# DATASETS
# ============================================================

train_paths = train_df[
    "wav_path"
].values

train_labels = train_df[
    "label"
].values.astype(
    np.float32
)

val_paths = val_df[
    "wav_path"
].values

val_labels = val_df[
    "label"
].values.astype(
    np.float32
)

print("\nCreating TensorFlow datasets...")

train_ds = tf.data.Dataset.from_tensor_slices(
    (
        train_paths,
        train_labels
    )
)

train_ds = train_ds.map(
    audio_to_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.shuffle(
    len(train_df),
    seed=SEED
)

train_ds = train_ds.batch(
    BATCH_SIZE
)

train_ds = train_ds.prefetch(
    tf.data.AUTOTUNE
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_paths,
        val_labels
    )
)

val_ds = val_ds.map(
    audio_to_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.batch(
    BATCH_SIZE
)

val_ds = val_ds.prefetch(
    tf.data.AUTOTUNE
)

# ============================================================
# PRETRAINED EFFICIENTNET
# ============================================================

print("\nLoading pretrained EfficientNetB0...")

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)

# Freeze pretrained layers for the first experiment.
base_model.trainable = False

# ============================================================
# CLASSIFICATION HEAD
# ============================================================

inputs = keras.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)

x = base_model(
    inputs,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(
    0.35
)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(
    0.30
)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = keras.Model(
    inputs,
    outputs
)

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0005
    ),

    loss="binary_crossentropy",

    metrics=[
        keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        keras.metrics.Precision(
            name="precision"
        ),
        keras.metrics.Recall(
            name="recall"
        ),
        keras.metrics.AUC(
            name="auc"
        )
    ]
)

print("\nTrainable parameters:")
print(
    sum(
        np.prod(v.shape)
        for v in model.trainable_variables
    )
)

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-6
    )
]

# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 65)
print("STARTING EFFICIENTNET PILOT")
print("=" * 65)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")

probabilities = model.predict(
    val_ds,
    verbose=1
).reshape(-1)

predictions = (
    probabilities >= 0.50
).astype(int)

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    val_labels,
    predictions
)

precision = precision_score(
    val_labels,
    predictions,
    zero_division=0
)

recall = recall_score(
    val_labels,
    predictions,
    zero_division=0
)

f1 = f1_score(
    val_labels,
    predictions,
    zero_division=0
)

auc = roc_auc_score(
    val_labels,
    probabilities
)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 65)
print("EFFICIENTNET SPECTROGRAM PILOT RESULTS")
print("=" * 65)

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC  : {auc * 100:.2f}%"
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        val_labels,
        predictions
    )
)

print("=" * 65)

print(
    "\nThis is a recording-level pilot."
)