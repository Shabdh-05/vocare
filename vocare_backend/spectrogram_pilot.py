import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "vocal_features.csv")
WAV_ROOT = r"C:\healthcare_model2\data"

SAMPLE_PER_CLASS = 1000
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 8
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 60)
print("SPECTROGRAM PILOT")
print("=" * 60)

# ============================================================
# LOAD CSV
# ============================================================

print("\nLoading labels...")

df = pd.read_csv(CSV_PATH)

print("CSV recordings:", len(df))
print("Patients:", df["patient_id"].nunique())

# ============================================================
# INDEX WAV FILES
# ============================================================

print("\nIndexing WAV files...")
print("This can take a little while...")

wav_map = {}

for root, dirs, files in os.walk(WAV_ROOT):
    for file in files:
        if file.lower().endswith(".wav"):
            wav_map[file.lower()] = os.path.join(root, file)

print("WAV files indexed:", len(wav_map))

# ============================================================
# MATCH CSV → WAV
# ============================================================

df["wav_path"] = df["file_path"].apply(
    lambda x: wav_map.get(
        os.path.basename(x).lower()
    )
)

df = df.dropna(
    subset=["wav_path"]
).copy()

df = df.reset_index(drop=True)

print("Matched labeled WAVs:", len(df))

# ============================================================
# BALANCED SAMPLE
# ============================================================

healthy = df[df["label"] == 0].copy()
disorder = df[df["label"] == 1].copy()

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
    frac=1.0,
    random_state=SEED
).reset_index(drop=True)

print("\nPilot sample:", len(sample))
print(
    "Healthy recordings:",
    int((sample["label"] == 0).sum())
)
print(
    "Disorder recordings:",
    int((sample["label"] == 1).sum())
)

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

# IMPORTANT:
# This pilot uses a recording-level split only to determine
# whether spectrograms contain useful signal.
#
# It is NOT the final patient-independent benchmark.

rng = np.random.RandomState(SEED)

indices = np.arange(len(sample))
rng.shuffle(indices)

split_point = int(
    len(indices) * 0.80
)

train_indices = indices[:split_point]
val_indices = indices[split_point:]

train_sample = sample.iloc[
    train_indices
].reset_index(drop=True)

val_sample = sample.iloc[
    val_indices
].reset_index(drop=True)

print("\nTraining recordings:", len(train_sample))
print("Validation recordings:", len(val_sample))

# ============================================================
# AUDIO → MEL SPECTROGRAM
# ============================================================

def audio_to_mel(path, label):

    # Read WAV
    audio_binary = tf.io.read_file(path)

    audio, sample_rate = tf.audio.decode_wav(
        audio_binary,
        desired_channels=1
    )

    audio = tf.squeeze(
        audio,
        axis=-1
    )

    # The dataset recordings are 16 kHz.
    # We use the first 2 seconds.
    target_samples = 32000

    audio = audio[:target_samples]

    # Pad recordings shorter than 2 seconds.
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

    spectrogram = tf.abs(stft)

    # --------------------------------------------------------
    # MEL FILTER BANK
    # --------------------------------------------------------

    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=128,
        num_spectrogram_bins=513,
        sample_rate=16000,
        lower_edge_hertz=50.0,
        upper_edge_hertz=8000.0
    )

    mel = tf.matmul(
        tf.square(spectrogram),
        mel_matrix
    )

    # Log-mel
    mel = tf.math.log(
        mel + 1e-6
    )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    mean = tf.reduce_mean(mel)

    std = tf.math.reduce_std(mel)

    mel = (
        mel - mean
    ) / (
        std + 1e-6
    )

    # --------------------------------------------------------
    # IMAGE FORMAT
    # --------------------------------------------------------

    mel = tf.expand_dims(
        mel,
        axis=-1
    )

    mel = tf.image.resize(
        mel,
        IMG_SIZE
    )

    label = tf.cast(
        label,
        tf.float32
    )

    return mel, label


# ============================================================
# CREATE TF DATASETS
# ============================================================

train_paths = train_sample[
    "wav_path"
].values

train_labels = train_sample[
    "label"
].values.astype(
    np.float32
)

val_paths = val_sample[
    "wav_path"
].values

val_labels = val_sample[
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
    audio_to_mel,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.shuffle(
    buffer_size=len(train_sample),
    seed=SEED,
    reshuffle_each_iteration=True
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
    audio_to_mel,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.batch(
    BATCH_SIZE
)

val_ds = val_ds.prefetch(
    tf.data.AUTOTUNE
)

# ============================================================
# BUILD CNN
# ============================================================

print("\nBuilding CNN...")

model = keras.Sequential(
    [
        layers.Input(
            shape=(
                IMG_SIZE[0],
                IMG_SIZE[1],
                1
            )
        ),

        # ----------------------------------------------------
        # Block 1
        # ----------------------------------------------------

        layers.Conv2D(
            32,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            pool_size=2
        ),

        # ----------------------------------------------------
        # Block 2
        # ----------------------------------------------------

        layers.Conv2D(
            64,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            pool_size=2
        ),

        # ----------------------------------------------------
        # Block 3
        # ----------------------------------------------------

        layers.Conv2D(
            128,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            pool_size=2
        ),

        # ----------------------------------------------------
        # Block 4
        # ----------------------------------------------------

        layers.Conv2D(
            256,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        layers.GlobalAveragePooling2D(),

        layers.Dropout(
            0.35
        ),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(
            0.30
        ),

        layers.Dense(
            1,
            activation="sigmoid"
        )
    ]
)

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
        )
    ]
)

print("\nModel summary:")
model.summary()

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
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

print("\n" + "=" * 60)
print("STARTING PILOT TRAINING")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("SPECTROGRAM PILOT RESULTS")
print("=" * 60)

results = model.evaluate(
    val_ds,
    return_dict=True,
    verbose=1
)

for name, value in results.items():

    print(
        f"{name}: {value * 100:.2f}%"
    )

print("=" * 60)

print("\nPilot complete.")
print(
    "These results are recording-level and are only"
    " being used to decide whether the spectrogram"
    " representation is worth pursuing."
)