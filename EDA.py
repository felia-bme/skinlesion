import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
except ImportError:  # Allow EDA / preprocessing without TensorFlow installed
    tf = None
    ImageDataGenerator = None
    TF_AVAILABLE = False

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
if TF_AVAILABLE:
    tf.random.set_seed(RANDOM_STATE)

DATA_DIR = "data"
METADATA_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
IMAGE_DIRS = [
    os.path.join(DATA_DIR, "HAM10000_images_part_1"),
    os.path.join(DATA_DIR, "HAM10000_images_part_2"),
]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
N_CLASSES = 7

# Nama panjang tiap kode kelas dx, untuk pelaporan dan visualisasi
DX_FULL_NAME = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc":   "Basal cell carcinoma",
    "bkl":   "Benign keratosis-like lesions",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic nevi",
    "vasc":  "Vascular lesions",
}

# 1. DATA UNDERSTANDING

def load_metadata(path: str = METADATA_PATH) -> pd.DataFrame:
    """Memuat metadata HAM10000 dan menampilkan ringkasan struktur data."""
    df = pd.read_csv(path)

    print("=" * 70)
    print("DATA UNDERSTANDING")
    print("=" * 70)
    print(f"Jumlah baris (citra)   : {df.shape[0]}")
    print(f"Jumlah kolom           : {df.shape[1]}")
    print(f"Kolom                  : {list(df.columns)}")
    print(f"Jumlah lesion_id unik  : {df['lesion_id'].nunique()}")
    print(f"Jumlah kelas dx        : {df['dx'].nunique()} -> {sorted(df['dx'].unique())}")
    print("-" * 70)
    print("Missing values per kolom:")
    print(df.isna().sum())
    print("-" * 70)
    print(f"Baris duplikat (lesion_id > 1 citra): "
          f"{(df['lesion_id'].value_counts() > 1).sum()} lesi memiliki >1 citra")
    print("=" * 70)
    return df


def resolve_image_path(image_id: str) -> str:
    """Mencari path file citra di antara folder part_1 / part_2."""
    for d in IMAGE_DIRS:
        candidate = os.path.join(d, f"{image_id}.jpg")
        if os.path.exists(candidate):
            return candidate
    return ""



# 2. DATA PREPROCESSING

def preprocess_metadata(df: pd.DataFrame):
    """
    Langkah preprocessing pada metadata:
      - Menangani missing values pada kolom 'age' (dipertahankan sebagai NaN,
        tidak diimputasi/dihapus.
      - Encoding label target 'dx' menjadi indeks numerik.
      - Menambahkan path citra untuk setiap baris & membuang baris yang citranya
        tidak ditemukan.

    Returns
    -------
    df : pd.DataFrame
    label_encoder : LabelEncoder
    """
    df = df.copy()

    n_missing_age = df["age"].isna().sum()
    print(f"[Preprocessing] {n_missing_age} baris memiliki age kosong -> "
          f"dipertahankan untuk training CNN (age tidak dipakai model), "
          f"dan dikecualikan otomatis dari analisis statistik usia di EDA.")

    # (b) Encoding label dx -> indeks numerik 0..6
    label_encoder = LabelEncoder()
    df["dx_label"] = label_encoder.fit_transform(df["dx"])
    print(f"[Preprocessing] Mapping label dx -> indeks: "
          f"{dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

    # (c) Encoding fitur kategorikal tambahan (opsional, untuk analisis / fitur tambahan)
    # Retain explicit "unknown" values for sex / localization (do not fabricate).
    df["sex"] = df["sex"].fillna("unknown")
    df["localization"] = df["localization"].fillna("unknown")
    df["sex_encoded"] = LabelEncoder().fit_transform(df["sex"])
    df["localization_encoded"] = LabelEncoder().fit_transform(df["localization"])

    # (d) Melengkapi path file citra & membuang baris tanpa citra valid
    df["image_path"] = df["image_id"].apply(resolve_image_path)
    n_before = len(df)
    df = df[df["image_path"] != ""].reset_index(drop=True)
    print(f"[Preprocessing] Membuang {n_before - len(df)} baris tanpa file citra valid.")

    return df, label_encoder


def _lesion_level_label(df: pd.DataFrame) -> pd.Series:
    """One dx label per lesion_id (first image's dx; lesions are single-diagnosis)."""
    return df.groupby("lesion_id")["dx"].first()


def _stratified_group_indices(df: pd.DataFrame, test_size: float, random_state: int):
    """
    Stratified split by lesion group: assign each lesion to train or hold-out while
    preserving class proportions of lesion-level dx labels.
    Falls back to GroupShuffleSplit if StratifiedGroupKFold is unavailable.
    """
    lesion_labels = _lesion_level_label(df)
    y = lesion_labels.to_numpy()

    try:
        from sklearn.model_selection import StratifiedGroupKFold

        # Prefer the fold whose hold-out size is closest to the requested test_size
        n_splits = max(2, int(round(1.0 / test_size)))
        n_splits = min(n_splits, len(lesion_labels))
        # Ensure every class can appear in folds when possible
        min_class = int(pd.Series(y).value_counts().min())
        n_splits = min(n_splits, min_class) if min_class >= 2 else 2

        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        groups = df["lesion_id"].to_numpy()
        y_row = df["dx"].to_numpy()
        best = None
        for train_idx, test_idx in sgkf.split(df, y_row, groups):
            holdout_frac = len(test_idx) / len(df)
            score = abs(holdout_frac - test_size)
            if best is None or score < best[0]:
                best = (score, train_idx, test_idx)
        return best[1], best[2]
    except Exception:
        gss = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        return next(gss.split(df, groups=df["lesion_id"]))


def group_aware_split(df: pd.DataFrame, test_size=0.15, val_size=0.15):
    """
    Split train/val/test grouped by lesion_id so all images of the same lesion
    stay in one subset (prevents leakage), with stratified sampling on dx.
    """
    train_val_idx, test_idx = _stratified_group_indices(df, test_size, RANDOM_STATE)
    train_val_df, test_df = df.iloc[train_val_idx], df.iloc[test_idx]

    relative_val = val_size / (1.0 - test_size)
    train_idx, val_idx = _stratified_group_indices(
        train_val_df, relative_val, RANDOM_STATE + 1
    )
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]

    print(f"[Split] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    for name, subset in (("train", train_df), ("val", val_df), ("test", test_df)):
        dist = subset["dx"].value_counts(normalize=True).sort_index()
        print(f"[Split] {name} dx proportions:\n{dist.to_string()}")
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_image_generators(train_df, val_df, test_df):
    """
    Membuat generator citra dengan normalisasi (rescale 1/255) dan augmentasi data
    pada set training untuk mengurangi dampak ketidakseimbangan kelas minoritas.
    """
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required for image generators. "
            "Install with: pip install tensorflow"
        )

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    common_args = dict(
        x_col="image_path", y_col="dx", target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical",
    )
    train_gen = train_datagen.flow_from_dataframe(train_df, **common_args, shuffle=True)
    val_gen = eval_datagen.flow_from_dataframe(val_df, **common_args, shuffle=False)
    test_gen = eval_datagen.flow_from_dataframe(test_df, **common_args, shuffle=False)
    return train_gen, val_gen, test_gen



# 3. EXPLORATORY DATA ANALYSIS (EDA)

def run_eda(df: pd.DataFrame, out_dir: str = "eda_outputs"):
    """Menghasilkan visualisasi EDA utama dan menyimpannya sebagai file gambar."""
    os.makedirs(out_dir, exist_ok=True)

    # (a) Distribusi kelas dx
    plt.figure(figsize=(8, 5))
    order = df["dx"].value_counts().index
    sns.countplot(data=df, x="dx", order=order, palette="viridis")
    plt.title("Distribusi Kelas Diagnosis (dx)")
    plt.xlabel("Kelas")
    plt.ylabel("Jumlah Citra")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_distribusi_kelas.png"))
    plt.close()

    # (b) Distribusi usia
    plt.figure(figsize=(8, 5))
    sns.histplot(df["age"].dropna(), bins=30, kde=True, color="teal")
    plt.title("Distribusi Usia Pasien")
    plt.xlabel("Usia")
    plt.ylabel("Frekuensi")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_distribusi_usia.png"))
    plt.close()

    # (c) Usia vs kelas dx (boxplot) -> melihat kelas mana yang cenderung di usia lebih tua
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="dx", y="age", order=order, palette="crest")
    plt.title("Sebaran Usia per Kelas Diagnosis")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "03_usia_per_kelas.png"))
    plt.close()

    # (d) Distribusi lokasi anatomi lesi
    plt.figure(figsize=(9, 5))
    loc_order = df["localization"].value_counts().index
    sns.countplot(data=df, y="localization", order=loc_order, palette="flare")
    plt.title("Distribusi Lokasi Anatomi Lesi")
    plt.xlabel("Jumlah Citra")
    plt.ylabel("Lokasi")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "04_lokasi_anatomi.png"))
    plt.close()

    # (e) Heatmap korelasi antar fitur numerik/encoded sederhana.
    #     dropna() di sini mengecualikan baris ber-age kosong HANYA dari
    #     perhitungan korelasi ini, bukan dari dataset/training CNN secara
    #     keseluruhan (age memang tidak diimputasi, lihat preprocess_metadata).
    plt.figure(figsize=(6, 5))
    corr_cols = ["age", "sex_encoded", "localization_encoded", "dx_label"]
    corr = df[corr_cols].dropna().corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Korelasi Antar Fitur (Encoded)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "05_heatmap_korelasi.png"))
    plt.close()

    print(f"[EDA] Visualisasi disimpan pada folder: {out_dir}/")


if __name__ == "__main__":
    raw = load_metadata()
    processed, _ = preprocess_metadata(raw)
    run_eda(processed)
