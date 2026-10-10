"""
CNN training pipeline for HAM10000 skin lesion classification.

Reuses preprocessing, group-aware split, generators, and EDA from EDA.py.
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2

from EDA import (
    RANDOM_STATE,
    IMG_SIZE,
    BATCH_SIZE,
    N_CLASSES,
    DX_FULL_NAME,
    load_metadata,
    preprocess_metadata,
    group_aware_split,
    build_image_generators,
    run_eda,
)

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

MODELS_DIR = "models"
OUTPUT_DIR = "training_outputs"
HIGH_RISK_CLASSES = {"mel", "bcc", "akiec"}

# Clinical characteristics for each dx code
DX_CHARACTERISTICS = {
    "akiec": "precancerous",
    "bcc": "malignant",
    "bkl": "benign",
    "df": "benign",
    "mel": "malignant",
    "nv": "benign",
    "vasc": "benign",
}


def compute_class_weights(train_df, class_indices: dict) -> dict:
    """Inverse-frequency class weights keyed by generator class index."""
    y = train_df["dx"].map(class_indices).values
    classes = np.arange(N_CLASSES)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weight = {int(i): float(w) for i, w in enumerate(weights)}
    print(f"[ClassWeight] {class_weight}")
    return class_weight


def build_model(num_classes: int = N_CLASSES, trainable_base: bool = False):
    """MobileNetV2 transfer-learning model with GAP + Dropout + Softmax head."""
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = trainable_base

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="skin_lesion_mobilenetv2")
    return model, base


def get_callbacks(checkpoint_path: str):
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


def train_phase1(model, train_gen, val_gen, class_weight, epochs=10):
    """Train with frozen base (feature extraction)."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    checkpoint = os.path.join(MODELS_DIR, "best_phase1.keras")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=get_callbacks(checkpoint),
    )
    return history


def train_phase2(model, base, train_gen, val_gen, class_weight, epochs=10, unfreeze_last_n=40):
    """Fine-tune by unfreezing the upper layers of the base model."""
    base.trainable = True
    for layer in base.layers[:-unfreeze_last_n]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    checkpoint = os.path.join(MODELS_DIR, "best_phase2.keras")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=get_callbacks(checkpoint),
    )
    return history


def evaluate_model(model, test_gen, class_indices: dict, out_dir: str = OUTPUT_DIR):
    """Full evaluation: metrics, confusion matrix, ROC AUC, threshold tuning."""
    os.makedirs(out_dir, exist_ok=True)

    # Index -> class name (respect generator class order)
    idx_to_class = {v: k for k, v in class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    y_prob = model.predict(test_gen, verbose=1)
    y_true = test_gen.classes
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(N_CLASSES)), zero_division=0
    )

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # ROC AUC (OvR)
    try:
        y_true_oh = tf.keras.utils.to_categorical(y_true, num_classes=N_CLASSES)
        auc_macro = roc_auc_score(y_true_oh, y_prob, average="macro", multi_class="ovr")
        auc_per_class = roc_auc_score(y_true_oh, y_prob, average=None, multi_class="ovr")
        print(f"ROC AUC (macro OvR): {auc_macro:.4f}")
        for name, auc_c in zip(class_names, auc_per_class):
            print(f"  AUC {name}: {auc_c:.4f}")
    except ValueError as exc:
        auc_macro = None
        auc_per_class = None
        print(f"[Eval] ROC AUC skipped: {exc}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

    # Decision threshold tuning for high-risk classes (raise recall)
    thresholds = tune_high_risk_thresholds(
        y_true, y_prob, class_indices, class_names, out_dir
    )

    metrics = {
        "accuracy": float(acc),
        "precision_per_class": {n: float(p) for n, p in zip(class_names, precision)},
        "recall_per_class": {n: float(r) for n, r in zip(class_names, recall)},
        "f1_per_class": {n: float(f) for n, f in zip(class_names, f1)},
        "support_per_class": {n: int(s) for n, s in zip(class_names, support)},
        "auc_macro": float(auc_macro) if auc_macro is not None else None,
        "auc_per_class": (
            {n: float(a) for n, a in zip(class_names, auc_per_class)}
            if auc_per_class is not None else None
        ),
        "high_risk_thresholds": thresholds,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[Eval] Artifacts saved to {out_dir}/")
    return metrics


def tune_high_risk_thresholds(y_true, y_prob, class_indices, class_names, out_dir):
    """
    Lower decision thresholds for mel / bcc / akiec to improve sensitivity.
    Searches thresholds in [0.1, 0.5] and picks the one maximizing recall
    while keeping precision >= 0.20 (accept higher FPR for high-risk classes).
    """
    thresholds = {}
    for dx in HIGH_RISK_CLASSES:
        if dx not in class_indices:
            continue
        idx = class_indices[dx]
        best_t, best_recall, best_prec = 0.5, 0.0, 0.0

        for t in np.arange(0.10, 0.51, 0.05):
            y_hat = (y_prob[:, idx] >= t).astype(int)
            y_bin = (np.array(y_true) == idx).astype(int)
            tp = int(((y_hat == 1) & (y_bin == 1)).sum())
            fp = int(((y_hat == 1) & (y_bin == 0)).sum())
            fn = int(((y_hat == 0) & (y_bin == 1)).sum())
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            if recall > best_recall and prec >= 0.20:
                best_t, best_recall, best_prec = float(t), float(recall), float(prec)

        thresholds[dx] = {
            "threshold": best_t,
            "recall": best_recall,
            "precision": best_prec,
        }
        print(
            f"[Threshold] {dx}: t={best_t:.2f} "
            f"recall={best_recall:.3f} precision={best_prec:.3f}"
        )

    with open(os.path.join(out_dir, "high_risk_thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    return thresholds


def save_artifacts(model, label_encoder, class_indices, scaler=None, thresholds=None):
    """Persist model, label encoder, class metadata, scaler, and thresholds."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "model.keras")
    model.save(model_path)
    print(f"[Save] Model -> {model_path}")

    le_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    with open(le_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"[Save] Label encoder -> {le_path}")

    # Scaler placeholder: image rescale is fixed (1/255); saved for pipeline completeness
    if scaler is None:
        scaler = {"rescale": 1.0 / 255, "img_size": list(IMG_SIZE)}
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[Save] Scaler -> {scaler_path}")

    meta = {
        "class_indices": class_indices,
        "dx_full_name": DX_FULL_NAME,
        "dx_characteristics": DX_CHARACTERISTICS,
        "high_risk_classes": sorted(HIGH_RISK_CLASSES),
        "img_size": list(IMG_SIZE),
    }
    meta_path = os.path.join(MODELS_DIR, "class_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[Save] Class metadata -> {meta_path}")

    # Keep tuned thresholds next to the model so the API can load them without
    # depending on training_outputs/ being present at serve time.
    if thresholds:
        thr_path = os.path.join(MODELS_DIR, "high_risk_thresholds.json")
        with open(thr_path, "w", encoding="utf-8") as f:
            json.dump(thresholds, f, indent=2)
        print(f"[Save] High-risk thresholds -> {thr_path}")


def plot_history(histories, out_dir: str = OUTPUT_DIR):
    """Plot combined phase-1 + phase-2 training curves."""
    os.makedirs(out_dir, exist_ok=True)
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc.extend(h.history.get("accuracy", []))
        val_acc.extend(h.history.get("val_accuracy", []))
        loss.extend(h.history.get("loss", []))
        val_loss.extend(h.history.get("val_loss", []))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(acc, label="train")
    axes[0].plot(val_acc, label="val")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].plot(loss, label="train")
    axes[1].plot(val_loss, label="val")
    axes[1].set_title("Loss")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"))
    plt.close()


def main(
    run_eda_flag: bool = True,
    phase1_epochs: int = 10,
    phase2_epochs: int = 10,
):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Steps 1–2: load + preprocess (from EDA.py) ---
    df_raw = load_metadata()
    df, label_encoder = preprocess_metadata(df_raw)

    if run_eda_flag:
        run_eda(df)

    # --- Step 3: group-aware split ---
    train_df, val_df, test_df = group_aware_split(df)

    # --- Steps 4–5: resize / normalize / augment via generators ---
    train_gen, val_gen, test_gen = build_image_generators(train_df, val_df, test_df)

    # Align LabelEncoder order with generator class_indices if they differ
    # Generators sort class names alphabetically by default.
    class_indices = train_gen.class_indices
    print(f"[Generator] class_indices: {class_indices}")

    # --- Step 6: class weights ---
    class_weight = compute_class_weights(train_df, class_indices)

    # --- Step 7: model construction & training ---
    model, base = build_model(num_classes=N_CLASSES, trainable_base=False)
    model.summary()

    print("\n=== Phase 1: Frozen base ===")
    hist1 = train_phase1(model, train_gen, val_gen, class_weight, epochs=phase1_epochs)

    print("\n=== Phase 2: Fine-tuning ===")
    hist2 = train_phase2(model, base, train_gen, val_gen, class_weight, epochs=phase2_epochs)

    plot_history([hist1, hist2])

    # Load best fine-tuned weights if available
    best_path = os.path.join(MODELS_DIR, "best_phase2.keras")
    if os.path.exists(best_path):
        model = tf.keras.models.load_model(best_path)
        print(f"[Train] Loaded best checkpoint: {best_path}")

    # --- Step 8: evaluation + threshold tuning ---
    metrics = evaluate_model(model, test_gen, class_indices)
    thresholds = metrics.get("high_risk_thresholds") if metrics else None

    # --- Step 9: save artifacts ---
    # Re-fit label encoder to match generator alphabetical order for consistent serving
    from sklearn.preprocessing import LabelEncoder as LE
    serving_le = LE()
    serving_le.fit(sorted(class_indices.keys()))
    save_artifacts(model, serving_le, class_indices, thresholds=thresholds)

    print("\n[Done] Training pipeline completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train HAM10000 skin lesion CNN")
    parser.add_argument("--no-eda", action="store_true", help="Skip EDA plots")
    parser.add_argument("--phase1-epochs", type=int, default=10)
    parser.add_argument("--phase2-epochs", type=int, default=10)
    args = parser.parse_args()

    main(
        run_eda_flag=not args.no_eda,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
    )
