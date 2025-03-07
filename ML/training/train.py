import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from models.ensemble import build_ensemble
from training.data_loader import load_dataset


def train_model(dataset_path, input_shape=(256, 256, 3), batch_size=32, epochs=100):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(base_dir, "best_model.keras")
    final_model_path = os.path.join(base_dir, "final_model.keras")

    train_ds, val_ds, class_names = load_dataset(dataset_path, input_shape, batch_size)

    num_classes = len(class_names)

    model = build_ensemble(num_classes=num_classes)

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint = ModelCheckpoint(
        checkpoint_path, save_best_only=True, monitor="val_accuracy", mode="max"
    )
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

    for layer in model.layers:
        if any(m in layer.name for m in ["vgg16", "densenet", "resnet"]):
            layer.trainable = False

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping],
    )

    model.save(final_model_path)
    return history, class_names


def main():
    config = {
        "dataset_path": "../cotton_leaves",
        "input_shape": (256, 256),
        "batch_size": 32,
        "epochs": 15,
    }

    history, class_names = train_model(
        dataset_path=config["dataset_path"],
        input_shape=config["input_shape"],
        batch_size=config["batch_size"],
        epochs=config["epochs"],
    )

    print("\nTraining completed!")
    print(f"Model saved as: final_model.h5")
    print(f"Class names: {class_names}")


if __name__ == "__main__":
    main()
