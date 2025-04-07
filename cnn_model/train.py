import argparse
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from srcVN.config import CLASSES
from srcVN.dataset import MyDataset
from srcVN.model import QuickDraw
from srcVN.utils import get_evaluation


def get_args():
    parser = argparse.ArgumentParser("Implementation of the Quick Draw model proposed by Google")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--total_images_per_class", type=int, default=10000)
    parser.add_argument("--ratio", type=float, default=0.8, help="train/test split ratio")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--es_patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args

def train(opt):
    #check GPU
    device = "/GPU:0" if tf.config.experimental.list_physical_devices('GPU') else "/CPU:0"
    print("---++#####")
    print(f"Training on: {device}")
    print("#####++---")

    # Data
    training_set = MyDataset(opt.data_path, opt.total_images_per_class, opt.ratio, "train").to_tf_dataset(batch_size=opt.batch_size)
    test_set = MyDataset(opt.data_path, opt.total_images_per_class, opt.ratio, "test").to_tf_dataset(batch_size=opt.batch_size, shuffle=False)

    print(f"Training set: {training_set.cardinality().numpy()} batches")
    print(f"Test set: {test_set.cardinality().numpy} batches")

    model = QuickDraw(num_classes=len(CLASSES))
    model.build(input_shape=(None, 28, 28, 1))
    model.summary()

    if opt.optimizer == 'adam':
        optimizer = Adam(learning_rate=opt.lr)
    elif opt.optimizer == 'sgd':
        optimizer = SGD(learning_rate=opt.lr, momentum=0.9)
    else:
        raise ValueError("Invalid optimizer")

    # Loss
    loss_fn = SparseCategoricalCrossentropy(from_logits=True)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.mkdir(opt.log_path)

    callbacks = [
        TensorBoard(log_dir=opt.log_path),
        ModelCheckpoint(filepath=os.path.join(opt.saved_path, "best_model.h5"), save_best_only=True),
        EarlyStopping(monitor="val_loss", patience=opt.es_patience, restore_best_weights=True)
    ]

    # Compile
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    model.fit(
        training_set,
        validation_data=test_set,
        epochs=opt.num_epochs,
        callbacks=callbacks
    )

    # Save
    model.save(os.path.join(opt.saved_path, "final_model.h5"))
    print("Training completed and model saved!")

if __name__ == "__main__":
    opt = get_args()
    train(opt)
