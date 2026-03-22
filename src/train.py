import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from model.model import CNNModel
from data_loader import load_data_from_config


def main():
    model_wrapper = CNNModel()
    data = load_data_from_config("./data.yaml")
    if data["X_train"].size > 0:
        print("Starting training...")
        model_wrapper.train(data=data, epochs=25, batch_size=16)
        model_wrapper.save()
        model_wrapper.evaluate(data["X_val"], data["y_val"])


if __name__ == "__main__":
    main()
