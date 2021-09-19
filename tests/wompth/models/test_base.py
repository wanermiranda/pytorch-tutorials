import os
import shutil

import pytest

from wompth.models.base import NeuralNetwork


@pytest.fixture
def model_path():
    path = os.path.join(os.getcwd(), "model_test.pth")
    yield path
    os.remove(path)


def test_model_build():
    nn = NeuralNetwork(layout=[28 * 28, 128, 128, 5])
    assert (
        nn.__str__()
        == "NeuralNetwork(\n  (_flatten): Flatten(start_dim=1, end_dim=-1)\n  (_loss_fn): CrossEntropyLoss()\n  (_network_stack): Sequential(\n    (0): Linear(in_features=784, out_features=128, bias=True)\n    (1): ReLU()\n    (2): Linear(in_features=128, out_features=128, bias=True)\n    (3): ReLU()\n    (4): Linear(in_features=128, out_features=5, bias=True)\n  )\n)"
    )


def test_training(training_loader_fMNIST, capsys):
    nn = NeuralNetwork(layout=[28 * 28, 256, 256, 10])
    nn.fit(train_loader=training_loader_fMNIST, epochs=2)
    out, _ = capsys.readouterr()
    assert out.find("Done!") > 0 and out.count("Epoch") == 2 and out.find("loss: ") > 0


def test_training_testing(training_loader_fMNIST, testing_loader_fMNIST, capsys):
    nn = NeuralNetwork(layout=[28 * 28, 256, 256, 10])
    nn.fit(
        train_loader=training_loader_fMNIST, test_loader=testing_loader_fMNIST, epochs=2
    )
    out, _ = capsys.readouterr()
    assert (
        out.find("Done!") > 0
        and out.count("Test Error") == 2
        and out.count("Accuracy") == 2
        and out.count("Avg loss") == 2
    )


def test_predict(training_loader_fMNIST, testing_loader_fMNIST):
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    nn = NeuralNetwork(layout=[28 * 28, 256, 256, 10], class_labels=classes)

    nn.fit(
        train_loader=training_loader_fMNIST,
        test_loader=testing_loader_fMNIST,
        epochs=5,
    )

    result = nn.predict(testing_loader_fMNIST)

    assert len(result) == len(testing_loader_fMNIST.dataset)

    correct_guesses = count_guesses(testing_loader_fMNIST, classes, result)

    print(f"Correct guesses {correct_guesses} out of 10")
    assert correct_guesses >= 5


def test_load_save(training_loader_fMNIST, testing_loader_fMNIST, model_path):
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    nn = NeuralNetwork(layout=[28 * 28, 256, 256, 10], class_labels=classes)

    nn.fit(
        train_loader=training_loader_fMNIST,
        test_loader=testing_loader_fMNIST,
        epochs=5,
    )
    nn.save(model_path)

    result = nn.predict(testing_loader_fMNIST)
    assert len(result) == len(testing_loader_fMNIST.dataset)
    correct_guesses = count_guesses(testing_loader_fMNIST, classes, result)
    assert correct_guesses >= 5

    nn_loaded = NeuralNetwork()
    nn_loaded.load(model_path)

    result_loaded = nn_loaded.predict(testing_loader_fMNIST)
    assert len(result_loaded) == len(testing_loader_fMNIST.dataset)
    correct_guesses = count_guesses(testing_loader_fMNIST, classes, result_loaded)
    assert correct_guesses >= 5

    assert result == result_loaded
    assert nn._layout == nn_loaded._layout
    assert nn._device == nn_loaded._device
    assert nn._class_labels == nn_loaded._class_labels


def count_guesses(testing_loader_fMNIST, classes, result):
    correct_guesses = 0
    for i in range(10):
        y = testing_loader_fMNIST.dataset[i][1]
        if result[i] == classes[y]:
            correct_guesses += 1
        print(f"guess={result[i]}, true={classes[y]}")

    return correct_guesses
