# Basic Deep Learning Fashion-MNIST

From-scratch NumPy neural network for [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) clothing classification, plus Black, Ruff, and MyPy for code quality.

## Dataset

This project uses **Fashion-MNIST**, a drop-in replacement for classic MNIST that keeps the same binary IDX layout and image size while posing a harder 10-class task.

| Property | Value |
|---|---|
| Image size | 28×28 grayscale |
| Training set | 60,000 images |
| Test set | 10,000 images |
| Classes | 10 fashion categories (labels `0`–`9`) |
| On-disk format | Same IDX files as MNIST (`*-images-idx3-ubyte`, `*-labels-idx1-ubyte`) |

Class labels:

| Label | Class |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

The IDX files live under `src/data/` and are loaded by the existing MNIST-compatible loader (`mnistnn`). The package and API names still say `mnist` because the file format is identical; the bytes on disk are Fashion-MNIST.

## Model

From-scratch NumPy MLP (no convolutions):

`784 → 512 → 256 → 128 → 10`

Hidden layers use ReLU with dropout `0.1 / 0.2 / 0.2`. Training uses SGD with momentum `0.9`, L2 weight decay `1e-4`, cosine learning-rate annealing from `0.03` to `1e-4` over 80 epochs, random horizontal flips, and test-time horizontal-flip averaging (TTA).

## Results

Evaluation metric is **top-1 accuracy on the official Fashion-MNIST test set** (10,000 images). Training loss is mean batch cross-entropy over the epoch. Numbers below are from completed default runs logged in `log/mnist-training.log` and can vary slightly by random seed.

| Metric | Typical value |
|---|---|
| Test accuracy (final epoch / saved model) | **~90.5–90.7%** |
| Test accuracy (best epoch during training) | **~90.8%** |
| Train loss (final epoch) | ~0.08 |
| Epochs | 80 |
| Wall time (CPU, approx.) | ~15–20 minutes |

**How to read these numbers:** report the **final-epoch** test accuracy as the result for the trained model returned by `train_mnist()`. Peak-over-epochs is slightly higher but is not checkpointed unless you add that yourself.

For context, a plain MLP on Fashion-MNIST often lands near the high 80s to low 90s. Small CNNs usually do better; this project intentionally stays fully connected / NumPy-only.

## Tooling

This boilerplate uses the following tools to maintain code quality:

### Black

[Black](https://github.com/psf/black) is the uncompromising Python code formatter. It enforces a strict and consistent code style, which helps to eliminate debates over formatting and makes code review faster. By using Black, you ensure that the code looks the same regardless of who wrote it.

### Ruff

[Ruff](https://github.com/astral-sh/ruff) is an extremely fast Python linter, written in Rust. It can check for a wide range of errors and style issues, from simple syntax mistakes to complex bugs. It also includes a code formatter. In this boilerplate, it is used for linting and fixing common issues automatically.

### MyPy

[MyPy](http://mypy-lang.org/) is a static type checker for Python. It helps you write cleaner, more robust code by adding type hints to your functions and variables. MyPy can catch type-related errors before you even run your code, which can save you a lot of time debugging.

## Automation

This project uses `make` and `invoke` to automate common development tasks.

### Makefile

A `Makefile` is included to provide a simple and familiar interface for common commands. You can see the available commands by running `make help`.

### Invoke & tasks.py

[Invoke](https://www.pyinvoke.org/) is a Python task execution tool. The `tasks.py` file defines a set of tasks that can be run from the command line. This allows for more complex logic than a `Makefile` and is written in Python. You can see the available tasks by running `invoke --list`.

## Train the neural network

Install dependencies first (`invoke install` or `make install`). Run every command from the project root. The package lives under `src/`, so Python needs `PYTHONPATH=src` (the Invoke and Make targets set this for you).

All of the commands below run `python -m mnistnn`, which calls `train_mnist()` with the defaults described under **Model** (80 epochs, batch size 64, `visualize=True`, pixels scaled to `[-0.5, 0.5]`). Training progress is written to `log/mnist-training.log`.

### Invoke

Works the same in bash, PowerShell, and cmd:

```text
invoke train
```

### Make

Use this in bash, Git Bash, or WSL. The Makefile uses Unix `PYTHONPATH=src python -m mnistnn`, so it is not for stock Windows cmd or PowerShell unless `make` is installed.

```bash
make train
```

### Python interpreter

Set `PYTHONPATH` in the current shell, then run the module.

bash, Git Bash, or WSL:

```bash
export PYTHONPATH=src
python -m mnistnn
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m mnistnn
```

cmd:

```bat
set PYTHONPATH=src
python -m mnistnn
```

To train without plot windows, set `PYTHONPATH` as above, then:

```text
python -c "from mnistnn.train_mnist import train_mnist; train_mnist(visualize=False)"
```

## Available Commands

You can use either `make` or `invoke` to run the following commands:

| Command | Makefile | Invoke | Description |
|---|---|---|---|
| **Train Network** | `make train` | `invoke train` | Trains the Fashion-MNIST network (`python -m mnistnn`). |
| **Format Code** | `make format` | `invoke format` | Formats the code using Black. |
| **Check Formatting** | `make format-check` | `invoke format-check` | Checks if the code is formatted correctly without making changes. |
| **Lint Code** | `make lint` | `invoke lint` | Lints the code with Ruff to find potential issues. |
| **Lint and Fix** | `make lint-fix` | `invoke lint-fix` | Lints the code with Ruff and automatically fixes any issues it can. |
| **Type Check** | `make typecheck` | `invoke typecheck` | Runs MyPy to check for type errors. |
| **Quality Assurance** | `make qa` | `invoke qa` | Runs all the quality checks: format, lint-fix, and typecheck. |
| **Install Dependencies**| `make install` | `invoke install` | Installs the required Python packages from `requirements.txt`. |
| **Clean Project** | `make clean` | `invoke clean` | Removes cache directories and temporary files. |
| **Help** | `make help` | `invoke help` | Shows the list of available commands. |
