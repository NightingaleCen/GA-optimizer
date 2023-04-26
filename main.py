from utils import *
from GA import GA

train_data = read_image("MINST/train-images-idx3-ubyte.gz")
train_label = read_label("MINST/train-labels-idx1-ubyte.gz")
test_data = read_image("MINST/t10k-images-idx3-ubyte.gz")
test_label = read_label("MINST/t10k-labels-idx1-ubyte.gz")

# architecture = [28 * 28, 512, 256, 128, 96, 48, 10]
architecture = [28 * 28, 100, 10]
ga = GA(
    train_data=train_data,
    train_label=train_label,
    test_data=test_data,
    test_label=test_label,
    M=400,
    Pc=0.8,
    Pm=0.5,
    epoch=16,
    batch_size=1000,
    mutation_rate=1e-1,
    mutation_ratio=0.05,
    MLP_architecture=architecture,
)
ga.run()
ga.fitness_plot()
