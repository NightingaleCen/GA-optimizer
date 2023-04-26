import numpy as np


class MLP:
    def __init__(self, chromosome: np.ndarray, layer_dims: list[int]) -> None:
        self.weights = []
        self.biases = []
        self.layer_dims = layer_dims
        self.__decode(chromosome)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """多层感知机的前向传播

        Args:
            x (np.ndarray): 多层感知机的输入

        Returns:
            np.ndarray: 最后一个线性层输出的logits
        """
        layer_num = len(self.layer_dims) - 1
        for index in range(layer_num):
            dim_in, dim_out = self.layer_dims[index], self.layer_dims[index + 1]
            weight = self.weights[index].reshape(dim_in, dim_out)
            bias = self.biases[index]
            x = x.dot(weight) + bias
            if index != layer_num - 1:
                x = relu(x)

        return x

    def __decode(self, chromosome: np.ndarray) -> None:
        """将个体染色体解码为各线性层的权重与偏置

        Args:
            chromosome (np.ndarray): 个体的染色体
        """
        layer_num = len(self.layer_dims) - 1
        for layer_index in range(layer_num):
            last_layer_dims = self.layer_dims[layer_index]
            current_layer_dims = self.layer_dims[layer_index + 1]
            self.weights.append(chromosome[: last_layer_dims * current_layer_dims])
            np.delete(chromosome, slice(0, last_layer_dims * current_layer_dims))
            self.biases.append(chromosome[:current_layer_dims])
            np.delete(chromosome, slice(0, current_layer_dims))


def relu(x: np.ndarray):
    out = np.fmax(x, np.zeros_like(x))
    return out


def cross_entropy_loss(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    """
    包含SoftMax函数的交叉熵损失

    Input:
    pred: 形状为(N, D)的Numpy数组，网络最终输出的pred
    label: 形状为(N, D)的Numpy数组，数据的标签，以独热码的形式表示

    Output:
    loss; 形状为(N,)的Numpy数组，预测值经过SoftMax后与标签之间的交叉熵损失

    """
    epsilon = 1e-12

    pred = pred - np.amax(pred, axis=1, keepdims=True)
    pred = np.exp(pred) / np.sum(np.exp(pred), axis=1, keepdims=True)  # softmax
    loss = (
        -np.sum(label * np.log(pred + epsilon)) / label.shape[0]
    )  # cross entropy loss

    return loss
