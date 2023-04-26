import numpy as np
import gzip


def read_label(file_path: str) -> np.ndarray:
    """读取MINST标签数据

    Args:
        file_path (str): MINST标签数据路径

    Returns:
        np.ndarray: 包含了MINST标签的独热码矩阵
    """
    with gzip.open(file_path) as f:
        raw_data = np.frombuffer(f.read(), "B", offset=8).astype(np.uint8)
    one_hot_matrix = np.zeros((raw_data.shape[0], 10))
    one_hot_matrix[np.arange(raw_data.shape[0]), raw_data] = 1
    return one_hot_matrix


def read_image(file_path: str) -> np.ndarray:
    """读取MINST图像数据

    Args:
        file_path (str): MINST图像数据路径

    Returns:
        np.ndarray: 包含了MINST图像数据的矩阵，每张图片的数据被拉伸至一维且像素取值属于[0, 1] #TODO: 尝试归一化到[-1, 1]
    """
    with gzip.open(file_path) as f:
        raw_data = np.frombuffer(f.read(), "B", offset=16).astype(np.float32)
    return raw_data.reshape(-1, 784) / 255


def calculate_accuracy(pred: np.ndarray, label: np.ndarray):
    """计算预测准确率

    Args:
        pred (np.ndarray): 模型预测结果
        label (np.ndarray): MINST标签

    Returns:
        _type_: 平均准确率
    """
    pred = np.argmax(pred, axis=-1)
    label = np.argmax(label, axis=-1)
    acc = np.where(pred == label, 1, 0).mean()
    return acc
