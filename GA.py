from classifier import MLP, cross_entropy_loss
from utils import calculate_accuracy
import asyncio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class GA:
    def __init__(
        self,
        train_data: np.ndarray,
        train_label: np.ndarray,
        test_data: np.ndarray,
        test_label: np.ndarray,
        M: int,
        Pc: float,
        Pm: float,
        epoch: int,
        batch_size: int,
        mutation_rate: float,
        mutation_ratio: float,
        MLP_architecture: list,
        seed: int = 42,
    ) -> None:
        self.M = M
        self.Pc = Pc
        self.Pm = Pm
        self.epoch = epoch
        self.batch_size = batch_size
        self.mutation_rate = mutation_rate
        self.mutation_ratio = mutation_ratio
        self.MLP_architecture = MLP_architecture

        np.random.seed(seed)  # 指定一个随机数种子

        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label

        self.chromosome_length = sum(
            [
                MLP_architecture[i - 1] * MLP_architecture[i] + MLP_architecture[i]
                for i in range(1, len(MLP_architecture))
            ]
        )  # 计算个体染色体长度

        self.population = (
            np.random.randn(M, self.chromosome_length) * 0.1
        )  # 从均值为0方差为0.1的高斯分布中随机初始化种群

        self.__best_individual = None
        self.__best_fitness = 0
        self.__loss_history = []
        self.__fitness_history = []

        asyncio.run(
            self.__eval_accuracy(
                self.train_data[:batch_size], self.train_label[:batch_size]
            )
        )

    def run(self):
        train_size = self.train_data.shape[0]
        steps = train_size // self.batch_size

        process_bar = tqdm(total=self.epoch * steps, desc="GA process")

        for epoch in range(self.epoch):
            shuffle_mask = np.random.permutation(self.train_data.shape[0])
            self.train_data = self.train_data[shuffle_mask]
            self.train_label = self.train_label[shuffle_mask]

            for i in range(steps):
                self.__select()
                self.__crossover_SBX()
                self.__mutate()

                x = self.train_data[
                    i * self.batch_size : min((i + 1) * self.batch_size, train_size)
                ]
                label = self.train_label[
                    i * self.batch_size : min((i + 1) * self.batch_size, train_size)
                ]

                asyncio.run(self.__eval_accuracy(x, label))
                process_bar.update(1)

        best_model = MLP(self.__best_individual, layer_dims=self.MLP_architecture)
        test_pred = best_model(self.test_data)

        self.test_accuracy = calculate_accuracy(test_pred, self.test_label)

    def get_best_model(self):
        assert self.__best_individual != None
        return self.__best_individual

    def fitness_plot(self):
        assert len(self.__fitness_history) != 0
        plt.plot(self.__fitness_history)
        plt.ylabel("Accuracy")
        plt.xlabel("Step")
        plt.title("Training Accuracy\nTest Accuracy: {:.4f}".format(self.test_accuracy))
        plt.show()

    async def __eval_accuracy(self, data: np.ndarray, label: np.ndarray):
        """评估种群适应度，适应度定义为在一个batch上的准确率

        Args:
            data (np.ndarray): 用于评估种群适应度的数据
            label (np.ndarray): 用于评估种群适应度的数据对应的标签

        Returns:
            np.ndarray: 种群的适应度数组
        """
        eval_tasks = [
            self.__eval_per_individual_accuracy(self.population[i], data, label)
            for i in range(self.M)
        ]

        loss = await asyncio.gather(*eval_tasks)
        fitness = np.array(loss)  # actually accuracy

        current_best_fitness = np.max(fitness)
        if current_best_fitness > self.__best_fitness:
            self.__best_individual = self.population[np.argmax(fitness)]
            self.__best_fitness = current_best_fitness

        self.fitness = fitness
        self.__fitness_history.append(np.mean(fitness))

    async def __eval_loss(self, data: np.ndarray, label: np.ndarray):
        """评估种群适应度，适应度定义为对应MLP交叉熵损失的倒数

        Args:
            data (np.ndarray): 用于评估种群适应度的数据
            label (np.ndarray): 用于评估种群适应度的数据对应的标签

        Returns:
            np.ndarray: 种群的适应度数组
        """
        eval_tasks = [
            self.__eval_per_individual_loss(self.population[i], data, label)
            for i in range(self.M)
        ]

        loss = await asyncio.gather(*eval_tasks)
        fitness = 1 / (np.array(loss) + 1e-6)

        self.__loss_history.append(np.mean(loss))

        current_best_fitness = np.max(fitness)
        if current_best_fitness > self.__best_fitness:
            self.__best_individual = self.population[np.argmax(fitness)]
            self.__best_fitness = current_best_fitness

        self.fitness = fitness
        self.__fitness_history.append(np.mean(fitness))

    async def __eval_per_individual_accuracy(
        self, chromosome: np.ndarray, data: np.ndarray, label: np.ndarray
    ):
        assert len(chromosome.shape) == 1
        model = MLP(chromosome, layer_dims=self.MLP_architecture)
        pred = model(data)
        accuracy = calculate_accuracy(pred, label)  # actually accuracy

        return accuracy

    async def __eval_per_individual_loss(
        self, chromosome: np.ndarray, data: np.ndarray, label: np.ndarray
    ):
        assert len(chromosome.shape) == 1
        model = MLP(chromosome, layer_dims=self.MLP_architecture)
        pred = model(data)
        loss = cross_entropy_loss(pred, label)

        return loss

    def __select(self):
        """使用轮盘赌方法对种群进行选择"""
        select_prob = self.fitness / np.sum(self.fitness)
        cum_prob = np.cumsum(select_prob)  # 计算累计概率
        p = np.random.rand(self.M)  # 随机给出M个概率
        select_mask = np.searchsorted(cum_prob, p)  # 轮盘赌得到选择的亲本下标
        self.population = self.population[select_mask]
        self.fitness = self.fitness[select_mask]

    def __crossover_linear(self):
        """使用线性加权对选定的亲本进行交叉"""
        p = np.random.rand(self.M)
        parents = self.population[p < self.Pc]
        if parents.shape[0] == 0:
            return
        parents = parents[np.random.permutation(parents.shape[0])]  # 随机打乱亲本

        parentA, parentB = np.vsplit(parents, [parents.shape[0] // 2])
        if parents.shape[0] % 2 != 0:  # 如果亲本数量非偶则需要特殊处理
            parentA = np.vstack((parentA, parentB[-1]))

        alpha = np.random.rand(1)
        childA = alpha * parentA + (1 - alpha) * parentB
        childB = (1 - alpha) * parentA + alpha * parentB

        if parents.shape[0] % 2 != 0:
            childA, _ = np.vsplit(childA, [childA.shape[0] - 1])

        self.population[p < self.Pc] = np.vstack((childA, childB))

    def __crossover_SBX(self):
        """使用模拟二进制交叉对选定的亲本进行交叉"""
        p = np.random.rand(self.M)
        parents = self.population[p < self.Pc]
        if parents.shape[0] == 0:
            return
        parents = parents[np.random.permutation(parents.shape[0])]  # 随机打乱亲本

        parentA, parentB = np.vsplit(parents, [parents.shape[0] // 2])
        if parents.shape[0] % 2 != 0:  # 如果亲本数量非偶则需要特殊处理
            parentA = np.vstack((parentA, parentB[-1]))

        eta = 20
        u = np.random.rand(parentA.shape[0])
        beta = np.where(
            u <= 0.5, (u * 2) ** (1 / (eta + 1)), (1 / (-2 * u + 2)) ** (1 / (eta + 1))
        ).reshape(-1, 1)

        childA = 0.5 * ((1 + beta) * parentA + (1 - beta) * parentB)
        childB = 0.5 * ((1 - beta) * parentA + (1 + beta) * parentB)

        if parents.shape[0] % 2 != 0:
            childA, _ = np.vsplit(childA, [childA.shape[0] - 1])

        self.population[p < self.Pc] = np.vstack((childA, childB))

    def __mutate(self):
        """使选定的亲本产生变异"""
        p = np.random.rand(self.M)
        parents = self.population[p < self.Pm]
        if parents.shape[0] == 0:
            return

        gene_num = parents.shape[1]
        mutation_gene_num = np.int32(gene_num * self.mutation_ratio)

        idx_pool = np.tile(np.arange(gene_num), (parents.shape[0], 1))
        choose = lambda row: np.random.choice(row, mutation_gene_num, replace=False)
        index = np.apply_along_axis(choose, 1, idx_pool)

        d = np.random.randn(*index.shape)  # 生成随机的变异方向
        # d = d / np.linalg.norm(d, axis=1, keepdims=True)
        parents[np.arange(parents.shape[0]).reshape(-1, 1), index] += (
            self.mutation_rate * d
        )

        self.population[p < self.Pm] = parents
