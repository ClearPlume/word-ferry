from typing import Iterator, Any

from torch.nn import Module, Dropout


class DropoutScheduler:
    """
    丢弃率调度器

    检测近几个epoch的训练loss和验证loss，发现有过拟合迹象时调整dropout
    """

    dropouts: dict[str, Dropout]
    initial_dropout: float
    factor: float
    max_dropout: float
    window: int
    cooldown: int

    current_dropout: float
    epochs_since_adjustment: int = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    def __init__(
            self,
            modules: Iterator[tuple[str, Module]],
            initial_dropout: float,
            factor: float,
            max_dropout: float = 0.5,
            window: int = 3,
            cooldown: int = 3,
    ):
        """
        :param window: 检测最近多少个epoch的loss状态以判定过拟合状态
        :param cooldown: 调整dropout后多少个 epoch 不再调整（冷却期）
        """

        self.dropouts = {}

        for name, module in modules:
            if isinstance(module, Dropout):
                self.dropouts[name] = module

        self.initial_dropout = initial_dropout
        self.factor = factor
        self.max_dropout = max_dropout
        self.window = window
        self.cooldown = cooldown

        self.current_dropout = initial_dropout

        # 设置所有dropout模块的初始值
        self._adjust_dropout(initial_dropout)

    def step(self, train_loss: float, val_loss: float) -> float:
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.epochs_since_adjustment += 1

        if len(self.train_losses) < self.window:
            return self.current_dropout

        if self.epochs_since_adjustment < self.cooldown:
            return self.current_dropout

        train_losses = self.train_losses[-self.window:]
        val_losses = self.val_losses[-self.window:]

        train_decreasing = all(train_losses[i] > train_losses[i + 1] for i in range(self.window - 1))
        val_increasing = all(val_losses[i] < val_losses[i + 1] for i in range(self.window - 1))

        if train_decreasing and val_increasing:
            new_dropout = min(self.current_dropout * (1 + self.factor), self.max_dropout)

            if new_dropout != self.current_dropout:
                self._adjust_dropout(new_dropout)
                self.current_dropout = new_dropout
                self.epochs_since_adjustment = 0

        return self.current_dropout

    def _adjust_dropout(self, dropout):
        for _, module in self.dropouts.items():
            module.p = dropout

    def state_dict(self) -> dict[str, Any]:
        return {
            "initial_dropout": self.initial_dropout,
            "factor": self.factor,
            "max_dropout": self.max_dropout,
            "window": self.window,
            "cooldown": self.cooldown,
            "epochs_since_adjustment": self.epochs_since_adjustment,
            "current_dropout": self.current_dropout,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.initial_dropout = state_dict["initial_dropout"]
        self.factor = state_dict["factor"]
        self.max_dropout = state_dict["max_dropout"]
        self.window = state_dict["window"]
        self.cooldown = state_dict["cooldown"]
        self.epochs_since_adjustment = state_dict["epochs_since_adjustment"]
        self.current_dropout = state_dict["current_dropout"]
        self.train_losses = state_dict["train_losses"]
        self.val_losses = state_dict["val_losses"]

        self._adjust_dropout(self.current_dropout)
