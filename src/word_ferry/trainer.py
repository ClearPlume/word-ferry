import time
from logging import Logger
from pathlib import Path

import sacrebleu
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from word_ferry.components.config import Config, ResumeStrategy
from word_ferry.components.dataset import BatchedTransSample
from word_ferry.components.dropout_scheduler import DropoutScheduler
from word_ferry.components.logger import setup_logger
from word_ferry.components.model import Model
from word_ferry.components.tokenizer import Tokenizer
from word_ferry.core.constants import PAD_TOKEN_ID
from word_ferry.core.tdr_guard import TDRGuard
from word_ferry.path import get_logs_dir, get_models_dir


class Trainer:
    """训练器"""

    model: Model
    config: Config
    tokenizer: Tokenizer
    train_loader: DataLoader
    val_loader: DataLoader
    optimizer: Optimizer
    lr_scheduler: ReduceLROnPlateau
    dp_scheduler: DropoutScheduler

    early_stop_count: int
    early_stop_patience: int
    train_name: str
    start_epoch: int
    epochs: int
    criterion: CrossEntropyLoss
    summary: SummaryWriter
    best_score: float
    best_epoch: int
    checkpoint_dir: Path
    logger: Logger
    tdr_guard: TDRGuard

    def __init__(
            self,
            model: Model,
            config: Config,
            tokenizer: Tokenizer,
            train_loader: DataLoader,
            val_loader: DataLoader,
            optimizer: Optimizer,
            lr_scheduler: ReduceLROnPlateau,
            dp_scheduler: DropoutScheduler,
            train_name: str,
            early_stop_patience: int = 6,
            epochs: int = 50,
    ):
        """
        :param train_name: 训练名称，将作为日志路径、模型保存路径的一部分
        """

        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.dp_scheduler = dp_scheduler

        self.early_stop_count = 0
        self.early_stop_patience = early_stop_patience
        self.train_name = train_name
        self.start_epoch = 1
        self.epochs = epochs
        self.criterion = CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, label_smoothing=0.05)
        self.summary = SummaryWriter(str(get_logs_dir() / train_name))
        self.best_score = 0
        self.best_epoch = 1
        self.checkpoint_dir = get_models_dir() / f"checkpoint/{train_name}"

        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger(train_name, "train")
        self.tdr_guard = TDRGuard()

    def load_checkpoint(
            self,
            checkpoint_name: str,
            resume_strategy: ResumeStrategy,
            reset_training_state: bool = False,
    ):
        """从存档点中恢复状态"""

        self.logger.info(f"▶ 尝试加载存档点: {checkpoint_name}")
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{checkpoint_name}_best.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")

        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint["model_state"])
        self.logger.info("✓ 模型权重已加载")

        if resume_strategy == ResumeStrategy.ALL_COMPONENTS:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state"])
            self.dp_scheduler.load_state_dict(checkpoint["dp_scheduler_state"])

            self.logger.info("✓ 完全恢复：优化器 + LR调度器 + 丢弃率调度器")

        elif resume_strategy == ResumeStrategy.EXCLUDE_OPTIMIZATION:
            self.dp_scheduler.load_state_dict(checkpoint["dp_scheduler_state"])

            self.logger.info(f"✓ 部分恢复：丢弃率调度器")
            self.logger.info(f"✗ 优化器和LR调度器使用新配置")

        elif resume_strategy == ResumeStrategy.EXCLUDE_REGULARIZATION:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state"])

            self.dp_scheduler.current_dropout = checkpoint["dp_scheduler_state"]["current_dropout"]
            self.dp_scheduler.train_losses = checkpoint["dp_scheduler_state"]["train_losses"]
            self.dp_scheduler.val_losses = checkpoint["dp_scheduler_state"]["val_losses"]

            self.logger.info(f"✓ 部分恢复：优化器 + LR调度器")
            self.logger.info(f"✗ 丢弃率调度器使用新配置，仅恢复checkpoint的dropout和历史loss")

        else:
            self.dp_scheduler.current_dropout = checkpoint["dp_scheduler_state"]["current_dropout"]
            self.dp_scheduler.train_losses = checkpoint["dp_scheduler_state"]["train_losses"]
            self.dp_scheduler.val_losses = checkpoint["dp_scheduler_state"]["val_losses"]

            self.logger.info(f"✗ 优化器、LR调度器使用新配置，仅恢复部分dropout状态")

        if not reset_training_state:
            self.best_score = checkpoint["best_score"]
            self.best_epoch = checkpoint["best_epoch"]
            self.early_stop_count = checkpoint["early_stop_count"]
            self.start_epoch = checkpoint["start_epoch"]

    def train(self):
        self.logger.info("=" * 60)
        self.logger.info("🚀 开始训练")
        self.logger.info(f"    Epochs: {self.start_epoch} -> {self.epochs}")
        self.logger.info(f"    模型架构: {self.config.arch_str}")
        self.logger.info(f"    训练参数: {self.config.train_str}")
        self.logger.info(f"    ⚠ 缓存敏感: batch_size={self.config.batch_size}, max_len={self.config.max_len}")
        self.logger.info(f"        — 若已修改，请确认缓存已清除")
        self.logger.info(f"    参数规模: {self.model.param_num}")
        self.logger.info(f"    样本规模: 180万")
        self.logger.info(f"    当前学习率: {self.lr_scheduler.get_last_lr()[0]:.2e}")
        self.logger.info(f"    当前Dropout: {self.dp_scheduler.current_dropout}")
        self.logger.info(f"    最佳分数: {self.best_score}")
        self.logger.info("=" * 60)

        total_start = time.perf_counter()

        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_start = time.perf_counter()

            train_loss = self.train_epoch(epoch)
            val_loss, score = self.validate_epoch(epoch)

            epoch_time = time.perf_counter() - epoch_start
            minutes, seconds = divmod(int(epoch_time), 60)

            self.logger.info(f"⏭️ Epoch {epoch}/{self.epochs}")
            self.logger.info(f"    Train loss: {train_loss:.8f}")
            self.logger.info(f"    Val loss: {val_loss:.8f}")
            self.logger.info(f"    BLEU: {score:.8f}")
            self.logger.info(f"    LR: {self.lr_scheduler.get_last_lr()[0]:.2e}")
            self.logger.info(f"    DP: {self.dp_scheduler.current_dropout}")
            self.logger.info(f"    Time: {minutes}m {seconds}s")

            old_lr = self.lr_scheduler.get_last_lr()[0]
            self.lr_scheduler.step(val_loss)
            new_lr = self.lr_scheduler.get_last_lr()[0]

            if old_lr != new_lr:
                self.logger.info(f"🔄 学习率调整 ({epoch}): {old_lr:.2e} -> {new_lr:.2e}")
                self.summary.add_text("Hyperparams", f"🔄 学习率调整 ({epoch}): {old_lr:.2e} -> {new_lr:.2e}", epoch)

            old_dp = self.dp_scheduler.current_dropout
            new_dp = self.dp_scheduler.step(train_loss, val_loss)

            if old_dp != new_dp:
                self.early_stop_count = 0

                self.logger.info(f"⚠️ 检测到过拟合趋势 ({epoch})")
                self.logger.info(f"🔄 Epoch {epoch}: Dropout {old_dp:.3f} → {new_dp:.3f}, 早停计数重置")

                self.summary.add_text(
                    "Hyperparams",
                    f"⚠️ 检测到过拟合趋势 ({epoch})\n🔄 Epoch {epoch}: Dropout {old_dp:.3f} → {new_dp:.3f}, 早停计数重置",
                    epoch,
                )

            self.summary.add_scalars("Training/Loss", {
                "train": train_loss,
                "val": val_loss,
            }, epoch)
            self.summary.add_scalar("Training/BLEU", score, epoch)
            self.summary.add_scalar("Hyperparams/LR", self.lr_scheduler.get_last_lr()[0], epoch)
            self.summary.add_scalar("Hyperparams/DP", self.dp_scheduler.current_dropout, epoch)

            if self._is_best(score, epoch):
                self.logger.info(f"✨ 新的最佳模型 (Epoch {self.best_epoch}, Score={score:.8f})")
                self.summary.add_text(
                    "BestModel",
                    f"✨ 新的最佳模型 (Epoch {self.best_epoch}, Score={score:.8f})",
                    epoch,
                )
                torch.save(
                    {
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "lr_scheduler_state": self.lr_scheduler.state_dict(),
                        "dp_scheduler_state": self.dp_scheduler.state_dict(),
                        "best_score": self.best_score,
                        "best_epoch": self.best_epoch,
                        "early_stop_count": self.early_stop_count,
                        "start_epoch": epoch + 1,
                    },
                    self.checkpoint_dir / f"checkpoint_{self.train_name}_best.pt",
                )

            self.summary.add_scalars("Early Stopping", {
                "early_stop_count": self.early_stop_count,
                "early_stop_patience": self.early_stop_patience,
            }, epoch)

            if self.early_stop_count >= self.early_stop_patience:
                self.logger.info(f"🚨 早停触发: 连续 {self.early_stop_patience} 个epoch无提升")
                self.logger.info(f"    最佳模型: Epoch {self.best_epoch}, Score={self.best_score:.8f}")
                self.summary.add_text(
                    "EarlyStop",
                    f"🚨 早停触发: 连续 {self.early_stop_patience} 个epoch无提升\n    最佳模型: Epoch {self.best_epoch}, Score={self.best_score:.8f}",
                    epoch,
                )
                break

        self.summary.close()
        self.logger.info("=" * 60)

        total_time = time.perf_counter() - total_start
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)

        self.logger.info(f"✅ 训练完成，总时长: {hours}h {minutes}m {seconds}s")
        self.logger.info(f"    最佳分数: {self.best_score:.8f} (Epoch {self.best_epoch})")

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        progress: tqdm[BatchedTransSample] = tqdm(self.train_loader, f"Epoch {epoch}/{self.epochs} [Train]")

        for idx, batch in enumerate(progress):
            loss, _, _ = self._predict(batch)
            total_loss += loss.item()

            loss.backward()

            if idx % 100 == 0:
                self._record_gradients(epoch)

            clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.tdr_guard.sync_if_needed()

            progress.set_postfix({
                "loss": f"{loss.item():.8f}",
                "lr": f"{self.lr_scheduler.get_last_lr()[0]:.2e}",
                "dp": f"{self.dp_scheduler.current_dropout}",
            })

        avg_loss = total_loss / len(self.train_loader)
        progress.set_postfix({
            "loss": f"{avg_loss:.8f}",
            "lr": f"{self.lr_scheduler.get_last_lr()[0]:.2e}",
            "dp": f"{self.dp_scheduler.current_dropout}",
        })
        return avg_loss

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        loss_progress: tqdm[BatchedTransSample] = tqdm(
            self.val_loader,
            f"Loss calculating, epoch {epoch}/{self.epochs} [Val]",
        )

        for idx, batch in enumerate(loss_progress):
            loss, src, target = self._predict(batch)
            total_loss += loss.item()

            self.tdr_guard.sync_if_needed()

            loss_progress.set_postfix({
                "loss": f"{loss.item():.8f}",
                "lr": f"{self.lr_scheduler.get_last_lr()[0]:.2e}",
                "dp": f"{self.dp_scheduler.current_dropout}",
            })

        avg_loss = total_loss / len(self.val_loader)
        loss_progress.set_postfix({
            "loss": f"{avg_loss:.8f}",
            "lr": f"{self.lr_scheduler.get_last_lr()[0]:.2e}",
            "dp": f"{self.dp_scheduler.current_dropout}",
        })

        bleu_data: list[BatchedTransSample] = []
        for idx, batch in enumerate(self.val_loader):
            if idx >= 10000 / self.config.batch_size:
                break
            bleu_data.append(batch)

        bleu_progress: tqdm[BatchedTransSample] = tqdm(
            bleu_data,
            f"BLEU forwarding, epoch {epoch}/{self.epochs} [Val]",
        )

        hypotheses = []
        references = []
        for idx, batch in enumerate(bleu_progress):
            generated = self.model.generate(
                batch.src.to(self.config.device),
                batch.src_attention_mask.to(self.config.device),
                batch.tgt_in.to(self.config.device)[:, 0],
            )

            # generated: [batch_size, seq_len]，每行是 [lang, t1, t2, ..., EOS, PAD, PAD, ...]
            for seq in generated:
                token_ids = seq.tolist()
                # 去掉语言标签，截断到 EOS
                if self.tokenizer.eos_token_id in token_ids:
                    token_ids = token_ids[1:token_ids.index(self.tokenizer.eos_token_id)]
                else:
                    token_ids = token_ids[1:]
                hypotheses.append(self.tokenizer.decode(token_ids))

            # references 从 batch.tgt_out 里解码
            for seq in batch.tgt_out:
                # 移除语言标签
                token_ids = seq.tolist()[1:]
                token_ids = token_ids[:token_ids.index(self.tokenizer.eos_token_id)]
                references.append(self.tokenizer.decode(token_ids))

            self.tdr_guard.sync_if_needed()

            # BLEU测试语句收集量
            bleu_progress.set_postfix({"samples": f"{len(hypotheses)}/{len(bleu_data) * self.config.batch_size}"})

        return avg_loss, sacrebleu.corpus_bleu(hypotheses, [references]).score

    def _predict(self, batch: BatchedTransSample) -> tuple[Tensor, Tensor, Tensor]:
        """
        模型预测
        
        :return tuple[loss, src, target]
        """

        src = batch.src.to(self.config.device)
        src_attention_masks = batch.src_attention_mask.to(self.config.device)
        tgt = batch.tgt_in.to(self.config.device)
        tgt_attention_masks = batch.tgt_attention_mask.to(self.config.device)

        target = batch.tgt_out.to(self.config.device)

        logits: Tensor = self.model(src, src_attention_masks, tgt, tgt_attention_masks)

        loss = self.criterion(logits.view(-1, self.tokenizer.vocab_size), target.view(-1))
        return loss, src, target

    def _is_best(self, score: float, epoch: int) -> bool:
        """判断并更新最佳记录和早停计数"""

        improvement = score - self.best_score
        if improvement > self.config.min_improvement:
            if self.early_stop_count > 0:
                self.logger.info(f"✔️ 早停计数重设 ({epoch})")
                self.summary.add_text("EarlyStop", f"✔️ 早停计数重设 ({epoch})", epoch)

            self.best_score = score
            self.best_epoch = epoch
            self.early_stop_count = 0

            return True
        else:
            self.early_stop_count += 1

            if self.early_stop_count < self.early_stop_patience // 2:
                symbol = "⏳"
            elif self.early_stop_count < self.early_stop_patience * 0.8:
                symbol = "⚠️"
            else:
                symbol = "🚨"

            self.logger.info(f"{symbol} 接近早停阈值 ({self.early_stop_count}/{self.early_stop_patience})")
            self.summary.add_text(
                "EarlyStop",
                f"{symbol} 接近早停阈值 ({self.early_stop_count}/{self.early_stop_patience})",
                epoch,
            )
            return False

    def _record_gradients(self, epoch: int):
        """记录关键组件的梯度范数到TensorBoard"""

        def get_component_group(component: str) -> str:
            """根据参数名称确定所属组件分组"""
            if component == "cls_parameter":
                return "cls_parameter"
            elif component == "embedding.weight":
                return "embedding"
            elif component.startswith("encoder.layers.0."):
                return "encoder_first"
            elif component.startswith(f"encoder.layers.{self.config.n_encoder_layers // 2}."):
                return "encoder_mid"
            elif component.startswith(f"encoder.layers.{self.config.n_encoder_layers - 1}."):
                return "encoder_last"
            else:
                return "classifier"

        # 初始化梯度分组
        grad_groups = {}

        # 遍历所有参数，收集梯度范数
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            group = get_component_group(name)
            if group is None:
                continue

            grad_norm = param.grad.norm().item()
            grad_groups.setdefault(group, []).append(grad_norm)

        # 写入TensorBoard（每组取平均）
        for group, norms in grad_groups.items():
            if norms:  # 确保列表不为空
                avg_norm = sum(norms) / len(norms)
                self.summary.add_scalar(f"GradNorm/{group}", avg_norm, epoch)

        # 记录全局梯度范数（所有参数）
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.norm().item()
                total_norm += param_norm ** 2
        total_norm = total_norm ** 0.5
        self.summary.add_scalar("GradNorm/global", total_norm, epoch)
