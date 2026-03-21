from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from word_ferry.components.collate import collate_fn
from word_ferry.components.config import Config, ResumeStrategy
from word_ferry.components.dataset import WordFerryDataset
from word_ferry.components.dropout_scheduler import DropoutScheduler
from word_ferry.components.model import Model
from word_ferry.components.sampler import LengthGroupSampler
from word_ferry.components.tokenizer import Tokenizer
from word_ferry.path import get_data_dir
from word_ferry.trainer import Trainer


def main():
    data_dir = get_data_dir()
    config = Config.default(
        learning_rate=5e-4,
        dropout=0.2,
        max_dropout=0.4,
        batch_size=40,
        max_len=256,
    )
    tokenizer = Tokenizer()

    dataset = WordFerryDataset(data_dir / "samples/samples.txt", tokenizer, config)
    train_dataset, val_dataset, _ = dataset.split()

    train_loader = DataLoader(
        dataset=train_dataset,
        collate_fn=collate_fn,
        batch_sampler=LengthGroupSampler("train", train_dataset, config.batch_size, True),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        collate_fn=collate_fn,
        batch_sampler=LengthGroupSampler("val", val_dataset, config.batch_size, True),
    )

    model = Model(tokenizer, config)

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=0.01,
    )

    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.9,
        patience=3,
    )

    dp_scheduler = DropoutScheduler(
        model.named_modules(),
        config.dropout,
        0.1,
        config.max_dropout,
        2,
    )
    trainer = Trainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dp_scheduler=dp_scheduler,
        train_name="2026-03-20",
    )

    trainer.load_checkpoint("2026-03-20", ResumeStrategy.EXCLUDE_REGULARIZATION)

    trainer.train()


if __name__ == '__main__':
    main()
