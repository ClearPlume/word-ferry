import pickle
import random
from typing import Iterator, cast

from torch.utils.data import Subset

from word_ferry.components.train.dataset import TokenizedTransSample, WordFerryDataset
from word_ferry.path import get_data_dir


class LengthGroupSampler:
    """按长度分组的批次采样器，减少padding开销"""

    dataset: Subset[WordFerryDataset]
    batch_size: int
    drop_last: bool
    length_groups: list[list[int]]

    def __init__(self, dataset_type: str, dataset: Subset[WordFerryDataset], batch_size: int, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        sample_file = (get_data_dir() / "samples/samples.txt")
        cache_file = sample_file.with_suffix(f".length-groups-{dataset_type}.pkl")

        if cache_file.exists() and cache_file.stat().st_mtime > sample_file.stat().st_mtime:
            with open(cache_file, "rb") as cache:
                self.length_groups = pickle.load(cache)
            return

        print(f"Sampler[{dataset_type}] initialing")
        self.length_groups = self._group_by_length()
        cache_file.write_bytes(pickle.dumps(self.length_groups))
        print(f"Sampler[{dataset_type}] initialed")

    def _group_by_length(self) -> list[list[int]]:
        """按序列长度对样本分组"""

        print("Grouping by length, fetching length-index pairs")
        # 获取所有样本的长度和索引对
        length_index_pairs: list[tuple[int, int]] = []
        for idx in range(len(self.dataset)):
            dataset = cast(TokenizedTransSample, cast(object, self.dataset[idx]))
            length_index_pairs.append((len(dataset.input), idx))

            if idx % 1000 == 0:
                print(f"Length-index fetched: {idx}/{len(self.dataset)}")
        print(f"Length-index pairs fetched: {len(length_index_pairs)}")

        # 按长度排序
        print("Length-index pairs sorting")
        length_index_pairs.sort(key=lambda p: p[0])
        print("Length-index pairs sorted")

        # 按长度分组
        print("Grouping by length")
        groups: list[list[int]] = []
        for idx in range(0, len(length_index_pairs), self.batch_size):
            group: list[int] = []
            for pair in length_index_pairs[idx: idx + self.batch_size]:
                group.append(pair[1])
            groups.append(group)

            if idx % 1000 == 0:
                print(f"Length-index pairs grouped: {idx}/{len(length_index_pairs)}")

        print("Grouped")
        return groups

    def __len__(self) -> int:
        """返回批次总数"""

        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return len(self.dataset)

    def __iter__(self) -> Iterator[list[int]]:
        """生成批次序列"""

        batches: list[list[int]] = []

        # 为每个长度组创建批次
        for indices in self.length_groups:
            # 处理 drop_last 逻辑
            if len(indices) < self.batch_size and self.drop_last:
                continue

            random.shuffle(indices)  # 组内随机
            batches.append(indices)

        random.shuffle(batches)  # 批次间随机
        return iter(batches)
