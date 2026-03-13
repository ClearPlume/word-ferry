import random
import shutil

from src.word_ferry.components.config import CorpusType
from src.word_ferry.core.constants import SAMPLES_PER_DIRECTION
from src.word_ferry.path import get_data_dir


def corpus_extractor(corpus_type: CorpusType):
    print(f"extracting {corpus_type} corpus...")
    type_ = corpus_type.value
    (source, target) = type_.split("-")

    print(f"source: {source} loading...")
    with open(fr"C:\Users\15203\Documents\衔言渡意\{type_}\UNv1.0.{type_}.{source}", "r", encoding="utf-8") as f:
        source_lines = f.readlines()
    print(f"source: {source} loaded")

    print(f"target: {target} loading...")
    with open(fr"C:\Users\15203\Documents\衔言渡意\{type_}\UNv1.0.{type_}.{target}", "r", encoding="utf-8") as f:
        target_lines = f.readlines()
    print(f"target: {target} loaded")

    indices = sorted(random.sample(range(0, len(source_lines)), k=SAMPLES_PER_DIRECTION * 2))
    source_target = indices[:SAMPLES_PER_DIRECTION]
    target_source = indices[SAMPLES_PER_DIRECTION:]

    # source -> target
    print(f"constructing {source}-{target} pairs...")
    sources = []
    for index in source_target:
        src = source_lines[index].strip()
        tgt = target_lines[index].strip()
        sources.append(f"{src}\n{tgt}")
    print(f"{source}-{target} pairs constructed")

    source_target_corpus_file = get_data_dir() / "corpus" / f"{source}-{target}.txt"
    with open(source_target_corpus_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(sources))
    print(f"{source}-{target} pairs saved at: {source_target_corpus_file}")

    # target -> source
    print(f"constructing {target}-{source} pairs...")
    targets = []
    for index in target_source:
        src = target_lines[index].strip()
        tgt = source_lines[index].strip()
        targets.append(f"{src}\n{tgt}")
    print(f"{target}-{source} pairs constructed")

    target_source_corpus_file = get_data_dir() / "corpus" / f"{target}-{source}.txt"
    with open(target_source_corpus_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(targets))
    print(f"{target}-{source} pairs saved at: {target_source_corpus_file}")

    print(f"corpus {corpus_type} extracted")


if __name__ == '__main__':
    corpus_dir = get_data_dir() / "corpus"

    if corpus_dir.exists():
        print(f"corpus_dir {corpus_dir} exists, recreating...")
        shutil.rmtree(corpus_dir)

    print(f"creating sample_dir")
    corpus_dir.mkdir(parents=True, exist_ok=True)
    print(f"sample_dir: {corpus_dir}")

    corpus_extractor(CorpusType.EN_ZH)
    corpus_extractor(CorpusType.EN_FR)
    corpus_extractor(CorpusType.FR_ZH)
