import random
import shutil

from word_ferry.path import get_data_dir


def sample_generator():
    corpus_dir = get_data_dir() / "corpus"
    sample_dir = get_data_dir() / "samples"

    if sample_dir.exists():
        print(f"sample_dir exists, recreating...")
        shutil.rmtree(sample_dir)

    print(f"creating sample_dir")
    sample_dir.mkdir(parents=True, exist_ok=True)
    print(f"sample_dir: {sample_dir}")

    samples = []
    for corpus in corpus_dir.iterdir():
        (source, target) = corpus.name.split(".")[0].split("-")
        print(f"processing {corpus}...")
        pairs = corpus.open(mode="r", encoding="utf-8").read().split("\n\n")

        for pair in pairs:
            (first, second) = pair.split("\n")
            samples.append(f"{first}\n<{target}>{second}")
        print(f"finished processing {corpus}")

    print(f"writing {len(samples)} samples")
    random.shuffle(samples)
    with (sample_dir / "samples.txt").open("w", encoding="utf-8") as f:
        f.write("\n\n".join(samples))

    print(f"finished writing {len(samples)} samples")


if __name__ == '__main__':
    sample_generator()
