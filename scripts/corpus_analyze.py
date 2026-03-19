from word_ferry.components.config import CorpusType


def corpus_analyze(corpus_type: CorpusType):
    print(f"analyzing {corpus_type}")
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

    assert len(source_lines) == len(target_lines)

    line_number = len(source_lines)
    print(corpus_type, line_number)


corpus_analyze(CorpusType.EN_ZH)
corpus_analyze(CorpusType.EN_FR)
corpus_analyze(CorpusType.FR_ZH)
