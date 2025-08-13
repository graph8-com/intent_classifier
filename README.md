## Intent Classifier (texts only)

Classify raw texts into the closest topics using `sentence-transformers` with caching and detailed timing logs.

### Usage

1) Prepare a `topics.csv` file with at least a `topic_name` column. Optional columns: `category`, `theme`.

2) Provide texts via a file (one per line) or as a comma-separated string.

Run:

```bash
python -m intent_classifier.main "path/to/texts.txt" --topk 5
# or
python -m intent_classifier.main "text one, another text, final text" --topk 3
```

The results are saved to `results.csv` with `topic_1..k`, `score_1..k`, and `text_index`.

### Notes

- The model is `intfloat/multilingual-e5-large`.
- Embeddings are cached to speed up repeated runs.

### Docker (AWS-friendly)

Build:

```bash
docker build -t intent-classifier:latest .
```

Run with a mounted `topics.csv` and input file:

```bash
docker run --rm -v $PWD/topics.csv:/app/topics.csv -v $PWD/texts.txt:/app/texts.txt \
  intent-classifier:latest /app/texts.txt --topk 5 --topics /app/topics.csv
```

Run with inline texts:

```bash
docker run --rm -v $PWD/topics.csv:/app/topics.csv \
  intent-classifier:latest "hello world,another sentence here" --topk 3 --topics /app/topics.csv
```

