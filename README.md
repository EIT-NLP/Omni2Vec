# Omni2Vec

Train and evaluate **Omni2Vec-3B** and **Omni2Vec-7B** for text, image, document, video, audio and agent retrieval. Both sizes use the same training script and data configuration. OmniSET adds an instruction reader and four modality directions to a frozen retrieval encoder.

## Install

Use Python 3.11 on Linux. Install PyTorch and torchvision for your CUDA runtime, then install the remaining dependencies from the project root.

```bash
python -m pip install -r requirements.txt
```

The default backbones are [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) and [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B). Use `--base` to select a local copy.

## Training data

[Source locations](configs/sources.yaml) lists public datasets, download patterns, and local directories. [Training configuration](configs/train.yaml) selects the sources. Fetch only the families you intend to prepare.

```bash
python -m prepare.download image pixmo colpali visrag
python -m prepare.download toolret toolrex evo
python -m prepare.build_toolret_toolrex_mix \
  --toolret-root data/tool/raw/toolret/ToolRet-Training-20w \
  --toolrex-json data/tool/raw/toolrex/Tool-DE_train_retriever_50k.json --output-dir data/tool/pairs
python -m prepare.evo --input data/memory/raw
```

| Family | Preparing training pairs |
|---|---|
| Image | Keep the MMEB training Parquet files and per-task image ZIPs together. The loader resolves members directly inside the ZIPs. |
| Documents | Read the public training shards directly. PixMo supplies multiple questions per page. The epoch selects one deterministically. ColPali and VisRAG pair queries with pages. |
| Text | Fetch `bge promptriever inf-ir` with `prepare.download`. Extract the BGE training archive into `data/text/raw/bge`, run `python -m prepare.bge --input data/text/raw/bge`, then `python -m prepare.build_text_320k_hn25_mix --text-root data/text --output-dir data/text/pairs`. The mixture includes 80,000 examples with supplied negatives. |
| Tool | The command above selects a released positive and negative for ToolRet and the supplied response for Tool-REX. |
| Memory | Evo uses the final user message as the query. The converter concatenates the annotated history turns into **one positive** in evidence-list order, separated by blank lines. It removes repeated indices, excludes the final turn and skips sessions without a query or evidence. |
| GUI | Fetch `gui` with `prepare.download` and place the corresponding screenshots under `data/gui/{AutoWebGLM,WebArena,WebLINX}/images`, preserving each record's relative filename. The recipe reads the three public `original.parquet` files and selects the last screenshot when a side contains several. [Metadata](https://huggingface.co/datasets/iMeanAI/GAE-Bench-lite) |
| Video | Prepare the six training relations described below, then run `python -m prepare.build_video_full_epoch --dataset-root data/video/raw --output-dir data/video/pairs`. |
| Audio | Use the upstream training splits to build one annotation manifest per source, then run `prepare.audio` as described below. |

**Video layout.** Under `data/video/raw`, the `MSR-VTT`, `MSVD` and `DiDeMo` directories each contain `raw_videos/` and their respective annotation file (`msrvtt_train_9k.json`, `msvd_train.json`, `didemo_train.json`). The annotation is a JSON list of objects with `video`, `video_id` and `caption` (string or list). For LLaVA-Hound, fetch `llava-hound`, extract the frame archives and retain the upstream `video_instruction/train/sft` caption and QA JSONL files. For Something-Something V2, use the training videos and action labels. For both sources create a JSONL manifest with `video_id` and `path` (video or frame directory, relative to the manifest). Supply `pos_text` for the action label. Preserve the video IDs used in the annotations.

```bash
python -m prepare.frames --manifest data/video/raw/llava_hound/train.jsonl \
  --output data/video/raw/llava_hound/parquet_8frames/train.parquet
python -m prepare.frames --manifest data/video/raw/SmthSmthV2/train.jsonl \
  --output data/video/raw/SmthSmthV2/parquet_8frames/train.parquet
```

**Audio layout.** Each JSONL record has `id`, `audio` and optional clip times `start` and `end` in seconds. For AudioCaps and Clotho, `texts` lists captions. For classification, `texts` lists labels. For SpeechCOCO and AVE, `images` lists the positive image or ordered video frames. Media paths are relative to the manifest. An AudioCaps example is `{"id":"clip-1","audio":"clips/1.wav","texts":["A dog barks."]}`.

```bash
python -m prepare.audio --source audiocaps --manifest data/audio/raw/audiocaps/train.jsonl \
  --output data/audio/audiocaps/train/train.parquet
```

## Train Omni2Vec

The reference epoch comprises **2,525,972 valid records** across 79 logical sources. Each global batch of 64 contains one source. The loss masks 1,900 padding positions at source boundaries, giving 39,498 optimizer steps. Selection and scheduling use seed 42. The source-level counts are in [epoch_counts.csv](configs/epoch_counts.csv). Upstream revisions, reconstructed media, and evidence conversions can change the selected records.

```bash
python -m prepare.epoch --config configs/train.yaml --check-counts configs/epoch_counts.csv
python -m prepare.pack
torchrun --standalone --nproc-per-node=8 train.py --size 3b --output runs/omni2vec-3b
```

For Omni2Vec-7B use `--size 7b --output runs/omni2vec-7b` with the same command. For a subset, keep its entries in a separate YAML and pass that file to `prepare.epoch` without `--check-counts`. The generated schedule records the actual counts. Packing writes a shuffled, normalized batch store under `data/epoch`. Keep relative media paths accessible when training.

The script runs this 2.53M-record epoch from the selected backbone or an initial `--adapter`. The reference checkpoints entered this epoch after a 640,000-exposure initialization phase. It trains rank-16 LoRA on the Thinker's language projections using masked mean pooling, L2 normalization and task-aware multi-positive contrastive loss. Candidate gradients flow across ranks. The peak AdamW learning rate is 1e-5 with linear warmup and decay. Video uses eight uniformly sampled frames. Training writes adapter checkpoints and a `final` adapter directory.

## Evaluate Omni2Vec

The evaluator accepts `queries`, `candidates` and `qrels` tables in Lance or JSONL. For the public MMEB image tasks, prepare the benchmark assets following the [MMEB-V3 dataset card](https://huggingface.co/datasets/VLM2Vec/MMEB-V3), then convert a test split and evaluate.

```bash
python -m prepare.mmeb_eval --metadata data/MMEB-V3/image-query/N24News \
  --images data/MMEB-V3/image-tasks/MMEB --output data/eval/N24News
python evaluate.py --size 3b --adapter runs/omni2vec-3b/final --task data/eval/N24News
```

A query or candidate needs an `id` and content in `text`, `images`, `video` or `audio`. JSONL media paths are relative to the task directory. A qrels row such as `{"query_id":"q1","mode":"exhaustive","candidate_ids":["d1","d2"],"candidate_scores":[2,0]}` defines a per-query pool and graded relevance. Use `mode: global` for full-corpus retrieval. Metrics are Hit@1, linear-gain nDCG@5 and Recall@10. For Lance exports with video frames in `images`, pass `--family video`. Use `--family visdoc` for document task prompts.

## Train and evaluate OmniSET

Fit shrinkage-LDA directions on distinct positive candidates from the **training epoch** and a four-class reader on instruction templates. The reader selects the requested modality using instruction text alone. Evaluation applies the selected direction to the query and ranks each benchmark pool with the same candidate embeddings. The default strength is 0.7.

```bash
python -m omniset.train --size 3b --adapter runs/omni2vec-3b/final \
  --epoch data/epoch --output runs/omniset/calibration.pt
hf download VLM2Vec/MMEB-V3 omniset.tar.gz --repo-type dataset --local-dir data
tar -xzf data/omniset.tar.gz -C data
python -m omniset.evaluate --size 3b --adapter runs/omni2vec-3b/final \
  --data data/omniset --calibration runs/omniset/calibration.pt
```

Point `--data` at the directory containing `omniset.jsonl`, `catalog.jsonl`, `val2014/`, `videos/` and `audios/`. Omit `--calibration` for base-encoder evaluation. Results cover all twelve directed retrieval tasks across text, image, video and audio.

Code is distributed under the [Apache 2.0 license](LICENSE). Model weights, media and datasets retain their upstream terms.
