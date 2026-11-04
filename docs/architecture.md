# Architecture

A farmer sends text, a photo, or both. `script_detect` picks the reply language. A photo goes through the plant filter and the calibrated classifier. If the classifier abstains, the system asks for a clearer photo instead of guessing. Otherwise its top-3 labels join the question for retrieval and prompting. The LLM answers from the retrieved chunks, and `safety` blocks any pesticide or dose not found in them and adds the extension-officer disclaimer.

```mermaid
flowchart TD
    U[Farmer: text and/or photo] --> S[script_detect: ur / ur-Latn / en]
    U -->|photo| PF[plant_filter]
    PF --> V[Vision model + temperature scaling]
    V --> A{Confidence over threshold?}
    A -->|no| C[Ask for a clearer photo]
    A -->|yes: top-3| R
    S --> R[Hybrid retriever: BM25 + bge-m3, optional rerank]
    KB[(KB chunks)] --> R
    R --> OT{Any relevant chunks?}
    OT -->|no| REF[Off-topic refusal]
    OT -->|yes| P[build_prompt: top-3 + question + chunks]
    P --> L[LLM generator]
    L --> SF{Pesticides and doses in chunks?}
    SF -->|no| B[Replace with could-not-verify message]
    SF -->|yes| OK[Answer]
    B --> D[Add disclaimer]
    OK --> D
```

## Offline flow

```mermaid
flowchart LR
    SRC[kb/source_register.csv] --> I[kb.ingest] --> CL[kb.clean] --> CH[kb.chunk] --> IDX[kb.build_index]
    QA[data/raw/qa] --> VAL[data.validate] --> SP[data.split_qa] --> SFT[nlp.finetune.build_sft_data] --> QL[nlp.finetune.qlora_train]
    IMG[data/raw/images_*] --> VAL2[data.validate] --> SP2[data.split_images] --> T[vision.train] --> CAL[vision.calibrate] --> EV[vision.eval]
```

## Evaluation

Retrieval: `eval.retrieval_metrics`. Vision: `eval.vision_metrics`. Answers: blinded expert ratings in `eval/rating_sheets/`, aggregated by `eval.expert_ratings`. Usability: `eval.sus` on the pilot log.
