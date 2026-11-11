# 2d_cattle
One-stage cattle identification from raw video frames, incorporating farm knowledge
            ```mermaid
flowchart TD
    A[UAV Images] --> B[Coat Pattern Segmentation]
    B --> C[Balanced Dataset Construction]
    C --> D[Dataset Optimization (F-ODS)]
    D --> E[Few-shot Learning]
    E --> F[YOLO]
    E --> G[Metric]
    E --> H[ViT]
    F & G & H --> I[Self-supervised Dynamic Learning]
    I --> J[Individual Cattle Identification]
