# AI Fashion Recommendation Search Engine

An end-to-end deep learning powered image similarity system that allows users to upload a fashion image and instantly retrieve visually similar products from a large-scale image dataset.

The system is built using a Triplet Network architecture, FAISS vector similarity search, GPU acceleration, and a Streamlit-based web interface.



## Live Demo

https://imagesimilarity-21.streamlit.app/



## Project Demo

https://drive.google.com/file/d/1ZlTBxIH45b0xpTjNCQnMQY-vLpDTKT1A/view



## Overview

This project uses deep metric learning to understand visual similarity between fashion products. Instead of classifying images into categories, the model learns an embedding space where visually similar products are located close together.

The workflow:

1. User uploads a fashion image
2. Image is converted into embeddings using a trained Triplet Network
3. FAISS performs high-speed vector similarity search
4. Similar images are retrieved from the indexed dataset
5. Results are displayed through the Streamlit interface

The system is optimized for large-scale fashion recommendation and similarity retrieval tasks.


## Features

- Deep learning image embeddings using ResNet50 Triplet Network
- Batch-hard mining for improved similarity learning
- Recall@K evaluation pipeline
- FAISS GPU-accelerated vector indexing
- Google Drive CDN based large-scale image hosting
- Streamlit web interface
- Scalable architecture supporting 50K+ fashion images



## System Architecture

```text
                ┌──────────────┐
                │  User Upload │
                └───────┬──────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Streamlit Web App │
              │  (User Interface) │
              └─────────┬─────────┘
                        │
                        ▼
            ┌─────────────────────────┐
            │ Triplet Embedding Model │
            │   (ResNet50 Backbone)   │
            └─────────┬───────────────┘
                      │
                      ▼
            ┌─────────────────────────┐
            │    FAISS Vector Index   │
            │   (Cosine Similarity)   │
            └─────────┬───────────────┘
                      │
                      ▼
     ┌────────────────────────────────────┐
     │ Google Drive CDN Image Repository  │
     │     (47,000+ Fashion Images)       │
     └────────────────────────────────────┘
```



## Tech Stack

| Component | Technology |
|---|---|
| Deep Learning Framework | PyTorch |
| Embedding Model | ResNet50 |
| Similarity Search | FAISS |
| Frontend | Streamlit |
| Dataset Hosting | Google Drive CDN |
| GPU Acceleration | CUDA |


## Project Structure

```text
app.py                         # Streamlit application

models/
├── embedding_model.py         # ResNet50 Triplet Network
└── embedding.pth              # Trained model weights

training/
├── train.py                   # Training pipeline
├── triplet_mydata.py          # Dataset loader
├── loss.py                    # Triplet margin loss
├── sampler.py                 # Batch-hard mining
├── recall_eval.py             # Recall@K evaluation
├── run_recall.py              # Recall evaluation runner
└── data/                      # Local training dataset

inference/
├── build_index_mydata.py      # Builds FAISS index
├── search.py                  # Similarity search engine
├── drive_urls.csv             # CDN image database
└── faiss/
    ├── index.bin              # FAISS vector index
    └── paths.npy              # Image CDN paths

tools/
├── drive_to_csv.py            # Export Drive dataset to CSV
├── convert_drive_links.py     # Convert Drive links to CDN
├── convert_to_cdn.py          # CDN link conversion
├── rebuild_paths_from_index.py # Repairs FAISS paths
├── client_secrets.example.json # OAuth template
└── settings.yaml              # Drive API configuration
```



## Model Architecture

The system uses a ResNet50 backbone trained using Triplet Loss.

### Training Strategy

- Anchor image
- Positive image (similar item)
- Negative image (different item)

The objective is to minimize the embedding distance between similar products while maximizing the distance between dissimilar products.

### Training Enhancements

- Batch-hard mining
- Metric learning
- Recall@K evaluation
- GPU acceleration using CUDA



## Search Pipeline

1. Extract embeddings from uploaded image
2. Normalize embedding vectors
3. Query FAISS vector index
4. Retrieve nearest neighbor embeddings
5. Display visually similar products

The retrieval process uses cosine similarity for efficient nearest-neighbor search.


## Installation

```bash
git clone https://github.com/Yogesh942134/image_similarity_.git

cd image_similarity_

pip install -r requirements.txt
```



## Run Locally

```bash
streamlit run app.py
```



## Recall@K Evaluation

```bash
python training/run_recall.py
```



## Build FAISS Index

```bash
python inference/build_index_mydata.py
```



## Performance Considerations

- FAISS enables ultra-fast similarity retrieval
- GPU acceleration improves embedding generation speed
- Google Drive CDN reduces dataset hosting cost
- Designed for scalable large-scale image retrieval systems
- Efficient vector indexing supports real-time recommendations



## Use Cases

- Fashion product recommendation
- Visual similarity search
- E-commerce recommendation systems
- Reverse image search
- Personalized shopping assistants

