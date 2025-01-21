# Document Clustering: From Traditional to Contemporary Approaches

## Abstract

Document clustering is an important unsupervised learning technique. It is widely used in optimizing information retrieval, improving knowledge management efficiency, and automatically mining document topics. The document clustering process is divided into three parts: text vectorization, high-dimensional data reduction and clustering algorithm application. In this paper, we focus on presenting, evaluating and comparing document clustering pipelines from traditional statistical methods to modern deep-learning approaches. Representative models of different development stages are chosen and combined. The results showed that, the combination of MiniLM and K-means balanced accuracy and efficiency and achieved the best overall performance.

## Key Features

The document clustering process is divided into three main parts:
- Text vectorization
- High-dimensional data reduction  
- Clustering algorithm application

## Project Structure

```
├── models/          # Model related files
├── results/         # Experimental results
├── src/            # Source code
├── main.py         # Main program entry
├── requirements.txt # Project dependencies
└── vectorized_data.pkl # Vectorized data
```

## Tech Stack

- Language: Python
- Text Vectorization: TFIDF、FastText、MiniLM
- Clustering Algorithm: K-means、HDBSCAN、BIRCH
- Other ML libraries

## Installation

1. Clone the repository
```bash
git clone https://github.com/xyFranzi/expKnowledgeProcessing.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the program
```bash
python main.py
```

## Experimental Process

1. Text Vectorization Processing
2. Dimensionally Reduction
3. Clustering Algorithm Application
4. Results Evaluation and Comparison

## Key Findings

The research demonstrated that the combination of MiniLM and K-means achieved the best overall performance, balancing accuracy and efficiency in document clustering tasks.

## Contributors

- Xiaosong Yu (@xyFranzi)
- Yue Yang (@Sherryyue24)

