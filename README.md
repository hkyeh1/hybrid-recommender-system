# A Hybrid Recommender System for Mobile App Stores: Tackling Sparsity in User Ratings

**By: William Yeh**  
*May 31, 2025*

A comprehensive implementation of a hybrid recommender system that combines collaborative filtering with content-based filtering to enhance recommendation accuracy for users with limited historical interactions in mobile app stores.

## 🎯 Project Overview

This project addresses the challenge of sparse user-item interaction matrices in mobile app recommendation systems. The hybrid approach combines:

- **Collaborative Filtering**: Matrix factorization using SVD
- **Content-Based Filtering**: TF-IDF vectorization with K-Nearest Neighbors
- **Hybrid Model**: Weighted combination optimized on validation data

### Key Results
- **Test RMSE**: 1.6228
- **Test MAE**: 1.4811
- **Dataset Sparsity**: 99.87% (realistic real-world scenario)
- **Sample Size**: 3,000 users from 19.3M interaction dataset

## 📊 Dataset

- **Source**: [MobileRec Dataset](https://huggingface.co/datasets/recmeapp/mobilerec) (Hugging Face)
- **Scale**: 19.3 million user-app interactions
- **Apps**: ~1,500 unique applications with metadata
- **Features**: User ratings, timestamps, app descriptions, categories, developer info

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scikit-surprise datasets huggingface-hub tqdm joblib
```

### Running the Notebook

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mobile-app-recommender.git
   cd mobile-app-recommender
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Open and run the notebook**:
   ```bash
   jupyter notebook "ML2_Final_Yeh (1).ipynb"
   ```

## 📈 Methodology

### Data Processing
1. **User Sampling**: Random selection of 3,000 users from the full dataset
2. **Temporal Splitting**: Chronological split (most recent interaction for testing)
3. **Text Processing**: Clean app names and descriptions for content-based filtering

### Model Architecture

#### Collaborative Filtering
- **Algorithm**: SVD (Singular Value Decomposition)
- **Hyperparameters**: Grid search over factors, regularization, learning rate
- **Best Config**: 100 factors, 0.05 regularization, 0.002 learning rate

#### Content-Based Filtering
- **Features**: TF-IDF vectors from app names and descriptions
- **Similarity**: Cosine similarity with K-Nearest Neighbors
- **Optimal K**: 60 neighbors (tuned on validation data)

#### Hybrid Model
- **Combination**: Weighted average of CF and CBF predictions
- **Optimal Weight**: 30% content-based, 70% collaborative filtering
- **Fallback**: Pure collaborative filtering when content features unavailable

### Evaluation Strategy
- **Temporal Split**: Train on past, validate on recent, test on most recent
- **Metrics**: RMSE (primary), MAE (secondary)
- **Analysis**: Error distribution, user data dependency, prediction range

## 🔍 Key Insights

1. **Sparsity Challenge**: 99.87% sparsity reflects real-world mobile app scenarios
2. **Conservative Predictions**: Model predicts ratings in 2.69-3.52 range (avoids extremes)
3. **Data Dependency**: Users with more training interactions have lower prediction errors
4. **Hybrid Benefits**: Combination approach helps with cold-start and sparse data issues

## 📊 Results Summary

### Model Performance Comparison (Validation RMSE)
| Model | RMSE | MAE |
|-------|------|-----|
| SVD (tuned) | 1.590 | 1.445 |
| SVD++ | 1.627 | 1.449 |
| Content-Based KNN | 1.594 | 1.433 |
| **Hybrid (final)** | **1.623** | **1.481** |

### Visualizations Included
- Rating distribution analysis
- App category breakdown
- Interaction timeline
- Prediction error analysis
- User data availability vs. error correlation
- Sample recommendations for users

## 🔮 Future Improvements

- **Scale**: Extend to larger user samples and full dataset
- **Features**: Add app install counts, update frequency, review sentiment
- **Models**: Explore neural collaborative filtering approaches
- **Evaluation**: Address conservative prediction range through loss function tuning

## 📁 Repository Structure

```
mobile-app-recommender/
├── ML2_Final_Yeh (1).ipynb    # Main notebook with complete implementation
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── data/                    # Data files (created when running notebook)
├── models/                  # Saved models (created when running notebook)
└── results/                 # Output plots and results
```

## 🛠️ Technical Details

### Dependencies
- **Data Processing**: pandas, numpy, datasets (Hugging Face)
- **Machine Learning**: scikit-learn, scikit-surprise
- **Visualization**: matplotlib, seaborn
- **Utilities**: tqdm, joblib

### Computational Requirements
- **Memory**: ~4GB RAM recommended for full dataset processing
- **Time**: ~30-45 minutes for complete pipeline execution
- **Storage**: ~500MB for data and model files

## 📝 Notebook Contents

The notebook is organized into clear sections:

1. **Introduction & Objectives**
2. **Data Loading & Preprocessing**
3. **Exploratory Data Analysis**
4. **Collaborative Filtering Models**
5. **Content-Based Filtering**
6. **Hybrid Model Development**
7. **Evaluation & Results**
8. **Sample Recommendations**
9. **Future Work Discussion**

Each section includes detailed explanations, code implementation, and result analysis.

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements. Areas for contribution:
- Performance optimizations
- Additional evaluation metrics
- Extended feature engineering
- Alternative model architectures

## 📚 References

- [Surprise Library Documentation](https://surprise.readthedocs.io/)
- [MobileRec Dataset](https://huggingface.co/datasets/recmeapp/mobilerec)
- [Hybrid Recommender Systems: Survey and Experiments](https://link.springer.com/article/10.1007/s10462-013-9406-y)

---

*This project demonstrates practical application of hybrid recommender systems in addressing real-world sparsity challenges in mobile app recommendation scenarios.*