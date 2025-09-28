# A Hybrid Recommender System for Mobile App Stores: Tackling Sparsity in User Ratings

A comprehensive implementation of a hybrid recommender system that combines collaborative filtering with content-based filtering to enhance recommendation accuracy for users with limited historical interactions in mobile app stores.

## 🎯 Project Overview

This project addresses the challenge of sparse user-item interaction matrices in mobile app recommendation systems. The hybrid approach combines:

- **Collaborative Filtering**: Matrix factorization using SVD
- **Content-Based Filtering**: TF-IDF vectorization with K-Nearest Neighbors
- **Hybrid Model**: Weighted combination optimized on validation data

### Key Results
- **Dataset Sparsity**: High sparsity (realistic real-world scenario)
- **Sample Size**: 3,000 users from 19.3M interaction dataset
- **Performance**: Competitive results across RMSE and MAE metrics

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
   jupyter notebook "hybrid_recommender_system.ipynb"
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
- **Optimization**: Tuned configuration selected via validation performance

#### Content-Based Filtering
- **Features**: TF-IDF vectors from app names and descriptions
- **Similarity**: Cosine similarity with K-Nearest Neighbors
- **Optimization**: K value tuned on validation data

#### Hybrid Model
- **Combination**: Weighted average of CF and CBF predictions
- **Weight Optimization**: Optimal balance determined via validation data
- **Fallback**: Pure collaborative filtering when content features unavailable

### Evaluation Strategy
- **Temporal Split**: Train on past, validate on recent, test on most recent
- **Metrics**: RMSE (primary), MAE (secondary)
- **Analysis**: Error distribution, user data dependency, prediction range

## 🔍 Key Insights

1. **Sparsity Challenge**: High sparsity reflects real-world mobile app scenarios
2. **Conservative Predictions**: Model tends to avoid extreme rating predictions
3. **Data Dependency**: Users with more training interactions have lower prediction errors
4. **Hybrid Benefits**: Combination approach helps with cold-start and sparse data issues

## 📊 Results Summary

### Model Performance Comparison
Multiple models were evaluated including:
- SVD (tuned via grid search)
- SVD++ (enhanced collaborative filtering)
- Content-Based KNN (TF-IDF similarity)
- **Hybrid Model** (optimal weighted combination)

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
hybrid-recommender-system/
├── hybrid_recommender_system.ipynb    # Main notebook with complete implementation
├── requirements.txt                   # Python dependencies
├── README.md                         # This file
├── .gitignore                       # Git ignore rules
├── data/                            # Data files (created when running notebook)
├── models/                          # Saved models (created when running notebook)
└── results/                         # Output plots and results
```

## 🛠️ Technical Details

### Dependencies
- **Data Processing**: pandas, numpy, datasets (Hugging Face)
- **Machine Learning**: scikit-learn, scikit-surprise
- **Visualization**: matplotlib, seaborn
- **Utilities**: tqdm, joblib


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


## 📚 References

- [Surprise Library Documentation](https://surprise.readthedocs.io/)
- [MobileRec Dataset](https://huggingface.co/datasets/recmeapp/mobilerec)
- [Hybrid Recommender Systems: Survey and Experiments](https://link.springer.com/article/10.1007/s10462-013-9406-y)

---

*This project demonstrates practical application of hybrid recommender systems in addressing real-world sparsity challenges in mobile app recommendation scenarios.*