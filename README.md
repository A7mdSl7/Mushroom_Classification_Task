# 🍄 Mushroom Classification Task

## 📌 Overview
This project focuses on building and evaluating multiple machine learning models to classify mushrooms as **edible** or **poisonous** based on their physical characteristics.  
The goal is to compare different algorithms, perform feature selection, and analyze the most important features that influence classification.

---

## 📊 Dataset
- **Name:** Mushroom Dataset  
- **Target Variable:** Edible (`e`) vs. Poisonous (`p`)  
- **Features:** Various categorical and numerical attributes describing mushroom properties (cap diameter, stem width, gill color, etc.).  
- **Preprocessing Steps:**
  - Handling missing values
  - Encoding categorical features
  - Feature scaling for numerical features
  - Balancing dataset using **SMOTE**

---

## 🔎 Project Workflow
1. **Data Preprocessing**
   - Encoding categorical variables
   - Scaling numerical features
   - Balancing dataset with SMOTE  

2. **Feature Selection**
   - Filter methods (Correlation, Chi-Square)
   - Wrapper methods (RFE)
   - Embedded methods (Tree-based models)

3. **Modeling**
   - Logistic Regression  
   - Decision Tree  
   - Random Forest  
   - Support Vector Machine (SVM)  
   - K-Nearest Neighbors (KNN)  
   - Naive Bayes  

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score  
   - Cross-validation for stability check  
   - Confusion Matrix  
   - Feature Importance (Decision Tree, Random Forest)  
   - ROC-AUC curves for model comparison  

5. **Model Saving**
   - Trained models are saved as `.pkl` files in `saved_models/` folder.  
   - Note: **KNN model** is large in size (>50MB). It is stored separately on Google Drive.

---

## 📈 Results
### Performance Comparison
Models were compared before and after feature selection.  
Metrics used: **Accuracy, Precision, Recall, F1-score, ROC-AUC**.  

Random Forest and KNN performed the best overall with very high accuracy and stability.

### ROC Curves
ROC-AUC curves were plotted to visualize performance of all models.
