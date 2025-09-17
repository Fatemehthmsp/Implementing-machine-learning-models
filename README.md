#  Student Academic Performance Analysis & Intelligent Dashboard

##  Overview
This project analyzes and predicts student academic performance using **data mining** and **machine learning** techniques.  
It also includes an interactive **Streamlit dashboard** for score prediction, performance classification, and personalized study recommendations.

## Dataset
- **Source:** Student Academic Performance file(csv)
- **Records:** 4,001 students
- **Features:**
  - Gender (Male/Female)
  - Hours studied per week
  - Tutoring status
  - Region (Urban/Rural)
  - Attendance percentage
  - Parental education level
  - Exam score (target)

## Methods & Models
- **Data Preprocessing:** Missing value handling, duplicate removal, one-hot encoding, standardization, target feature engineering (Low/Medium/High classes).
- **Regression:** Linear Regression, Decision Tree
- **Classification:** Logistic Regression, Decision Tree
- **Clustering:** K-Means
- **Recommendation Systems:** User-based and item-based
- **Association Rule Mining:** Apriori

## Key Features
- Predict **exact exam scores** and classify performance levels.
- Identify similar students for peer-based insights.
- Provide tailored study recommendations.

To run the web application, type and run the following line in the terminal:
streamlit run student_app.py
