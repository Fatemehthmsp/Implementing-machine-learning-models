import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.markdown("""
    <style>
    body { background: #1b1e23 !important; color: #fff !important }
    .stApp { background-color: #1b1e23; }
    .block-container { background: #22252A !important; border-radius: 18px; padding: 24px 18px}
    h1, h2, h3, h4 { color: #53baff; }
    </style>
""", unsafe_allow_html=True)

df = pd.read_csv('student-academic-performance.csv').dropna()

def exam_class(score):
    if score >= 85:
        return 'High'
    elif score >= 60:
        return 'Medium'
    else:
        return 'Low'
df['ScoreClass'] = df['Exam_Score'].apply(exam_class)
class_label_map = {'Low':0, 'Medium':1, 'High':2}
df['ScoreClassNum'] = df['ScoreClass'].map(class_label_map)

categoricals = ['Gender', 'Tutoring', 'Region', 'Parent Education']
df_encoded = pd.get_dummies(df, columns=categoricals, drop_first=True)
X = df_encoded.drop(['Exam_Score', 'ScoreClass', 'ScoreClassNum'], axis=1)
y_reg = df_encoded['Exam_Score']
y_clf = df_encoded['ScoreClassNum']

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.3, random_state=32, stratify=y_clf
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression().fit(X_train_scaled, y_reg_train)
y_pred_reg = lr.predict(X_test_scaled)

clf = LogisticRegression(max_iter=1000, multi_class='auto', solver='lbfgs')
clf.fit(X_train_scaled, y_clf_train)
y_pred_clf = clf.predict(X_test_scaled)

X_clu = df[['HoursStudied/Week', 'Attendance(%)']]
scaler_clu = StandardScaler()
X_clu_scaled = scaler_clu.fit_transform(X_clu)
kmeans = KMeans(n_clusters=4, random_state=12, n_init=10)
labels = kmeans.fit_predict(X_clu_scaled)
centers = scaler_clu.inverse_transform(kmeans.cluster_centers_)

genders = df['Gender'].unique().tolist()
tutoring_opts = df['Tutoring'].unique().tolist()
regions = df['Region'].unique().tolist()
parent_edu = df['Parent Education'].unique().tolist()

st.title("Student Academic Performance Dashboard")
st.header("Enter Student Information")
with st.form("user_form"):
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        gender = st.selectbox('Gender', genders)
        tutoring = st.selectbox('Tutoring', tutoring_opts)
    with c2:
        region = st.selectbox('Region', regions)
        parent_education = st.selectbox('Parent Education', parent_edu)
    with c3:
        hours = st.slider('Hours Studied (per week)', 0.0, 20.0, 8.0)
        attendance = st.slider('Attendance (%)', 50.0, 100.0, 75.0)
    submitted = st.form_submit_button("Analyze")

if submitted:
    user_dict = {'HoursStudied/Week': [hours], 'Attendance(%)': [attendance]}
    for col in X_train.columns:
        if 'Gender_' in col:
            user_dict[col] = [1 if f'Gender_{gender}' == col else 0]
        elif 'Tutoring_' in col:
            user_dict[col] = [1 if f'Tutoring_{tutoring}' == col else 0]
        elif 'Region_' in col:
            user_dict[col] = [1 if f'Region_{region}' == col else 0]
        elif 'Parent Education_' in col:
            user_dict[col] = [1 if f'Parent Education_{parent_education}' == col else 0]
    user_X = pd.DataFrame(user_dict)
    for col in X_train.columns:
        if col not in user_X:
            user_X[col] = 0
    user_X = user_X[X_train.columns]
    user_X_scaled = scaler.transform(user_X)
    user_score_pred = lr.predict(user_X_scaled)[0]
    user_class_pred_num = clf.predict(user_X_scaled)[0]
    user_class_pred = {v:k for k,v in class_label_map.items()}[user_class_pred_num]
    class_colors = {'High':'#00D26A', 'Medium':'#FFC300', 'Low':'#FF595E'}

    st.markdown(f"""### 🏆 Predicted Exam Score: <span style='color:#80cfff;font-size:2rem;font-weight:bold'>{user_score_pred:.1f}</span>""", unsafe_allow_html=True)
    st.markdown(f"""### 🎯 Predicted Class: <span style='background-color:{class_colors[user_class_pred]};color:black; border-radius:10px;padding:6px 20px;font-size:1.1rem;font-weight:bold'>{user_class_pred}</span>""", unsafe_allow_html=True)
    st.caption("Class labels: Low (<60), Medium (60-85), High (≥85)")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(7, 5), facecolor='#171a1c')
        palette = sns.color_palette("dark", 4)
        sns.set_theme(style="darkgrid")
        sns.regplot(x=y_reg_test, y=y_pred_reg, color='#26A69A', scatter=False, ax=ax1,
                    line_kws={'label':'Regression Line', 'lw':2, 'color':'#53baff'})
        sns.scatterplot(x=y_reg_test, y=y_pred_reg, ax=ax1, s=55, color='#212F3D', edgecolor='#53baff',
                        linewidth=0.7, alpha=0.95, label='Test Samples')
        ax1.scatter([user_score_pred], [user_score_pred], color=class_colors[user_class_pred],
                    s=220, marker='*', edgecolor='white', linewidths=2, label='You')
        ax1.set_facecolor('#22262b')
        ax1.set_xlabel('Actual Exam Score', fontsize=13, fontweight='bold', color='white')
        ax1.set_ylabel('Predicted Exam Score', fontsize=13, fontweight='bold', color='white')
        ax1.set_title('Actual vs Predicted Exam Score', fontsize=15, weight='bold', color='#53baff', pad=12)
        ax1.tick_params(colors='white')
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_color('white')
        legend1 = ax1.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
        for text in legend1.get_texts():
            text.set_color('white')
        sns.despine(ax=ax1)
        fig1.tight_layout()
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(7, 5), facecolor='#171a1c')
        sns.set_theme(style="darkgrid")
        palette = sns.color_palette("dark", 4)
        sns.scatterplot(
            x=X_clu['HoursStudied/Week'],
            y=X_clu['Attendance(%)'],
            hue=labels, palette=palette, ax=ax2,
            alpha=0.68, s=44, edgecolor='white', legend=None)
        ax2.scatter(centers[:, 0], centers[:, 1], c="#F6F930", marker='X', s=180,
                    label='Cluster Center', linewidths=2, edgecolors='black')
        ax2.scatter(hours, attendance, color=class_colors[user_class_pred], marker='*',
                    s=220, edgecolor='white', linewidths=2, label="You", zorder=10)
        ax2.set_facecolor('#22262b')
        ax2.set_xlabel('Hours Studied per Week', fontsize=13, fontweight='bold', color='white')
        ax2.set_ylabel('Attendance (%)', fontsize=13, fontweight='bold', color='white')
        ax2.set_title('KMeans Clustering of Students', fontsize=15, weight='bold', color='#53baff', pad=12)
        ax2.tick_params(colors='white')
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_color('white')
        legend2 = ax2.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.01, 1),
                             frameon=False, labelcolor='white')
        for text in legend2.get_texts():
            text.set_color('white')
        sns.despine(ax=ax2)
        fig2.tight_layout()
        st.pyplot(fig2)

    st.markdown("#### Most Similar Students to You")
    DX = df.copy()
    DX["diff"] = ((DX['HoursStudied/Week']-hours)**2 + (DX['Attendance(%)']-attendance)**2 )**0.5
    neighbors = DX.sort_values("diff").head(5)
    st.dataframe(neighbors[['Gender','Tutoring','HoursStudied/Week','Attendance(%)',
                            'Parent Education','Exam_Score','ScoreClass']]
                            .reset_index(drop=True).round(2))

    accuracy = accuracy_score(y_clf_test, y_pred_clf)
    st.markdown(
        f"<span style='color:#fff; background:#00D26A; border-radius:13px; padding: 8px 22px; font-size:1.1rem; font-weight:bold;'>"
        f"Classification Accuracy: {accuracy:.2%} </span>",
        unsafe_allow_html=True,
    )

    with st.expander("📚 Personalized Recommendations", expanded=True):
        recomend_list = []
        if user_class_pred == "0" or user_score_pred < 60:
            recomend_list.append("⏫ Increase your weekly study hours.")
            recomend_list.append("✅ Improve your attendance rate in all classes.")
            recomend_list.append("👨‍🏫 Ask for help from teachers or peers.")
            recomend_list.append("👥 Participate in group study sessions for better focus.")
        elif user_class_pred == "Medium":
            recomend_list.append("🔄 Your performance is average, aim for improvement through practice.")
            recomend_list.append("📒 Review your notes regularly and focus on weak points.")
            recomend_list.append("👂 Engage more in classroom discussions.")
        else:
            recomend_list.append("🌟 Excellent! Keep up the current strategies.")
            recomend_list.append("💡 Try advanced exercises or help others to reinforce learning.")
            recomend_list.append("🚀 Share your successful strategies with classmates.")

        if tutoring == "No" and (user_class_pred in ["Low", "Medium"] or user_score_pred < 85):
            recomend_list.append("📚 Consider enrolling in additional tutoring sessions for faster improvement.")

        for rec in recomend_list:
            st.markdown(f"<li style='font-size:1.08rem;color:#42E9F5;font-weight:500'>{rec}</li>", unsafe_allow_html=True)
