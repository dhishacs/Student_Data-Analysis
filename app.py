import streamlit as st
from data_prep import load_and_preprocess_data
import viz
import mining

DATA_PATH = "Student Mental health.csv"  # Adjust if your CSV is in a different location
data = load_and_preprocess_data(DATA_PATH)

st.sidebar.write("**Columns:**", data.columns.tolist())
st.sidebar.write("**Null counts:**")
null_counts = data.isnull().sum().to_frame(name='Null Count')
st.sidebar.dataframe(null_counts)


# Display Data Head
st.header("Data Preview")
st.dataframe(data.head(),hide_index=True)

# Tabs for visualization
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Demographics", 
    "Mental Health Prevalence", 
    "CGPA & Year Analysis", 
    "By Course", 
    "Age Group & Treatment", 
    "Data Mining"
])

with tab1:
    st.subheader("Treemap: Student Distribution by Course")
    st.pyplot(viz.course_treemap(data))
    st.subheader("CGPA Bar Plot")
    st.pyplot(viz.cgpa_barh(data))
    st.subheader("Age Histogram")
    st.pyplot(viz.age_hist(data))
    st.subheader("Gender Bar Chart")
    st.pyplot(viz.gender_barh(data))
    st.subheader("Year of Study Bar Chart")
    st.pyplot(viz.year_study_barh(data))
    st.subheader("Gender Pie Chart")
    st.pyplot(viz.gender_pie(data))

with tab2:
    st.subheader("Depression by Year of Study")
    st.pyplot(viz.countplot_by_year(data, 'Depression'))
    st.subheader("Anxiety by Year of Study")
    st.pyplot(viz.countplot_by_year(data, 'Anxiety'))
    st.subheader("Panic Attacks by Year of Study")
    st.pyplot(viz.countplot_by_year(data, 'Panic Attack'))
    st.subheader("Treatment by Year of Study")
    st.pyplot(viz.countplot_by_year(data, 'Treatment'))
    st.subheader("Correlation Heatmap")
    st.pyplot(viz.correlation_heatmap(data))
    st.subheader("Conditions by Gender")
    st.pyplot(viz.conditions_by_gender(data))

with tab3:
    st.subheader("CGPA by Depression")
    st.pyplot(viz.cgpa_by_condition(data, 'Depression'))
    st.subheader("CGPA by Anxiety")
    st.pyplot(viz.cgpa_by_condition(data, 'Anxiety'))
    st.subheader("CGPA by Panic Attack")
    st.pyplot(viz.cgpa_by_condition(data, 'Panic Attack'))
    st.subheader("Depression by Year (Plotly)")
    st.plotly_chart(viz.depression_year_plotly(data), use_container_width=True)

with tab4:
    st.subheader("Depression Rate by Course")
    st.pyplot(viz.course_bar_by_condition(data, 'Depression', 'Average Depression Rate by Course', 'Proportion of Depressed Students'))
    st.subheader("Anxiety Rate by Course")
    st.pyplot(viz.course_bar_by_condition(data, 'Anxiety', 'Average Anxiety Rate by Course', 'Proportion of Students with Anxiety'))
    st.subheader("Panic Attack Rate by Course")
    st.pyplot(viz.course_bar_by_condition(data, 'Panic Attack', 'Average Panic Attack Rate by Course', 'Proportion of Students with Panic Attack'))
    st.subheader("Treatment Rate by Course")
    st.pyplot(viz.course_bar_by_condition(data, 'Treatment', 'Average Treatment Rate by Course', 'Proportion of Students taking Treatment'))

with tab5:
    st.subheader("Treatment-Seeking: Depression")
    st.pyplot(viz.treatment_by_condition_bar(data, 'Depression'))
    st.subheader("Treatment-Seeking: Anxiety")
    st.pyplot(viz.treatment_by_condition_bar(data, 'Anxiety'))
    st.subheader("Treatment-Seeking: Panic Attack")
    st.pyplot(viz.treatment_by_condition_bar(data, 'Panic Attack'))
    st.subheader("Anxiety by Age Group")
    st.pyplot(viz.age_group_countplot(data, 'Anxiety'))
    st.subheader("Depression by Age Group")
    st.pyplot(viz.age_group_countplot(data, 'Depression'))
    st.subheader("Panic Attack by Age Group")
    st.pyplot(viz.age_group_countplot(data, 'Panic Attack'))
    st.subheader("Treatment by Age Group")
    st.pyplot(viz.age_group_countplot(data, 'Treatment'))

with tab6:
    st.subheader("Association Rule Mining (Apriori)")
    rules = mining.association_rule_mining(data)
    st.dataframe(rules)
    st.subheader("K-Means Clustering (3D Visualization)")
    clustered_data = mining.kmeans_clustering(data, n_clusters=3)
    st.plotly_chart(viz.scatter_3d(clustered_data, 'Age', 'CGPA', 'Anxiety', color='Cluster', title='Clusters by Age, CGPA, Anxiety'), use_container_width=True)
    st.plotly_chart(viz.scatter_3d(clustered_data, 'Age', 'CGPA', 'Depression', color='Cluster', title='Clusters by Age, CGPA, Depression'), use_container_width=True)
    st.plotly_chart(viz.scatter_3d(clustered_data, 'Age', 'CGPA', 'Panic Attack', color='Cluster', title='Clusters by Age, CGPA, Panic Attack'), use_container_width=True)
