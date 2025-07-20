import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import pandas as pd
import plotly.express as px

def course_treemap(data):
    course_counts = data['Course'].value_counts()
    fig, ax = plt.subplots(figsize=(12, 6))
    squarify.plot(sizes=course_counts.values, label=course_counts.index, alpha=0.8, ax=ax)
    plt.title('Student Distribution by Course (Treemap)')
    plt.axis('off')
    return fig

def cgpa_barh(data):
    fig, ax = plt.subplots()
    data.groupby('CGPA').size().plot(kind='barh', color=sns.color_palette('Set2'), ax=ax)
    ax.set_title('CGPA Distribution')
    return fig

def age_hist(data):
    fig, ax = plt.subplots()
    data['Age'].plot(kind='hist', bins=20, ax=ax, alpha=0.7)
    ax.set_title('Age')
    return fig

def gender_barh(data):
    fig, ax = plt.subplots()
    data.groupby('Gender').size().plot(kind='barh', color=sns.color_palette('Dark2'), ax=ax)
    ax.set_title('Gender Distribution')
    return fig

def year_study_barh(data):
    fig, ax = plt.subplots()
    data.groupby('Year Study').size().plot(kind='barh', color=sns.color_palette('Dark2'), ax=ax)
    ax.set_title('Year of Study Distribution')
    return fig

'''def gender_pie(data):
    gen_count = pd.DataFrame(data['Gender'].value_counts().reset_index())
    gen_count = gen_count.rename(columns = {'index':'Gender','Gender':'Number of Students'})
    gen_count['Gender'] = gen_count['Gender'].replace({0:'Female', 1:'Male'})
    fig, ax = plt.subplots(figsize=(8,6))
    ax.pie(gen_count['Number of Students'], explode=(0.015,0),
           labels=gen_count['Gender'],
           colors=['bisque','lightcoral'], autopct='%1.2f%%',
           startangle=120)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    ax.set_title("Gender Distribution")
    return fig'''

def gender_pie(data):
    # Map gender for labels
    label_map = {0: 'Female', 1: 'Male'}
    
    # Count and map gender values
    gen_count = data['Gender'].value_counts().rename_axis('Gender_Code').reset_index(name='Number of Students')
    gen_count['Gender'] = gen_count['Gender_Code'].map(label_map)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.pie(gen_count['Number of Students'], explode=(0.015, 0),
           labels=gen_count['Gender'],
           colors=['bisque', 'lightcoral'], autopct='%1.2f%%',
           startangle=120)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    ax.set_title("Gender Distribution")
    return fig

def countplot_by_year(data, condition):
    fig, ax = plt.subplots()
    sns.countplot(data=data, x='Year Study', hue=condition, palette='Set2', ax=ax)
    ax.set_title(f'{condition} by Year of Study')
    ax.set_ylabel('Number of Students')
    return fig

def correlation_heatmap(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data[['Age', 'Depression', 'Anxiety', 'Panic Attack', 'Treatment']].corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    return fig

def conditions_by_gender(data):
    gender_group = data.groupby(['Gender'])[['Depression', 'Anxiety', 'Panic Attack']].sum()
    gender_group.index = gender_group.index.map({0: 'Female', 1: 'Male'})
    fig, ax = plt.subplots()
    gender_group.plot(kind='bar', stacked=True, colormap='viridis', ax=ax)
    ax.set_title('Mental Health Conditions by Gender')
    ax.set_ylabel('Number of Cases')
    return fig

def cgpa_by_condition(data, condition):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x=condition, y='CGPA', palette='pastel', ax=ax)
    xticklabels = {
        'Depression': ['No Depression', 'Depressed'],
        'Anxiety': ['No Anxiety', 'Anxiety'],
        'Panic Attack': ['No Panic Attack', 'Panic Attack']
    }
    ax.set_xticklabels(xticklabels.get(condition, ['No', 'Yes']))
    ax.set_title(f'CGPA Distribution by {condition} Status')
    return fig

def course_bar_by_condition(data, condition, title, ylabel):
    fig, ax = plt.subplots(figsize=(12, 6))
    course_stat = data.groupby('Course')[condition].mean().sort_values(ascending=False)
    course_stat.plot(kind='bar', color='steelblue', ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Course')
    plt.xticks(rotation=90)
    return fig

def treatment_by_condition_bar(data, condition):
    ct = pd.crosstab(data[condition], data['Treatment'], normalize='index')
    fig, ax = plt.subplots()
    ct.plot(kind='bar', stacked=True, colormap='Accent', ax=ax)
    ax.set_title(f'{condition} vs Seeking Treatment')
    ax.set_ylabel('Proportion')
    ax.set_xticklabels(['No', 'Yes'], rotation=0)
    ax.legend(title='Sought Treatment', labels=['No', 'Yes'])
    return fig

def age_group_countplot(data, condition):
    fig, ax = plt.subplots()
    sns.countplot(data=data, x='Age Group', hue=condition, palette='coolwarm', ax=ax)
    ax.set_title(f'{condition} Cases Across Age Groups')
    return fig

def depression_year_plotly(data):
    fig = px.histogram(data, x='Year Study', color='Depression',
                       barmode='group', title='Depression Cases by Year of Study',
                       color_discrete_map={0:'lightblue', 1:'red'})
    return fig

def scatter_3d(data, x, y, z, color='Cluster', title=''):
    fig = px.scatter_3d(data, x=x, y=y, z=z, color=color, title=title)
    return fig
