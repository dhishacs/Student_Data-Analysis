import pandas as pd

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    #data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    # Rename columns for easier handling
    data_column = {
        'Choose your gender':'Gender','What is your course?':'Course',
        'Your current year of Study':'Year Study','What is your CGPA?':'CGPA',
        'Do you have Depression?':'Depression','Marital status':'Marital',
        'Do you have Anxiety?':'Anxiety','Do you have Panic attack?':'Panic Attack',
        'Did you seek any specialist for a treatment?':'Treatment'
    }
    data.rename(columns=data_column, inplace=True)

    # Encode categorical variables
    data['Depression'] = data['Depression'].replace(['Yes', 'No'], [1, 0]).infer_objects(copy=False)
    data['Anxiety'] = data['Anxiety'].replace(['Yes', 'No'], [1, 0]).infer_objects(copy=False)
    data['Panic Attack'] = data['Panic Attack'].replace(['Yes', 'No'], [1, 0]).infer_objects(copy=False)
    data['Marital'] = data['Marital'].replace(['Yes', 'No'], [1, 0]).infer_objects(copy=False)
    data['Treatment'] = data['Treatment'].replace(['Yes', 'No'], [1, 0]).infer_objects(copy=False)
    data['Gender'] = data['Gender'].replace(['Male', 'Female'], [1, 0]).infer_objects(copy=False)
    data['Year Study'] = data['Year Study'].replace(
        ['year 1', 'year 2', 'year 3', 'year 4'],
        ['Year 1', 'Year 2', 'Year 3', 'Year 4']
    )
    # Replace course inconsistencies
    data['Course'].replace(
        ['koe','Laws','Koe','Irkhs','Kirkhs','Engine','Benl','engin','Pendidikan islam','psychology','Fiqh','Islamic education'],
        ['KOE','Law','KOE','KIRKHS','KIRKHS','Engineering','BENL','Engineering','Pendidikan Islam','Psychology','Fiqh fatwa','Islamic Education'],
        inplace=True
    )
    # Fill missing age
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Age'] = data["Age"].astype(int)

    # Drop Timestamp if present
    if 'Timestamp' in data.columns:
        data.drop('Timestamp', axis=1, inplace=True)

    # Age group column for later visualization
    data['Age Group'] = pd.cut(data['Age'], bins=[15, 18, 21, 24, 27], labels=['15-18','19-21','22-24','25-27'])
    return data
