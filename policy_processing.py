import pandas as pd

def create_data(df, sid):
    data = df[df.site_id == sid]
    return data


def get_policy_texts(data):
    return list(data['policy_text'])


def clean_text(documents, cleaning_function):
    cleaned_documents = []
    
    for document in documents:
        cleaned_documents.append(cleaning_function(document))
    
    return cleaned_documents