from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json


# tempate_query = """
# {{
#     "uid": "{uid}"
    
# }}
# """

category_tag = ['price', 'category_1', 'category_2', 'category_3', 'category_4',
       'category_5', 'category_6', 'tag_accessories', 'tag_clothing',
       'tag_decoration', 'tag_gift/flower basket', 'tag_sports',
       'tag_technology', 'tag_toys']

# temp = tempate_query.format(uid="0000CM")
# print(temp)

app = Flask(__name__)

def get_recommendations(clf, input_features, threshold=0.01):
    input_df = input_features.copy()
    predicted_labels = clf.predict(input_df)
    predicted_probs = clf.predict_proba(input_df)
    indices = np.where(predicted_probs > threshold)[1]
    labels = clf.classes_[indices]
    probs = predicted_probs[:, indices].ravel()
    results_df = pd.DataFrame({'name': labels, 'probability': probs})
    results_df = results_df.sort_values(by='probability', ascending=False)
    return list(results_df['name'][:5])



@app.route('/recommend', methods=['POST'])
def recommend_products():
   # Load model and label encoder object
    with open('model.pickle10', 'rb') as f:
        clf, le= pickle.load(f)
    
    # Get the input data from the request
    input_data = request.get_json()
    input_data =json.loads(input_data)
    # preprocessing
    s = ""
    # input_data= preprocessing(input_data)
    
    input_data = pd.DataFrame(input_data, index=[0], columns=category_tag)
    input_data = input_data.fillna(0)
    
    # Get the recommendations
    recommended_products = get_recommendations(clf, input_data)
    recommended_products = le.inverse_transform(recommended_products)
    # Return the recommendations as a JSON response
    print(recommended_products.tolist())
    return jsonify({'recommended_products': recommended_products.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
