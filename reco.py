import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

class SkinCareRecommender:
    def __init__(self, data_path):
        # Load the dataset
        self.df = pd.read_csv(data_path)

        # Preprocessing skin types and concerns
        self.df['Skintype'] = self.df['Skintype'].fillna("").apply(lambda x: [i.strip().lower() for i in x.split(',')])
        self.df['Concern'] = self.df['Concern'].fillna("").apply(lambda x: [i.strip().lower() for i in x.split(',')])

        # One-hot encode the skin types and concerns using MultiLabelBinarizer
        self.mlb_skin = MultiLabelBinarizer()
        self.mlb_concern = MultiLabelBinarizer()

        # Encoding the skin types and concerns
        self.skin_encoded = pd.DataFrame(self.mlb_skin.fit_transform(self.df['Skintype']), columns=self.mlb_skin.classes_)
        self.concern_encoded = pd.DataFrame(self.mlb_concern.fit_transform(self.df['Concern']), columns=self.mlb_concern.classes_)

        # Normalize the ratings
        self.scaler = MinMaxScaler()
        self.ratings_scaled = self.scaler.fit_transform(self.df[['Rating']].fillna(0))

         # Combine all features
        self.features_df = pd.concat([
            self.skin_encoded.reset_index(drop=True),
            self.concern_encoded.reset_index(drop=True),
            pd.DataFrame(self.ratings_scaled, columns=['Rating']).reset_index(drop=True)
        ], axis=1)
        
    def get_recommended_products(self, skin_type, concerns, top_n=5):
         # Create feature vector for input
        skin_input = [skin_type.lower()]  # Make sure it is a list
        concerns_input = [c.lower() for c in concerns]
         
        skin_encoded = pd.DataFrame(self.mlb_skin.transform([skin_input]), columns=self.mlb_skin.classes_)
        concern_encoded = pd.DataFrame(self.mlb_concern.transform([concerns_input]), columns=self.mlb_concern.classes_)

        # Combine input features
        input_features = pd.concat([skin_encoded, concern_encoded, pd.DataFrame([[0]], columns=['Rating'])], axis=1)

        # Calculate similarity
        similarity_scores = cosine_similarity(input_features, self.features_df)

        # Get top N products
        top_indices = similarity_scores[0].argsort()[-top_n:][::-1]

        recommended_products = self.df.iloc[top_indices]

        return recommended_products[['Product', 'product_url', 'product_pic', 'Rating']]
