from cleaner import Clean_text
from load_data import LoadDataset
from sentiment import Sentiment

class Pipeline :
    def __init__(self,path):
        #Load dataset
        load = LoadDataset(path)
        data = load.load_dataset()
        self.data = data

    def fit(self,col:str):
        #Clean data
        cleaner = Clean_text()
        self.data[col] = self.data[col].apply(cleaner.clean)

        #Sentiment analyse
        sent = Sentiment()
        self.data['sentiment'] = self.data[col].astype(str).apply(sent.result)
        self.data['confidence'] = self.data[col].apply(sent.confidence)
        return self.data
    
    def top(self,df,n):
        negative_df = df[df['sentiment'] == 'Negative']
    
        top_negative = negative_df.sort_values(
        by='confidence', ascending=False
        ).head(n)

        # ✅ Better format for API
        return top_negative.to_dict(orient="records")


    
# a = Pipeline(r"data\test_reviews.csv")
# result = a.fit()
# t = a.top(result,3)

# # print(result)
# print(t)

        
        
        
        

        
        
      



