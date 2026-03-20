from textblob import TextBlob
import numpy as np

class Sentiment:
    # def __init__(self,text):
    #     self.text = text

    def classify(self, sent):
        if sent >= 0.20:
           return "Positive"
        elif sent <= -0.20:
           return "Negative"
        else:
           return "Neutral"
        
    def analyse(self,data):
        #Indentify sentiment
        analyse = TextBlob(data)
        sent = analyse.polarity
        return sent
    
    def result(self,data):
        sent = self.analyse(data)
        result = self.classify(sent)
        return result
    
    def confidence(self,data):
        sent = self.analyse(data)
        conf = np.round(abs(sent * 100),2)
        return conf


# s1 = Sentiment('Hello i am confident !')
# print(s1.analyse())