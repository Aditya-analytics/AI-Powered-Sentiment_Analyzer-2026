import re

class Clean_text:
    # def __init__(self,text):
    #     self.text = text
    
    def clean(self,data):
        #Lowercase text
        cleaned_text = str(data).lower()
        cleaned_text = re.sub(r"http\S+", "", cleaned_text)  # remove links
        cleaned_text = re.sub(r"[^a-z\s]", "", cleaned_text)  # remove special chars
        #remove numbers
        cleaned_text = re.sub(r"\d+", "", cleaned_text)
        #remove extra whitespace
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        #Remove extra spaces
        cleaned_text = cleaned_text.split()
        text = " ".join(cleaned_text)
        return text
    

# text = Clean_text('Sentiment : Positive😊 Hello i am confident !')

# print(text.clean())