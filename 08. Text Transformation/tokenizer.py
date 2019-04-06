from nltk.stem.porter import PorterStemmer

def tokenizer(text):
    return text.split()

def porter_tokenizer(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]
