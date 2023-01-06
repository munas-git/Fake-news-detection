import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from django.shortcuts import render, redirect


##################################################################
# Instantiating and reading in necessary objects
sess = {
    "result": ""
}

stemmer = PorterStemmer()
stop_words = stopwords.words("english")
model = pickle.load(open("../pred-model.pkl", "rb"))
transformer = pickle.load(open("../news-transformer.pkl", "rb"))
##################################################################

# Create your views here.
def landing_page(request):

    if request.method == "POST":
        # Extracting data from HTML form.
        news_text = request.POST["news-text"]
        
        # Text pre-processing.
        # Dropping text case.
        news_text = news_text.lower()
        # Removal of numbers.
        news_text = re.sub(r'[0-9]+', "", news_text)
        # Removal of symbols/special characters.
        news_text = news_text.replace(",", " ").replace(":", " ").replace(".", "").replace("'", " ").replace("%", " ").replace(";", " ").replace("/", " ").replace('"', " ").replace("(", " ").replace(")", " ").replace("!", " ").replace("+", " ").replace("@", " ").replace("*", " ").replace("$", " ").replace("&", " ").replace("-", " ").replace("?", " ").replace("#", " ").replace("[", " ").replace("]", " ").replace("“", " ").replace("”", " ").replace(r"‘", " ").replace(r"’", " ").replace("'", " ").replace("  ", " ").replace("    ", " ").replace("_", " ").replace("\n", " ").replace("`", " ").replace("\t", " ")
        # Splitting text, removing stopwords and words of irrelevant length then stemming.
        news_text = " ".join([stemmer.stem(word.strip()) for word in news_text.split() if word not in stop_words if len(word) >= 3])
        # Text transformation with saved transformer (TF-IDF)
        news_text = transformer.transform([news_text])
        # Passing transformed data into model for result
        prediction = model.predict(news_text)
        result = prediction[0]
        # Saving result to session
        sess["result"] = result
        return redirect("/result")
    else:
        return render(request, 'fake_news_app/index.html')


def result_page(request):
    result = sess["result"]

    if result == 0:
        result = "Fake News"
        message = """Fake news is false or misleading information presented as news. Fake news often has the aim of damaging the reputation of a person or entity, or making money through advertising revenue.[1][2] Although false news has always been spread throughout history, the term "fake news" was first used in the 1890s when sensational reports in newspapers were common. """
    else:
        result = "Real News"
        message = """This news is true and not misleading. In order to avoid fake news, it is necessary to verify the source and that the content is true. To verify news, you can go onlie to your trusted news service providers or simply make use of this smart AI system to analyze the news."""

    return render(request, 'fake_news_app/result.html', {"result_": result, "message_": message})