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
        message = """What is fake news? it is intentionally misleading or fabricated information presented as if it were real news. It is often spread through social media or other online platforms, and can be harmful because it can mislead people and spread false information. It is important for people to be able to critically evaluate the sources of the information they encounter and to be able to distinguish between reliable sources and fake news.

How to verify your news
    1. Use me, I am a smart AI that is capable of analyzing any piece of news and returning accurate predictions for you
    2. Check the source: Is the source of the news a reputable one? Is it known for producing reliable information?
    3. Look for other sources: Check other sources for same news. It is more likely to be true if multiple reputable sources are reporting it
    4. Consider the purpose: Is the news trying to sell you something? Is it trying to promote a particular agenda or point of view? This could affect the objectivity of the news."""
    else:
        result = "Real News"
        message = """Real news is accurate, fact-based information that is reported by reliable sources. It is intended to inform the public about events and issues that are important and relevant to society. Real news is typically produced by journalists who follow ethical principles and standards, such as truthfulness, accuracy, fairness, and independence. These principles help to ensure that the information being reported is reliable and trustworthy.

Sources of real news include:
    1. Newspapers: Many newspapers have a long history of producing accurate and reliable news.
    2. Television news: Many television news programs are produced by respected organizations and have a reputation for accuracy.
    3. Radio news: Radio news programs are often produced by respected organizations and can be a good source of real news
    4. Online news: Online news sources such as websites, apps, and social media accounts run by reputable news organizations"""

    return render(request, 'fake_news_app/result.html', {"result_": result, "message_": message})