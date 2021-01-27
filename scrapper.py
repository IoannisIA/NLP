from newspaper import Article
import pandas as pd


# Get articles and text data from links!

text_file = open("googlenews_text_from_links.csv", "w", encoding="utf-8")
data = pd.read_csv('googlenews_clusters_and_links.csv',
                   error_bad_lines=False, delimiter=';',
                   names=['cluster_id', 'title', 'publisher', 'link']).drop_duplicates(subset=['link'])
links = data['link'].values.tolist()
titles = data['title'].values.tolist()
actual_labels = data['cluster_id'].values.tolist()
for index, link in enumerate(links):
    try:
        article = Article(link)
        article.download()
        article.parse()
        text_file.write(str(actual_labels[index])+";"+
                        titles[index]+
                        ";"+article.text.replace('\n', ' ').replace('\r', '')+"\n")
    except:
        print("error")

text_file.close()
