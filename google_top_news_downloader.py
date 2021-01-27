import time
from pygooglenews import GoogleNews

# Get news links every 8 hours!

gn = GoogleNews(lang='gr', country='GR')
links_file = open("googlenews_clusters_and_links.csv", "w", encoding="utf-8")
group_id = 0
while 1:

    top = gn.top_news()
    for value in top['entries']:
        if len(value['sub_articles']) != 0:
            for i in value['sub_articles']:
                links_file.write(str(group_id) + ";" + i['title'] + ";" + i['publisher'] + ";" + i['url']+"\n")
            group_id += 1

    time.sleep(28000)

links_file.close()

