#!/usr/bin/env python
# coding: utf-8

# ## Info-box for Toy-story 3 

# In[1]:


# Import necessary libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import sklearn
import keras 
import nltk 
import gensim
from bs4 import BeautifulSoup as bs
import requests
import cython


# In[2]:


## load the web page
r = requests.get("https://en.wikipedia.org/wiki/Toy_Story_3")
# convert it to a beautifulsoup object 
soup = bs(r.content)
# print out the html 
contents = soup.prettify()
print(contents)


# In[5]:


info_box= soup.find(class_="infobox vevent")
info_rows = info_box.find_all("tr")
for row in info_rows:
    print(row.prettify())


# In[11]:


movie_info = {}

def get_content_value(row_data):
    if row_data.find("li"):
        return [li.get_text(" ", strip=True).replace("\xa0"," ") for li in row_data.find_all("li")]
    else:
        return row_data.get_text(" ", strip=True).replace("\xa0"," ")
    

for index, row in enumerate(info_rows):
    if index == 0:
        movie_info["title"] = row.find("th").get_text(" ", strip=True)
    elif index == 1:
        continue
    else:
        content_key = row.find("th").get_text(" " , strip =True)
        content_value = get_content_value(row.find("td"))
        movie_info[content_key] = content_value
        

movie_info


# ## Getting Info-box for all movies 

# In[12]:


## load the web page
r = requests.get("https://en.wikipedia.org/wiki/List_of_Walt_Disney_Pictures_films")
# convert it to a beautifulsoup object 
soup = bs(r.content)
# print out the html 
contents = soup.prettify()
print(contents)


# In[21]:


movies = soup.select(".wikitable.sortable i")
movies[:10]


# In[14]:


def get_content_value(row_data):
    if row_data.find("li"):
        return [li.get_text(" ", strip=True).replace("\xa0"," ") for li in row_data.find_all("li")]
    elif row_data.find("br"):
        return [text for text in row_data.stripped_strings]
    else:
        return row_data.get_text(" ", strip=True).replace("\xa0"," ")

    
def clean_tags(soup):
    for tag in soup.find_all(["sup","span"]):
        tag.decompose()

def get_info_box(url):
    ## load the web page
    r = requests.get(url)
    # convert it to a beautifulsoup object 
    soup = bs(r.content)
    info_box= soup.find(class_="infobox vevent")
    info_rows = info_box.find_all("tr")    
    clean_tags(soup)

    movie_info = {}
    for index, row in enumerate(info_rows):
        if index == 0:
            movie_info["title"] = row.find("th").get_text(" ", strip=True)
        else:
            header = row.find("th")
            if header:
                content_key = row.find("th").get_text(" " , strip =True)
                content_value = get_content_value(row.find("td"))
                movie_info[content_key] = content_value
            

    return movie_info
        


# In[15]:


get_info_box("https://en.m.wikipedia.org/wiki/One_Little_Indian_(film)")


# In[16]:


## load the web page
r = requests.get("https://en.wikipedia.org/wiki/List_of_Walt_Disney_Pictures_films")
# convert it to a beautifulsoup object 
soup = bs(r.content)
movies = soup.select(".wikitable.sortable i a") # iterate through all the italics 

base_path="https://en.wikipedia.org/"

movie_info_list = []
for index,movie in enumerate(movies):
    if index % 10 == 0:
        print(index)
    try:
        relative_path = movie["href"]
        full_path = base_path + relative_path 
        title = movie["title"]
        
        movie_info_list.append(get_info_box(full_path))
        
    except Exception as e:
        print(movie.get_text())
        print(e)
    


# In[17]:


len(movie_info_list)


# ## Save /Reload the Data 

# In[4]:


import json

def save_data(title,data):
    with open(title,"w", encoding= "utf-8") as f:
        json.dump(data,f,ensure_ascii = False, indent=2)


# In[5]:


import json 

def load_data(title):
    with open(title, encoding = "utf-8") as f:
        return json.load(f)


# In[19]:


save_data("disney_data_cleaned.json", movie_info_list)


# ## Clean our data! 

# In[4]:


movie_info_list = load_data("disney_data_cleaned.json")


# ### mini tasks :
# 
# 1) clean up references --done
# 
# 2) convert running time into an integer ---done
# 
# 3) convert dates into date-time object 
# 
# 4) split up long strings -- done
# 
# 5) convert budget and box office to numbers --done
# 

# In[20]:


movie_info_list[-10]


# In[21]:


[movie.get("Running time") for movie in movie_info_list]


# In[26]:


def minutes_to_integer(running_time):
    if running_time == "N/A":
        return None
    if isinstance(running_time, list):
        entry = running_time[0]
        return int(entry.split(" ")[0])

    else:
        #is a string 
        return int(running_time.split(" ")[0])

for movie in movie_info_list:
    movie["Running time(int)"] = minutes_to_integer(movie.get("Running time","N/A"))
    


# In[27]:


movie_info_list[-10]


# In[29]:


[movie.get("Running time(int)","N/A") for movie in movie_info_list]


# In[27]:


movie_info_list[-40]


# In[31]:


print([movie.get("Budget") for movie in movie_info_list])


# In[47]:


import re

amounts= r"thousand|million|billion"
number = r"\d+(,\d{3})*\.*\d*"

word_re = rf"\${number}(-|\sto\s|–)?({number})?\s({amounts})"
value_re = rf"\${number}"

def word_to_value(word):
    value_dict = {"thousand": 1000, "million": 1000000, "billion": 1000000000}
    return value_dict[word]

def parse_word_syntax(string):
    
    value_string = re.search(number,string).group()
    value = float(value_string.replace(",",""))
    word = re.search(amounts,string, flags=re.I).group().lower()
    word_value = word_to_value(word)
    return value*word_value
   
def parse_value_syntax(string):
    value_string = re.search(number,string).group()
    value = float(value_string.replace(",",""))
    return value

def money_conversion(money):
    if money == "N/A":
        return None
    
    if isinstance(money,list):
        money = money[0]
    word_syntax = re.search(word_re,money, flags=re.I)
    value_syntax = re.search(value_re, money)
    
    if word_syntax:
        return parse_word_syntax(word_syntax.group())
    elif value_syntax:
        return parse_value_syntax(value_syntax.group())
    else:
        return None
       
        

    


# In[51]:


for movie in movie_info_list:
    movie["Budget(float)"] = money_conversion(movie.get("Budget","N/A"))
    movie["Box office(float)"] = money_conversion(movie.get("Box office","N/A"))


# In[53]:


movie_info_list[-40]


# In[56]:


# convert Dates into datetimes
print([movie.get("Release date") for movie in movie_info_list])


# In[55]:


movie_info_list[-50]


# In[61]:


# general format : June 28, 1950 
from datetime import datetime

dates= [movie.get("Release date","N/A") for movie in movie_info_list]

def clean_date(date):
    return date.split("(")[0].strip()

def date_conversion(date):
    if isinstance(date,list):
        date = date[0]
        
    if date == "N/A":
        return None 
    date_str = clean_date(date)
 
    
    fmts = ["%B %d, %Y", "%d %B %Y"]
    for fmt in fmts:
        try:      
            return datetime.strptime(date_str,fmt)
        except:
            pass
    return None
        
        

    


# In[64]:


for movie in movie_info_list:
    movie["Release date (datetime)"] = date_conversion(movie.get("Release date","N/A"))


# In[65]:


movie_info_list[50]


# In[69]:


# saving using pickle as jason cannot serialize datetimne object
import pickle
def save_data_pickle(name,data):
    with open(name, 'wb') as f:
        pickle.dump(data, f)


# In[70]:


import pickle
def load_data_pickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


# In[71]:


save_data_pickle("disney_movie_data_cleaned_more.pickle", movie_info_list)


# In[72]:


a = load_data_pickle("disney_movie_data_cleaned_more.pickle")


# In[73]:


a[5]


# ### Attach IMDB, Rotten Tomtoes,Meta scores along with Genre and number of user ratings 

# In[74]:


movie_info_list = load_data_pickle("disney_movie_data_cleaned_more.pickle")


# In[75]:


movie_info_list[-60]


# In[76]:


# API key
#key = 75c72f70
#base_path = http://www.omdbapi.com/?apikey=[yourkey]&


# In[113]:


import requests
import urllib

def get_omdb_info(title):
    base_url = "http://www.omdbapi.com/?"
    parameters = {"apikey":"ae291949", "t": title}
    params_encoded = urllib.parse.urlencode(parameters)
    full_url = base_url+params_encoded
    return requests.get(full_url).json()

def get_rotten_tomatoe_score(omdb_info):
    ratings = omdb_info.get("Ratings",[])
    for rating in ratings:
        if rating["Source"] == "Rotten Tomatoes":
            return rating["Value"]
    return None
get_omdb_info("into the woods")


# In[114]:


for movie in movie_info_list:
    title = movie["title"]
    omdb_info = get_omdb_info(title)
    movie["imdb"] = omdb_info.get("imdbRating",None)
    movie["metascore"] = omdb_info.get("Metascore",None)
    movie["rotten_tmatoes"] = get_rotten_tomatoe_score(omdb_info)
    movie["plot"] = omdb_info.get("Plot",None)
    movie["genre"] = omdb_info.get("Genre",None)
    movie["user_ratings"] = omdb_info.get("imdbVotes",None)


# In[85]:


movie_info_list[-30]


# In[115]:


save_data_pickle("disney_movie_data_final.pickle",movie_info_list)


# ### save data as csv

# In[87]:


movie_info_list[50]


# In[116]:


movie_info_copy = [movie.copy() for movie in movie_info_list]


# In[117]:


for movie in movie_info_copy:
    current_date = movie["Release date (datetime)"]
    if current_date:
        movie["Release date (datetime)"] = current_date.strftime("%B %d %Y")
    else:
        movie["Release date (datetime)"] = None


# In[118]:


save_data("disney_data_final.json", movie_info_copy)


# In[119]:


import pandas as pd


# In[120]:


df = pd.DataFrame(movie_info_list)


# In[121]:


df.head()


# In[122]:


df.to_csv("disney_movie_data_final.csv")


# In[123]:


df.info()


# In[124]:


rate = df.sort_values(["user_ratings"],ascending = False)
rate.head()


# In[7]:


def minutes_to_integer(running_time):
    if running_time == "N/A":
        return None
    if isinstance(running_time, list):
        entry = running_time[0]
        return int(entry.split(" ")[0])

    else:
        #is a string 
        return int(running_time.split(" ")[0])

for movie in movie_info_list:
    movie["Running time(int)"] = minutes_to_integer(movie.get("Running time","N/A"))
    


# In[6]:


movie_info_list = load_data("disney_data_final.json")


# In[8]:


movie_info_list[-50]


# In[9]:


df = pd.DataFrame(movie_info_list)


# In[10]:


df.to_csv("disney_movie_data_final_edit.csv")


# In[ ]:




