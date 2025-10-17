# cmse830_fds

Link to streamlit app: https://nflpenalties.streamlit.app/


What each file is and how to use it:

CMSE_830_Midtern_Project.ipynb: This is the main jupyter notebook file for the project. All of the IDA, EDA, and streamlit app construction is done in this file. Not all of the IDA and EDA is shown in the streamlit app. In order to run it properly CMSE_830_Midterm_dataset1.csv and CMSE_830_Midterm_dataset2.csv must be able to be loaded into this file. This file outputs merged_df_imputed.csv and app.py. I used python kernal version 3.12 (ipykernal) to run this code.

CMSE_830_Midterm_dataset1.csv: One of the csv files used to conduct the data analysis. Should be saved in same file as CMSE_830_Midtern_Project.ipynb so it can be loaded into the jupyter notebook. https://www.kaggle.com/datasets/nickcantalupa/nfl-team-data-2003-2023

CMSE_830_Midterm_dataset2.csv: One of the csv files used to conduct the data analysis. Should be saved in same file as CMSE_830_Midtern_Project.ipynb so it can be loaded into the jupyter notebook. https://www.kaggle.com/datasets/mattop/nfl-penalties-data-2009-2022-season?select=games.csv

merged_df_imputed.csv: This is the final dataframe used for the project. CMSE_830_Midtern_Project.ipynb creates and exports this csv file at the end of the notebook so it can be used in app.py for creating the streamlit app

app.py: This file contains all of the code for the streamlit app. CMSE_830_Midtern_Project.ipynb creates and runs this streamlit app at the end of the notebook.

requirements.txt: this file contains a list of all the libraries required to run the streamlit app. This is necessary to have in the reopsitory for the app to work.


Why I chose these datasets?

I am a huge fan of the NFL and it has a ton of publically avaliable data. Also, I was an ice hockey referee for 10 years. So, when I found a dataset containing data on penalty flags in the NFL it felt like a natural fit. I have always wondered, how much do penalties actually affect a sports game? How important is it to avoid getting penalties? Do they give a significant enough disadvantage that usually causes you to lose a game? Or, is the effect small enough that you don't need to worry about penalties? As soon as I saw this dataset all of these question began coming up in my head. The other dataset I chose contains team stats for whole seasons. I figured this would be a good way to compare penalties to other statistics that affect winning. 

What I've learned from IDA/EDA?

There were a few challenges that came up while doing IDA. First of all the dataset with penalty flag info was grouped by game and each row had information for the home team and away team. This mean't that in order to combine the datasets I would have to first separate the home and away data from each other to make sure the information was linked to the correct team. Then, I had to sum up all the data and combine the datasets so that each teams totals as home teams and away teams were combined to show totals for the whole season. The next big issue was that the teams were labeled differently in each dataset. Also, some teams changed locations and names during the period the dataset looks at. For this reason I had to map all these unique names to whatever the current name of the team is. Then, apply these new names to the dataframes and finally combine them using the team and year. This resulted in a dataframe with one row for each combination of year and team. Another portion of the IDA for this project was imputing missing penalty data. The team statistics dataset included more years than the penalty dataset did. For this reason when I combined them there was a bunch of missing values for the penalty data. I chose to use stochastic regression to fit the missing data. For EDA I started by looking at a correlation matrix including all of the numerical variables in my dataset. It was immediatly obvious that penalties had very small correlation with all of the variables related to winning. This was somewhat expected but not to such an extreme extent. Although the correlations were very weak every single one relating was negative meaning that there is some relationship it might just be very small. Penalties in football always result in a loss of yards. Total yards had a medium correlation with winning. 

What preprocessing steps I've completed?

I explained all the steps I completed for preprocessing in the IDA section above. I have pasted the same response here. First of all the dataset with penalty flag info was grouped by game and each row had information for the home team and away team. This mean't that in order to combine the datasets I would have to first separate the home and away data from each other to make sure the information was linked to the correct team. Then, I had to sum up all the data and combine the datasets so that each teams totals as home teams and away teams were combined to show totals for the whole season. The next big issue was that the teams were labeled differently in each dataset. Also, some teams changed locations and names during the period the dataset looks at. For this reason I had to map all these unique names to whatever the current name of the team is. Then, apply these new names to the dataframes and finally combine them using the team and year. This resulted in a dataframe with one row for each combination of year and team. Another portion of the IDA for this project was imputing missing penalty data. The team statistics dataset included more years than the penalty dataset did. For this reason when I combined them there was a bunch of missing values for the penalty data. I chose to use stochastic regression to fit the missing data.

What I've tried with Streamlit so far?

In streamlit I have put four of the visualization I felt were most important for highlighting the relationship between penalties and winning. I also, included a snapshot of the dataset I used and included some writing to explain the visualizations and how they relate to the overarching question. To make the visualizations interactive I made it so that all the scatterplot show you the team and the year when you hover over a datapoint. It also tels you what the exact values are. By this I mean it tells you what the exact value is on the x and the y axis because it is often hard to tell especially when it is a large scale. For the histogram I added a slidebar that changes the number of bins in the histogram. 
