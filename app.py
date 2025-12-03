# creating a .py file for the app
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go



# loading in final dataframes
stats = pd.read_csv("stats.csv")
flags_orignal = pd.read_csv("CMSE_830_Midterm_dataset2.csv")
flags = pd.read_csv("flags.csv")
merged_df_imputed = pd.read_csv("merged_df_imputed.csv")
merged_df = pd.read_csv("merged_df.csv")
superbowls = pd.read_csv("superbowls.csv")
engineered_features = pd.read_csv("engineered_features.csv")

st.title("How Do Penalties Affect a Teams Success in the NFL?")

tabs = st.tabs(["Introduction", "IDA", "EDA", "FE/ML", "Results/Discussion", "Conclusion"])

with tabs[0]:
    st.write("# Introduction")
    st.write("##### I am a huge NFL fan and I love watching games with my friends. People can get very emotional when watching a game especially when their favorite team is playing. Often times, this leads to very big reactions when something positive or negative happens during the game including penalties. I have always wondered, when a friend of mine starts kicking and screaming on the floor after a penalty is called against his team is this an overreaction or is there some merit to his reaction. My goal for this project is to explore the impact of penalties on a teams chances of winning. To do this I will investigate the impact of penalties on winning indivudually and compared to other metrics that may contribute to winning. ")
    st.write("##### The purpose of this app is to educate people on whether penalties actually have a significant affect on a teams chances of winning. It goes through the process of using data to find an answer to this question. This information can be used any time you encounter someone claiming that a penalty is the reason their team is going to lose.")
    st.write("##### In these next sections I will cover my initial data analysis, exploratory data analysis, feature engineering and machine learning, discuss the results, and wrap things up with a conclusion. Please go through the tabs from left to right to follow the flow of the project from start to finish.")

with tabs[1]: # Starting the IDA tab

    st.write("# IDA")

    st.write("### Data Sets")
    st.write("##### The first step of this project was reading in all three of the data sets.")
    st.write("##### This data set has team statistics for every year and every team for the years 2003-2023.")
    st.dataframe(stats.head())
    st.write("##### This data set has penalty totals for each game from 2009-2022.")
    st.dataframe(flags_orignal.head())
    st.write("##### This dataset shows which teams won the superbowl from 2003-2023.")
    st.dataframe(superbowls.head())
    st.write("### Data Processing/Transformation")
    st.write("##### The goal is to combine the first two data sets so that we have penalty and stat totals for each team from each season spanning from 2003-2023. The first step is changing the flags data set from per game to per season and by team. To do this the home team data needs to be separated from the away team data making two different data sets. Then the numerical columns should be grouped by team and year summing all the numerical columns. This left me with two datasets each having penalty sums for each team in each year. The last step was adding together the home team and away team datasets to get full season totals. The resulting dataset included team penalty totals for each year as shown below:")
    st.dataframe(flags.head())
    st.write("##### The next step was combining the first two datasets by team and year. The year had the same format in both data sets but the teams names were different. There was an additional level of complexity here because some teams have changed names and locations over the time period that our data is from. This means some teams had up to 5 different names over this span of time. For this reason the names in both data sets needed to be standardized before the data sets could be combined.")

    # Compacting team naming system 
    name_map = {
        # AFC East
        **dict.fromkeys(['Patriots', 'New England', 'New England Patriots'], 'New England Patriots'),
        **dict.fromkeys(['Dolphins', 'Miami', 'Miami Dolphins'], 'Miami Dolphins'),
        **dict.fromkeys(['Bills', 'Buffalo', 'Buffalo Bills'], 'Buffalo Bills'),
        **dict.fromkeys(['Jets', 'N.Y. Jets', 'NY Jets', 'New York Jets'], 'New York Jets'),

        # AFC North
        **dict.fromkeys(['Ravens', 'Baltimore', 'Baltimore Ravens'], 'Baltimore Ravens'),
        **dict.fromkeys(['Bengals', 'Cincinnati', 'Cincinnati Bengals'], 'Cincinnati Bengals'),
        **dict.fromkeys(['Steelers', 'Pittsburgh', 'Pittsburgh Steelers'], 'Pittsburgh Steelers'),
        **dict.fromkeys(['Browns', 'Cleveland', 'Cleveland Browns'], 'Cleveland Browns'),

        # AFC South
        **dict.fromkeys(['Colts', 'Indianapolis', 'Indianapolis Colts'], 'Indianapolis Colts'),
        **dict.fromkeys(['Titans', 'Tennessee', 'Tennessee Titans', 'Houston Oilers'], 'Tennessee Titans'),
        **dict.fromkeys(['Jaguars', 'Jacksonville', 'Jacksonville Jaguars'], 'Jacksonville Jaguars'),
        **dict.fromkeys(['Texans', 'Houston', 'Houston Texans'], 'Houston Texans'),

        # AFC West
        **dict.fromkeys(['Chiefs', 'Kansas City', 'Kansas City Chiefs'], 'Kansas City Chiefs'),
        **dict.fromkeys(['Broncos', 'Denver', 'Denver Broncos'], 'Denver Broncos'),
        **dict.fromkeys(['Raiders', 'Oakland', 'Oakland Raiders', 'Los Angeles Raiders', 'Las Vegas', 'Las Vegas Raiders'], 'Las Vegas Raiders'),
        **dict.fromkeys(['Chargers', 'San Diego', 'San Diego Chargers', 'LA Chargers', 'Los Angeles Chargers'], 'Los Angeles Chargers'),

        # NFC East
        **dict.fromkeys(['Eagles', 'Philadelphia', 'Philadelphia Eagles'], 'Philadelphia Eagles'),
        **dict.fromkeys(['Cowboys', 'Dallas', 'Dallas Cowboys'], 'Dallas Cowboys'),
        **dict.fromkeys(['Commanders', 'Washington', 'Washington Redskins', 'Washington Football Team', 'Washington Commanders'], 'Washington Commanders'),
        **dict.fromkeys(['Giants', 'N.Y. Giants', 'NY Giants', 'New York Giants'], 'New York Giants'),

        # NFC North
        **dict.fromkeys(['Packers', 'Green Bay', 'Green Bay Packers'], 'Green Bay Packers'),
        **dict.fromkeys(['Vikings', 'Minnesota', 'Minnesota Vikings'], 'Minnesota Vikings'),
        **dict.fromkeys(['Bears', 'Chicago', 'Chicago Bears'], 'Chicago Bears'),
        **dict.fromkeys(['Lions', 'Detroit', 'Detroit Lions'], 'Detroit Lions'),

        # NFC South
        **dict.fromkeys(['Panthers', 'Carolina', 'Carolina Panthers'], 'Carolina Panthers'),
        **dict.fromkeys(['Saints', 'New Orleans', 'New Orleans Saints'], 'New Orleans Saints'),
        **dict.fromkeys(['Buccaneers', 'Tampa Bay', 'Tampa Bay Buccaneers'], 'Tampa Bay Buccaneers'),
        **dict.fromkeys(['Falcons', 'Atlanta', 'Atlanta Falcons'], 'Atlanta Falcons'),

        # NFC West
        **dict.fromkeys(['Cardinals', 'Arizona', 'Arizona Cardinals'], 'Arizona Cardinals'),
        **dict.fromkeys(['Seahawks', 'Seattle', 'Seattle Seahawks'], 'Seattle Seahawks'),
        **dict.fromkeys(['49ers', 'San Francisco', 'San Francisco 49ers'], 'San Francisco 49ers'),
        **dict.fromkeys(['Rams', 'St. Louis', 'St. Louis Rams', 'LA Rams', 'Los Angeles Rams'], 'Los Angeles Rams'),
    }
    st.write("##### Type or select any version of a team name — historical or short — and get the standardized full name. Make sure to capitalize the team name you use.")

    # Adding a dropdown or free text input
    user_input = st.text_input("Enter a team name (Make sure it is capitalized):", placeholder="e.g. NY Jets, Oakland, Washington Redskins")

    if user_input:
        normalized = name_map.get(user_input.strip(), "Unknown Team")
        st.success(f"**Standardized Name:** {normalized}")

    st.write("##### This is a list of all of the final standardized names used for both datasets")

    with st.expander("Show all standardized team names"):
        st.write(sorted(set(name_map.values())))

    st.write("##### After combining the first two data sets this is what the result looks like.")
    st.dataframe(merged_df.head())
    st.write("### Data Imputation")
    st.write("##### Now that the first two data sets have been combined there are many rows with missing values. This is because the stats dataset covers years from 2003-2023 while the flags dataset only goes from 2009-2022. To fix this I will use stochastic regression to impute the missing values. This will allow for the data to be imputed accurately while still having same variance. The standard deviation of the imputed columns are as follows: Accepted_Penalty_Count residual std = 5.9150, Penalty_Yards residual std = 52.3693, Team_Penalty_Count residual std = 4.1602, Team_Penalty_Yards residual std = 38.1223. After imputation the data set looks like this and it is ready for EDA. I also added the Superbowl Champion column from the superbowls data set to the final data frame to use for visualization")
    st.dataframe(merged_df_imputed.head())

with tabs[2]:
    st.write("# EDA")
    st.write("##### In this section I will explore the data using a variety of plots. All plots will be shown here regardless of if they turn out to be useful for answering the overarching project question or not. The plots that turn out to be the most relevant will be used in the results/discussion section.")

    st.write("##### When doing exploratory data analysis I always like to start by looking at a correlation matrix to see where I might be able to find relationships within the data. As you can see here, the penalty variables have very small correlations with all the other variables so finding obvious relationships may be a difficult task. For this correlation matrix there are a lot of variables and the small size makes it hard to read. To help with this you can hover your cursor over the matrix and it will tell you the variables you are hovering over and their correlation.")
    # Computing correlation matrix
    corr = merged_df_imputed.corr(numeric_only=True)

    # Creating interactive Plotly heatmap
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale='RdBu_r',
        aspect='auto',
        title="Correlation Plot of Numeric Variables"
    )

    fig.update_layout(
        width=1000,
        height=700,
        title_x=0.25,
        title_font_size=20
    )

    # Displaying in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.write("##### This is a more user friendly option than the massive correlation matrix. Here you can select the variables you want to look at and it will show you the distributions of those variables and the correlation scatterplot for each combination.")

    # selecting variables
    default_vars = ['Team_Penalty_Count', 'Team_Penalty_Yards', 'wins', 'win_loss_perc', 'points_diff']
    available_vars = list(merged_df_imputed.select_dtypes(include='number').columns)

    selected_vars = st.multiselect(
        "Select numeric variables to include in the pairplot:",
        options=available_vars,
        default=default_vars,
        key="pairplot_vars"
    )

    # Generating the plot only if at least 2 variables selected 
    if len(selected_vars) >= 2:
        fig = sns.pairplot(merged_df_imputed[selected_vars], diag_kind='kde')
        fig.fig.suptitle("Relationships Among Selected Variables", y=1.02)
        st.pyplot(fig)
    else:
        st.warning("Please select at least two numeric variables to display the pairplot.")

    st.write("##### I am most interested in the relatinship between penalty yards and winning because these two variables are directly related to my overall project question. For this reason I chose to plot the correlation scatter plot for these two variables and include the correlation line. You can hover over each point to see the team, year, and specific values for that point.")

    # making a correlation plot for wins and penalty yards
    x = merged_df_imputed['wins']
    y = merged_df_imputed['Team_Penalty_Yards']

    slope, intercept = np.polyfit(x, y, 1)
    r_value = np.corrcoef(x, y)[0, 1]
    r_squared = r_value**2
    equation = f"y = {slope:.2f}x + {intercept:.2f}   |   correlation = {r_value:.2f}"

    # Computing fitted line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    fig = px.scatter(
        merged_df_imputed,
        x='wins',
        y='Team_Penalty_Yards',
        hover_data=['Team', 'Year'],  # show team + year on hover
        opacity=0.7,
        title="Relationship Between Wins and Team Penalty Yards",
    )

    # Adding regression line
    fig.add_scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        name='Regression Line',
        line=dict(color='red')
    )

    # Annotating the equation on the plot
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.05, y=0.95,
        text=equation,
        showarrow=False,
        font=dict(size=14, color="red")
    )

    fig.update_layout(
        xaxis_title="Wins",
        yaxis_title="Team Penalty Yards",
        template="simple_white",
        hovermode="closest"
    )

    st.plotly_chart(fig, use_container_width=True, key="corr")
    st.write(f"**Regression Equation:** {equation}")

    st.write("##### This plot show the distribution of the percentage of total yards that are lost due to penalties. This information is helpful for grasping how much losing penalty yards affects a teams total yards. You can use the slider to change how many bins the histogram has.")

    # Calculate metric
    merged_df_imputed['percent_yards_lost'] = (
        merged_df_imputed['Team_Penalty_Yards'] /
        (merged_df_imputed['Team_Penalty_Yards'] + merged_df_imputed['total_yards'])
    )

    # Creating the initial plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        merged_df_imputed['percent_yards_lost'],
        bins=20,  # temporary default
        kde=True,
        color='green',
        ax=ax
    )
    ax.set_title("Distribution of Percent of Yards Lost Due to Penalties", fontsize=14)
    ax.set_xlabel("Percent of Total Yards Lost")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle='--', alpha=0.4)

    # Adding controls directly below the plot
    st.markdown("##### Plot Settings")
    bins = st.slider("Number of bins", min_value=5, max_value=50, value=20, key="bins_main")
    show_kde = st.checkbox("Show KDE Curve", value=True, key="checkbox_main")

    # Updating the plot with new settings
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        merged_df_imputed['percent_yards_lost'],
        bins=bins,
        kde=show_kde,
        color='green',
        ax=ax
    )
    ax.set_title("Distribution of Percent of Yards Lost Due to Penalties", fontsize=14)
    ax.set_xlabel("Percent of Total Yards Lost")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle='--', alpha=0.4)

    # Displaying figure
    st.pyplot(fig)

    # Displaying summary info
    mean_ratio = merged_df_imputed['percent_yards_lost'].mean()
    st.write(f"**Average percent of yards lost to penalties:** {mean_ratio:.2%}")

    st.write("##### This plot is related to the histogram above. It shows how the percent of total yards lost due to penalties has changed over the seasons. As you can see the it remains between 13% and 17% making it a pretty consistent time trend. You can use the sliding bar above the plot to adjust which years are shown in the plot.")

    # Adding user controls 
    year_range = st.slider(
        "Select Year Range",
        min_value=int(merged_df_imputed['Year'].min()),
        max_value=int(merged_df_imputed['Year'].max()),
        value=(2003, 2023),
        key="year_slider"
    )

    show_grid = st.checkbox("Show Grid", value=True, key="show_grid")
    show_points = st.checkbox("Show Data Points", value=True, key="show_points")

    # Computing yearly ratios
    yearly_ratio = (
        merged_df_imputed.groupby('Year')['Team_Penalty_Yards'].sum() /
        merged_df_imputed.groupby('Year')['total_yards'].sum()
    )

    # Filtering by year range
    yearly_ratio = yearly_ratio.loc[
        (yearly_ratio.index >= year_range[0]) & (yearly_ratio.index <= year_range[1])
    ]

    # Creating plot 
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        yearly_ratio.index,
        yearly_ratio.values * 100,  # convert to percent
        marker='o' if show_points else '',
        color='steelblue',
        linewidth=2
    )
    ax.set_title("League-Wide Trend of Yards Lost to Penalties (2003–2023)", fontsize=14)
    ax.set_ylabel("Percent of Yards Lost (%)", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.grid(show_grid, linestyle='--', alpha=0.5)

    # Displaying plot in Streamlit 
    st.pyplot(fig)

    st.write("##### This plot shows the average number of wins for each team and the average number of penalty yards for each team over the 20 year span of the data. This graph clearly shows that the teams with the most penalty yards typically also have the most losses. To see precise values hover your cursor over the graph.")
    # Aggregating by team
    team_summary = merged_df_imputed.groupby('Team').agg({
        'Team_Penalty_Yards': 'mean',
        'wins': 'mean'
    }).reset_index()

    # Sorting by penalties
    team_summary = team_summary.sort_values('Team_Penalty_Yards')

    # Computing linear regression for straight trend line
    x = team_summary['Team_Penalty_Yards']
    y = team_summary['wins']
    slope, intercept = np.polyfit(x, y, 1)
    trend_line = slope * x + intercept

    # creating a plot showing wins and penalty yards for each team over all seasons
    fig = go.Figure()

    # creating bar chart for penalties
    fig.add_trace(go.Bar(
        x=team_summary['Team'],
        y=team_summary['Team_Penalty_Yards'],
        name='Avg Team Penaltiy Yards',
        marker_color='salmon',
        yaxis='y1',
        hovertemplate="<b>%{x}</b><br>Avg Penalties: %{y:.2f}<extra></extra>"
    ))

        # adding line for average wins
    fig.add_trace(go.Scatter(
        x=team_summary['Team'],
        y=team_summary['wins'],
        mode='lines+markers',
        name='Avg Wins',
        marker=dict(color='blue', size=8),
        line=dict(color='blue', width=2),
        yaxis='y2',
        hovertemplate="<b>%{x}</b><br>Avg Wins: %{y:.2f}<extra></extra>"
    ))
    # adding regression trend line to highlight trend
    fig.add_trace(go.Scatter(
        x=team_summary['Team'],
        y=trend_line,
        mode='lines',
        name='Trend Line',
        line=dict(color='black', dash='dash'),
        yaxis='y2',
        hovertemplate="<b>%{x}</b><br>Trend Wins: %{y:.2f}<extra></extra>"
    ))

        # customizing the layout
    fig.update_layout(
            title=dict(
                text="Average Team Penalty Yards vs Wins (2003–2023)",
                x=0.15,
                font=dict(size=20)
            ),
            xaxis=dict(title='Team', tickangle=45),
            yaxis=dict(title='Avg Team Penalty Yards', color='salmon'),
            yaxis2=dict(
                title='Avg Wins',
                overlaying='y',
                side='right',
                color='blue'
            ),
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.1,
                xanchor='center',
                x=0.5,
                font=dict(size=14)
            ),
            hovermode='x unified',
            template='plotly_white',
            width=2200,
            height=600,
            margin=dict(l=80, r=80, t=100, b=150)
        )

    st.plotly_chart(fig, use_container_width=True, key = "plot")

    st.write("##### This plot shows the distributions of team penalty counts across seasons using box plots. This plot would be useful for comparing a few teams to see how their penalty counts differ. You can select which teams you would like included in the plot.")
    # Letting user choose variables interactively 
    teams = merged_df_imputed['Team'].unique()
    selected_teams = st.multiselect(
        "Select teams to include:",
        options=teams,
        default=teams,
        key="team_boxplot"
    )

    if len(selected_teams) > 0:
        filtered_df = merged_df_imputed[merged_df_imputed['Team'].isin(selected_teams)]

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(
            data=filtered_df,
            x='Team',
            y='Team_Penalty_Count',
            palette='coolwarm',
            ax=ax
        )
        ax.set_title("Distribution of Team Penalty Counts Across Seasons", fontsize=14)
        ax.set_xlabel("Team")
        ax.set_ylabel("Penalty Count per Season")
        plt.xticks(rotation=90)
        st.pyplot(fig)
    else:
        st.warning("Please select at least one team to display the boxplot.")

    st.write("##### This plot shows the scatter plot comparing wins and team penalty yards. The unique thing about this plot is that it highlights all of the super bowl winners. The super bowl winner is considered to be one of the best teams from each season so this plot helps highlight how the best teams differ from all the other teams. You can choose which super bowl winners to display by choosing the year that each champion won.")
    # Super Bowl winners data 
    super_bowl_winners = [
        {"Year": 2003, "Team": "Tampa Bay Buccaneers", "SuperBowl_Winner": 1},
        {"Year": 2004, "Team": "New England Patriots", "SuperBowl_Winner": 1},
        {"Year": 2005, "Team": "New England Patriots", "SuperBowl_Winner": 1},
        {"Year": 2006, "Team": "Pittsburgh Steelers", "SuperBowl_Winner": 1},
        {"Year": 2007, "Team": "Indianapolis Colts", "SuperBowl_Winner": 1},
        {"Year": 2008, "Team": "New York Giants", "SuperBowl_Winner": 1},
        {"Year": 2009, "Team": "Pittsburgh Steelers", "SuperBowl_Winner": 1},
        {"Year": 2010, "Team": "New Orleans Saints", "SuperBowl_Winner": 1},
        {"Year": 2011, "Team": "Green Bay Packers", "SuperBowl_Winner": 1},
        {"Year": 2012, "Team": "New York Giants", "SuperBowl_Winner": 1},
        {"Year": 2013, "Team": "Baltimore Ravens", "SuperBowl_Winner": 1},
        {"Year": 2014, "Team": "Seattle Seahawks", "SuperBowl_Winner": 1},
        {"Year": 2015, "Team": "New England Patriots", "SuperBowl_Winner": 1},
        {"Year": 2016, "Team": "Denver Broncos", "SuperBowl_Winner": 1},
        {"Year": 2017, "Team": "New England Patriots", "SuperBowl_Winner": 1},
        {"Year": 2018, "Team": "Philadelphia Eagles", "SuperBowl_Winner": 1},
        {"Year": 2019, "Team": "New England Patriots", "SuperBowl_Winner": 1},
        {"Year": 2020, "Team": "Kansas City Chiefs", "SuperBowl_Winner": 1},
        {"Year": 2021, "Team": "Tampa Bay Buccaneers", "SuperBowl_Winner": 1},
        {"Year": 2022, "Team": "Los Angeles Rams", "SuperBowl_Winner": 1},
        {"Year": 2023, "Team": "Kansas City Chiefs", "SuperBowl_Winner": 1},
    ]

    # Merging data 
    winners_df = pd.DataFrame(super_bowl_winners)
    df = pd.merge(merged_df_imputed, winners_df, on=["Year", "Team"], how="left")
    df["SuperBowl_Winner"] = df["SuperBowl_Winner"].fillna(0).astype(int)

    # Adding inline filter controls 
    st.markdown("### 🔍 Filter Options")
    col1, col2 = st.columns([1, 3])

    with col1:
        st.write("")  # spacer

    with col2:
        years = sorted(df["Year"].unique())
        selected_years = st.multiselect(
            "Select Years to Display",
            options=years,
            default=years,
            key="year_selector_scatter"
        )

    # Filtering data 
    filtered_df = df[df["Year"].isin(selected_years)]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=filtered_df,
        x="wins",
        y="Team_Penalty_Yards",
        hue="SuperBowl_Winner",
        palette={0: "green", 1: "red"},
        alpha=0.7,
        ax=ax
    )
    ax.set_title("Wins vs Team Penalty Yards (Super Bowl Winners Highlighted)", fontsize=14)
    ax.set_xlabel("Wins")
    ax.set_ylabel("Team Penalty Yards")
    ax.grid(True, linestyle="--", alpha=0.5)

    st.pyplot(fig)

with tabs[3]:

    importances_df1 = pd.read_csv("importances_df1.csv")
    importances_df2 = pd.read_csv("importances_df2.csv")

    st.write("# Feature Engineering and Machine Learning")
    st.write("### Feature Engineering")
    st.write("##### For feature engineering I created a few new variables that could help in predicting a teams wins. The first of which is passing yards per attempt (passing yards/passing attempts). This new variable is good for expressing how efficient the pass game is for an offense. The second was Rushing yards per attempt (rushing yards/rushing attempts). This new variable is good for expressing how efficient the running game is for an offense. The third was total touchdowns (passing touchdowns + rushing touchdowns) and the fourth was total first downs (passing first downs + rushing first downs). These new variables are good for looking at the success of an offense as a whole rather than the passing and rushing games individually.")
    st.write("##### I also conducted data encoding. The only categorical variable I have in the data is the team name. In order to include this variable in the machine learning models it had to be converted into a form that is digestible for the models. This added a column to the data set for each team and included a value of true or false based on whether that row belonged to that team or not. This way the categorical variable could be treated like a numerical variable.")
    st.write("##### Below is a snapshot of the dataframe that includes all of the engineered features:")
    st.dataframe(engineered_features.head())

    st.write("### Machine Learning")
    st.write("##### After conducting EDA I was able to see that there is a small relationship between between winning and penalty yards. The question now is how much of a relationship is there. The goal of this section is to quantify that relationship into a numeric value that can tell us exactly how much penalty yards affects winning compared to the other stats.")
    st.write("#### Linear regression:")
    st.write("##### I fit a linear regression model using the entire data set to see how much penalty yards contributed to predicting wins. When scaled to help with comparability, the coefficient for penalty yards was 0.0247. Since it was scaled this means that for every 40 standard deviations in penalty yards you should expect to see an increase of one win. This means that based on the Linear model, penalty yards have little to no affect on winning.")
    st.write("#### Random Forest/Gradient Boosted Regressor:")
    st.write("##### The results of the Random Forest and Gradient Boosted Regressor models were very similar so I will show their results here. The random forest model found that penalty yards had a 0.003993% affect on predicting wins in the model. Similarly the Gradient Boosted Regressor model found that penalty yards had a 0.003115% affect on predicting wins in the model. These percent values indicated what number of trees that the model created, included penalty yards on them. Since these numbers are so low they indicate that the penalty yards has little to no affect on predicting wins.")
    st.write("##### The plots below show the importance of each variable in the models. Take note that I used a log scale for this plot because one variable dominate all of the rest making it hard to see a difference between the rest of the variables on the plot. The red line highlights our variable of interest penalty yards. As you can see penalty yards ranks somewhere in the middle of the plot indicating that there are worse variables in terms of importance but considering the log scale none of them have a significant importance except for the top variable.")

    # Sort importance values
    importances_sorted = importances_df1.sort_values(by="Importance", ascending=False)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 14))

    # Colors: highlight penalty yards
    colors = ["red" if feat == "Team_Penalty_Yards" else "gray" 
              for feat in importances_sorted["Feature"]]

    # Plot
    ax.barh(importances_sorted["Feature"], importances_sorted["Importance"], color=colors)
    ax.invert_yaxis()  # largest at top
    ax.set_xscale("log")  # log scale

    # Formatting
    ax.set_yticks(range(len(importances_sorted)))
    ax.set_yticklabels(importances_sorted["Feature"], fontsize=10)

    ax.set_title("Feature Importances for Random Forest Model (Log Scale)", fontsize=14)
    ax.set_xlabel("Importance (log scale)")
    ax.set_ylabel("Feature")

    plt.tight_layout()

    # Streamlit display
    st.pyplot(fig)

    # Sort importance values
    importances_sorted = importances_df2.sort_values(by="Importance", ascending=False)

   # Create figure
    fig, ax = plt.subplots(figsize=(10, 14))

    # Colors: highlight penalty yards
    colors = ["red" if feat == "Team_Penalty_Yards" else "gray" 
              for feat in importances_sorted["Feature"]]

    # Plot
    ax.barh(importances_sorted["Feature"], importances_sorted["Importance"], color=colors)
    ax.invert_yaxis()  # largest at top
    ax.set_xscale("log")  # log scale

    # Formatting
    ax.set_yticks(range(len(importances_sorted)))
    ax.set_yticklabels(importances_sorted["Feature"], fontsize=10)

    ax.set_title("Feature Importances for Random Forest Model (Log Scale)", fontsize=14)
    ax.set_xlabel("Importance (log scale)")
    ax.set_ylabel("Feature")

    plt.tight_layout()

    # Streamlit display
    st.pyplot(fig)

with tabs[4]:
    # showing the dataframe
    st.write("# Results/Discussion")
    st.write("### Dataset Preview")
    st.write("##### This is the final dataset after conducting IDA. Each row in the dataset includes information for one team for one year. This is the data that was used to see how penalties during a season impacts a teams outcome.")
    st.dataframe(merged_df_imputed.head())

    # making a correlation plot for wins and penalty yards
    x = merged_df_imputed['wins']
    y = merged_df_imputed['Team_Penalty_Yards']

    slope, intercept = np.polyfit(x, y, 1)
    r_value = np.corrcoef(x, y)[0, 1]
    r_squared = r_value**2
    equation = f"y = {slope:.2f}x + {intercept:.2f}   |   correlation = {r_value:.2f}"

    # Computing fitted line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    fig = px.scatter(
        merged_df_imputed,
        x='wins',
        y='Team_Penalty_Yards',
        hover_data=['Team', 'Year'],  # show team + year on hover
        opacity=0.7,
        title="Relationship Between Wins and Team Penalty Yards",
    )

    # Adding regression line
    fig.add_scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        name='Regression Line',
        line=dict(color='red')
    )

    # Annotating the equation on the plot
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.05, y=0.95,
        text=equation,
        showarrow=False,
        font=dict(size=14, color="red")
    )

    fig.update_layout(
        xaxis_title="Wins",
        yaxis_title="Team Penalty Yards",
        template="simple_white",
        hovermode="closest"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.write(f"**Regression Equation:** {equation}")

    st.write("##### This scatterplot shows that there is almost no correlation between wins and Penalty yards directly. However, this does not necessarily mean that there is no causation. In the next plot we will show this same relationship with a different method.")

    # Aggregating by team
    team_summary = merged_df_imputed.groupby('Team').agg({
        'Team_Penalty_Yards': 'mean',
        'wins': 'mean'
    }).reset_index()

    # Sorting by penalties
    team_summary = team_summary.sort_values('Team_Penalty_Yards')

    # Computing linear regression for straight trend line
    x = team_summary['Team_Penalty_Yards']
    y = team_summary['wins']
    slope, intercept = np.polyfit(x, y, 1)
    trend_line = slope * x + intercept

    # creating a plot showing wins and penalty yards for each team over all seasons
    fig = go.Figure()

    # creating bar chart for penalties
    fig.add_trace(go.Bar(
        x=team_summary['Team'],
        y=team_summary['Team_Penalty_Yards'],
        name='Avg Team Penaltiy Yards',
        marker_color='salmon',
        yaxis='y1',
        hovertemplate="<b>%{x}</b><br>Avg Penalties: %{y:.2f}<extra></extra>"
    ))

        # adding line for average wins
    fig.add_trace(go.Scatter(
        x=team_summary['Team'],
        y=team_summary['wins'],
        mode='lines+markers',
        name='Avg Wins',
        marker=dict(color='blue', size=8),
        line=dict(color='blue', width=2),
        yaxis='y2',
        hovertemplate="<b>%{x}</b><br>Avg Wins: %{y:.2f}<extra></extra>"
    ))

        # adding regression trend line to highlight trend
    fig.add_trace(go.Scatter(
        x=team_summary['Team'],
        y=trend_line,
        mode='lines',
        name='Trend Line',
        line=dict(color='black', dash='dash'),
        yaxis='y2',
        hovertemplate="<b>%{x}</b><br>Trend Wins: %{y:.2f}<extra></extra>"
    ))

        # customizing the layout
    fig.update_layout(
            title=dict(
                text="Average Team Penalty Yards vs Wins (2003–2023)",
                x=0.15,
                font=dict(size=20)
            ),
            xaxis=dict(title='Team', tickangle=45),
            yaxis=dict(title='Avg Team Penalty Yards', color='salmon'),
            yaxis2=dict(
                title='Avg Wins',
                overlaying='y',
                side='right',
                color='blue'
            ),
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.1,
                xanchor='center',
                x=0.5,
                font=dict(size=14)
            ),
            hovermode='x unified',
            template='plotly_white',
            width=2200,
            height=600,
            margin=dict(l=80, r=80, t=100, b=150)
        )

    st.plotly_chart(fig, use_container_width=True)

    st.write("##### This plot clearly shows that the teams with more penalty yards have less wins. The dashed black line smooths out the noisy data to reveal the underlying trend in the data. This plot proves that there is at least some relationship between the penalty yards and wins. This relationship may be better explained by including total yards in the discussion.")

    # showing the relationship between total yards and wins
    x = merged_df_imputed['wins']
    y = merged_df_imputed['total_yards']

    slope, intercept = np.polyfit(x, y, 1)
    r_value = np.corrcoef(x, y)[0, 1]
    r_squared = r_value**2
    equation = f"y = {slope:.2f}x + {intercept:.2f}   |   correlation = {r_value:.2f}"

    # Computing fitted line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    fig = px.scatter(
        merged_df_imputed,
        x='wins',
        y='total_yards',
        hover_data=['Team', 'Year'],  # show team + year on hover
        opacity=0.7,
        title="Relationship Between Wins and Total Yards",
    )

    # Adding regression line
    fig.add_scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        name='Regression Line',
        line=dict(color='red')
    )

    # Annotating the equation on the plot
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.05, y=0.95,
        text=equation,
        showarrow=False,
        font=dict(size=14, color="red")
    )
    fig.update_layout(
        xaxis_title="Wins",
        yaxis_title="total_yards",
        template="simple_white",
        hovermode="closest"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("##### The consequence of getting a penalty is always a loss of yards but the amount of yard varies based on the type of penalty. This means that getting a penalty directly reduces the total yards of a team, which in turn indirectly affects if the team wins or not. ")
    st.write("##### On average 13.7% of the yards a team earn end up being lost due to penalties. A team loses an average of 747 yards due to penalties each season. However, teams earn an average of 5464 yards per season. This goes to show that teams earn way more yards than they lose due to penalties which is why penalty yards has a small correlation with winning.")

    # Calculating metrics
    merged_df_imputed['percent_yards_lost'] = (
        merged_df_imputed['Team_Penalty_Yards'] /
        (merged_df_imputed['Team_Penalty_Yards'] + merged_df_imputed['total_yards'])
    )

    # Displaying summary info
    mean_ratio = merged_df_imputed['percent_yards_lost'].mean()
    st.write(f"**Average percent of yards lost to penalties:** {mean_ratio:.2%}")

    # Creating the initial plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        merged_df_imputed['percent_yards_lost'],
        bins=20,  # temporary default
        kde=True,
        color='green',
        ax=ax
    )
    ax.set_title("Distribution of Percent of Yards Lost Due to Penalties", fontsize=14)
    ax.set_xlabel("Percent of Total Yards Lost")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle='--', alpha=0.4)

    # Adding controls directly below the plot
    st.markdown("##### Plot Settings")
    bins = st.slider("Number of bins", min_value=5, max_value=50, value=20, key="bins_secondary")
    show_kde = st.checkbox("Show KDE Curve", value=True, key="checkbox_secondary")

    # Updating the plot with new settings
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        merged_df_imputed['percent_yards_lost'],
        bins=bins,
        kde=show_kde,
        color='green',
        ax=ax
    )
    ax.set_title("Distribution of Percent of Yards Lost Due to Penalties", fontsize=14)
    ax.set_xlabel("Percent of Total Yards Lost")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle='--', alpha=0.4)

    # displaying figure
    st.pyplot(fig)

    st.write("##### The results of the Machine Learning models show that penalty yards have about 0.0035% importance in predicting wins when all possible variables are considered. This suggests that while the relationship between penalty yards and wins does exist it is obviously not a direct relationship. The variable with the highest importance in the machine learning model was point differential. Penalty yards is connected to point differential through a series of factors. Penalty yards contributes to a teams total yards, a teams total yards contributes to how many points a team scores, and how many points a team scores contributed to the teams point differential. This is a helpful way of thinking about why penalty yards has such a small impact on winning. There are a lot a variables that are in effect during a football game and penalty yards is just one of them. So while there may be a connection between penalty yards and winning this connection is a very distant one.")

with tabs[5]:
    st.write("# Conclusion")
    st.write("##### In conclusion, it turns out that penalties on their own do not have a significant impact on teams chances of winning. There are a lot of factors that go into winning a football game. Penalties are a very small portion of what goes on during a football game. Also, both teams get penalized so penalty yards often cancel each other out or get close to even by the end of the game. I have found that for these reasons penalties do not have any significant impact on winning a football game. So, next time one of my friends throws a fit over a penalty during a football game I will show them this data and remind them that it really doesn't matter and that they should probably spend their time worrying about other things.")



