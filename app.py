# creating a .py file for the app
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# loading in final dataframe
merged_df_imputed = pd.read_csv("merged_df_imputed.csv")

st.title("How Do Penalties Affect a Teams Success in the NFL?")

# showing the dataframe
st.write("### Dataset Preview")
st.write("##### This dataset includes team metrics for full seasons. Each row in the dataset includes information for one team for one year. This data will be used to see how penalties during a season impacts a teams outcome.")

st.write("Below is a list of all dataset columns with short descriptions:")

# Creating dictionary of column descriptions
column_descriptions = {
    "Year": "The season year.",
    "Accepted_Penalty_Count": "Total number of accepted penalties in all games.",
    "Penalty_Yards": "Total yards penalized in the season.",
    "Team_Penalty_Count": "Total penalties committed by the team.",
    "Team_Penalty_Yards": "Total penalty yards against the team.",
    "wins": "Number of games won by the team.",
    "losses": "Number of games lost by the team.",
    "win_loss_perc": "Winning percentage for the season.",
    "points": "Total points scored by the team.",
    "points_opp": "Total points scored by opponents.",
    "points_diff": "Point differential (points scored − points allowed).",
    "g": "Total games played in the season.",
    "total_yards": "Total offensive yards gained by the team.",
    "plays_offense": "Total number of offensive plays run.",
    "yds_per_play_offense": "Average yards gained per offensive play.",
    "turnovers": "Total number of turnovers committed (interceptions + fumbles lost).",
    "fumbles_lost": "Number of times the team lost possession via fumble.",
    "first_down": "Total number of first downs achieved.",
    "pass_cmp": "Number of completed passes.",
    "pass_att": "Number of pass attempts.",
    "pass_yds": "Total passing yards gained.",
    "pass_td": "Number of passing touchdowns.",
    "pass_int": "Number of interceptions thrown.",
    "pass_net_yds_per_att": "Net yards gained per pass attempt (including sacks).",
    "pass_fd": "Number of first downs achieved via passing.",
    "rush_att": "Total rushing attempts.",
    "rush_yds": "Total rushing yards gained.",
    "rush_td": "Number of rushing touchdowns.",
    "rush_yds_per_att": "Average rushing yards per attempt.",
    "rush_fd": "Number of first downs achieved via rushing.",
    "score_pct": "Percentage of drives that resulted in scores.",
    "turnover_pct": "Percentage of drives that ended in turnovers.",
    "exp_pts_tot": "Total expected points contributed by the team.",
    "Team": "Team name."
}

# Displaying all columns vertically
for col in merged_df_imputed.columns:
    desc = column_descriptions.get(col, "Description not available.")
    st.markdown(f"**{col}** — {desc}")
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

# creating an interative plot that shows the distribution of the percentage of total yards lost die to penalties
merged_df_imputed['percent_yards_lost'] = (
    merged_df_imputed['Team_Penalty_Yards'] /
    (merged_df_imputed['Team_Penalty_Yards'] + merged_df_imputed['total_yards'])
)

st.sidebar.header("Plot Settings")
bins = st.sidebar.slider("Number of bins", min_value=5, max_value=50, value=20)
show_kde = st.sidebar.checkbox("Show KDE Curve", value=True)

mean_ratio = merged_df_imputed['percent_yards_lost'].mean()
st.write(f"**Average percent of yards lost to penalties:** {mean_ratio:.2%}")

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

st.pyplot(fig)

