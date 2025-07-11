"""
SURVIVOR WINNERS ANALYSIS

This analysis examines data from the reality TV show Survivor to identify patterns
and characteristics that distinguish winners from other contestants. Using gameplay
statistics, demographic data, and strategic metrics, we aim to uncover what makes
the "ultimate" Survivor winner.

RESEARCH QUESTIONS:
- What gameplay strategies are most effective for winning?
- How do winner characteristics vary across different eras of the show?
- What demographic factors correlate with success?
- Is there a definitive "winning formula" or is success highly variable?

METHODOLOGY:
The analysis uses contestant data including voting records, challenge performance,
advantage usage, and demographic information. Players are categorized by play style
(Physical, Strategic, Social, Advantage-Heavy) and game statistics are normalized
per tribal council to account for varying game lengths.

KEY METRICS:
- Votes Received per Tribal: Measures how much of a target a player was
- Correct Vote Rate: Strategic awareness and information gathering
- Individual Immunity Rate: Physical capability and clutch performance
- Advantage Usage: Modern game adaptation and resource management
- Tribal Attendance: Game longevity and deep run capability

The analysis reveals both consistent patterns among winners and significant variance
that demonstrates Survivor's unpredictable nature.
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import ttest_ind
from job_categories import job_categories

print("FINDING THE ULTIMATE SURVIVOR WINNER")
print("=" * 50)

filepath = r"C:\Users\becca\OneDrive\Desktop\Portfolio\Survivor\Voting Stats Plus.csv"
print("\nInitial Data Exploration\n")

# Read in CSV
survivors = pd.read_csv(filepath, encoding='latin-1')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

# Drop unnamed index column if it exists
if 'Unnamed: 0' in survivors.columns:
    survivors = survivors.drop('Unnamed: 0', axis=1)

print(f"Dimensions: {survivors.shape}")
print(f"\nFirst 5 rows:")
print(survivors.head())

print(f"\nColumn names: {survivors.columns.tolist()}")
print(f"\nData types:\n{survivors.dtypes}")
print(f"\nNull values:\n{survivors.isnull().sum()}")
print(f"\nUnique values:\n{survivors.nunique()}")
print(f"\nNumerical description:\n{survivors.describe()}")

# Remove empty columns
# survivors = survivors.drop(columns=['mergetribecolor', 'advantagesplayed'], axis=1)

# Fill in null values
# survivors = survivors.fillna()

# Add job type category
job_to_category = {}
for category, jobs in job_categories.items():
    for job in jobs:
        job_to_category[job] = category

survivors['jobcategory'] = survivors['occupation'].map(job_to_category)

# Add grouped ages to df
def group_ages(row):
    if row['age'] < 20:
        return 'Teens'
    elif row['age'] < 30:
        return '20s'
    elif row['age'] < 40:
        return '30s'
    elif row['age'] < 50:
        return '40s'
    else:
        return '50+'

survivors['agegroup'] = survivors.apply(group_ages, axis=1)

# Add column for play style
def style_of_play(row):
    if row['individualimmunites'] >= 3 or row['tribeimmunities'] >= 4:
        return 'Physical'
    elif row['votesnegated'] >= 1 or row['advantagesplayed'] >= 1:
        return 'Advantage-Heavy'
    elif row['votescast'] >= 6 or (row['correctlyvoted'] / (row['votescast'] + 1)) > 0.65:
        return 'Strategic'
    else:
        return 'Social'

survivors['style'] = survivors.apply(style_of_play, axis=1)

# Map season names to numbers
season_map = {'Borneo': 1, 'Australian Outback': 2, 'Africa': 3, 'Marquesas': 4, 'Thailand': 5,
    'The Amazon': 6, 'Pearl Islands': 7, 'All-Stars': 8, 'Vanuatu': 9, 'Palau': 10, 'Guatemala': 11,
    'Panama': 12, 'Cook Islands': 13, 'Fiji': 14, 'China': 15, 'Micronesia': 16, 'Gabon': 17,
    'Tocantins': 18, 'Samoa': 19, 'Heroes vs. Villains': 20, 'Nicaragua': 21, 'Redemption Island': 22,
    'South Pacific': 23, 'One World': 24, 'Philippines': 25, 'Caramoan': 26, 'Blood vs. Water': 27,
    'Cagayan': 28, 'San Juan del Sur': 29, 'Worlds Apart': 30, 'Cambodia': 31, 'Kaôh R?ng': 32, 'Millenials vs. Gen X': 33,
    'Game Changers': 34, 'Heroes vs. Healers vs. Hustlers': 35, 'Ghost Island': 36, 'David vs. Goliath': 37,
    'Edge of Extinction': 38, 'Island of the Idols': 39, 'Winners at War': 40, 'Survivor 41': 41,
    'Survivor 42': 42, 'Survivor 43': 43, 'Survivor 44': 44, 'Survivor 45': 45, 'Survivor 46': 46,
    'Survivor 47': 47, 'Survivor 48': 48
}

survivors['seasonnum'] = survivors['seasonplayed'].map(season_map)

# Classify seasons by eras
def classify_era(season):
    if season <= 11:
        return 'Old School'
    elif season <= 26:
        return 'Dark'
    elif season <= 40:
        return 'Advantage'
    else:
        return 'New'

survivors['era'] = survivors['seasonnum'].apply(classify_era)

# Normalize data based on how long player is in game
survivors['votesreceived_pertribal'] = survivors['votesrecieved'] / survivors['tribalsattended']
survivors['votescast_pertribal'] = survivors['votescast'] / survivors['tribalsattended']
survivors['correctvote_rate'] = survivors['correctlyvoted'] / (survivors['votescast'] + 1)
survivors['advantagesplayed_pertribal'] = survivors['advantagesplayed'] / survivors['tribalsattended']
survivors['immunities_pertribal'] = survivors['individualimmunites'] / survivors['tribalsattended']

print(f"After cleaning: {survivors.shape}\n")

# Separate winners vs. non-winners
survivors['results'] = np.where(survivors['finalplacement'] == 1, 'winner', 'nonwinner')

# Create separate data set for winners vs. nonwinners
winners = survivors[survivors['results'] == 'winner']
nonwinners = survivors[survivors['results'] == 'nonwinner']

# Do winners perform differently vs. nonwinners?
grouped = (survivors.groupby('results')[[
    'votesreceived_pertribal',
    'votescast_pertribal',
    'correctvote_rate',
    'advantagesplayed_pertribal',
    'immunities_pertribal',
    'timesswapped',
    'tribalsattended',
    'playersonseason']]
           .mean().round(3))

print("\nComparison of Winners vs. Nonwinners Actions:\n")
print(grouped.T)

plot_df = survivors.melt(
    id_vars='results',
    value_vars=[
        'votesreceived_pertribal',
        'votescast_pertribal',
        'correctvote_rate',
        'advantagesplayed_pertribal',
        'immunities_pertribal',
        'timesswapped',
        'tribalsattended',
        'playersonseason'
    ],
    var_name='metric',
    value_name='value'
)

# Plot of winners vs. nonwinners actions
plt.figure(figsize=(10, 6))
sns.barplot(data=plot_df, x='metric', y='value', hue='results', errorbar='sd')
plt.xticks(rotation=45)
plt.title("Winners vs Non-Winners: Survivor Gameplay Metrics")
plt.ylabel("Average per Action")
plt.xlabel("Metric")
plt.legend(title='Player Type')
plt.tight_layout()
plt.show()

# Snapshot of a typical winner's behavior
win_stats = winners[['votesreceived_pertribal',
    'votescast_pertribal',
    'correctvote_rate',
    'advantagesplayed_pertribal',
    'immunities_pertribal',
    'timesswapped',
    'tribalsattended',
    'playersonseason']].describe()

print(f"\nWinner action stats:\n{win_stats}")

# Find the elite winners
elite_winners = winners[
    (winners['correctvote_rate'] > 0.8) &
    (winners['votesreceived_pertribal'] < 1) &
    (winners['immunities_pertribal'] > 0.2)
]

elite_winners = elite_winners.reset_index(drop=True)
print(f"\n Elite Survivor winners:\n{elite_winners}")

# Snapshot of an elite winner's behavior
elite_stats = elite_winners[['votesreceived_pertribal',
    'votescast_pertribal',
    'correctvote_rate',
    'advantagesplayed_pertribal',
    'immunities_pertribal',
    'timesswapped',
    'tribalsattended',
    'playersonseason']].describe()

print(f"\nElite winner action stats:\n{elite_stats}")

avg_winner = win_stats.loc['mean']
avg_elite = elite_stats.loc['mean']

comparison = pd.DataFrame({
    'avg_winner': avg_winner,
    'avg_elite_winner': avg_elite,
    'difference': (avg_elite - avg_winner).round(3)
})

print("\nElite vs. Average Winner Comparison:")
print(comparison)

# Actions by style of play
style_stats = survivors.groupby('style')[[
    'votesreceived_pertribal',
    'correctvote_rate',
    'advantagesplayed_pertribal',
    'immunities_pertribal'
]].mean().round(3)
print(f"\nAction averages by style of play:\n{style_stats}")

# How does chance of winning change based on play-style?
players_style = survivors['style'].value_counts()
win_style = winners['style'].value_counts()

style_win_rate = pd.DataFrame({
    'total_players': players_style,
    'winners': win_style
})

style_win_rate['win_rate'] = (style_win_rate['winners'] / style_win_rate['total_players']).round(3)
style_win_rate = style_win_rate.sort_values('win_rate', ascending=False)
print(f"\nWinners by style of play:\n{style_win_rate}")

# Do the qualities that make a good winner change based on era?
era_wins = (
    winners[winners['finalplacement'] == 1]
    .groupby('era')[['correctlyvoted', 'votesrecieved', 'individualimmunites', 'tribeimmunities', 'tribalsattended']]
    .mean()
)
print(f"\nWin averages by era:\n {era_wins}\n")

# TODO: break down ideal winner by era

# all_stats = survivors.groupby('era')['correctlyvoted'].mean()
# winner_stats = survivors[survivors['finalplacement'] == 1].groupby('era')['correctlyvoted'].mean()
#
# comparison = pd.DataFrame({'all_players': all_stats, 'results': winner_stats})
# comparison['difference'] = comparison['results'] - comparison['all_players']
#
# # Have winners gained more immunities over time?
# sns.lmplot(data=survivors[survivors['finalplacement'] == 1], x='seasonnum', y='individualimmunites')
# plt.show()

# Any similar jobs amongst winners?
job_wins = winners.groupby(['occupation']).count().sort_values(by='id', ascending=False)
print(job_wins.head())

cat_wins = winners.groupby(['jobcategory']).count().sort_values(by='id', ascending=False)
print(cat_wins.head())

# What age are winners?
age_wins = winners.groupby(['agegroup']).count().sort_values(by='id', ascending=False)
print(age_wins.head())

print("\nWinner Profile Summary")
profile_summary = pd.DataFrame({
    'Characteristic': [
        'Age Group (Most Common)',
        'Job Category (Most Common)',
        'Play Style (Highest Win Rate)',
        'Votes Received per Tribal',
        'Correct Vote Rate',
        'Individual Immunities per Tribal',
        'Votes Cast per Tribal',
        'Advantages Played per Tribal',
        'Average Tribals Attended'
    ],
    'Winner Profile': [
        f"{age_wins.index[0]} ({age_wins.iloc[0]['id']} winners)",
        f"{cat_wins.index[0]} ({cat_wins.iloc[0]['id']} winners)",
        f"{style_win_rate.index[0]} ({style_win_rate.iloc[0]['win_rate']:.1%} win rate)",
        f"{win_stats.loc['mean', 'votesreceived_pertribal']:.3f}",
        f"{win_stats.loc['mean', 'correctvote_rate']:.3f}",
        f"{win_stats.loc['mean', 'immunities_pertribal']:.3f}",
        f"{win_stats.loc['mean', 'votescast_pertribal']:.3f}",
        f"{win_stats.loc['mean', 'advantagesplayed_pertribal']:.3f}",
        f"{win_stats.loc['mean', 'tribalsattended']:.1f}"
    ]
})

total_winners = len(winners)
print(profile_summary.to_string(index=False))

# 2. Elite Winners Summary
print(f"\nElite Winners Summary")
print(f"Total Elite Winners: {len(elite_winners)}")
print(f"Criteria: >80% correct votes, <1 vote per tribal, >20% immunity rate")

if len(elite_winners) > 0:
    print(f"\nElite Winners:")
    for idx, winner in elite_winners.iterrows():
        print(f"  • {winner['playername']} ({winner['seasonplayed']}, Season {winner['seasonnum']})")

print(f"\nPlay Style Success Summary")
for style in style_win_rate.index:
    count = style_win_rate.loc[style, 'winners']
    if pd.isna(count):
        count = 0
    else:
        count = int(count)
    percentage = (count / total_winners) * 100
    print(f"{style:15} {count:2d} winners ({percentage:4.1f}%)")

print(f"\nWinner Evolution by Era")
for era in era_wins.index:
    correct_votes = era_wins.loc[era, 'correctlyvoted']
    votes_received = era_wins.loc[era, 'votesrecieved']
    immunities = era_wins.loc[era, 'individualimmunites']
    print(f"{era:12} Correct Votes: {correct_votes:4.1f} | Votes Received: {votes_received:4.1f} | Immunities: {immunities:4.1f}")

print(f"\n*****THE ULTIMATE WINNER FORMULA*****")

# Key insights from existing analysis
top_age = age_wins.index[0]
top_job = cat_wins.index[0]
best_style = style_win_rate.index[0]
best_style_rate = style_win_rate.iloc[0]['win_rate']

winner_analysis = f"""
DEMOGRAPHIC PROFILE:
The typical Survivor winner is in their {top_age.lower()}, most commonly working in 
{top_job.lower()}. 

GAMEPLAY STRATEGY:
Winners excel at the "{best_style.lower()}" play style, which has a {best_style_rate:.1%} 
success rate. The data shows that winners typically:
• Stay Under the Radar: Receive only {win_stats.loc['mean', 'votesreceived_pertribal']:.2f} votes per tribal council
• Vote Strategically: Have a {win_stats.loc['mean', 'correctvote_rate']:.1%} correct voting rate  
• Show Physical Strength: Win {win_stats.loc['mean', 'immunities_pertribal']:.1%} of individual immunities
• Play Actively: Cast {win_stats.loc['mean', 'votescast_pertribal']:.2f} votes per tribal council
• Use Advantages Sparingly: Play {win_stats.loc['mean', 'advantagesplayed_pertribal']:.3f} advantages per tribal

ELITE WINNER CHARACTERISTICS:
The most dominant winners (elite tier) separate themselves by:
- Voting correctly over 80% of the time
- Receiving less than 1 vote per tribal council  
- Winning individual immunities at a 20%+ rate
"""

print(winner_analysis)

print(f"\nVariance in winning play")

print("\nWinner metric variability:")
variance_metrics = ['votesreceived_pertribal', 'correctvote_rate', 'immunities_pertribal', 'votescast_pertribal']
for metric in variance_metrics:
    mean_val = win_stats.loc['mean', metric]
    std_val = win_stats.loc['std', metric]
    min_val = win_stats.loc['min', metric]
    max_val = win_stats.loc['max', metric]
    cv = (std_val / mean_val) * 100  # coefficient of variation
    print(f"{metric:25} Mean: {mean_val:.3f} ± {std_val:.3f} (Range: {min_val:.3f}-{max_val:.3f}) CV: {cv:.1f}%")

print(f"\nOutlier winners")

# Winners who don't fit the typical profile
outlier_winners = []

# High votes received but still won
high_votes = winners[winners['votesreceived_pertribal'] > win_stats.loc['75%', 'votesreceived_pertribal']]
if len(high_votes) > 0:
    outlier_winners.append(("High Target", high_votes.iloc[0]['playername'], high_votes.iloc[0]['seasonplayed'],
                           f"{high_votes.iloc[0]['votesreceived_pertribal']:.2f} votes/tribal"))

# Low immunity but still won
low_immunity = winners[winners['immunities_pertribal'] < win_stats.loc['25%', 'immunities_pertribal']]
if len(low_immunity) > 0:
    outlier_winners.append(("Low Physical", low_immunity.iloc[0]['playername'], low_immunity.iloc[0]['seasonplayed'],
                           f"{low_immunity.iloc[0]['immunities_pertribal']:.2f} immunity rate"))

# Poor voting record but still won
poor_voting = winners[winners['correctvote_rate'] < win_stats.loc['25%', 'correctvote_rate']]
if len(poor_voting) > 0:
    outlier_winners.append(("Poor Voting", poor_voting.iloc[0]['playername'], poor_voting.iloc[0]['seasonplayed'],
                           f"{poor_voting.iloc[0]['correctvote_rate']:.1%} correct votes"))

for category, name, season, stat in outlier_winners:
    print(f"  {category:12} {name} ({season}) - {stat}")

# Play style diversity among winners
print(f"\nDiversity by play style")
for style in style_win_rate.index:
    count = style_win_rate.loc[style, 'winners']
    if pd.isna(count):  # Handle NaN values
        count = 0
    else:
        count = int(count)
    percentage = (count / total_winners) * 100
    print(f"{style:15} {count:2d} winners ({percentage:4.1f}%)")

# Age and job diversity
print(f"\nDemographic diversity")
age_diversity = len(age_wins[age_wins['id'] > 0])  # Number of different age groups with winners
job_diversity = len(cat_wins[cat_wins['id'] > 0])  # Number of different job categories with winners

print(f"Age Groups with Winners: {age_diversity}")
print(f"Job Categories with Winners: {job_diversity}")

# Show era differences
print(f"\nEra variance")
era_variance = era_wins.std().round(2)
print("Standard Deviation Across Eras:")
for metric in era_variance.index:
    print(f"  {metric:20} {era_variance[metric]:.2f}")

variance_message = f"""
While patterns exist, the data reveals HIGH VARIANCE in winner characteristics:

METRIC VARIABILITY:
• Coefficient of variation ranges from {((win_stats.loc['std'] / win_stats.loc['mean']) * 100).min():.0f}% to {((win_stats.loc['std'] / win_stats.loc['mean']) * 100).max():.0f}%
• Winners span the full range of ages, jobs, and play styles
• Even "elite" winners represent only {len(elite_winners)} of {len(winners)} total winners

OUTLIER SUCCESS:
• Winners exist who received heavy targeting but still won
• Champions with poor voting records have succeeded  
• Weak physical players and strategic misfits have claimed victory

BOTTOM LINE:
The "ultimate winner profile" represents tendencies, not requirements. 
Survivor's beauty lies in its unpredictability - anyone can win with the 
right combination of strategy, luck, and timing. The variance in the data 
proves that while some approaches are more successful on average, there's 
no single path to victory.
"""

print(variance_message)