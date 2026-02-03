# -*- coding: utf-8 -*-
"""
Article Code: "Dynamic emotional updating as a computational marker of well-being"
@Code Author: Noa Nagar


Code 2- One Session Statistical Analysis
Time plots (Mood & RPE) with statistical analyses of t-tests and correlations for each condition in the one session experiment. 

First, run code 1 ('OneSession_Preprocessing') or download the 'HighMoodCon_preprocessed.csv' and 'LowMoodCon_preprocessed.csv' files from the 'onseSessionData' folder.
Then run this code.

"""

#%% Import Libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

#%% Import Data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "oneSessionData")

# High Mood Target
HighMoodCon_DIR = os.path.join(DATA_DIR, "HighMoodCon_preprocessed.csv")
high_df = pd.read_csv(HighMoodCon_DIR)

# Low Mood Target
LowMoodCon_DIR = os.path.join(DATA_DIR, "LowMoodCon_preprocessed.csv")
low_df = pd.read_csv(LowMoodCon_DIR)

# Data preparation
# High Mood Target
high_df_Ratings = high_df.dropna(subset=['happySlider.response']).copy()
high_df_RPE = high_df.dropna(subset=['RPE']).copy()

# Low Mood Target
low_df_Ratings = low_df.dropna(subset=['happySlider.response']).copy()
low_df_RPE = low_df.dropna(subset=['RPE']).copy()

#%% Analysis over Time (Mood & RPE)

def get_group_mean_std_matrices(df, data_col):
    """
    Compute per-trial mean and standard deviation of a given variable for each symptom
    group ('High Symptoms' / 'Low Symptoms') and return the results as group-by-trial matrices.

    Parameters:
        df (pd.DataFrame): Trial-level data.
        data_col (str): Name of the column for which mean and std are computed.

    Returns:
        mean_df (pd.DataFrame): A matrix where each row represents a group and each column represents
                                a rating index, containing the mean values of data_col.
        std_df (pd.DataFrame): A matrix with the same structure as mean_df,
                               containing standard deviations of data_col.
    """

    group_means = {}
    group_stds = {}

    for group_name, group_df in df.groupby('Group'):
        group_df_clean = group_df.dropna(subset=[data_col]).copy()
        group_df_clean.loc[:, 'rating_index'] = group_df_clean.groupby('participant').cumcount() + 1

        # Compute mean and std for each rating index
        stats = group_df_clean.groupby('rating_index')[data_col].agg(['mean', 'std'])
        group_means[group_name] = stats['mean'].values
        group_stds[group_name] = stats['std'].values

    # Convert to DataFrames (groups as rows, rating indices as columns)
    mean_df = pd.DataFrame.from_dict(group_means, orient='index')
    std_df = pd.DataFrame.from_dict(group_stds, orient='index')

    return mean_df, std_df

# Mood- High Mood Target
high_moodMean, high_moodSTD = get_group_mean_std_matrices(high_df, 'happySlider.response')

high_moodMean_lows = high_moodMean.loc['Low Symptoms']
high_moodMean_highs = high_moodMean.loc['High Symptoms']

# RPE- High Mood Target
high_rpeMean, high_rpeSTD = get_group_mean_std_matrices(high_df, 'RPE')

high_rpeMean_lows = high_rpeMean.loc['Low Symptoms']
high_rpeMean_highs = high_rpeMean.loc['High Symptoms']

# Mood- Low Mood Target - 30%
low_moodMean, low_moodSTD = get_group_mean_std_matrices(low_df, 'happySlider.response')

low_moodMean_lows = low_moodMean.loc['Low Symptoms']
low_moodMean_highs = low_moodMean.loc['High Symptoms']

# RPE- Low Mood Target - 30%
low_rpeMean, low_rpeSTD = get_group_mean_std_matrices(low_df, 'RPE')

low_rpeMean_lows = low_rpeMean.loc['Low Symptoms']
low_rpeMean_highs = low_rpeMean.loc['High Symptoms']

# Plot
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharey='row')
fig.suptitle('Group Differences Over Time', fontsize = 20, fontweight='bold')

# 1- Momentary Mood Ratings - Healthy
ax = axes[0, 0]
ax.plot(high_moodMean_lows, label = 'High Mood Target (85%)', color = 'darkorange', alpha=0.7, linewidth=2.5)
ax.plot(low_moodMean_lows ,label = 'Low Mood Target (30%)', color = '#FFBF00', alpha=0.7, linewidth=2.5)
ax.set_title('Low Symptoms', fontsize=18)
ax.set_xlabel('Trial Index', fontsize=16)
ax.set_ylabel('Momentary Mood Ratings', fontsize=16)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_xticks([0, 10, 20, 30])
ax.tick_params(axis='both', labelsize=16)
ax.legend(loc= 'center left', bbox_to_anchor=((-0.5, -0.01)))

# 2- Momentary Mood Ratings - Depression
ax = axes[0, 1]
ax.plot(high_moodMean_highs, label = 'High Mood Target (85%)', color = 'darkorange', alpha=0.7,linewidth=2.5)
ax.plot(low_moodMean_highs, label = 'Low Mood Target (30%)', color = '#FFBF00', alpha=0.7, linewidth=2.5)
ax.set_title('High Symptoms', fontsize=18)
ax.set_xlabel('Trial Index', fontsize=16)
ax.set_ylabel('', fontsize=16)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_xticks([0, 10, 20, 30])
ax.tick_params(axis='both', labelsize=16)

# 3- RPE - Healthy
ax = axes[1, 0]
ax.plot(high_rpeMean_lows, color = 'darkorange', alpha=0.7, linewidth=2.5)
ax.plot(low_rpeMean_lows, color = '#FFBF00', alpha=0.7, linewidth=2.5)
ax.set_title('Low Symptoms', fontsize=18)
ax.set_xlabel('Trial Index', fontsize=16)
ax.set_ylabel('RPE', fontsize=16)
ax.set_yticks([-20, 0, 20])
ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
ax.tick_params(axis='both', labelsize=16)

# 4- RPE - Depression
ax = axes[1, 1]
ax.plot(high_rpeMean_highs, color = 'darkorange', alpha=0.7, linewidth=2.5)
ax.plot(low_rpeMean_highs, color = '#FFBF00', alpha=0.7, linewidth=2.5)
ax.set_title('High Symptoms', fontsize=18)
ax.set_xlabel('Trial Index', fontsize=16)
ax.set_ylabel('', fontsize=16)
ax.set_yticks([-20, 0, 20,])
ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
ax.tick_params(axis='both', labelsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#%% t-test

# t-test by group

def ttest_first_vs_last_three(df):
    """
    Perform a paired t-test comparing, for each participant, the mean of the
    first three momentary ratings with the mean of the last three momentary ratings.
    
    Parameters:
        df (pd.DataFrame): Trial-level data for a specific group.
    
    Returns:
        t_stat (float): T-statistic from the paired t-test.
        p_value (float): P-value from the paired t-test.
        ratings_3_df (pd.DataFrame): A DataFrame containing, for each participant,
                                     the mean of the first three ratings and the mean
                                     of the last three ratings.
    """
    df = df.sort_values(by=['participant', 'getAnswer.stopped']) # Sort the DataFrame by participant and time ('getAnswer.stopped')
    rating = df.dropna(subset=['happySlider.response'])
    grouped = rating.groupby('participant') # Correctly identifying the first and last three ratings for each subject individually
    
    first_3_rating = grouped.head(3).groupby('participant')['happySlider.response'].mean()
    last_3_rating = grouped.tail(3).groupby('participant')['happySlider.response'].mean()

    ratings_3_df = pd.DataFrame({
          'first_three_mean': first_3_rating,
          'last_three_mean': last_3_rating})
    
    t_stat, p_value= stats.ttest_rel(ratings_3_df['first_three_mean'], ratings_3_df['last_three_mean'])
    
    return t_stat, p_value, ratings_3_df

# High Mood Target
high_t_HSymp_threeRatings, high_p_HSymp_threeRatings, high_df_HSymp_threeRatings = ttest_first_vs_last_three(high_df[high_df['Group'] == 'High Symptoms'])
print(f"\nT-statistic between the mean of the first and last three ratings for participants with High symptoms: {high_t_HSymp_threeRatings}, P-value: {high_p_HSymp_threeRatings}")

high_t_LSymp_threeRatings, high_p_LSymp_threeRatings, high_df_LSymp_threeRatings = ttest_first_vs_last_three(high_df[high_df['Group'] == 'Low Symptoms'])
print(f"\nT-statistic between the mean of the first and last three ratings for participants with Low symptoms: {high_t_LSymp_threeRatings}, P-value: {high_p_LSymp_threeRatings}")

# t-test between the groups

# t-test between the first rating and the group
def ttest_first_rating(df):
    """
    Perform an independent t-test comparing the first momentary rating between two symptom groups ('High Symptoms' vs 'Low Symptoms').
    The function also performs Levene's test to determine whether to assume equal variance in the t-test.
    
    Parameters:
        df (pd.DataFrame): Trial-level data.
    
    Returns:
        t_stat (float): T-statistic from the independent t-test.
        p_value (float): P-value from the independent t-test.
        first_rating_df (pd.DataFrame): DataFrame containing participant, group, CESD10 score, and first rating.
    """
    # Data preparation
    df = df.sort_values(by=['participant', 'getAnswer.stopped'])
    first_rating = df.groupby('participant')['happySlider.response'].first() # Extract first rating for each participant
    cesd_info = df[['participant', 'Group', 'CESD10_Score']].drop_duplicates() # Extract participant-level info
    first_rating_df =  cesd_info.merge(first_rating.rename('first_rating'), on='participant') # Merge to create full per-participant dataset

    # Levene's Test for Variance Equality
    levene_stat, levene_p  = stats.levene(first_rating_df[first_rating_df['Group'] == 'High Symptoms']['first_rating'], first_rating_df[first_rating_df['Group'] == 'Low Symptoms']['first_rating'])

    # Independent t-test
    if levene_p < 0.05:
        t_stat, p_value =  stats.ttest_ind(first_rating_df[first_rating_df['Group'] == 'High Symptoms']['first_rating'], first_rating_df[first_rating_df['Group'] == 'Low Symptoms']['first_rating'], equal_var=False)
        print(f"\nT-statistic for the first rating between participants with High symptoms and with Low symptoms: {t_stat}, P-value: {p_value}")
    else:
        t_stat, p_value =  stats.ttest_ind(first_rating_df[first_rating_df['Group'] == 'High Symptoms']['first_rating'], first_rating_df[first_rating_df['Group'] == 'Low Symptoms']['first_rating'])
        print(f"\nT-statistic for the first rating between participants with High symptoms and with Low symptoms: {t_stat}, P-value: {p_value}")
     
    return t_stat, p_value, first_rating_df 
      
# High Mood Target      
high_t_first, high_p_first, high_first_rating_df = ttest_first_rating(high_df_Ratings)

# t-test between task RPE and the group
def ttest_meanRPE(df):
    """
    Perform independent t-tests comparing the mean RPE and RPE variability (standard deviation)
    between participants with High Symptoms and Low Symptoms. The function also performs 
    Levene's tests to assess equality of variances before selecting the appropriate t-test.

    Parameters:
        df (pd.DataFrame): Trial-level data.
        
    Returns:
        t_mean (float): T-statistic for the comparison of mean RPE between groups.
        p_mean (float): P-value for the comparison of mean RPE between groups.
        t_std (float): T-statistic for the comparison of RPE standard deviation between groups.
        p_std (float): P-value for the comparison of RPE standard deviation between groups.
        RPE_df (pd.DataFrame): A participant-level DataFrame containing: mean_rpe, std_rpe, Group, CESD10_Score.
    """
    df = df.sort_values(by=['participant', 'getAnswer.stopped'])
    
    meanrpe = df.groupby('participant')['RPE'].mean()
    stdrpe = df.groupby('participant')['RPE'].std()
    cesd_info = df[['participant', 'Group', 'CESD10_Score']].drop_duplicates() # Extract participant-level info
    RPE_df =  cesd_info.merge(meanrpe.rename('mean_rpe'), on='participant') # Merge to create full per-participant dataset
    RPE_df =  RPE_df.merge(stdrpe.rename('std_rpe'), on='participant') # Merge to create full per-participant dataset
    
    # Levene's Test for Variance Equality
    levene_mean, levene_p_mean  = stats.levene(RPE_df[RPE_df['Group'] == 'High Symptoms']['mean_rpe'], RPE_df[RPE_df['Group'] == 'Low Symptoms']['mean_rpe'])

    # Independent t-test
    if levene_p_mean < 0.05:
        t_mean, p_mean =  stats.ttest_ind(RPE_df[RPE_df['Group'] == 'High Symptoms']['mean_rpe'], RPE_df[RPE_df['Group'] == 'Low Symptoms']['mean_rpe'], equal_var=False)
        print(f"\nT-statistic for the mean RPE between participants with High symptoms and with Low symptoms: {t_mean}, P-value: {p_mean}")
    else:
        t_mean, p_mean =  stats.ttest_ind(RPE_df[RPE_df['Group'] == 'High Symptoms']['mean_rpe'], RPE_df[RPE_df['Group'] == 'Low Symptoms']['mean_rpe'])
        print(f"\nT-statistic for the mean RPE between participants with High symptoms and with Low symptoms: {t_mean}, P-value: {p_mean}")
      
      
    # Levene's Test for Variance Equality
    levene_std, levene_p_std  = stats.levene(RPE_df[RPE_df['Group'] == 'High Symptoms']['std_rpe'], RPE_df[RPE_df['Group'] == 'Low Symptoms']['std_rpe'])

    # Independent t-test
    if levene_p_std < 0.05:
        t_std, p_std =  stats.ttest_ind(RPE_df[RPE_df['Group'] == 'High Symptoms']['std_rpe'], RPE_df[RPE_df['Group'] == 'Low Symptoms']['std_rpe'], equal_var=False)
        print(f"\nT-statistic for RPE std between participants with High symptoms and with Low symptoms: {t_std}, P-value: {p_std}")
    else:
        t_std, p_std =  stats.ttest_ind(RPE_df[RPE_df['Group'] == 'High Symptoms']['std_rpe'], RPE_df[RPE_df['Group'] == 'Low Symptoms']['std_rpe'])
        print(f"\nT-statistic for RPE std between participants with High symptoms and with Low symptoms: {t_std}, P-value: {p_std}")
      
    return t_mean, p_mean, t_std, p_std, RPE_df 

# High Mood Target     
high_t_meanRPE, high_p_meanRPE, high_t_stdRPE, high_p_stdRPE, high_RPE = ttest_meanRPE(high_df_RPE)


# t-test for the ratings mean between the conditions
def par_mean_byGroup(high_df, low_df, group):
    """
    Perform an independent t-test comparing the mean mood ratings between two experimental 
    conditions (High Mood Target vs. Low Mood Target), within a specified symptom group.

    Parameters:
        high_df (pd.DataFrame): Trial-level data for the High Mood Target condition.
        low_df (pd.DataFrame): Trial-level data for the Low Mood Target condition.
        group (str): Symptom group to analyze (e.g. "High Symptoms" or "Low Symptoms").

    Returns:
        t_mean (float): T-statistic for the comparison of participant-level mean ratings between conditions.
        p_mean (float): P-value associated with the t-test.
    """
    # Filter for Group
    high_group = high_df[high_df['Group'] == group].dropna(subset = ['happySlider.response'])
    low_group = low_df[low_df['Group'] == group].dropna(subset = ['happySlider.response'])

    # Compute participant-wise means
    high_mean = high_group.groupby('participant')['happySlider.response'].mean().reset_index(name= 'high_mean_ratings')
    low_mean = low_group.groupby('participant')['happySlider.response'].mean().reset_index(name= 'low_mean_ratings')
        
    # Levene's Test for Variance Equality
    levene_mean, levene_p_mean  = stats.levene(high_mean['high_mean_ratings'], low_mean['low_mean_ratings'])

    # Independent t-test
    if levene_p_mean < 0.05:
        t_mean, p_mean =  stats.ttest_ind(high_mean['high_mean_ratings'], low_mean['low_mean_ratings'], equal_var=False)
        print(f"\nT-statistic for the mean ratings between conditions: {t_mean}, P-value: {p_mean}")
    else:
        t_mean, p_mean =  stats.ttest_ind(high_mean['high_mean_ratings'], low_mean['low_mean_ratings'])
        print(f"\nT-statistic for the mean ratings between conditions: {t_mean}, P-value: {p_mean}")
      
    return  t_mean, p_mean

# Low Symptoms
t_mean_ls, p_mean_ls = par_mean_byGroup(high_df, low_df, 'Low Symptoms')

# High Symptoms
t_mean_hs, p_mean_hs = par_mean_byGroup(high_df, low_df, 'High Symptoms')

#%% Correlations

# Correlation between CESD-10 score and the change in mood

def mood_change3_correlation(df, floor=False):
    """
    Compute the correlation between CESD-10 scores and the change in mood ratings
    (defined as the difference between the mean of the last three ratings and the mean of the first three ratings) for each participant.

    Optionally, participants with low overall mean ratings can be excluded.

    Parameters:
        df (pd.DataFrame): Trial-level data.
        floor (bool): If True, excludes participants whose mean mood ratin is ≤ 0.2 before calculating correlations.

    Returns:
        corr (float): Pearson correlation coefficient between mood change and CESD-10 score.
        p_value (float): P-value associated with the correlation.
        threediff_df (pd.DataFrame): Participant-level DataFrame containing: CESD10_Score, three_diff (last 3 mean − first 3 mean mood rating)
    """
    
    df = df.sort_values(by=['participant', 'getAnswer.stopped'])
    
    # Filter participants if needed
    if floor:
       means= df.groupby('participant')['happySlider.response'].mean() # Compute mean per participant
       valid_participants = means[means > 0.2].index
       df = df[df['participant'].isin(valid_participants)] # Keep only those participants

    # Calculating the gap between the average of the last 3 ratings and the first 3 ratings.
    def calc_diff(group):
        first3_mean = group['happySlider.response'].head(3).mean()
        last3_mean = group['happySlider.response'].tail(3).mean()
        return pd.Series({'three_diff': last3_mean - first3_mean})

    diffs = df.groupby('participant').apply(calc_diff).reset_index()
    
    cesd_info = df[['participant', 'CESD10_Score']].drop_duplicates() # Extract participant-level info
    threediff_df = cesd_info.merge(diffs, on='participant') # Merge to create full per-participant dataset
    
    # Pearson correlation
    corr, p_value = stats.pearsonr(threediff_df['three_diff'], threediff_df['CESD10_Score'])
    print(f"\nPearson correlation between mood change (3 last ratings mean - 3 first ratings mean) and CESD-10 score:{corr}, P-value: {p_value}")
  
    return corr, p_value, threediff_df

# Low Mood Target
low_corr_fl3, low_p_fl3, low_fl3_df = mood_change3_correlation(low_df_Ratings)
low_corr_fl3floor, low_p_fl3floor, low_fl3floor_df = mood_change3_correlation(low_df_Ratings, floor=True)

# Correlation between CESD-10 score and standart deviation of the second half of the mood ratings

def std_mood_correlation(df):
    """
    Calculate the correlation between CESD-10 scores and the standard deviation
    of mood ratings during the second half of the task for each participant.

    Parameters:
        df (pd.DataFrame): Trial-level data.
        
    Returns:
        corr (float): Pearson correlation coefficient between CESD-10 score and second-half rating variability.
        p_value (float): P-value associated with the correlation.
        std_df (pd.DataFrame): Participant-level DataFrame containing: CESD10_Score, ratings_std (STD of second-half mood ratings)
    """
    
    df['trial_index'] = df.groupby('participant').cumcount() + 1 # Create trial index per participant
    df = df[df['trial_index'] >= 16]

    std = df.groupby('participant')['happySlider.response'].std().reset_index(name='ratings_std') # Calculate the STD for each participant
    cesd_info = df[['participant', 'CESD10_Score']].drop_duplicates() # Extract participant-level info

    std_df = cesd_info.merge(std, on='participant')
    
    # Pearson correlation
    corr, p_value = stats.pearsonr(std_df['CESD10_Score'], std_df['ratings_std'])
    print(f"\nPearson correlation between the second half ratings std and CESD-10 score: {corr},P value: {p_value}")
    
    return corr, p_value, std_df

# High Mood Target
high_corr_stdr, high_p_stdr, high_stdr_df = std_mood_correlation(high_df_Ratings)
high_p_text_stdr= f"p={high_p_stdr:.4f}" if high_p_stdr > 0.001 else "p<0.001"

# Plot the correlation:
plt.figure(figsize=(6,4))
sns.regplot(x= 'CESD10_Score', y = 'ratings_std', data= high_stdr_df, color ='black')
plt.title(f'Pearson Correlation between the Second Half Ratings STD and CESD-10 \nr={high_corr_stdr:.2f}, {high_p_text_stdr}')
plt.xlabel('CESD-10 Score')
plt.ylabel('Half Trails Ratings STD')
plt.tight_layout()
plt.show()

# Correlation between CESD-10 score and Mean Absolute Difference of the second half of the mood ratings
 
def mad_mood_correlation(df):
    """
    Calculate the correlation between CESD-10 scores and the mean absolute 
    difference (MAD) of mood ratings during the second half of the task for each participant.
    
    Parameters:
        df (pd.DataFrame): Trial-level data.
        
    Returns:
        corr (float): Pearson correlation coefficient between CESD-10 scores and second-half mood rating MAD.
        p_value (float): P-value associated with the correlation.
        mad_df (pd.DataFrame): Participant-level DataFrame containing: CESD-10 scores, the MAD of second-half mood ratings ('MAD_half').
    """
    
    df['trial_index'] = df.groupby('participant').cumcount() + 1 # Create trial index per participant
    df = df.sort_values(by = ['participant', 'trial_index'])

    # Compute MAD for the second half of the task.
    def compute_mad_half(mood_series):
        mood_half = mood_series.iloc[16:]
        diffs = np.abs(np.diff(mood_half))
        return np.mean(diffs)

    mad_half = df.groupby('participant')['happySlider.response'].apply(compute_mad_half).reset_index()
    mad_half.columns = ['participant','MAD_half']

    cesd_info = df[['participant', 'CESD10_Score']].drop_duplicates() # Extract participant-level info
    mad_df = cesd_info.merge(mad_half, on='participant', how='left')
    
    # Pearson correlation
    corr, p_value = stats.pearsonr(mad_df['CESD10_Score'], mad_df['MAD_half'])
    print(f"\nPearson correlation between half MAD and CESD-10 score: {corr}, P value: {p_value}")
          
    return corr, p_value, mad_df

# High Mood Target
high_corr_mad, high_p_mad, high_mad_df = mad_mood_correlation(high_df_Ratings)
high_p_text_mad= f"p={high_p_mad:.4f}" if high_p_mad > 0.001 else "p<0.001"

# Plot the correlation:
plt.figure(figsize=(6,4))
sns.regplot(x= 'CESD10_Score', y = 'MAD_half', data= high_mad_df, color ='black')
plt.title(f'Pearson Correlation between the MAD of the Second Half Ratings and CESD-10 \nr={high_corr_mad:.2f}, {high_p_text_mad}')
plt.xlabel('CESD-10 Score')
plt.ylabel('Half MAD')
plt.tight_layout()
plt.show()

# Correlation between mean RPE and CESD-10 score

def mean_rpe_correlation(df):
    """
    Calculate the correlation between CESD-10 scores and the RPE for each participant.
    
    Parameters:
        df (pd.DataFrame): Trial-level data.
        
    Returns:
        corr (float): Pearson correlation coefficient between CESD-10 scores and mean RPE.
        p_value (float): P-value associated with the correlation.
        meanrpe_df (pd.DataFrame): Participant-level DataFrame containing: CESD-10 scores, mean RPE.
    """
    
    mean = df.groupby('participant')['RPE'].mean().reset_index(name='mean_rpe') #  Calculate the average RPE for each participant
    
    cesd_info = df[['participant', 'CESD10_Score']].drop_duplicates() # Extract participant-level info
    meanrpe_df = cesd_info.merge(mean, on='participant')
    
    # Pearson correlation
    corr, p_value = stats.pearsonr(meanrpe_df['mean_rpe'], meanrpe_df['CESD10_Score'])
    print(f"\nPearson correlation between mean RPE and CESD_10 score: {corr}, P value: {p_value}")
    
    return corr, p_value, meanrpe_df

# High Mood Target
high_corr_meanrpe, high_p_meanrpe, high_meanrpe_df = mean_rpe_correlation(high_df_RPE)
high_p_text_rpe= f"p={high_p_meanrpe:.4f}" if high_p_meanrpe > 0.001 else "p<0.001"

# Plot the correlation:
plt.figure(figsize=(6,4))
sns.regplot(x= 'CESD10_Score', y = 'mean_rpe', data= high_meanrpe_df, color ='black')
plt.title(f'Pearson Correlation between Mean RPE and CESD-10 \nr={high_corr_meanrpe:.2f}, {high_p_text_rpe}')
plt.xlabel('CESD-10 Score')
plt.ylabel('Mean RPE')
plt.tight_layout()
plt.show()   
    
# Low Mood Target
low_corr_meanrpe, low_p_meanrpe, low_meanrpe_df = mean_rpe_correlation(low_df_RPE)








