# -*- coding: utf-8 -*-
"""
Article Code: "Dynamic emotional updating as a computational marker of well-being"
@Code Author: Noa Nagar


Code 1- One Session Preprocessing
Import data and preliminary preprocessing for each condition in the one session experiment. 

First, download each participant's data from the 'ParticipantsData' folder located in either the:
    'HighMoodCon' folder (for subjects who participated in the high mood target condition or the 'LowMoodCon' folder (for subjects who participated in the low mood target condition).

Then, execute this code.

"""

#%% Import Libraries

import os
import glob
import pandas as pd

#%% Import Data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "oneSessionData")

# High Mood Target
HighMoodCon_DIR = os.path.join(DATA_DIR, "HighMoodCon")
high_files = sorted(glob.glob(os.path.join(HighMoodCon_DIR, "*.csv")))

# Low Mood Target
LowMoodCon_DIR = os.path.join(DATA_DIR, "LowMoodCon")
low_files = sorted(glob.glob(os.path.join(LowMoodCon_DIR, "*.csv")))


#%% Preprocessing- Process and combine multiple participant CSV files into a single DataFrame.

def process_and_combine_files(file_list):
    """
    Process and combine multiple participant CSV files into a single DataFrame.
    
    Parameters:
        file_list (list): List of file paths (strings).

    Returns:
        pd.DataFrame: Combined DataFrame with all processed participant data.
    """
    
    dfs = []

    for filename in file_list:
        
        # Read CSV file 
        df = pd.read_csv(filename)

        # Extract participant name from the third row of 'textboxProlific_IDA.text'
        participant_name = df['textboxProlific_IDA.text'].iloc[2]
        df['participant'] = participant_name

        # Arranging values- Combining the first momentary rating with the other momentary ratings.
        value_to_move = df.at[10, 'blockHappySlider.response'].copy()
        df.at[10, 'happySlider.response'] = value_to_move
        df.at[10, 'trialHappy.stopped'] = 1

        # Interpolation of momentary ratings.
        happy_res = df.dropna(subset=['trialHappy.stopped'])
        happy_res_df = happy_res[['participant', 'happySlider.response']]
        happy_res_df['happySlider.response'] = happy_res_df['happySlider.response'].interpolate(method='linear', limit_direction='both')
        df['happySlider.response'] = happy_res_df['happySlider.response']

        # Check Alertness Q - include only participants who answered correctly
        if ('Alertness_test_KeyResp.keys' in df.columns and 85 in df.index and df['Alertness_test_KeyResp.keys'].iloc[85] == 2):
            dfs.append(df)  # Add DataFrame to list

    # Adding a Source Column to each DataFrame
    for i, df in enumerate(dfs):
        df['Source'] = f'File {i+1}'

    # Concatenate all DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    return combined_df

high_combined_df = process_and_combine_files(high_files)
low_combined_df = process_and_combine_files(low_files)

# Preprocessing- CESD-10 

def compute_cesd_scores(combined_df):
    """
    Extract CESD-10 questionnaire responses, compute CESD-10 scores,
    and assign participants into groups (Healthy / Risk of Depression).
    
    Parameters:
       combined_df : pd.DataFrame. The full dataframe containing all participants' trial-level data.
    
    Returns:
        combined_df : pd.DataFrame. Updated combined_df with two new columns: CESD10_Score, Group
    """   

    # Creating a new data frame containing the answers to the questions
    CESD_question = combined_df.dropna(subset=['CESD_Q1Slider.response'])
    CESD_question_df = CESD_question[['participant', 'CESD_Q1Slider.response', 'CESD_Q2Slider.response', 'CESD_Q3Slider.response', 'CESD_Q4Slider.response', 'CESD_Q5Slider.response', 'CESD_Q6Slider.response', 'CESD_Q7Slider.response', 'CESD_Q8Slider.response', 'CESD_Q9Slider.response', 'CESD_Q10Slider.response']]
    CESD_question_df = CESD_question_df.rename(columns=lambda x:x.replace("CESD_Q","Q").replace("Slider.response", ""))

    # Calculate the CESD-10 score as the sum of the 10 questions
    reg_questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q6', 'Q7', 'Q9', 'Q10'] # Questions according to normal scoring
    CESD_question_df[reg_questions] = CESD_question_df[reg_questions] - 1 # Subtract 1 from the CESD scores to align with 0-3 scale

    # Q5 & Q8 are reverse scored questions (1 = 3, 2 = 2, 3 = 1, 4 = 0)
    reverse_map = {1: 3, 2: 2, 3: 1, 4: 0}
    CESD_question_df[['Q5', 'Q8']] = CESD_question_df[['Q5', 'Q8']].apply(lambda col: col.map(reverse_map))

    CESD_question_df['CESD10_Score'] = CESD_question_df.sum(axis=1, numeric_only=True)

    combined_df = pd.merge(combined_df, CESD_question_df[['participant', 'CESD10_Score']], on='participant', how='left') # Ensure all participants in combined_df are retained

    # Add a 'Group' column directly to combined_df
    combined_df['Group'] = combined_df['CESD10_Score'].apply(lambda x: 'High Symptoms' if x >= 10 else 'Low Symptoms' if pd.notnull(x) else 'Unknown')
    
    return combined_df

high_combined_df = compute_cesd_scores(high_combined_df)
low_combined_df = compute_cesd_scores(low_combined_df)

# Save file

higt_output_path = os.path.join(DATA_DIR, "HighMoodCon_preprocessed.csv")
high_combined_df.to_csv(higt_output_path, index=False)

low_output_path = os.path.join(DATA_DIR, "LowMoodCon_preprocessed.csv")
low_combined_df.to_csv(low_output_path, index=False)
