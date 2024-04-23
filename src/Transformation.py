## Helper functions
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder


def get_event_num(arrays, event_Code):
    """Get away corners in the game sequences

    :param arrays: list of game sequences
    :param event_Code: (int), either 1:goal_home, 2:goal_away, 3:corner_home, 4:corner_away
    """
    return [np.count_nonzero(arr == event_Code) for arr in arrays]       

def event_freq(num_event, arrays):
    """Get freuqency of the event in comparison to game duration

    :param num_event: (list) list of integers:calculated occurances of the event so far
    :param arrays: list of game sequences
    """
    sizes = np.array([arr.size for arr in arrays])
    return num_event/sizes

def mean_std_intervals(arr, event_code):
    """mean and standard deviation of distances of consecutive occurrences

    :param arr: 2d matrix of game sequences
    :param event_Code: (int), either 1:goal_home, 2:goal_away, 3:corner_home, 4:corner_away
    """
    mean_lst = []
    std_lst = []
    intervals = []
    for row in arr:
        intervals = []
        fours_indices = np.where(row == event_code)[0] # find all occurances indices of event
        if len(fours_indices) >= 2: # if more than 2 occurances
            intervals.extend(np.diff(fours_indices))
        else:
            intervals = [0]
        mean_lst.append(np.mean(intervals))
        std_lst.append(np.std(intervals))
        
    events = ['goal_h', 'goal_a', 'corner_h', 'corner_a']
    return {f"mean_{events[event_code-1]}_intervals": mean_lst, f"std_{events[event_code-1]}_intervals": std_lst}

def events_feature_generation(arrays, game_id_list):
    """Generate all event based features for the given set of sequences

    :param arrays: 2d matrix of game sequences
    :param game_id_list: (list of strings) list of game ids to be concated to the dataframe
    """
    data = pd.DataFrame({
        'home_goals':get_event_num(arrays, 1),
        'away_goals':get_event_num(arrays, 2),
        'home_corners':get_event_num(arrays, 3),
        'away_corners':get_event_num(arrays, 4)
    })
    data = pd.concat([data, pd.DataFrame(mean_std_intervals(arrays, 1))], axis = 1)
    data = pd.concat([data, pd.DataFrame(mean_std_intervals(arrays, 2))], axis = 1)
    data = pd.concat([data, pd.DataFrame(mean_std_intervals(arrays, 3))], axis = 1)
    data = pd.concat([data, pd.DataFrame(mean_std_intervals(arrays, 4))], axis = 1)
    data["goal_h_freq"] = event_freq(data["home_goals"].values, arrays)
    data["goal_a_freq"] = event_freq(data["away_goals"].values, arrays)
    data["corner_h_freq"] = event_freq(data["home_corners"].values, arrays)
    data["corner_a_freq"] = event_freq(data["away_corners"].values, arrays)
    data['game_id'] = game_id_list
    return data


def extract_two_strings(match_string, splitter):
    """Extracts the two strings from athe given match string

    :param match_string: (str) the string containing two words separated by splitter.
    :param splitter: (str) the string which is used for splitting
    :return: (list) list of two strings, the two words before and after splitter.
    Example:
    >>> extract_team_names("Deportivo Saprissa v Club Sport Heredian", " v ")
    ['Deportivo Saprissa', 'Club Sport Heredian']

    >>> extract_team_names("a vs b", " v ")
    [a vs b]
    >>> extract_team_names("a b v c", " v ")
    ['a b', 'c']
    """
    # Split the match string at splitter and strip whitespace from team names
    names = [name.strip() for name in match_string.split(splitter)]
    
    # edge case if the two words are not extracted, return empty string
    if len(names) != 2:
        return match_string
    
    return names[:2]  # MAGIC: 2 to make sure the first two are only returned


def preprocess_events(events_df):
    """Inputs events_df and preforms preprocessing on it"""
    # Drop the Unnamed column
    events_df = events_df.loc[:, "game_id":]

    # Make all string columns lower case to be consistent
    events_df.state = events_df.state.str.lower()
    events_df.type = events_df.type.str.lower()
    events_df.sort_values(by=["game_id", "minute", "second"], inplace=True)

    # Encode event types into numerical values, with 0 as no event
    event_encoding = {'goal_home': 1, 'goal_away': 2, 'corner_home': 3, 'corner_away': 4}
    events_df['type_encoded'] = events_df['type'].map(event_encoding)
    return events_df

def preprocess_games(games_df):
    """Inputs games_df and preforms preprocessing on it"""
    # drop the Unnamed column
    games_df = games_df.loc[:, "game_id":]

    # make all string columns lowe case to be consistent
    games_df.description = games_df.description.str.lower()
    games_df.league = games_df.league.str.lower()

    games_df.sort_values(by=["game_id","date"], inplace=True)
    return games_df

def generate_sequences(events_df):
    """Inputs the events_df and outputs sequences per game with encoded events

    :param events_df: (dataframe)
    :return: (array) one row per game with events encoded as values
    """
    # Get corresponding index for each game id to optimize the sequence creation with arrays
    events_df['idx'], _ = pd.factorize(events_df['game_id'])

    #initialize the sequence with zero as no events
    sequences = np.zeros((len(events_df['game_id'].unique()), events_df["minute"].max()))

    # Loop through each event and update the corresponding minute in the sequence with type_encoded
    for idx, row in events_df.iterrows():
        sequences[row['idx'], row['minute'] - 1] = row['type_encoded']

    sequences = sequences.astype(int)
    print(f'the generated sequences shape is {sequences.shape}')
    return sequences

def games_features(games_df):
    """EXTRACT features from games_df
    """
    ## Get team names from the description column
    games_df[['team1', 'team2']] = games_df['description'].apply(lambda x: pd.Series(extract_two_strings(x, " v ")))
    label_encoder = LabelEncoder() # econde team columms in place
    games_df['encoded_team1'] = label_encoder.fit_transform(games_df['team1'])
    games_df['encoded_team2'] = label_encoder.fit_transform(games_df['team2'])

    ## Get features from leauge description column
    games_df['friendly'] = games_df['league'].apply(lambda x: 1 if "friendly" in x else 0)
    games_df['u21'] = games_df['league'].apply(lambda x: 1 if "u21" in x else 0)
    games_df['cup'] = games_df['league'].apply(lambda x: 1 if "cup" in x else 0)

    # keep only features columns
    games_df = games_df[['game_id', 'encoded_team1', 'encoded_team2', 'friendly', 'u21', 'cup']]
    return games_df

def generate_input_output_seqs(sequences, max_time):
    """Given pred points, extract features from sequences dataframe"""
    random.seed(42)
    # timestamp of prediction point, resulting in complementary input and output seqs, with different sizes
    pred_point = [random.randint(10, max_time-10) for _ in range(sequences.shape[0])]
    input_sequence = [seq[0:point] for seq, point in zip(sequences, pred_point)]

    input_events_df = events_feature_generation(input_sequence, events_df["game_id"].unique())
    input_events_df["time_gone"] = np.array(pred_point)
    input_events_df["time_left"] = max_time - np.array(pred_point)

    # Generate ouput dataframe, we mainly use the home goals and away goals from this dataset as y
    output_sequence = [seq[point:max_time+1] for seq, point in zip(sequences, pred_point)]
    output_events_df = events_feature_generation(output_sequence, events_df["game_id"].unique())

    return input_events_df, output_events_df

if __name__ == "__main__":
    
    events_df = pd.read_csv('../data/events.csv')
    events_df = preprocess_events(events_df)
    sequences = generate_sequences(events_df)
    seq_df = pd.DataFrame(sequences) 
    seq_df['game_id'] = events_df['game_id'].unique() # add the game_id to the created feature dataframe
    max_time = 90  # MAGIC: Investigating only first 90 mins
    input_events_df, output_events_df = generate_input_output_seqs(sequences, max_time)
    print(f'input and outpur sequences generated.')


    # Read and preprocess games_df
    games_df = pd.read_csv('../data/games.csv')
    games_df = preprocess_games(games_df)
    games_df = games_features(games_df)
    print(f'Feature generation from games_df is done')


    X = pd.merge(input_events_df, games_df, on='game_id')
    X.sort_values('game_id', inplace=True)
    X.set_index("game_id", inplace=True, drop=True)
    print(f'X dataframe with shape {X.shape} is craeted.')

    y = output_events_df[["game_id", "home_goals", "away_goals"]].copy() 
    y.sort_values('game_id', inplace=True)
    y.set_index("game_id", inplace=True, drop=True)
    print(f'y dataframe with shape {y.shape} is craeted.')

    # Drop outliers
    X, y = drop_outliers(X, y)
    
    ## store the datasets
    X.to_csv('../data/X.csv')
    y.to_csv('../data/y.csv')
    print('X and y files saved')