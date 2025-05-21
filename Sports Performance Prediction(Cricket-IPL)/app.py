from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
import random

app = Flask(__name__)

# Load datasets
processed_data_path = 'datasets/processed'

# Load processed data files with column verification
try:
    batting_stats = pd.read_csv(os.path.join(processed_data_path, 'batting_stats.csv'))
    bowling_stats = pd.read_csv(os.path.join(processed_data_path, 'bowling_stats.csv'))
    # Verify columns
    expected_batting_cols = ['player_name', 'team', 'matches_played', 'runs', 'balls_faced', 'dismissals', 'average', 'strike_rate', 'highest_score']
    expected_bowling_cols = ['player_name', 'team', 'matches_bowled', 'wickets', 'economy', 'best_figures']
    if not all(col in batting_stats.columns for col in expected_batting_cols):
        print("Warning: batting_stats.csv columns do not match expected structure")
        print(f"Actual columns: {batting_stats.columns.tolist()}")
    if not all(col in bowling_stats.columns for col in expected_bowling_cols):
        print("Warning: bowling_stats.csv columns do not match expected structure")
        print(f"Actual columns: {bowling_stats.columns.tolist()}")
    # Rename columns
    batting_stats = batting_stats.rename(columns={'matches_played': 'matches'})
    bowling_stats = bowling_stats.rename(columns={'matches_bowled': 'matches', 'balls_bowled': 'balls_delivered'})
    print(f"Loaded {len(batting_stats)} batting records and {len(bowling_stats)} bowling records")
except Exception as e:
    print(f"Error loading processed data files: {e}")
    batting_stats = pd.DataFrame()
    bowling_stats = pd.DataFrame()

# Load raw data files with fallback
try:
    matches_df = pd.read_csv(os.path.join(processed_data_path, 'matches.csv'))
    deliveries_df = pd.read_csv(os.path.join(processed_data_path, 'deliveries.csv'))
    print("Loaded raw matches and deliveries data")
except Exception as e:
    print(f"Error loading raw data files: {e}")
    matches_df = pd.DataFrame()
    deliveries_df = pd.DataFrame()
    print("Continuing with processed data only")

# Load ML models with fallback
try:
    match_model = joblib.load('models/match_prediction_model.joblib')
    player_model = joblib.load('models/player_prediction_model.joblib')
    print("ML models loaded successfully")
except Exception as e:
    print(f"Error loading ML models: {e}")
    match_model = None
    player_model = None
    print("Continuing without ML models")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/player-performance')
def player_performance_options():
    return render_template('player_performance_options.html')

@app.route('/player-stats')
def player_stats():
    return render_template('player_stats.html')

@app.route('/venue-player-prediction')
def venue_player_prediction():
    return render_template('venue_player.html')

@app.route('/team-best-players')
def team_best_players_page():
    return render_template('team_best_players.html')

@app.route('/team-best-players', methods=['POST'])
def team_best_players():
    try:
        data = request.get_json()
        if not data or 'team' not in data:
            return jsonify({'status': 'error', 'message': 'Team not provided'}), 400

        team = data['team']
        if batting_stats.empty or bowling_stats.empty:
            return jsonify({'status': 'error', 'message': 'Required data not loaded'}), 500

        # Get team's batting stats
        team_batting = batting_stats[batting_stats['team'] == team].copy()
        # Get team's bowling stats
        team_bowling = bowling_stats[bowling_stats['team'] == team].copy()

        if team_batting.empty and team_bowling.empty:
            return jsonify({'status': 'error', 'message': 'No data found for the selected team'}), 404

        batsmen_data = []
        bowlers_data = []

        # Calculate batting statistics
        if not team_batting.empty:
            # Normalize metrics to 0-100 scale
            max_runs = team_batting['runs'].max()
            max_avg = team_batting['average'].max()
            max_sr = team_batting['strike_rate'].max()
            
            # Calculate normalized components (0-100 scale)
            runs_score = (team_batting['runs'] / max_runs * 100) if max_runs > 0 else 0
            avg_score = (team_batting['average'] / max_avg * 100) if max_avg > 0 else 0
            sr_score = (team_batting['strike_rate'] / max_sr * 100) if max_sr > 0 else 0
            
            # Calculate final batting score (0-100 scale)
            team_batting['batting_score'] = (
                runs_score * 0.6 +      # Runs (60%)
                avg_score * 0.25 +      # Average (25%)
                sr_score * 0.15         # Strike Rate (15%)
            )
            
            # Get top 3 batsmen
            top_batsmen = team_batting.nlargest(3, 'batting_score')
            for _, batsman in top_batsmen.iterrows():
                batsmen_data.append({
                    'player_name': batsman['player_name'],
                    'matches': int(batsman['matches']),
                    'runs': int(batsman['runs']),
                    'average': round(float(batsman['average']), 2),
                    'strike_rate': round(float(batsman['strike_rate']), 2),
                    'performance_score': round(float(batsman['batting_score']), 2)
                })

        # Calculate bowling statistics
        if not team_bowling.empty:
            # Normalize metrics to 0-100 scale
            max_wickets = team_bowling['wickets'].max()
            min_economy = team_bowling['economy'].min()
            max_economy = team_bowling['economy'].max()
            max_wpm = (team_bowling['wickets'] / team_bowling['matches']).max()
            
            # Calculate normalized components (0-100 scale)
            wickets_score = (team_bowling['wickets'] / max_wickets * 100) if max_wickets > 0 else 0
            # Economy score (inverse - lower is better)
            economy_score = ((max_economy - team_bowling['economy']) / (max_economy - min_economy) * 100) if (max_economy - min_economy) > 0 else 0
            wpm_score = ((team_bowling['wickets'] / team_bowling['matches']) / max_wpm * 100) if max_wpm > 0 else 0
            
            # Calculate final bowling score (0-100 scale)
            team_bowling['bowling_score'] = (
                wickets_score * 0.6 +    # Wickets (60%)
                economy_score * 0.25 +    # Economy (25%)
                wpm_score * 0.15         # Wickets per match (15%)
            )
            
            # Get top 3 bowlers
            top_bowlers = team_bowling.nlargest(3, 'bowling_score')
            for _, bowler in top_bowlers.iterrows():
                bowlers_data.append({
                    'player_name': bowler['player_name'],
                    'matches': int(bowler['matches']),
                    'wickets': int(bowler['wickets']),
                    'economy': round(float(bowler['economy']), 2),
                    'balls_delivered': int(bowler['balls_delivered']),
                    'performance_score': round(float(bowler['bowling_score']), 2)
                })

        if not batsmen_data and not bowlers_data:
            return jsonify({'status': 'error', 'message': 'No player statistics available'}), 404

        return jsonify({
            'status': 'success',
            'bestBatsmen': batsmen_data,
            'bestBowlers': bowlers_data
        })

    except Exception as e:
        print(f"Error in team_best_players: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/rankings')
def rankings_options():
    return render_template('rankings_options.html')

@app.route('/rankings/batting')
def batting_rankings():
    return render_template('batting_rankings.html')

@app.route('/rankings/bowling')
def bowling_rankings():
    return render_template('bowling_rankings.html')

@app.route('/rankings/allrounders')
def allrounder_rankings():
    return render_template('allrounder_rankings.html')

@app.route('/match-outcome')
def match_outcome():
    return render_template('match_outcome.html')

@app.route('/team-prediction')
def team_prediction():
    return render_template('team_prediction.html')

@app.route('/player-prediction')
def player_prediction():
    return render_template('player_prediction.html')

@app.route('/get_team_venues', methods=['GET'])
def get_team_venues():
    try:
        if not matches_df.empty:
            team_venues = {}
            for _, match in matches_df.iterrows():
                team1, team2, venue = match['team1'], match['team2'], match['venue']
                if pd.notna(venue):  # Filter out NaN venues
                    team_venues.setdefault(team1, set()).add(venue)
                    team_venues.setdefault(team2, set()).add(venue)
            available_teams = [team for team in team_venues.keys() if team in batting_stats['team'].unique()]
            return jsonify({'status': 'success', 'team_venues': {k: sorted(list(v)) for k, v in team_venues.items() if k in available_teams}})
        return jsonify({'status': 'error', 'message': 'Matches data not loaded'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_team_players')
def get_team_players():
    team = request.args.get('team')
    if not team:
        return jsonify({'status': 'error', 'message': 'Team parameter is required'})

    try:
        # Get unique players for the team (both batting and bowling)
        team_players = set()
        
        # Add batsmen from batting stats
        if not batting_stats.empty:
            batsmen = batting_stats[batting_stats['team'] == team]['player_name'].unique()
            team_players.update(batsmen)
        
        # Add bowlers from bowling stats
        if not bowling_stats.empty:
            bowlers = bowling_stats[bowling_stats['team'] == team]['player_name'].unique()
            team_players.update(bowlers)
        
        # Convert set to sorted list
        players_list = sorted(list(team_players))
        
        if not players_list:
            return jsonify({
                'status': 'error',
                'message': f'No players found for team {team}'
            })
        
        return jsonify({
            'status': 'success',
            'players': players_list
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/get_common_venues', methods=['GET'])
def get_common_venues():
    try:
        team1, team2 = request.args.get('team1'), request.args.get('team2')
        if not team1 or not team2:
            return jsonify({'status': 'error', 'message': 'Both teams are required'}), 400
        if not matches_df.empty:
            team1_venues = set(matches_df[(matches_df['team1'] == team1) | (matches_df['team2'] == team1)]['venue'].dropna().unique())
            team2_venues = set(matches_df[(matches_df['team1'] == team2) | (matches_df['team2'] == team2)]['venue'].dropna().unique())
            venues = sorted(list(team1_venues & team2_venues))
            return jsonify({'status': 'success', 'venues': venues})
        return jsonify({'status': 'error', 'message': 'Matches data not loaded'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/team_player_stats', methods=['POST'])
def team_player_stats():
    try:
        data = request.get_json()
        if not data or 'team' not in data:
            return jsonify({'status': 'error', 'message': 'Team not provided'}), 400

        team = data['team']
        if not batting_stats.empty and not bowling_stats.empty:
            # Get batting stats for the team
            team_batting = batting_stats[batting_stats['team'] == team]
            # Get bowling stats for the team
            team_bowling = bowling_stats[bowling_stats['team'] == team]

            # Combine stats for each player
            players = []
            for _, batsman in team_batting.iterrows():
                player_data = {
                    'player_name': batsman['player_name'],
                    'runs': int(batsman['runs']),
                    'average': round(float(batsman['average']), 2),
                    'strike_rate': round(float(batsman['strike_rate']), 2),
                    'matches': int(batsman['matches']),
                    'highest_score': int(batsman['highest_score']),
                    'wickets': 0,
                    'economy': 0.0
                }
                
                # Add bowling stats if player is also a bowler
                bowler = team_bowling[team_bowling['player_name'] == batsman['player_name']]
                if not bowler.empty:
                    # Format best figures as 'wickets/runs'
                    best_figures = bowler.iloc[0]['best_figures']
                    if isinstance(best_figures, str) and '/' not in best_figures:
                        # If best_figures is a date or incorrect format, set to '0/0'
                        best_figures = '0/0'
                    
                    player_data.update({
                        'wickets': int(bowler.iloc[0]['wickets']),
                        'economy': round(float(bowler.iloc[0]['economy']), 2),
                        'best_figures': best_figures
                    })
                
                players.append(player_data)

            # Add players who are only bowlers
            bowlers_only = team_bowling[~team_bowling['player_name'].isin(team_batting['player_name'])]
            for _, bowler in bowlers_only.iterrows():
                # Format best figures as 'wickets/runs'
                best_figures = bowler['best_figures']
                if isinstance(best_figures, str) and '/' not in best_figures:
                    # If best_figures is a date or incorrect format, set to '0/0'
                    best_figures = '0/0'
                
                players.append({
                    'player_name': bowler['player_name'],
                    'runs': 0,
                    'average': 0.0,
                    'strike_rate': 0.0,
                    'matches': int(bowler['matches']),
                    'highest_score': 0,
                    'wickets': int(bowler['wickets']),
                    'economy': round(float(bowler['economy']), 2),
                    'best_figures': best_figures
                })

            return jsonify({
                'status': 'success',
                'players': sorted(players, key=lambda x: (x['runs'] + x['wickets'] * 20), reverse=True)
            })
        else:
            return jsonify({'status': 'error', 'message': 'Data not loaded'}), 500
    except Exception as e:
        print(f"Error in team_player_stats: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/venue_best_player', methods=['POST'])
def venue_best_player():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400

        team1 = data.get('team1')
        team2 = data.get('team2')
        venue = data.get('venue')
        selected_team = data.get('selectedTeam')

        if not all([team1, team2, venue, selected_team]):
            return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400

        if not matches_df.empty and not deliveries_df.empty:
            # Get matches at the venue between these teams
            venue_matches = matches_df[
                (matches_df['venue'] == venue) &
                ((matches_df['team1'].isin([team1, team2])) & (matches_df['team2'].isin([team1, team2])))
            ]

            if venue_matches.empty:
                return jsonify({'status': 'error', 'message': 'No matches found for these teams at this venue'}), 404

            # Get deliveries for these matches
            venue_deliveries = deliveries_df[
                deliveries_df['match_id'].isin(venue_matches['id']) &
                (deliveries_df['batting_team'] == selected_team)
            ]

            if venue_deliveries.empty:
                return jsonify({'status': 'error', 'message': 'No batting data found for selected team at this venue'}), 404

            # Calculate batting statistics
            batting_stats = venue_deliveries.groupby('batsman').agg({
                'batsman_runs': ['sum', 'count'],
                'match_id': 'nunique'
            }).reset_index()

            batting_stats.columns = ['player_name', 'runs', 'balls', 'matches']
            batting_stats['average'] = batting_stats['runs'] / batting_stats['matches']
            batting_stats['strike_rate'] = (batting_stats['runs'] / batting_stats['balls']) * 100

            # Get bowling statistics for the same matches
            bowling_deliveries = deliveries_df[
                deliveries_df['match_id'].isin(venue_matches['id']) &
                (deliveries_df['bowling_team'] == selected_team)
            ]

            bowling_stats = None
            if not bowling_deliveries.empty:
                # Calculate bowling statistics
                bowling_stats = bowling_deliveries.groupby('bowler').agg({
                    'player_dismissed': lambda x: x.notna().sum(),  # Count wickets
                    'total_runs': 'sum',
                    'match_id': 'nunique',
                    'ball': 'count'
                }).reset_index()
                bowling_stats.columns = ['player_name', 'wickets', 'runs_conceded', 'matches', 'balls']
                bowling_stats['economy'] = (bowling_stats['runs_conceded'] / (bowling_stats['balls'] / 6))
                bowling_stats['bowling_average'] = bowling_stats.apply(
                    lambda x: x['runs_conceded'] / x['wickets'] if x['wickets'] > 0 else float('inf'),
                    axis=1
                )

            if batting_stats.empty and (bowling_stats is None or bowling_stats.empty):
                return jsonify({'status': 'error', 'message': 'No player statistics available'}), 404

            # Calculate overall performance score considering both batting and bowling
            all_players = pd.DataFrame()
            
            # Process batting statistics
            if not batting_stats.empty:
                batting_stats['batting_score'] = (
                    (batting_stats['runs'] * batting_stats['strike_rate'] / 100) *  # Run scoring ability
                    (batting_stats['matches'] / venue_matches.shape[0]) *  # Experience at venue
                    (batting_stats['average'] / batting_stats['average'].mean())  # Relative performance
                )
                all_players = batting_stats[['player_name', 'runs', 'balls', 'matches', 'average', 'strike_rate', 'batting_score']]
                all_players['role'] = 'Batsman'

            # Process bowling statistics
            if bowling_stats is not None and not bowling_stats.empty:
                bowling_stats['bowling_score'] = (
                    (bowling_stats['wickets'] * 20) *  # Wicket-taking ability
                    (bowling_stats['matches'] / venue_matches.shape[0]) *  # Experience at venue
                    (1 / (bowling_stats['economy'] + 1))  # Economy rate factor
                )
                
                # Merge bowling stats with existing players or create new entries
                if not all_players.empty:
                    # Update existing players with bowling stats
                    bowling_data = bowling_stats[['player_name', 'wickets', 'economy', 'bowling_score', 'bowling_average']]
                    all_players = all_players.merge(bowling_data, on='player_name', how='outer')
                    all_players['role'] = all_players.apply(
                        lambda x: 'All-Rounder' if pd.notna(x.get('batting_score')) and pd.notna(x.get('bowling_score'))
                        else ('Batsman' if pd.notna(x.get('batting_score')) else 'Bowler'),
                        axis=1
                    )
                else:
                    # Create new entries for bowlers only
                    all_players = bowling_stats[['player_name', 'wickets', 'economy', 'bowling_score', 'bowling_average', 'matches']]
                    all_players['role'] = 'Bowler'

            # Fill NaN values with 0
            all_players = all_players.fillna(0)

            # Calculate total score
            all_players['total_score'] = all_players.get('batting_score', 0) + all_players.get('bowling_score', 0)

            # Find best player based on total score
            best_player_row = all_players.loc[all_players['total_score'].idxmax()]

            # Calculate predicted score and confidence
            predicted_score = 0
            confidence = 0
            venue_stats = {}

            if best_player_row['role'] in ['Batsman', 'All-Rounder']:
                # Batting prediction - More realistic for T20
                base_score = min(best_player_row['average'], 80)  # Cap average at 80
                form_factor = min(best_player_row['strike_rate'] / 150, 1.5)  # Normalize strike rate relative to 150
                venue_factor = min(best_player_row['matches'] / len(venue_matches), 1.2)  # Cap venue experience boost
                
                # Calculate predicted score with realistic constraints
                raw_prediction = int(base_score * form_factor * venue_factor)
                predicted_score = min(max(raw_prediction, 20), 100)  # Keep prediction between 20 and 100 runs

                venue_stats.update({
                    'matches': int(best_player_row['matches']),
                    'runs': int(best_player_row.get('runs', 0)),
                    'average': round(float(best_player_row['average']), 2),
                    'strikeRate': round(float(best_player_row['strike_rate']), 2)
                })

            if best_player_row['role'] in ['Bowler', 'All-Rounder']:
                # More realistic bowling stats
                venue_stats.update({
                    'wickets': int(best_player_row.get('wickets', 0)),
                    'economy': round(float(best_player_row.get('economy', 0)), 2),
                    'bowling_average': round(float(best_player_row.get('bowling_average', 0)), 2)
                })

            # Calculate confidence based on multiple factors with adjusted weights
            experience_confidence = min(35, (best_player_row['matches'] / len(venue_matches)) * 100)
            recent_form = min(35, (best_player_row['total_score'] / all_players['total_score'].max()) * 100)
            role_bonus = 20 if best_player_row['role'] == 'All-Rounder' else 15
            
            # Additional confidence modifiers based on realistic stats
            if best_player_row['role'] in ['Batsman', 'All-Rounder']:
                if best_player_row['average'] > 40:
                    role_bonus += 5
                if best_player_row['strike_rate'] > 140:
                    role_bonus += 5
            
            confidence = min(90, int(experience_confidence + recent_form + role_bonus))  # Cap at 90%

            return jsonify({
                'status': 'success',
                'bestPlayer': best_player_row['player_name'],
                'role': best_player_row['role'],
                'predictedScore': predicted_score,
                'venueStats': venue_stats,
                'totalMatches': len(venue_matches),
                'confidence': confidence
            })
        else:
            return jsonify({'status': 'error', 'message': 'Required data not loaded'}), 500

    except Exception as e:
        print(f"Error in venue_best_player: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/top_batsmen', methods=['GET'])
def top_batsmen():
    if not batting_stats.empty:
        top_batsmen = batting_stats.nlargest(10, 'runs').to_dict('records')
        return jsonify({'status': 'success', 'batsmen': top_batsmen})
    return jsonify({'status': 'error', 'message': 'Data not loaded'})

@app.route('/top_bowlers', methods=['GET'])
def top_bowlers():
    if not bowling_stats.empty:
        top_bowlers = bowling_stats.nlargest(10, 'wickets').to_dict('records')
        return jsonify({'status': 'success', 'bowlers': top_bowlers})
    return jsonify({'status': 'error', 'message': 'Data not loaded'})

@app.route('/top_allrounders', methods=['GET'])
def top_allrounders():
    if not batting_stats.empty and not bowling_stats.empty:
        allrounders = pd.merge(batting_stats, bowling_stats, on=['player_name', 'team'], how='inner')
        top_allrounders = allrounders.nlargest(10, ['runs', 'wickets'], keep='all').to_dict('records')
        return jsonify({'status': 'success', 'allrounders': top_allrounders})
    return jsonify({'status': 'error', 'message': 'Data not loaded'})

@app.route('/player_details', methods=['POST'])
def player_details():
    player_name = request.form.get('playerName')
    if not batting_stats.empty and not bowling_stats.empty:
        player_batting = batting_stats[batting_stats['player_name'] == player_name].iloc[0].to_dict() if player_name in batting_stats['player_name'].values else {}
        player_bowling = bowling_stats[bowling_stats['player_name'] == player_name].iloc[0].to_dict() if player_name in bowling_stats['player_name'].values else {}
        return jsonify({
            'status': 'success',
            'player': {**player_batting, **player_bowling}
        })
    return jsonify({'status': 'error', 'message': 'Data not loaded'})

def calculate_win_probability(team1, team2, venue):
    """Calculate win probability based on historical head-to-head and venue performance"""
    if not matches_df.empty:
        # Get head-to-head matches
        h2h_matches = matches_df[
            ((matches_df['team1'] == team1) & (matches_df['team2'] == team2)) |
            ((matches_df['team1'] == team2) & (matches_df['team2'] == team1))
        ]
        
        # Get venue matches
        venue_matches = matches_df[
            (matches_df['venue'] == venue) &
            ((matches_df['team1'] == team1) | (matches_df['team2'] == team1))
        ]
        
        # Calculate head-to-head win rate
        team1_h2h_wins = h2h_matches[
            ((h2h_matches['team1'] == team1) & (h2h_matches['winner'] == team1)) |
            ((h2h_matches['team2'] == team1) & (h2h_matches['winner'] == team1))
        ].shape[0]
        h2h_win_rate = team1_h2h_wins / len(h2h_matches) if len(h2h_matches) > 0 else 0.5
        
        # Calculate venue win rate
        team1_venue_wins = venue_matches[venue_matches['winner'] == team1].shape[0]
        venue_win_rate = team1_venue_wins / len(venue_matches) if len(venue_matches) > 0 else 0.5
        
        # Calculate final probability (weighted average)
        win_prob = (h2h_win_rate * 0.6) + (venue_win_rate * 0.4)
        
        # Ensure probability is between 30% and 70% for more realistic predictions
        win_prob = min(max(win_prob * 100, 30), 70)
        
        return round(win_prob)
    else:
        # Default to slightly favoring team1 if no historical data
        return 52

@app.route('/predict_match', methods=['POST'])
def predict_match():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400

        team1 = data.get('team1')
        team2 = data.get('team2')
        venue = data.get('venue')

        if not all([team1, team2, venue]):
            return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400

        # Calculate win probabilities
        team1_win_prob = calculate_win_probability(team1, team2, venue)
        
        # Predict toss winner based on historical toss data
        toss_winner = predict_toss_winner(team1, team2)
        
        # Get key players for both teams
        team1_players = get_key_players(team1)
        team2_players = get_key_players(team2)

        # Calculate realistic score ranges based on venue and team averages
        team1_score_range = calculate_score_range(team1, venue)
        team2_score_range = calculate_score_range(team2, venue)

        return jsonify({
            'status': 'success',
            'team1': team1,
            'team2': team2,
            'team1_win_probability': team1_win_prob,
            'toss_winner': toss_winner,
            'team1_score_range': team1_score_range,
            'team2_score_range': team2_score_range,
            'team1_key_batsmen': team1_players['batsmen'][:2],
            'team1_key_bowlers': team1_players['bowlers'][:2],
            'team2_key_batsmen': team2_players['batsmen'][:2],
            'team2_key_bowlers': team2_players['bowlers'][:2]
        })

    except Exception as e:
        print(f"Error in predict_match: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def calculate_score_range(team, venue):
    """Calculate expected score range based on team's performance at venue"""
    if not matches_df.empty and not deliveries_df.empty:
        # Get team's matches at the venue
        venue_matches = matches_df[
            (matches_df['venue'] == venue) &
            ((matches_df['team1'] == team) | (matches_df['team2'] == team))
        ]
        
        if not venue_matches.empty:
            # Get all innings scores for the team at this venue
            team_scores = []
            for _, match in venue_matches.iterrows():
                match_deliveries = deliveries_df[deliveries_df['match_id'] == match['id']]
                if match['team1'] == team:
                    score = match_deliveries[match_deliveries['batting_team'] == team]['total_runs'].sum()
                    team_scores.append(score)
                elif match['team2'] == team:
                    score = match_deliveries[match_deliveries['batting_team'] == team]['total_runs'].sum()
                    team_scores.append(score)
            
            if team_scores:
                avg_score = sum(team_scores) / len(team_scores)
                # Create a realistic range: -20% to +20% of average score
                min_score = max(140, int(avg_score * 0.8))  # minimum 140
                max_score = min(220, int(avg_score * 1.2))  # maximum 220
                return f"{min_score}-{max_score}"
    
    # Default range if no data available
    return "160-180"

def get_key_players(team):
    """Get key batsmen and bowlers for a team"""
    key_players = {
        'batsmen': [],
        'bowlers': []
    }
    
    if not batting_stats.empty and not bowling_stats.empty:
        # Get top batsmen by runs
        team_batsmen = batting_stats[batting_stats['team'] == team].nlargest(2, 'runs')
        key_players['batsmen'] = team_batsmen['player_name'].tolist()
        
        # Get top bowlers by wickets
        team_bowlers = bowling_stats[bowling_stats['team'] == team].nlargest(2, 'wickets')
        key_players['bowlers'] = team_bowlers['player_name'].tolist()
    
    return key_players

@app.route('/predict_player', methods=['POST'])
def predict_player():
    player_name = request.form.get('playerName')
    team1 = request.form.get('team1')
    team2 = request.form.get('team2')
    venue = request.form.get('venue')
    if player_model and not batting_stats.empty and not bowling_stats.empty:
        player_batting = batting_stats[batting_stats['player_name'] == player_name].iloc[0] if player_name in batting_stats['player_name'].values else None
        player_bowling = bowling_stats[bowling_stats['player_name'] == player_name].iloc[0] if player_name in bowling_stats['player_name'].values else None

        if player_batting is None and player_bowling is None:
            return jsonify({'status': 'error', 'message': 'Player not found'})

        # Feature engineering for player prediction
        features = np.array([
            [player_batting['runs'] if player_batting is not None else 0,
             player_batting['average'] if player_batting is not None else 0,
             player_batting['strike_rate'] if player_batting is not None else 0,
             player_bowling['wickets'] if player_bowling is not None else 0,
             player_bowling['economy'] if player_bowling is not None else 0]
        ])
        prediction = player_model.predict(features)[0]  # Predicted performance score
        confidence = max(70, min(95, player_model.predict_proba(features)[0].max() * 100))  # Confidence based on max probability

        # Dynamic performance prediction
        predicted_runs = int(prediction * 0.5) if player_batting is not None else 0  # Scaled runs
        predicted_wickets = int(prediction * 0.5) if player_bowling is not None else 0  # Scaled wickets

        return jsonify({
            'status': 'success',
            'prediction': {
                'player_name': player_name,
                'role': 'All-Rounder' if player_batting is not None and player_bowling is not None else ('Batsman' if player_batting is not None else 'Bowler'),
                'batting_stats': {
                    'matches': int(player_batting['matches']) if player_batting is not None else 0,
                    'runs': predicted_runs,
                    'average': round(player_batting['average'], 2) if player_batting is not None else 0,
                    'strike_rate': round(player_batting['strike_rate'], 2) if player_batting is not None else 0,
                    'centuries': int(player_batting['runs'] // 100) if player_batting is not None else 0,
                    'fifties': int((player_batting['runs'] % 100) // 50) if player_batting is not None else 0
                },
                'bowling_stats': {
                    'matches': int(player_bowling['matches']) if player_bowling is not None else 0,
                    'wickets': predicted_wickets,
                    'average': round(player_bowling['average'], 2) if player_bowling is not None and pd.notna(player_bowling['average']) else 0,
                    'economy': round(player_bowling['economy'], 2) if player_bowling is not None else 0,
                    'best': player_bowling['best_figures'] if player_bowling is not None else '0/0'
                },
                'confidence': confidence
            }
        })
    elif not batting_stats.empty and not bowling_stats.empty:
        # Fallback placeholder logic
        player_batting = batting_stats[batting_stats['player_name'] == player_name].iloc[0] if player_name in batting_stats['player_name'].values else None
        player_bowling = bowling_stats[bowling_stats['player_name'] == player_name].iloc[0] if player_name in bowling_stats['player_name'].values else None

        if player_batting is None and player_bowling is None:
            return jsonify({'status': 'error', 'message': 'Player not found'})

        predicted_runs = int(player_batting['runs'] * 1.05) if player_batting is not None else 0
        predicted_wickets = int(player_bowling['wickets'] * 1.05) if player_bowling is not None else 0

        return jsonify({
            'status': 'success',
            'prediction': {
                'player_name': player_name,
                'role': 'All-Rounder' if player_batting is not None and player_bowling is not None else ('Batsman' if player_batting is not None else 'Bowler'),
                'batting_stats': {
                    'matches': int(player_batting['matches']) if player_batting is not None else 0,
                    'runs': predicted_runs,
                    'average': round(player_batting['average'] * 1.05, 2) if player_batting is not None else 0,
                    'strike_rate': round(player_batting['strike_rate'] * 1.05, 2) if player_batting is not None else 0,
                    'centuries': int(player_batting['runs'] // 100) if player_batting is not None else 0,
                    'fifties': int((player_batting['runs'] % 100) // 50) if player_batting is not None else 0
                },
                'bowling_stats': {
                    'matches': int(player_bowling['matches']) if player_bowling is not None else 0,
                    'wickets': predicted_wickets,
                    'average': round(player_bowling['average'] * 0.95, 2) if player_bowling is not None and pd.notna(player_bowling['average']) else 0,
                    'economy': round(player_bowling['economy'] * 0.95, 2) if player_bowling is not None else 0,
                    'best': player_bowling['best_figures'] if player_bowling is not None else '0/0'
                },
                'confidence': 70
            }
        })
    return jsonify({'status': 'error', 'message': 'Data not loaded'})

@app.route('/predict_player_performance', methods=['POST'])
def predict_player_performance():
    try:
        data = request.get_json()
        player_name = data.get('player_name')
        team1 = data.get('team1')
        team2 = data.get('team2')
        venue = data.get('venue')

        if not all([player_name, team1, team2, venue]):
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields'
            }), 400

        # Validate player belongs to one of the teams
        player_team = None
        if player_name in batting_stats[batting_stats['team'] == team1]['player_name'].values:
            player_team = team1
        elif player_name in batting_stats[batting_stats['team'] == team2]['player_name'].values:
            player_team = team2
        
        if not player_team:
            return jsonify({
                'status': 'error',
                'message': f'Player {player_name} not found in either {team1} or {team2}'
            }), 404

        # Get opponent team
        opponent = team2 if player_team == team1 else team1

        # Get player's stats
        player_batting = batting_stats[batting_stats['player_name'] == player_name].copy()
        player_bowling = bowling_stats[bowling_stats['player_name'] == player_name].copy()

        # Get venue-specific performance
        venue_matches = matches_df[
            (matches_df['venue'] == venue) & 
            ((matches_df['team1'].isin([team1, team2])) & (matches_df['team2'].isin([team1, team2])))
        ]

        venue_deliveries = deliveries_df[deliveries_df['match_id'].isin(venue_matches['id'])]
        
        prediction_result = {
            'player_name': player_name,
            'player_team': player_team,
            'opponent': opponent,
            'venue': venue
        }

        # Calculate batting predictions
        if not player_batting.empty:
            # Get venue stats
            venue_batting = venue_deliveries[venue_deliveries['batsman'] == player_name]
            venue_runs = int(venue_batting['batsman_runs'].sum()) if not venue_batting.empty else 0
            venue_matches_count = len(venue_batting['match_id'].unique()) if not venue_batting.empty else 0
            
            # Calculate predicted runs
            overall_avg = float(player_batting['average'].iloc[0])
            venue_avg = float(venue_runs / venue_matches_count if venue_matches_count > 0 else overall_avg)
            
            # Calculate form factor based on recent matches
            recent_matches = venue_deliveries[venue_deliveries['batsman'] == player_name].tail(5)
            recent_avg = float(recent_matches['batsman_runs'].mean()) if not recent_matches.empty else overall_avg
            
            # Weighted prediction
            predicted_runs = int(
                overall_avg * 0.4 +    # Overall form: 40%
                venue_avg * 0.4 +      # Venue performance: 40%
                recent_avg * 0.2       # Recent form: 20%
            )
            
            # Add batting stats
            prediction_result.update({
                'predicted_runs': predicted_runs,
                'batting_stats': {
                    'overall_average': round(float(overall_avg), 2),
                    'venue_average': round(float(venue_avg), 2),
                    'venue_matches': int(venue_matches_count),
                    'strike_rate': round(float(player_batting['strike_rate'].iloc[0]), 2)
                }
            })

        # Calculate bowling predictions
        if not player_bowling.empty:
            # Get venue bowling stats
            venue_bowling = venue_deliveries[venue_deliveries['bowler'] == player_name]
            venue_wickets = int(venue_bowling['player_dismissed'].notna().sum()) if not venue_bowling.empty else 0
            venue_matches_count = len(venue_bowling['match_id'].unique()) if not venue_bowling.empty else 0
            
            # Calculate predicted wickets
            overall_wickets_per_match = float(player_bowling['wickets'].iloc[0] / player_bowling['matches'].iloc[0])
            venue_wickets_per_match = float(venue_wickets / venue_matches_count if venue_matches_count > 0 else overall_wickets_per_match)
            
            # Weighted prediction
            predicted_wickets = int(
                overall_wickets_per_match * 0.5 +     # Overall form: 50%
                venue_wickets_per_match * 0.5         # Venue performance: 50%
            )
            
            # Add bowling stats
            prediction_result.update({
                'predicted_wickets': predicted_wickets,
                'bowling_stats': {
                    'overall_economy': round(float(player_bowling['economy'].iloc[0]), 2),
                    'venue_wickets': int(venue_wickets),
                    'venue_matches': int(venue_matches_count)
                }
            })

        # Calculate success probability
        base_probability = 60  # Base 60% chance
        form_boost = min(20, (prediction_result.get('predicted_runs', 0) / 30 * 10 +  # Up to 10% for runs
                             prediction_result.get('predicted_wickets', 0) / 3 * 10))  # Up to 10% for wickets
        venue_boost = 10 if venue_matches_count >= 3 else (venue_matches_count * 3)
        
        prediction_result['success_probability'] = float(min(90, base_probability + form_boost + venue_boost))

        return jsonify({
            'status': 'success',
            'prediction': prediction_result
        })

    except Exception as e:
        print(f"Error in predict_player_performance: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/teams')
def get_teams():
    try:
        if not matches_df.empty:
            teams = sorted(list(set(matches_df['team1'].unique()) | set(matches_df['team2'].unique())))
            return jsonify({'status': 'success', 'teams': teams})
        return jsonify({'status': 'error', 'message': 'Matches data not loaded'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/venues')
def get_venues():
    try:
        if not matches_df.empty:
            venues = sorted(matches_df['venue'].dropna().unique().tolist())
            return jsonify({'status': 'success', 'venues': venues})
        return jsonify({'status': 'error', 'message': 'Matches data not loaded'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/rankings/batting')
def get_batting_rankings():
    try:
        if not batting_stats.empty:
            top_batsmen = batting_stats.nlargest(10, 'runs')[['player_name', 'team', 'matches', 'runs', 'average', 'strike_rate', 'highest_score']].to_dict('records')
            return jsonify({'status': 'success', 'rankings': top_batsmen})
        return jsonify({'status': 'error', 'message': 'Data not loaded'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/rankings/bowling')
def get_bowling_rankings():
    try:
        if not bowling_stats.empty:
            top_bowlers = bowling_stats.nlargest(10, 'wickets')[['player_name', 'team', 'matches', 'wickets', 'economy', 'balls_delivered', 'runs_conceded']].to_dict('records')
            return jsonify({'status': 'success', 'rankings': top_bowlers})
        return jsonify({'status': 'error', 'message': 'Data not loaded'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/rankings/allrounders')
def get_allrounder_rankings():
    try:
        if not batting_stats.empty and not bowling_stats.empty:
            # Ensure consistent column names before merging
            batting_cols = ['player_name', 'team', 'matches', 'runs', 'average', 'strike_rate']
            bowling_cols = ['player_name', 'team', 'matches', 'wickets', 'economy']
            
            # Select and rename columns for merging
            batsmen = batting_stats[batting_cols].copy()
            bowlers = bowling_stats[bowling_cols].copy()
            
            # Merge batting and bowling stats
            allrounders = pd.merge(batsmen, bowlers, on=['player_name', 'team'], suffixes=('_batting', '_bowling'))
            
            # Calculate normalized batting metrics
            allrounders['runs_score'] = (allrounders['runs'] / allrounders['runs'].max() * 100) if allrounders['runs'].max() > 0 else 0
            allrounders['avg_score'] = (allrounders['average'] / allrounders['average'].max() * 100) if allrounders['average'].max() > 0 else 0
            allrounders['sr_score'] = (allrounders['strike_rate'] / allrounders['strike_rate'].max() * 100) if allrounders['strike_rate'].max() > 0 else 0
            
            # Calculate batting score (max 50 points)
            allrounders['batting_score'] = (
                allrounders['runs_score'] * 0.3 +      # Runs (30 points)
                allrounders['avg_score'] * 0.1 +       # Average (10 points)
                allrounders['sr_score'] * 0.1          # Strike Rate (10 points)
            )
            
            # Calculate normalized bowling metrics
            allrounders['wickets_score'] = (allrounders['wickets'] / allrounders['wickets'].max() * 100) if allrounders['wickets'].max() > 0 else 0
            max_economy = allrounders['economy'].max()
            min_economy = allrounders['economy'].min()
            allrounders['economy_score'] = ((max_economy - allrounders['economy']) / (max_economy - min_economy) * 100) if (max_economy - min_economy) > 0 else 0
            
            # Calculate bowling score (max 50 points)
            allrounders['bowling_score'] = (
                allrounders['wickets_score'] * 0.3 +   # Wickets (30 points)
                allrounders['economy_score'] * 0.2     # Economy (20 points)
            )
            
            # Calculate final performance index (0-100)
            allrounders['performance_index'] = allrounders['batting_score'] + allrounders['bowling_score']
            
            # Filter for minimum qualification criteria
            qualified_allrounders = allrounders[
                (allrounders['runs'] >= 200) &         # Minimum runs
                (allrounders['wickets'] >= 15)         # Minimum wickets
            ]
            
            # Get top 10 all-rounders
            top_allrounders = qualified_allrounders.nlargest(10, 'performance_index')
            
            # Format the output
            result = []
            for idx, player in top_allrounders.iterrows():
                result.append({
                    'rank': len(result) + 1,
                    'player': player['player_name'],
                    'team': player['team'],
                    'matches': int(player['matches_batting']),
                    'runs': int(player['runs']),
                    'batting_avg': round(float(player['average']), 2),
                    'wickets': int(player['wickets']),
                    'economy': round(float(player['economy']), 2),
                    'performance_index': round(float(player['performance_index']), 2)
                })
            
            return jsonify({
                'status': 'success',
                'rankings': result
            })
            
        return jsonify({'status': 'error', 'message': 'Required data not loaded'}), 500
        
    except Exception as e:
        print(f"Error in get_allrounder_rankings: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/players', methods=['GET'])
def get_players():
    try:
        team1 = request.args.get('team1')
        team2 = request.args.get('team2')
        
        if not team1 or not team2:
            return jsonify({'status': 'error', 'message': 'Both teams are required'}), 400
            
        if batting_stats.empty or bowling_stats.empty:
            return jsonify({'status': 'error', 'message': 'Player data not loaded'}), 500
            
        # Get players from both teams
        team1_players = set(batting_stats[batting_stats['team'] == team1]['player_name'].dropna().tolist() + 
                          bowling_stats[bowling_stats['team'] == team1]['player_name'].dropna().tolist())
                          
        team2_players = set(batting_stats[batting_stats['team'] == team2]['player_name'].dropna().tolist() + 
                          bowling_stats[bowling_stats['team'] == team2]['player_name'].dropna().tolist())
        
        # Combine and sort players
        all_players = sorted(list(team1_players | team2_players))
        
        if not all_players:
            return jsonify({'status': 'error', 'message': 'No players found for the selected teams'}), 404
            
        return jsonify({
            'status': 'success',
            'players': all_players
        })
        
    except Exception as e:
        print(f"Error getting players: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def predict_toss_winner(team1, team2):
    """Predict toss winner based on historical toss win rates"""
    if not matches_df.empty:
        # Get toss statistics for both teams
        team1_tosses = matches_df[
            ((matches_df['team1'] == team1) | (matches_df['team2'] == team1))
        ]
        team1_toss_wins = team1_tosses[team1_tosses['toss_winner'] == team1].shape[0]
        team1_toss_rate = team1_toss_wins / len(team1_tosses) if len(team1_tosses) > 0 else 0.5

        team2_tosses = matches_df[
            ((matches_df['team1'] == team2) | (matches_df['team2'] == team2))
        ]
        team2_toss_wins = team2_tosses[team2_tosses['toss_winner'] == team2].shape[0]
        team2_toss_rate = team2_toss_wins / len(team2_tosses) if len(team2_tosses) > 0 else 0.5

        # Return team with better toss win rate
        return team1 if team1_toss_rate > team2_toss_rate else team2
    else:
        # If no historical data, return random team
        return team1 if random.random() > 0.5 else team2

if __name__ == '__main__':
    app.run(debug=True)