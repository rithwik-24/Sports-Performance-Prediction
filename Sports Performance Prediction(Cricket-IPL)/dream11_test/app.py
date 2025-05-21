from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
from itertools import combinations

app = Flask(__name__)
app.secret_key = 'dream11_secret'

# Load datasets with error handling
try:
    batting_stats = pd.read_csv('batting_stats.csv')
    bowling_stats = pd.read_csv('bowling_stats.csv')
    matches = pd.read_csv('matches.csv')
    deliveries = pd.read_csv('deliveries.csv')
except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
except Exception as e:
    print(f"Error loading datasets: {e}")

# Define IPL teams and updated 2025 squads with predefined roles
ipl_teams_2025 = {
    'CSK': {
        'Batsmen': ['Ruturaj Gaikwad', 'Devon Conway', 'Rahul Tripathi', 'Shaik Rasheed'],
        'Wicket-Keepers': ['MS Dhoni', 'Vansh Bedi'],
        'All-Rounders': ['Ravindra Jadeja', 'Shivam Dube', 'Rachin Ravindra', 'Vijay Shankar', 
                         'Sam Curran', 'Jamie Overton', 'Deepak Hooda', 'Anshul Kamboj', 'Ramakrishna Ghosh'],
        'Bowlers': ['Matheesha Pathirana', 'Noor Ahmad', 'Ravichandran Ashwin', 'Khaleel Ahmed', 
                    'Mukesh Choudhary', 'Gurjapneet Singh', 'Nathan Ellis', 'Kamlesh Nagarkoti', 
                    'Shreyas Gopal', 'Andre Siddarth']
    },
    'DC': {
        'Batsmen': ['Jake Fraser-McGurk', 'Karun Nair', 'Faf du Plessis', 'Abhishek Porel', 'Tristan Stubbs'],
        'Wicket-Keepers': ['KL Rahul'],
        'All-Rounders': ['Axar Patel', 'Sameer Rizvi', 'Ashutosh Sharma', 'Darshan Nalkande', 'Vipraj Nigam', 
                         'Ajay Mandal', 'Manvanth Kumar', 'Tripurana Vijay', 'Madhav Tiwari', 'Donovan Ferreira', 
                         'Dushmantha Chameera'],
        'Bowlers': ['Kuldeep Yadav', 'T Natarajan', 'Mitchell Starc', 'Mohit Sharma', 'Mukesh Kumar']
    },
    'GT': {
        'Batsmen': ['Shubman Gill', 'Jos Buttler', 'Shahrukh Khan', 'Sai Sudharsan'],
        'Wicket-Keepers': ['Wriddhiman Saha', 'Matthew Wade'],
        'All-Rounders': ['Vijay Shankar', 'Azmatullah Omarzai', 'Abhinav Manohar', 'Mahipal Lomror', 
                         'Washington Sundar', 'Rahul Tewatia', 'Sandeep Warrier', 'Nishant Sindhu', 'Manav Suthar'],
        'Bowlers': ['Rashid Khan', 'Mohammed Siraj', 'Josh Little', 'Umesh Yadav', 'Noor Ahmad', 'Mohit Sharma', 
                    'Prasidh Krishna', 'Ishant Sharma', 'Gurnoor Brar', 'Kulwant Khejroliya', 'Kumar Kushagra']
    },
    'KKR': {
        'Batsmen': ['Ajinkya Rahane', 'Rinku Singh', 'Quinton de Kock', 'Rahmanullah Gurbaz', 'Angkrish Raghuvanshi', 
                    'Manish Pandey', 'Rovman Powell', 'Luvnith Sisodia'],
        'Wicket-Keepers': ['Phil Salt'],
        'All-Rounders': ['Venkatesh Iyer', 'Andre Russell', 'Sunil Narine', 'Anukul Roy', 'Moeen Ali', 
                         'Ramandeep Singh', 'Spencer Johnson'],
        'Bowlers': ['Varun Chakravarthy', 'Harshit Rana', 'Vaibhav Arora', 'Anrich Nortje', 'Mayank Markande', 
                    'Chetan Sakariya']
    },
    'LSG': {
        'Batsmen': ['Rishabh Pant', 'David Miller', 'Aiden Markram', 'Devdutt Padikkal', 'Himmat Singh'],
        'Wicket-Keepers': ['Nicholas Pooran'],
        'All-Rounders': ['Marcus Stoinis', 'Krunal Pandya', 'Deepak Hooda', 'Ayush Badoni', 'Arshin Kulkarni', 
                         'Shahbaz Ahmed', 'Rajvardhan Hangargekar', 'Yuvraj Chaudhary', 'Matthew Breetzke', 
                         'Shamar Joseph', 'Prince Yadav'],
        'Bowlers': ['Avesh Khan', 'Mayank Yadav', 'Ravi Bishnoi', 'Yash Thakur', 'Akash Deep', 'Shardul Thakur', 
                    'Akash Singh', 'Digvesh Singh']
    },
    'MI': {
        'Batsmen': ['Rohit Sharma', 'Suryakumar Yadav', 'Tilak Varma', 'Tim David', 'Naman Dhir', 'Will Jacks', 
                    'Robin Minz', 'Ryan Rickelton'],
        'Wicket-Keepers': ['Ishan Kishan'],
        'All-Rounders': ['Hardik Pandya', 'Deepak Chahar', 'Mitchell Santner', 'Raj Angad Bawa', 'Vignesh Puthur', 
                         'Shrijith Krishnan', 'Bevon Jacobs', 'Venkat Satyanarayana Raju', 'Ashwani Kumar'],
        'Bowlers': ['Jasprit Bumrah', 'Trent Boult', 'Arjun Tendulkar', 'Reece Topley', 'Karn Sharma', 
                    'Mujeeb-ur-Rahman', 'Corbin Bosch']
    },
    'PBKS': {
        'Batsmen': ['Shreyas Iyer', 'Prabhsimran Singh', 'Jonny Bairstow', 'Liam Livingstone', 'Shashank Singh', 
                    'Ashutosh Sharma', 'Priyansh Arya', 'Harnoor Pannu', 'Nehal Wadhera', 'Suryansh Shedge'],
        'Wicket-Keepers': ['Jitesh Sharma', 'Josh Inglis'],
        'All-Rounders': ['Sam Curran', 'Marcus Stoinis', 'Glenn Maxwell', 'Harpreet Brar', 'Marco Jansen', 
                         'Azmatullah Omarzai', 'Aaron Hardie', 'Musheer Khan', 'Pyla Avinash'],
        'Bowlers': ['Arshdeep Singh', 'Nathan Ellis', 'Kagiso Rabada', 'Rahul Chahar', 'Harshal Patel', 
                    'Vijaykumar Vyshak', 'Yash Thakur', 'Lockie Ferguson', 'Kuldeep Sen', 'Xavier Bartlett', 'Pravin Dubey']
    },
    'RR': {
        'Batsmen': ['Yashasvi Jaiswal', 'Shimron Hetmyer', 'Shubham Dubey', 'Vaibhav Suryavanshi'],
        'Wicket-Keepers': ['Sanju Samson', 'Dhruv Jurel', 'Kunal Rathore'],
        'All-Rounders': ['Riyan Parag', 'Nitish Rana', 'Yudhvir Charak'],
        'Bowlers': ['Ravichandran Ashwin', 'Trent Boult', 'Yuzvendra Chahal', 'Avesh Khan', 'Kuldeep Sen', 
                    'Navdeep Saini', 'Jofra Archer', 'Sandeep Sharma', 'Tushar Deshpande', 'Akash Madhwal', 
                    'Maheesh Theekshana', 'Wanindu Hasaranga', 'Fazalhaq Farooqi', 'Kwena Maphaka', 'Ashok Sharma', 
                    'Kumar Kartikeya Singh']
    },
    'RCB': {
        'Batsmen': ['Virat Kohli', 'Rajat Patidar', 'Faf du Plessis', 'Devdutt Padikkal', 'Tim David', 'Will Jacks', 
                    'Manoj Bhandage', 'Swastik Chhikara'],
        'Wicket-Keepers': ['Jitesh Sharma'],
        'All-Rounders': ['Glenn Maxwell', 'Liam Livingstone', 'Krunal Pandya', 'Swapnil Singh', 'Romario Shepherd', 
                         'Jacob Bethell', 'Mohit Rathee'],
        'Bowlers': ['Josh Hazlewood', 'Yash Dayal', 'Bhuvneshwar Kumar', 'Rasikh Dar', 'Nuwan Thushara', 
                    'Lungi Ngidi', 'Abhinandan Singh']
    },
    'SRH': {
        'Batsmen': ['Travis Head', 'Abhishek Sharma', 'Atharva Taide', 'Abhinav Manohar', 'Sachin Baby'],
        'Wicket-Keepers': ['Heinrich Klaasen', 'Ishan Kishan'],
        'All-Rounders': ['Nitish Kumar Reddy', 'Washington Sundar', 'Shahbaz Ahmed', 'Kamindu Mendis', 
                         'Wiaan Mulder', 'Aniket Verma', 'Eshan Malinga'],
        'Bowlers': ['Pat Cummins', 'Mohammed Shami', 'Harshal Patel', 'Rahul Chahar', 'Adam Zampa', 
                    'Simarjeet Singh', 'Zeeshan Ansari', 'Jaydev Unadkat']
    }
}

# Flatten the player lists for easier access
for team in ipl_teams_2025.values():
    team['players'] = (team['Batsmen'] + team['Wicket-Keepers'] + 
                       team['All-Rounders'] + team['Bowlers'])

# Define team colors
team_colors = {
    'CSK': {'primary': '#FFC107', 'secondary': '#2E6DB4'},
    'DC': {'primary': '#004C93', 'secondary': '#EF2121'},
    'GT': {'primary': '#1C2526', 'secondary': '#D4A017'},
    'KKR': {'primary': '#3F2587', 'secondary': '#FFD700'},
    'LSG': {'primary': '#2B4F9F', 'secondary': '#00A651'},
    'MI': {'primary': '#0052A5', 'secondary': '#FFD700'},
    'PBKS': {'primary': '#ED1C24', 'secondary': '#C0C0C0'},
    'RR': {'primary': '#EE2E7B', 'secondary': '#1B2E7A'},
    'RCB': {'primary': '#D11F2F', 'secondary': '#000000'},
    'SRH': {'primary': '#F26522', 'secondary': '#000000'}
}

# Assign player roles based on predefined squad roles
def assign_player_role(player, batting_df, bowling_df):
    for team_name, team in ipl_teams_2025.items():
        if player in team['Batsmen']:
            return 'Batsman'
        elif player in team['Wicket-Keepers']:
            return 'Wicket-Keeper'
        elif player in team['All-Rounders']:
            return 'All-Rounder'
        elif player in team['Bowlers']:
            return 'Bowler'
    return 'Batsman'  # Default fallback

player_roles = {player: assign_player_role(player, batting_stats, bowling_stats)
                for team in ipl_teams_2025.values() for player in team['players']}

# Calculate player credits
def calculate_credits(player, batting_df, bowling_df):
    batting = batting_df[batting_df['player_name'] == player]
    bowling = bowling_df[bowling_df['player_name'] == player]
    
    runs = batting['runs'].sum() if not batting.empty else 0
    wickets = bowling['wickets'].sum() if not bowling.empty else 0
    matches = max(batting['matches_played'].sum() if not batting.empty else 0,
                  bowling['matches_bowled'].sum() if not bowling.empty else 0)
    
    base_credit = 7.5
    runs_credit = min(runs / 100, 2.0)
    wickets_credit = min(wickets / 5, 2.0)
    consistency_credit = min(matches / 10, 1.0)
    
    total_credit = base_credit + runs_credit + wickets_credit + consistency_credit
    return round(min(total_credit, 10.0), 1)

player_credits = {player: calculate_credits(player, batting_stats, bowling_stats) for player in player_roles}

# Performance score
def performance_score(player, batting_df, bowling_df, deliveries_df):
    batting = batting_df[batting_df['player_name'] == player]
    bat_score = batting['runs'].sum() / max(batting['matches_played'].sum(), 1) if not batting.empty else 0
    
    bowling = bowling_df[bowling_df['player_name'] == player]
    bowl_score = bowling['wickets'].sum() / max(bowling['matches_bowled'].sum(), 1) * 25 if not bowling.empty else 0
    
    player_deliveries = deliveries_df[
        ((deliveries_df['batsman'] == player) | (deliveries_df['bowler'] == player))
    ]
    runs_scored = player_deliveries[player_deliveries['batsman'] == player]['batsman_runs'].sum()
    wickets_taken = player_deliveries[player_deliveries['bowler'] == player]['player_dismissed'].notna().sum()
    
    return bat_score + bowl_score + (runs_scored / 10) + (wickets_taken * 25)

# Optimize Dream11 team with constraints
def optimize_dream11_team(team1_players, team2_players, batting_df, bowling_df, deliveries_df):
    all_players = team1_players + team2_players
    scores = {p: performance_score(p, batting_df, bowling_df, deliveries_df) for p in all_players}
    
    best_team = None
    best_score = -1
    best_composition = None
    
    for wk in range(1, 2):  # At least 1 Wicket-Keeper
        for bat in range(2, 4):  # 2 to 3 Batsmen
            for bowl in range(2, 4):  # 2 to 3 Bowlers
                ar = 11 - (wk + bat + bowl)  # Remaining as All-Rounders
                if ar < 0 or ar > 5:  # Max 5 All-Rounders
                    continue
                
                wicket_keepers = [p for p in all_players if player_roles[p] == 'Wicket-Keeper']
                batsmen = [p for p in all_players if player_roles[p] == 'Batsman']
                bowlers = [p for p in all_players if player_roles[p] == 'Bowler']
                all_rounders = [p for p in all_players if player_roles[p] == 'All-Rounder']
                
                if (len(wicket_keepers) < wk or len(batsmen) < bat or len(bowlers) < bowl or len(all_rounders) < ar):
                    continue
                
                for wk_comb in combinations(wicket_keepers, wk):
                    for bat_comb in combinations(batsmen, bat):
                        for bowl_comb in combinations(bowlers, bowl):
                            for ar_comb in combinations(all_rounders, ar):
                                team = list(wk_comb) + list(bat_comb) + list(bowl_comb) + list(ar_comb)
                                team1_count = sum(1 for p in team if p in team1_players)
                                team2_count = sum(1 for p in team if p in team2_players)
                                if team1_count > 7 or team2_count > 7:
                                    continue
                                total_credits = sum(player_credits[p] for p in team)
                                if total_credits > 100:
                                    continue
                                team_score = sum(scores[p] for p in team)
                                if team_score > best_score:
                                    best_score = team_score
                                    best_team = team
                                    best_composition = (bat, bowl, ar, wk)
    
    if not best_team:
        return None, None, None, None
    
    team_scores = [(p, scores[p]) for p in best_team]
    team_scores.sort(key=lambda x: x[1], reverse=True)
    captain = team_scores[0][0]
    vice_captain = team_scores[1][0]
    
    return best_team, captain, vice_captain, best_composition

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    teams = list(ipl_teams_2025.keys())
    if request.method == 'POST':
        team1 = request.form.get('team1')
        team2 = request.form.get('team2')
        
        if not team1 or not team2 or team1 == team2:
            flash('Please select two different teams.')
            return redirect(url_for('index'))
        
        return redirect(url_for('select_players', team1=team1, team2=team2))
    
    return render_template('index.html', teams=teams)

@app.route('/select_players/<team1>/<team2>', methods=['GET', 'POST'])
def select_players(team1, team2):
    team1_players = ipl_teams_2025[team1]['players']
    team2_players = ipl_teams_2025[team2]['players']
    
    # Organize players by role
    roles = ['Batsman', 'Bowler', 'All-Rounder', 'Wicket-Keeper']
    team1_by_role = {role: [p for p in team1_players if player_roles[p] == role] for role in roles}
    team2_by_role = {role: [p for p in team2_players if player_roles[p] == role] for role in roles}
    
    # Get team colors
    team1_colors = team_colors[team1]
    team2_colors = team_colors[team2]
    
    if request.method == 'POST':
        team1_selected = request.form.getlist('team1_players')
        team2_selected = request.form.getlist('team2_players')
        
        if len(team1_selected) != 11 or len(team2_selected) != 11:
            flash('Each team must have exactly 11 players.')
            return redirect(url_for('select_players', team1=team1, team2=team2))
        
        # Store selections and predict
        dream11_team, captain, vice_captain, composition = optimize_dream11_team(
            team1_selected, team2_selected, batting_stats, bowling_stats, deliveries
        )
        
        if not dream11_team:
            flash('Unable to form a valid Dream11 team with 1+ Wicket-Keeper, 2-3 Bowlers, and 2-3 Batsmen.')
            return redirect(url_for('select_players', team1=team1, team2=team2))
        
        # Captain/Vice-Captain for bowling first
        bowlers_ar = [p for p in dream11_team if player_roles[p] in ['Bowler', 'All-Rounder']]
        if len(bowlers_ar) >= 2:
            captain_bowl = max(bowlers_ar, key=lambda p: performance_score(p, batting_stats, bowling_stats, deliveries))
            vice_captain_bowl = max([p for p in bowlers_ar if p != captain_bowl], key=lambda p: performance_score(p, batting_stats, bowling_stats, deliveries), default=vice_captain)
        else:
            captain_bowl = captain
            vice_captain_bowl = vice_captain
        
        # Prepare Dream11 team with team info for coloring
        dream11_team_with_info = []
        for p in dream11_team:
            team = team1 if p in ipl_teams_2025[team1]['players'] else team2
            dream11_team_with_info.append((p, player_roles[p], player_credits[p], team))
        
        return render_template('result.html',
                               team1=team1,
                               team2=team2,
                               dream11_team=dream11_team_with_info,
                               composition=composition,
                               captain=captain,
                               vice_captain=vice_captain,
                               captain_bowl=captain_bowl,
                               vice_captain_bowl=vice_captain_bowl,
                               team_colors=team_colors)
    
    return render_template('select_players.html',
                           team1=team1,
                           team2=team2,
                           team1_by_role=team1_by_role,
                           team2_by_role=team2_by_role,
                           team1_colors=team1_colors,
                           team2_colors=team2_colors)

if __name__ == '__main__':
    app.run(debug=True, port=8080)