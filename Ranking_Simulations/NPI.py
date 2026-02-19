import pandas as pd
import numpy as np


def calculate_npi(
    df,
    win_dial=0.2,
    sos_dial=0.8,
    qwb_mult=0.5,
    qwb_threshold=54.0,
    max_wins=8.0,
    max_iterations=100,
    tolerance=1e-6
):
    """
    Calculate Net Performance Index (NPI) for teams in a league.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: home_team, away_team, home_score, away_score
    win_dial : float, default=0.2
        Weight for win/loss/tie result
    sos_dial : float, default=0.8
        Weight for strength of schedule (opponent NPI)
    qwb_mult : float, default=0.5
        Multiplier for quality win bonus
    qwb_threshold : float, default=54.0
        NPI threshold for quality win bonus
    max_wins : float, default=8.0
        Maximum number of wins to count (ties count as 0.5 wins)
    max_iterations : int, default=100
        Maximum iterations for convergence
    tolerance : float, default=1e-6
        Convergence tolerance
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: team, npi, games_played
    """
    
    # Get all unique teams
    teams = pd.concat([df['home_team'], df['away_team']]).unique()
    
    # Initialize NPI scores (start with neutral 50)
    npi_scores = {team: 50.0 for team in teams}
    
    # Pre-compute game results for each team (optimization)
    team_game_results = {team: [] for team in teams}
    
    for _, game in df.iterrows():
        home = game['home_team']
        away = game['away_team']
        home_score = game['home_score']
        away_score = game['away_score']
        
        # Store result from each team's perspective
        home_score_diff = home_score - away_score
        away_score_diff = away_score - home_score
        
        team_game_results[home].append(home_score_diff)
        team_game_results[away].append(away_score_diff)
    
    # Iterate until convergence
    for iteration in range(max_iterations):
        old_npi = npi_scores.copy()
        game_npis = {team: [] for team in teams}
        
        # Process each game
        for _, game in df.iterrows():
            home = game['home_team']
            away = game['away_team']
            home_score = game['home_score']
            away_score = game['away_score']
            
            # Calculate results
            if home_score > away_score:
                home_result = 100
                away_result = 0
                home_won = True
                away_won = False
                home_tied = False
                away_tied = False
            elif home_score < away_score:
                home_result = 0
                away_result = 100
                home_won = False
                away_won = True
                home_tied = False
                away_tied = False
            else:
                home_result = 50  # Placeholder, will compute both win and loss versions
                away_result = 50
                home_won = False
                away_won = False
                home_tied = True
                away_tied = True
            
            # Get opponent NPIs
            away_npi = old_npi[away]
            home_npi = old_npi[home]
            
            # Calculate quality win bonus
            home_qwb = 0
            away_qwb = 0
            
            if home_won and away_npi > qwb_threshold:
                home_qwb = (away_npi - qwb_threshold) * qwb_mult
            
            if away_won and home_npi > qwb_threshold:
                away_qwb = (home_npi - qwb_threshold) * qwb_mult
            
            # Calculate game NPI for each team
            if home_tied:
                # For ties, compute both win and loss versions
                home_game_npi_win = win_dial * 100 + sos_dial * away_npi
                if away_npi > qwb_threshold:
                    home_game_npi_win += (away_npi - qwb_threshold) * qwb_mult
                home_game_npi_loss = win_dial * 0 + sos_dial * away_npi
                game_npis[home].append(('tie', home_game_npi_win, home_game_npi_loss))
            else:
                home_game_npi = win_dial * home_result + sos_dial * away_npi + home_qwb
                game_npis[home].append(('win' if home_won else 'loss', home_game_npi, None))
            
            if away_tied:
                # For ties, compute both win and loss versions
                away_game_npi_win = win_dial * 100 + sos_dial * home_npi
                if home_npi > qwb_threshold:
                    away_game_npi_win += (home_npi - qwb_threshold) * qwb_mult
                away_game_npi_loss = win_dial * 0 + sos_dial * home_npi
                game_npis[away].append(('tie', away_game_npi_win, away_game_npi_loss))
            else:
                away_game_npi = win_dial * away_result + sos_dial * home_npi + away_qwb
                game_npis[away].append(('win' if away_won else 'loss', away_game_npi, None))
        
        # Update NPI scores (weighted average of all game NPIs)
        for team in teams:
            if game_npis[team]:
                # Separate wins/ties and losses
                wins_ties = []  # (game_npi_for_sorting, win_value, game_npi_win, game_npi_loss)
                losses = []  # (game_npi, weight)
                
                # Process each game
                for i, game_data in enumerate(game_npis[team]):
                    result_type = game_data[0]
                    
                    if result_type == 'win':
                        game_npi = game_data[1]
                        wins_ties.append((game_npi, 1.0, game_npi, None))
                    elif result_type == 'tie':
                        game_npi_win = game_data[1]
                        game_npi_loss = game_data[2]
                        # Tie: 0.5 win value, sort by win NPI, also has loss NPI
                        wins_ties.append((game_npi_win, 0.5, game_npi_win, game_npi_loss))
                    else:  # loss
                        game_npi = game_data[1]
                        losses.append((game_npi, 1.0))
                
                # !!! Sort wins/ties by win NPI (best first)
                wins_ties.sort(key=lambda x: x[0], reverse=True)
                
                # Assign weights based on max_wins threshold
                weighted_games = []
                win_count = 0.0
                
                for sort_npi, win_value, npi_win, npi_loss in wins_ties:
                    remaining_capacity = max_wins - win_count
                    
                    if remaining_capacity <= 0:
                        # Beyond max_wins - include if it improves the average
                        if npi_win > old_npi[team]:
                            win_weight = 1.0
                        else:
                            win_weight = 0.0
                    elif win_value <= remaining_capacity:
                        # The entire win/tie fits within remaining capacity
                        win_weight = 1.0
                    else:
                        # include if improves average
                        if npi_win > old_npi[team]:
                            win_weight = 1.0
                        else:
                            # Only part of this win/tie fits
                            # Scale the weight by how much fits
                            win_weight = remaining_capacity / win_value
                    
                    # Add the win portion
                    weighted_games.append((npi_win, win_weight * win_value))
                    
                    # For ties, also add the loss portion (always 0.5 weight)
                    if npi_loss is not None:
                        weighted_games.append((npi_loss, 0.5))
                    
                    win_count += win_value
                
                # Add losses that make score worse. Eliminate losses that improve score
                for loss in losses:
                    loss_npi = loss[0]
                    if loss_npi <= old_npi[team]:
                        weighted_games.append(loss)
                
                # Calculate weighted average NPI
                if weighted_games:
                    total_weighted = sum(npi * weight for npi, weight in weighted_games)
                    total_weight = sum(weight for _, weight in weighted_games)
                    npi_scores[team] = total_weighted / total_weight if total_weight > 0 else 50.0
                else:
                    npi_scores[team] = 50.0
        
        # Check for convergence
        max_change = max(abs(npi_scores[team] - old_npi[team]) for team in teams)
        if max_change < tolerance:
            break
    
    # Create results DataFrame
    results = []
    for team in teams:
        results.append({
            'team': team,
            'npi': npi_scores[team],
            'games_played': len(game_npis[team])
        })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('npi', ascending=False).reset_index(drop=True)
    
    return result_df


def get_team_detail_report(df, team_name, win_dial=0.2, sos_dial=0.8, qwb_mult=0.5, qwb_threshold=54.0, max_wins=8.0):
    """
    Generate a detailed report for a specific team showing all games and NPI calculations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: home_team, away_team, home_score, away_score
    team_name : str
        Name of the team to analyze
    (other parameters same as calculate_npi)
    
    Returns:
    --------
    pd.DataFrame
        Detailed game-by-game breakdown with NPI calculations
    """
    
    # First run the full NPI calculation to get final NPIs
    npi_results = calculate_npi(df, win_dial, sos_dial, qwb_mult, qwb_threshold, max_wins)
    npi_dict = dict(zip(npi_results['team'], npi_results['npi']))
    
    # Also need the team's final NPI for comparison
    team_final_npi = npi_dict.get(team_name, 50.0)
    
    # Get all games for this team
    team_games = df[(df['home_team'] == team_name) | (df['away_team'] == team_name)].copy()
    
    if len(team_games) == 0:
        print(f"No games found for team: {team_name}")
        return pd.DataFrame()
    
    # Build detailed report
    game_details = []
    
    for idx, game in team_games.iterrows():
        if game['home_team'] == team_name:
            opponent = game['away_team']
            team_score = game['home_score']
            opp_score = game['away_score']
            location = 'Home'
        else:
            opponent = game['home_team']
            team_score = game['away_score']
            opp_score = game['home_score']
            location = 'Away'
        
        # Determine result
        score_diff = team_score - opp_score
        if score_diff > 0:
            result = 'W'
            result_value = 100
            win_value = 1.0
        elif score_diff == 0:
            result = 'T'
            result_value = 50
            win_value = 0.5
        else:
            result = 'L'
            result_value = 0
            win_value = 0.0
        
        # Get opponent NPI
        opp_npi = npi_dict.get(opponent, 50.0)
        
        # For ties, calculate both win and loss versions
        if result == 'T':
            # Win version
            qwb_win = 0
            if opp_npi > qwb_threshold:
                qwb_win = (opp_npi - qwb_threshold) * qwb_mult
            game_npi_win = win_dial * 100 + sos_dial * opp_npi + qwb_win
            
            # Loss version
            game_npi_loss = win_dial * 0 + sos_dial * opp_npi
            
            game_details.append({
                'opponent': opponent,
                'location': location,
                'score': f"{team_score}-{opp_score}",
                'result': result,
                'opp_npi': round(opp_npi, 2),
                'result_contrib_win': round(win_dial * 100, 2),
                'result_contrib_loss': round(win_dial * 0, 2),
                'sos_contrib': round(sos_dial * opp_npi, 2),
                'qwb_win': round(qwb_win, 2),
                'qwb_loss': 0.0,
                'game_npi_win': round(game_npi_win, 2),
                'game_npi_loss': round(game_npi_loss, 2),
                'win_value': win_value
            })
        else:
            # Win or loss - single NPI value
            qwb = 0
            if result == 'W' and opp_npi > qwb_threshold:
                qwb = (opp_npi - qwb_threshold) * qwb_mult
            
            game_npi = win_dial * result_value + sos_dial * opp_npi + qwb
            game_loss_npi = sos_dial * opp_npi
            
            game_details.append({
                'opponent': opponent,
                'location': location,
                'score': f"{team_score}-{opp_score}",
                'result': result,
                'opp_npi': round(opp_npi, 2),
                'result_contrib_win': round(win_dial * result_value, 2),
                'result_contrib_loss': None,
                'sos_contrib': round(sos_dial * opp_npi, 2),
                'qwb_win': round(qwb, 2),
                'qwb_loss': None,
                'game_npi_win': round(game_npi, 2),
                'game_npi_loss': round(game_loss_npi, 2),
                'win_value': win_value
            })
    
    # Create DataFrame
    detail_df = pd.DataFrame(game_details)
    
    # Sort wins/ties by game_npi_win to determine which count toward max_wins
    wins_ties_indices = detail_df[detail_df['result'].isin(['W', 'T'])].index.tolist()
    wins_ties_data = [(i, detail_df.loc[i, 'game_npi_win'], detail_df.loc[i, 'win_value']) 
                      for i in wins_ties_indices]
    wins_ties_data.sort(key=lambda x: x[1], reverse=True)  # Sort by game_npi_win
    
    # Determine weights for win portion
    detail_df['weight_win'] = 0.0
    detail_df['weight_loss'] = 0.0
    
    # Losses get weight 1 if they make score works
    detail_df.loc[((detail_df['result'] == 'L') &
                  (detail_df['game_npi_loss'] <= team_final_npi)), 
                  'weight_loss'] = 1.0
    
    win_count = 0.0
    
    for idx, game_npi_val, win_val in wins_ties_data:
        remaining_capacity = max_wins - win_count
        
        if remaining_capacity <= 0:
            # Beyond max_wins - check if it improves the average
            if game_npi_val > team_final_npi:
                win_weight = 1.0
                counted = 'Yes (Improves Avg)'
            else:
                win_weight = 0.0
                counted = 'No'
        elif win_val <= remaining_capacity:
            # The entire win/tie fits within remaining capacity
            win_weight = 1.0
            counted = 'Yes'
        else:
            if game_npi_val > team_final_npi:
                win_weight = 1.0
                counted = 'Yes (Improves Avg)'
            else:
                win_weight = remaining_capacity / win_val
                counted = f'Partial ({win_weight:.2f})'
        
        # Apply the weight, scaled by win_value
        detail_df.loc[idx, 'weight_win'] = win_weight * win_val
        detail_df.loc[idx, 'counted_toward_max'] = counted
        
        # For ties, also add loss weight (always 0.5)
        if detail_df.loc[idx, 'result'] == 'T':
            detail_df.loc[idx, 'weight_loss'] = 0.5
            if counted == 'No':
                detail_df.loc[idx, 'counted_toward_max'] = 'No (Win) + 0.5 Loss'
            else:
                detail_df.loc[idx, 'counted_toward_max'] = f'{counted} (Win) + 0.5 Loss'
        
        win_count += win_val
    
    # Add "counted" column for pure losses
    detail_df.loc[detail_df['result'] == 'L', 'counted_toward_max'] = 'Yes (Loss)'
    
    # Calculate final NPI using both win and loss components
    total_weighted = 0.0
    total_weight = 0.0
    
    # Add game contribution column
    detail_df['game_contribution'] = 0.0
    
    for idx, row in detail_df.iterrows():
        if row['result'] == 'T':
            # Tie: add both win and loss portions
            contribution = row['game_npi_win'] * row['weight_win'] + row['game_npi_loss'] * row['weight_loss']
            total_weighted += contribution
            total_weight += row['weight_win'] + row['weight_loss']
        else:
            # Win or loss: single NPI value
            weight = row['weight_win'] + row['weight_loss']
            contribution = row['game_npi_win'] * weight
            total_weighted += contribution
            total_weight += weight
        
        detail_df.loc[idx, 'game_contribution'] = contribution
    

    # Drop unwanted columns
    columns_to_drop = ['result_contrib_win', 'result_contrib_loss', 'sos_contrib', 'qwb_loss', 'counted_toward_max']
    detail_df = detail_df.drop(columns=[col for col in columns_to_drop if col in detail_df.columns])
    
    
    return detail_df

