import pyodbc

def create_conn():
    server = 'idsscout.database.windows.net' 
    database = 'idsscout' 
    username = 'idssdb@idsscout'
    password = '.99#wA&47R' 
    driver= '{ODBC Driver 17 for SQL Server}'
    conn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
    return conn

# Establish connection
connection = create_conn()

def getPlayerInfo(conn):
    cursor = conn.cursor()
    getPlayerSql = "SELECT [general_player_statistic].[id], [general_player_statistic].[season], [general_player_statistic].[name], [general_player_statistic].[first_name], [general_player_statistic].[last_name], [general_player_statistic].[age], [general_player_statistic].[nationality], [general_player_statistic].[height], [general_player_statistic].[weight], [general_player_statistic].[injured], [general_player_statistic].[photo], [general_player_statistic].[market_value_in_eur], [general_player_statistic].[market_value_date], [statistics].[id] AS [statistics.id], [statistics->game_infos].[id] AS [statistics.game_infos.id], [statistics->game_infos].[appearences] AS [statistics.game_infos.appearences], [statistics->game_infos].[lineups] AS [statistics.game_infos.lineups], [statistics->game_infos].[minutes] AS [statistics.game_infos.minutes], [statistics->game_infos].[position] AS [statistics.game_infos.position], [statistics->game_infos].[rating] AS [statistics.game_infos.rating], [statistics->game_infos].[captain] AS [statistics.game_infos.captain], [statistics->duels].[id] AS [statistics.duels.id], [statistics->duels].[total] AS [statistics.duels.total], [statistics->duels].[won] AS [statistics.duels.won], [statistics->fouls].[id] AS [statistics.fouls.id], [statistics->fouls].[drawn] AS [statistics.fouls.drawn], [statistics->fouls].[committed] AS [statistics.fouls.committed], [statistics->cards].[id] AS [statistics.cards.id], [statistics->cards].[yellow] AS [statistics.cards.yellow], [statistics->cards].[yellowred] AS [statistics.cards.yellowred], [statistics->cards].[red] AS [statistics.cards.red], [statistics->dribbles].[id] AS [statistics.dribbles.id], [statistics->dribbles].[attempts] AS [statistics.dribbles.attempts], [statistics->dribbles].[success] AS [statistics.dribbles.success], [statistics->goals].[id] AS [statistics.goals.id], [statistics->goals].[total] AS [statistics.goals.total], [statistics->goals].[conceded] AS [statistics.goals.conceded], [statistics->goals].[assists] AS [statistics.goals.assists], [statistics->goals].[saves] AS [statistics.goals.saves], [statistics->passes].[id] AS [statistics.passes.id], [statistics->passes].[total] AS [statistics.passes.total], [statistics->passes].[key] AS [statistics.passes.key], [statistics->passes].[accuracy] AS [statistics.passes.accuracy], [statistics->penalties].[id] AS [statistics.penalties.id], [statistics->penalties].[won] AS [statistics.penalties.won], [statistics->penalties].[committed] AS [statistics.penalties.committed], [statistics->penalties].[scored] AS [statistics.penalties.scored], [statistics->penalties].[missed] AS [statistics.penalties.missed], [statistics->penalties].[saved] AS [statistics.penalties.saved], [statistics->shots].[id] AS [statistics.shots.id], [statistics->shots].[total] AS [statistics.shots.total], [statistics->shots].[on] AS [statistics.shots.on], [statistics->substitutes].[id] AS [statistics.substitutes.id], [statistics->substitutes].[in] AS [statistics.substitutes.in], [statistics->substitutes].[out] AS [statistics.substitutes.out], [statistics->substitutes].[bench] AS [statistics.substitutes.bench], [statistics->tackles].[id] AS [statistics.tackles.id], [statistics->tackles].[total] AS [statistics.tackles.total], [statistics->tackles].[blocks] AS [statistics.tackles.blocks], [statistics->teams].[id] AS [statistics.teams.id], [statistics->teams].[team_id] AS [statistics.teams.team_id], [statistics->teams].[name] AS [statistics.teams.name], [statistics->teams].[logo] AS [statistics.teams.logo] FROM [general_player_statistics] AS [general_player_statistic] LEFT OUTER JOIN [player_statistics_by_seasons] AS [statistics] ON [general_player_statistic].[statistics_id] = [statistics].[id] LEFT OUTER JOIN [game_infos] AS [statistics->game_infos] ON [statistics].[game_infos_id] = [statistics->game_infos].[id] LEFT OUTER JOIN [duels] AS [statistics->duels] ON [statistics].[duels_id] = [statistics->duels].[id] LEFT OUTER JOIN [fouls] AS [statistics->fouls] ON [statistics].[fouls_id] = [statistics->fouls].[id] LEFT OUTER JOIN [cards] AS [statistics->cards] ON [statistics].[cards_id] = [statistics->cards].[id] LEFT OUTER JOIN [dribbles] AS [statistics->dribbles] ON [statistics].[dribbles_id] = [statistics->dribbles].[id] LEFT OUTER JOIN [goals] AS [statistics->goals] ON [statistics].[goals_id] = [statistics->goals].[id] LEFT OUTER JOIN [passes] AS [statistics->passes] ON [statistics].[passes_id] = [statistics->passes].[id] LEFT OUTER JOIN [penalties] AS [statistics->penalties] ON [statistics].[penalties_id] = [statistics->penalties].[id] LEFT OUTER JOIN [shots] AS [statistics->shots] ON [statistics].[shots_id] = [statistics->shots].[id] LEFT OUTER JOIN [substitutes] AS [statistics->substitutes] ON [statistics].[substitutes_id] = [statistics->substitutes].[id] LEFT OUTER JOIN [tackles] AS [statistics->tackles] ON [statistics].[tackles_id] = [statistics->tackles].[id] LEFT OUTER JOIN [teams] AS [statistics->teams] ON [statistics].[teams_id] = [statistics->teams].[id];"
    cursor.execute(getPlayerSql)

def insertPredictedValue(conn,playerData):
    cursor = conn.cursor()
    insertPredictedValueSql = "INSERT INTO [dbo].[predicted_value] ([player_id], [predicted_value], [predicted_date]) VALUES (?, ?, ?)"
    cursor.execute(insertPredictedValueSql, (1, 1000000, '2020-01-01'))
    conn.commit()

def insertModelMetrics(conn,modelMetricData):
    cursor = conn.cursor()
    insertModelMetricsSql = "INSERT INTO [dbo].[model_metrics] ([model_name], [model_type], [model_accuracy], [model_precision], [model_recall], [model_f1_score], [model_roc_auc_score], [model_date]) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    cursor.execute(insertModelMetricsSql, ('model_name', 'model_type', 0.1, 0.2, 0.3, 0.4, 0.5, '2020-01-01'))
    conn.commit()