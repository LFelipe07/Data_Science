import streamlit as st
import pandas as pd
import numpy as np 
from scipy.stats import poisson

st.title("Predicting World Cup matches with AI")

teams = pd.read_excel("DadosCopaDoMundoQatar2022.xlsx", sheet_name='selecoes', index_col=0)

# Definindo as forças do time
# Defining team strengths
fifa = teams['PontosRankingFIFA']
a, b = min(fifa), max(fifa)
# definindo os intervalos que eu quero das seleções
# defining the ranges I want from the selections
fa, fb = 0.15, 1
# transformação linear de escala numerica
# Numerical scale linear transformation
b1 = (fb - fa)/(b-a)
b0 = fb - b*b1
strenght = b0 + b1*fifa

# Função para definir vitoria, empate e derrota
# Function to define victory, draw and defeat
def Results(goals1, goals2):
    if goals1 > goals2:
        results = 'victory'
    elif goals2 > goals1:
        results = 'defeat'
    else:
        results = 'draw'
    return results

# Usando a metrica POISSON para calcular as probabilidades
# Using the POISSON metric to calculate the probability
def PoissonMean(team1, team2):
    strenght1 = strenght[team1]
    strenght2 = strenght[team2]
    goalsm= 2.75
    l1 = goalsm * strenght1/(strenght1 + strenght2)
    l2 = goalsm - l1
    return [l1, l2]

# Calcula a probabilidade dos times de marcar gols com a metrica POISSON
# Calculate the probability of the teams score with POISSON metrics
def Distribution(media):
    probs = []
    for i in range(7):
        probs.append(poisson.pmf(i, media))
    probs.append(1 - sum(probs))
    return pd.Series(probs, index = ['0', '1', '2', '3', '4', '5', '6', '7+'])

# Função de calculo das probabilidades de gols, vitoria, empate e derrota e pontos das duas seleções
# Function to calculate the goals probabilities, victory, draw and defeat and the the points of the teams
def MatchProbability(team1, team2):
    l1, l2 = PoissonMean(team1, team2)
    d1, d2 = Distribution(l1), Distribution(l2)
    matrix = np.outer(d1, d2)
    victory = np.tril(matrix).sum()-np.trace(matrix) #Soma o triangulo inferior / sum the lower triangle
    defeat = np.triu(matrix).sum()-np.trace(matrix) #Soma o triangulo inferior / sum the upper triangle
    draw = 1 - (victory + defeat)
    
    probs = np.around([victory, draw, defeat], 3)
    probsp = [f'{100*i:.1f}%' for i in probs]
    
    names = ['0','1','2','3','4','5','6','7+']
    matrix = pd.DataFrame(matrix, columns = names, index = names)
    matrix.index = pd.MultiIndex.from_product([[team1], matrix.index])
    matrix.columns = pd.MultiIndex.from_product([[team2], matrix.columns])
    
    output = {'team1': team1, 'team2': team2,
              'f1': strenght[team1], 'f2': strenght[team2],
              'media1': l1, 'media2': l2,
              'probabilities': probsp, 'matrix': matrix}
    
    return output

# Função de sistema de pontos na tabela apos a predição da partida para Vitoria empate e derrota
# Function of table system points after the match prediction to victory, draw and defeat  
def Points(goals1, goals2):
    result = Results(goals1, goals2)
    if result == 'V':
        points1, points2 = 3, 0
    elif result == 'D':
        points1, points2 = 0, 3
    else:
        points1, points2 = 1, 1
    return[points1, points2, result]


# Função para retornar o valor do placar do jogo(saldo de gols)
# Function to return the match scoreboard values(goals difference)
def Game(team1, team2):
    l1, l2 = PoissonMean(team1, team2) 
    goals1 = int(np.random.poisson(lam = l1, size = 1))
    goals2 = int(np.random.poisson(lam = l2, size = 1))
    diff1 = goals1 - goals2
    diff2 = goals2 - goals1 # Ou -saldo1
    points1, points2, result = Points(goals1, goals2)
    placar ='{}X{}'.format(goals1, goals2)
    return [goals1, goals2, diff1, diff2, points1, points2, result, placar]


# app

teamlist1 = teams.index.tolist()
teamlist1.sort()
teamlist2 = teamlist1.copy()

j1, j2 = st.columns(2)
team1  = j1.selectbox("Choose the first team", teamlist1)
teamlist2.remove(team1)
team2 = j2.selectbox("Choose the second team", teamlist2, index = 1)
st.markdown("-----")

# Probabilidades
# Probabilities
game = MatchProbability(team1, team2)
prob = game['probabilities']
matrix = game['matrix']

# Criando colunas para a porcentagem de vitorias e para as seleções
# Creating columns for win percentage and the country teams
col1, col2, col3, col4, col5 = st.columns(5)
col1.image(teams.loc[team1, 'LinkBandeiraGrande'])
col2.metric(team1, prob[0])
col3.metric('Draw', prob[1])
col4.metric(team2, prob[2])
col5.image(teams.loc[team2, 'LinkBandeiraGrande'])


st.markdown("----")
st.markdown("Scoreboard probabilities")

def aux(x):
    return f'{str(round(100*x,1))}%'
st.table(matrix.applymap(aux))


st.markdown("----")
st.markdown("World Cup Match probabilities")

wcmatch = pd.read_excel("OutputWorldCupGameEstimates.xlsx", index_col=0)
st.table(wcmatch[['grupo', 'seleção1', 'seleção2', 'Victory', 'Draw','Defeat']])
