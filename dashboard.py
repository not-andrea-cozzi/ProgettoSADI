import dash
from dash import dcc, html, Input, Output
import dash_cytoscape as cyto
import pandas as pd
import json

# 1. Carica i dati
df = pd.read_csv("dataset/analisi_matti.csv")

app = dash.Dash(__name__)

# 2. Definizione del Layout (GUI)
app.layout = html.Div([
    html.H1("Analizzatore Grafi di Matto (GNN Ready)"),
    
    html.Label("Seleziona una posizione di matto:"),
    dcc.Dropdown(
        id='game-dropdown',
        options=[{'label': f"Game {row['Game_ID']} - Mossa {row['Mossa_SAN']} (Mate in {row['SF_Mate']})", 
                  'value': i} for i, row in df.iterrows()],
        value=0
    ),
    
    cyto.Cytoscape(
        id='cytoscape-graph',
        # 'grid' con 8x8 righe/colonne è perfetto per la scacchiera
        layout={'name': 'grid', 'rows': 8, 'cols': 8},
        style={'width': '100%', 'height': '700px', 'background-color': '#2a2a2a'},
        stylesheet=[
            {'selector': 'node', 'style': {'label': 'data(id)', 'color': 'white', 'text-valign': 'center', 'background-color': '#444'}},
            {'selector': '[has_piece=1]', 'style': {'background-color': '#0074D9', 'shape': 'rectangle'}},
            {'selector': '[piece_type=6]', 'style': {'background-color': '#FF4136'}}, # Re in rosso
            {'selector': 'edge', 'style': {'curve-style': 'bezier', 'target-arrow-shape': 'triangle'}},
            {'selector': '[edge_type="legal_move"]', 'style': {'line-color': '#aaa'}},
            {'selector': '[edge_type="attack"]', 'style': {'line-color': '#FF851B', 'line-style': 'dashed'}},
            {'selector': '[edge_type="pin"]', 'style': {'line-color': '#FFDC00', 'width': 3}}
        ]
    )
])

# 3. Callback unico per gestire l'aggiornamento
@app.callback(
    Output('cytoscape-graph', 'elements'),
    Input('game-dropdown', 'value')
)
def update_graph(index):
    row = df.iloc[index]
    graph_data = json.loads(row["Graph_JSON"])
    
    # 1. Estrazione Nodi
    nodes = []
    for node_dict in graph_data.get('nodes', []):
        n_data = node_dict.copy()
        n_id = str(n_data.pop('id', 'unknown'))
        nodes.append({'data': {'id': n_id, **n_data}})
    
    # 2. Estrazione Archi (Controllo se si chiama 'edges' o 'links')
    edges = []
    # Usiamo 'edges' che è la chiave presente nel tuo JSON
    lista_archi = graph_data.get('edges') or graph_data.get('links', [])
    
    for link_dict in lista_archi:
        l_data = link_dict.copy()
        s = str(l_data.pop('source', ''))
        t = str(l_data.pop('target', ''))
        # Manteniamo anche il 'edge_type' che ti serve per il CSS
        edges.append({'data': {'source': s, 'target': t, **l_data}})
    
    return nodes + edges
if __name__ == '__main__':
    # debug=False evita che il server ricarichi il file due volte creando conflitti
    app.run(debug=False)