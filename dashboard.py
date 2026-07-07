import dash
from dash import dcc, html, Input, Output, State
import dash_cytoscape as cyto
import torch
import chess
from collections import defaultdict

data_list = torch.load("dataset/games.pt", weights_only=False)
PIECE_NAMES = {0: "-", 1: "Pedone", 2: "Cavallo", 3: "Alfiere", 4: "Torre", 5: "Regina", 6: "Re"}
EDGE_TYPES = {0: "legal_move", 1: "attack", 2: "pin"}

groups = defaultdict(list)
for i, d in enumerate(data_list):
    gid = getattr(d, "puzzle_id", None) or getattr(d, "game_id", "unknown")
    groups[str(gid)].append(i)

app = dash.Dash(__name__)
BORDER = "#d0d7de"
ACCENT = "#0969da"

app.layout = html.Div(style={'backgroundColor': '#fff', 'minHeight': '100vh', 'padding': '24px',
                              'fontFamily': 'Inter, sans-serif'}, children=[
    html.H1("VISUALIZZA I GAME"),

    html.Div(style={'display': 'flex', 'gap': '16px', 'marginBottom': '16px'}, children=[
        html.Div(style={'flex': 1}, children=[
            html.Label("Partita / Puzzle"),
            dcc.Dropdown(id='game-dropdown',
                         options=[{'label': gid, 'value': gid} for gid in groups],
                         value=list(groups.keys())[0], clearable=False),
        ]),
        html.Div(style={'flex': 2}, children=[
            html.Label("Mossa"),
            dcc.Slider(id='move-slider', min=0, max=0, step=1, value=0, marks={},
                       tooltip={"placement": "bottom"}),
        ]),
    ]),

    html.Div(style={'display': 'flex', 'gap': '16px'}, children=[
        html.Div(style={'flex': 1, 'border': f'1px solid {BORDER}', 'borderRadius': '8px',
                         'padding': '12px', 'height': 'fit-content'}, children=[
            html.B("Legenda"),
            dcc.Checklist(
                id='edge-filter',
                options=[{'label': ' Legal move', 'value': 'legal_move'},
                         {'label': ' Attack', 'value': 'attack'},
                         {'label': ' Pin', 'value': 'pin'}],
                value=['legal_move', 'attack', 'pin'],
                labelStyle={'display': 'block', 'margin': '8px 0'}
            ),
        ]),
        html.Div(style={'flex': 4, 'border': f'1px solid {BORDER}', 'borderRadius': '8px'}, children=[
            cyto.Cytoscape(
                id='cytoscape-graph',
                layout={'name': 'grid', 'rows': 8, 'cols': 8, 'spacingFactor': 2.2},
                style={'width': '100%', 'height': '900px'},
                stylesheet=[
                    {'selector': 'node', 'style': {
                        'label': 'data(label)', 'font-size': '11px', 'text-valign': 'center',
                        'background-color': '#f6f8fa', 'border-width': 1, 'border-color': BORDER,
                        'width': 50, 'height': 50}},
                    {'selector': '[has_piece=1]', 'style': {'background-color': ACCENT, 'shape': 'round-rectangle', 'color': '#fff'}},
                    {'selector': '[color=0][has_piece=1]', 'style': {'background-color': '#57606a'}},
                    {'selector': '[piece_type=6]', 'style': {'background-color': '#cf222e'}},
                    {'selector': 'edge', 'style': {'curve-style': 'bezier', 'target-arrow-shape': 'triangle',
                                                    'arrow-scale': 0.8, 'opacity': 0.6}},
                    {'selector': '[edge_type="legal_move"]', 'style': {'line-color': '#8c959f', 'width': 1}},
                    {'selector': '[edge_type="attack"]', 'style': {'line-color': '#bf8700', 'line-style': 'dashed'}},
                    {'selector': '[edge_type="pin"]', 'style': {'line-color': '#cf222e', 'width': 2}},
                ]
            ),
        ]),
        html.Div(id='node-info', style={
            'flex': 1, 'border': f'1px solid {BORDER}', 'borderRadius': '8px',
            'padding': '16px', 'fontFamily': 'monospace', 'fontSize': '13px',
            'height': '900px', 'overflowY': 'auto'
        })
    ]),

    dcc.Store(id='current-data-idx')
])


@app.callback(
    Output('move-slider', 'max'), Output('move-slider', 'marks'), Output('move-slider', 'value'),
    Input('game-dropdown', 'value')
)
def update_slider(gid):
    idxs = groups[gid]
    return len(idxs) - 1, {i: str(i) for i in range(len(idxs))}, 0


@app.callback(
    Output('cytoscape-graph', 'elements'), Output('current-data-idx', 'data'),
    Input('game-dropdown', 'value'), Input('move-slider', 'value'), Input('edge-filter', 'value')
)
def update_graph(gid, move_i, edge_filter):
    idxs = groups[gid]
    d = data_list[idxs[move_i]]

    nodes = []
    for sq in range(64):
        has_piece, piece_type, color, clock = d.x[sq].tolist()
        nodes.append({'data': {'id': str(sq), 'label': chess.square_name(sq),
                                'has_piece': int(has_piece), 'piece_type': int(piece_type),
                                'color': int(color), 'clock': round(clock, 3)}})

    edges = []
    for (src, tgt), etype in zip(d.edge_index.t().tolist(), d.edge_attr.tolist()):
        etype_name = EDGE_TYPES.get(etype, 'unknown')
        if etype_name in edge_filter:
            edges.append({'data': {'source': str(src), 'target': str(tgt), 'edge_type': etype_name}})

    return nodes + edges, idxs[move_i]


@app.callback(
    Output('node-info', 'children'),
    Input('cytoscape-graph', 'tapNodeData'), State('current-data-idx', 'data')
)
def show_node_info(node_data, data_idx):
    if not node_data or data_idx is None:
        return html.Div("Clicca un nodo per i dettagli.", style={'color': '#57606a'})

    d = data_list[data_idx]
    sq = int(node_data['id'])
    has_piece, piece_type, color, clock = d.x[sq].tolist()

    def row(label, val):
        return html.Div([html.Span(label, style={'color': '#57606a'}), html.Span(val)],
                         style={'display': 'flex', 'justifyContent': 'space-between', 'padding': '4px 0'})

    return [
        html.Div(chess.square_name(sq), style={'fontSize': '20px', 'color': ACCENT, 'marginBottom': '10px'}),
        row("Pezzo", "si" if has_piece else "no"),
        row("Tipo", PIECE_NAMES.get(int(piece_type), '-')),
        row("Colore", 'Bianco' if color == 1 else 'Nero' if color == 0 else '-'),
        row("Clock norm.", f"{clock:.3f}"),
        html.Hr(style={'borderColor': BORDER, 'margin': '12px 0'}),
        row("Mate in", int(d.mate_n.item())),
        row("Best move idx", int(d.y.item())),
        row("ID", getattr(d, 'puzzle_id', getattr(d, 'game_id', '?'))),
    ]


if __name__ == '__main__':
    app.run(debug=False)