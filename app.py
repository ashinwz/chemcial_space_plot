'''
Descripttion: 
version: 
Author: John Wang
Date: 2022-05-04 10:19:26
LastEditors: John Wang
LastEditTime: 2022-05-04 23:17:27
'''

from sre_constants import SUCCESS
import dash
import pandas as pd
import base64
import io
import datetime

from io import BytesIO
import pathlib
from dash import html, dcc
import plotly.express as px

from dash.dependencies import Input, Output, State
from dash import no_update, dash_table

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdDepictor
from rdkit.Chem import MACCSkeys
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server



'''
function layer
'''
def smi_to_fp_ecfp(smi):
    fp = AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(smi), 2, nBits=1024)
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def smi_to_fp_macc(smi):
    fps = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smi))
    fp_arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fps,fp_arr)

    return np.array(fp_arr)

def smi2svg(smi):
    mol = Chem.MolFromSmiles(smi)
    buffered = BytesIO()

    img = Draw.MolToImage(mol)
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    img_str = f"data:image/png;base64,{repr(img_str)[2:-1]}"

    return img_str

def do_PCA(dataset, value):
    pca = PCA(n_components=2)
    if value == "ECFP4":
        fps = np.array([smi_to_fp_ecfp(smi) for smi in dataset['smiles']])
        components = pca.fit_transform(fps.reshape(-1, 1024))

    elif value == "MACCS":
        fps = np.array([smi_to_fp_macc(smi) for smi in dataset['smiles']])
        components = pca.fit_transform(fps.reshape(-1, 167))
        
    
    dataset['PCA-1'] = components[:, 0]
    dataset['PCA-2'] = components[:, 1]

    return dataset

def do_TSNE(dataset, value):
    pca = PCA(n_components=30)
    if value == "ECFP4":
        fps = np.array([smi_to_fp_ecfp(smi) for smi in dataset['smiles']])
        components = pca.fit_transform(fps.reshape(-1, 1024))

    elif value == "MACCS":
        fps = np.array([smi_to_fp_macc(smi) for smi in dataset['smiles']])
        components = pca.fit_transform(fps.reshape(-1, 167))

    tsne_model = TSNE(n_components=2, random_state=0, perplexity=30, n_iter=5000)
    tsne_pca   = tsne_model.fit_transform(components)

    dataset['TSNE-1'] = tsne_pca[:, 0]
    dataset['TSNE-2'] = tsne_pca[:, 1]

    return dataset

def do_umap(dataset, value):
    pca = PCA(n_components=30)
    if value == "ECFP4":
        fps = np.array([smi_to_fp_ecfp(smi) for smi in dataset['smiles']])
        components = pca.fit_transform(fps.reshape(-1, 1024))

    elif value == "MACCS":
        fps = np.array([smi_to_fp_macc(smi) for smi in dataset['smiles']])
        components = pca.fit_transform(fps.reshape(-1, 167))

    

def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    with open('temp.csv', 'w') as f:
        f.write(decoded.decode('utf-8'))
    return pd.read_csv("temp.csv")



'''
layout layer
'''
app.layout = html.Div(
    [
            html.Div(
                [html.Img(src=app.get_asset_url("dash-logo.png"))], className="app__banner"
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        "Chemical Space Visualization",
                                        className="uppercase title",
                                    ),
                                    html.Span("Upload ", className="uppercase bold"),
                                    html.Span(
                                        "a dataset with SMILES to visualize the chemcial space."
                                    ),
                                    html.Br(),
                                    html.Span("Hover ", className="uppercase bold"),
                                    html.Span(
                                        "over a plot in the graph to see its structure."
                                    ),
                                    html.Br(),
                                    html.Span("Select ", className="uppercase bold"),
                                    html.Span(
                                        "both Fingerprints such as ECFP4, MACCS and dimension methods such as T-SNE, PCA"
                                    ),
                                ]
                            )
                        ],
                        className="app__header",
                    ),
                    
                    html.Div([
                        dcc.ConfirmDialog(
                            id='alert-dialog',
                            message='Please upload file with more rows!!!',
                        ),
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div(
                                ['Drag and Drop File or ',
                                html.A('Select Files')
                                ]),
                                style={
                                        'width': '100%',
                                        'height': '60px',
                                        'lineHeight': '60px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                        'margin': '10px'
                                },
                                multiple=True
                            ),
                    ], className="app__dropdown"),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="method_dropdown",
                                value="PCA",
                                options=[{"label": "PCA", "value": "PCA"}, {"label": "TSNE", "value": "TSNE"}]
                            ),
                            dcc.Dropdown(
                                id="fp_dropdown",
                                value="ECFP4",
                                options=[{"label": "ECFP4", "value": "ECFP4"}, {"label": "MACCS", "value": "MACCS"}]
                            ),
                            dcc.Dropdown(
                                id="color_dropdown",
                                value=''
                            )
                        ],
                        className="app__dropdown",
                    ),
                    html.Div(
                        [
                            dcc.Loading(
                                id="loading-1",
                                type="dot",
                                children=html.Div([
                                    dcc.Graph(
                                        id="clickable-graph",
                                        className="table__container",
                                        clear_on_unhover=True
                                    )
                                ],
                                id="loading-output-1")
                            ),
                             dcc.Tooltip(id="graph-tooltip")
                        ],
                        className="container bg-white p-0",
                    ),
                ],
                className="app__container",
            )
    ]
)


'''
callback layer

'''
@app.callback(
    [Output('color_dropdown','options'),
    Output('color_dropdown','value')],
    Input('upload-data','contents')
)
def color_by_column(contents):
    try:
        df = parse_contents(contents[0])
    except Exception:
        df = pd.read_csv("delaney-processed.csv")

    color_cols = df.columns.values

    options = []
    for each in color_cols:
        
        each_dir = {"label": each, "value": each}
        options.append(each_dir)

    return options, options[0]["label"]
    

# generate a scatter plot
@app.callback(
    [Output('clickable-graph', 'figure'),
    Output('alert-dialog', 'displayed')],
    [Input('upload-data','contents'),
    Input('method_dropdown', 'value'),
    Input('fp_dropdown', 'value'),
    Input('color_dropdown', 'value')]
)
def update_output(contents, md_value, fp_value, color_name):
    try:
        df = parse_contents(contents[0])
    except Exception:
        df = pd.read_csv("delaney-processed.csv")


    if df.shape[0] > 10:
        SUCCESS_flag = False

        if md_value == "PCA":
            fig = px.scatter(do_PCA(df, fp_value), x="PCA-1", y="PCA-2", color=color_name, height=800)
        else:
            fig = px.scatter(do_TSNE(df, fp_value), x="TSNE-1", y="TSNE-2", color=color_name, height=800)

        fig.update_traces(hoverinfo="none", hovertemplate=None)
    else:
        fig = None
        SUCCESS_flag = True

    return fig, SUCCESS_flag

# add the structure tooltip
@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    [Input('upload-data','contents'),
    Input("clickable-graph", "hoverData")]
)
def display_hover(contents, hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]

    try:
        df = parse_contents(contents[0])
    except Exception:
        df = pd.read_csv("delaney-processed.csv")

    df_row = df.iloc[num]
    img_src = smi2svg(df_row["smiles"])

    children = [
        html.Div([
            html.Img(src=img_src, style={"width": "100%"})
        ], style={'width': '200px', 'white-space': 'normal'})
    ]

    return True, bbox, children

if __name__ == "__main__":
    app.run_server(debug=True)