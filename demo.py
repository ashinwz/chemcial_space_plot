'''
Descripttion: 
version: 
Author: John Wang
Date: 2022-05-02 10:14:36
LastEditors: John Wang
LastEditTime: 2022-05-04 22:23:57
'''

import pandas as pd
from requests import options
import plotly.express as px
import molplotly
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.decomposition import PCA

from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)
app.layout = html.Div([
    dcc.Dropdown(options=['NYC', 'MTL', 'SF'], value=['NYC'], id='demo-dropdown', multi=True),
    html.Div(id='dd-output-container')
])


@app.callback(
    Output('dd-output-container', 'children'),
    Input('demo-dropdown', 'value')
)
def update_output(value):
    return f'You have selected {value}'


if __name__ == '__main__':
    app.run_server(debug=True)



def smi_to_fp(smi):
    fp = AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(smi), 2, nBits=1024)
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# load a DataFrame with smiles
df_esol = pd.read_csv('delaney-processed.csv')
df_esol['y_pred'] = df_esol['ESOL predicted log solubility in mols per litre']
df_esol['y_true'] = df_esol['measured log solubility in mols per litre']

esol_fps = np.array([smi_to_fp(smi) for smi in df_esol['smiles']])
pca = PCA(n_components=2)
components = pca.fit_transform(esol_fps.reshape(-1, 1024))
df_esol['PCA-1'] = components[:, 0]
df_esol['PCA-2'] = components[:, 1]

# generate a scatter plot
@app.callback(
    Output('dd-output-container', 'children'),
    Input('demo-dropdown', 'value')
)
def update_output(value):
    fig_pca = px.scatter(df_esol,
                        x="PCA-1",
                        y="PCA-2",
                        color='y_true',
                        title='ESOL PCA of morgan fingerprints',
                        labels={'y_true': 'Measured Solubility'},
                        width=1200,
                        height=800)

    # app = molplotly.add_molecules(fig=fig_pca,
    #                                 df=df_esol.rename(columns={'y_true': 'Measured Solubility'}),
    #                                 smiles_col='smiles',
    #                                 title_col='Compound ID',
    #                                 caption_cols=['Measured Solubility'],
    #                                 caption_transform={'Measured Solubility': lambda x: f"{x:.2f}"},
    #                                 color_col='Measured Solubility',
    #                                 show_coords=False)

    return fig_pca

# run Dash app inline in notebook (or in an external server)
app.run_server(port=8700, height=1000)