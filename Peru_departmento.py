import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from scipy import stats

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

def process_data(df, department, case_type):
    df = df[df['departamento'] == department]
    df.sort_values('fecha_resultado', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df[['fecha_resultado', case_type]]

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data)-seq_length-1):
        seq = data[i:(i+seq_length)]
        X.append(seq[:-1])
        y.append(seq[-1])
    return np.array(X), np.array(y)

departments = sorted(list(set([file.split('_')[0] for file in os.listdir('df')])))

app = dash.Dash(__name__)

app.layout = html.Div(
    style={'background-image': 'url("assets/white.jpg")', 'background-size': 'cover', 'height': '100vh'},
    children=[
        html.Div(
            className="header",
            children=[
                html.Div(
                    className="logo",
                    children=[
                        html.Img(src='assets/ou.png', style={'height': '90px'}),
                    ],
                ),
                html.Div(
                    className="title",
                    children=[
                        html.H1(
                            "Strengthening Global Health Surveillance - PERU",
                            style={
                                'text-align': 'center',
                                'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif',
                                'font-size': '2.5em',
                                'color': 'BLACK',
                                'margin-top': '10px',
                                'margin-bottom': '10px',
                            },
                        ),
                    ],
                ),
                
            ],
        ),
        html.Div(
            className="content",
            children=[
                dcc.Dropdown(
                    id="department-dropdown",
                    options=[{'label': i, 'value': i} for i in departments],
                    placeholder="Select a department",
                ),
                dcc.Dropdown(
                    id="predicted-day-dropdown",
                    options=[{'label': f'Day {i}', 'value': i} for i in range(1, 15)],
                    multi=True,
                    placeholder="Select predicted days (max 3)",
                ),
                dcc.Graph(id='num-death-cases-graph'),
                dcc.Graph(id='num-positive-cases-graph'),
            ],
        ),
    ],
)

@app.callback(
    Output('num-death-cases-graph', 'figure'),
    Output('num-positive-cases-graph', 'figure'),
    Input('department-dropdown', 'value'),
    Input("predicted-day-dropdown", "value"),
)
def update_graphs(department, predicted_days):
    if not department or not predicted_days:
        return go.Figure(), go.Figure()

    case_types = ['num_death_cases', 'num_positive_cases']
    colors = ['gold', 'darkorange', 'crimson']

    figures = []

    for case_type in case_types:
        fig = go.Figure()
        filename = f"df/{department}_{case_type}_predictions.csv"
        df = pd.read_csv(filename)

        fig.add_trace(go.Scatter(x=df['fecha_resultado'], y=df[case_type], mode='lines', name=f'Actual', line=dict(color='blue'),
                                 hovertemplate='<b>Date</b>: %{x}<br><b>Value</b>: %{y}'))

        if len(predicted_days) > 3:
            predicted_days = predicted_days[:3]

        for day, color in zip(predicted_days, colors):
            fig.add_trace(go.Scatter(
                x=df['fecha_resultado'],
                y=df[f'Predicted Day {day}'],
                mode='lines',
                name=f'Predicted Day {day}',
                line=dict(color=color),
                hovertemplate='<b>Date</b>: %{x}<br><b>Value</b>: %{y}',
            ))

        fig.update_layout(
            plot_bgcolor='light blue',
            title=f"{department} {case_type} - Actual vs Predicted",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode="x",
        )

        for trace in fig['data']:
            trace.hovertemplate = '<b>Date</b>: %{x}<br><b>Value</b>: %{y}'

        figures.append(fig)

    return figures

if __name__ == "__main__":
    seq_length = 14
    df = pd.read_csv('department_death_positive_cases.csv')

    if df.isnull().sum().sum() != 0:
        print("Missing values detected!")
        return

    z_scores = stats.zscore(df[['num_death_cases', 'num_positive_cases']])
    outliers = df[(np.abs(z_scores) > 3).any(axis=1)]
    if not outliers.empty:
        print("Outliers detected!")
        print(outliers)

    departments = df['departamento'].unique()
    print(f"Available departments: {departments}")

    department = input("Please enter the department: ")
    if department not in departments:
        print(f"Department {department} is not found in the available departments!")
        return

    case_type = input("Please enter the case type (num_death_cases or num_positive_cases): ")
    if case_type not in ['num_death_cases', 'num_positive_cases']:
        print(f"Case type {case_type} is not valid!")
        return

    case_df = process_data(df, department, case_type)

    if case_df is None:
        print(f"No data found for department {department} and case type {case_type}")
        return

    data = case_df[case_type].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    X, y = create_sequences(data, seq_length)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(seq_length))
    model.compile(optimizer=Adam(), loss='mse')

    history = model.fit(X, y, epochs=100, shuffle=False)

    pred = model.predict(X)
    pred = scaler.inverse_transform(pred)

    dates = case_df['fecha_resultado'].values[-len(pred):]

    columns = ['Predicted Day ' + str(i) for i in range(1, seq_length + 1)]
    pred_df = pd.DataFrame(index=dates, data=pred, columns=columns)

    pred_df.insert(0, 'fecha_resultado', pred_df.index)
    pred_df.insert(1, case_type, df.loc[df['departamento'] == department, case_type].values[seq_length:-1])

    filename = f"df/{department}_{case_type}_predictions.csv"
    pred_df.to_csv(filename, index=False)
    print(pred_df)

    app.run_server(debug=True)
