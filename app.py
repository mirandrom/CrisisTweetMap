import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import sqlite3
import json
import numpy as np

from dash.dependencies import Input, Output, State
from plotly import graph_objs as go
from plotly.graph_objs import *
from datetime import datetime as dt
import dash_auth

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

DB = "./coronavirus.db"
TABLE = "live_tweets"

with open("dashboard_usr_pw_pairs.json", "r") as f:
    VALID_USERNAME_PASSWORD_PAIRS = json.load(f)
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

def get_data(db : str, table: str, num_hours: int = 0, num_minutes: int = 30):
    """Extracts tweets from the last num_hours:num_minutes
    """
    con = sqlite3.connect(db)
    df_plot = pd.read_sql_query(f"SELECT * FROM {table} WHERE datetime(created_at) >= datetime('now', '-{num_hours} hours', '-{num_minutes} minutes')", con)
    return df_plot

LABELS = [
    "affected_people",
    "other_useful_information",
    "disease_transmission",
    "disease_signs_or_symptoms",
    "prevention",
    "treatment",
    "not_related_or_irrelevant",
    "deaths_reports"
]

COORDS = [
    "tweet",
    "place",
    "content_cities",
    "user_cities",
    "content_countries",
    "user_countries"
]

colorVal = [
    "#F4EC15",
    "#DAF017",
    "#BBEC19",
    "#9DE81B",
    "#80E41D",
    "#66E01F",
    "#4CDC20",
    "#34D822",
    "#24D249",
    "#25D042",
    "#26CC58",
    "#28C86D",
    "#29C481",
    "#2AC093",
    "#2BBCA4",
    "#2BB5B8",
    "#2C99B4",
    "#2D7EB0",
    "#2D65AC",
    "#2E4EA4",
    "#2E38A4",
    "#3B2FA0",
    "#4E2F9C",
    "#603099",
]

def to_color(val, max_val):
    i = int((len(colorVal) - 1)* val / max_val)
    return colorVal[i]

LABEL_COLORS = {l: to_color(i, len(LABELS)-1) for i, l in enumerate(LABELS)}

# Plotly mapbox public token
mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNqdnBvNDMyaTAxYzkzeW5ubWdpZ2VjbmMifQ.TXcBE-xg9BFdV2ocecc_7g"

# Layout of Dash App
app.layout = html.Div(
    children=[
        html.Div(
            className="row",
            children=[
                # Column for user controls
                html.Div(
                    className="four columns div-user-controls",
                    children=[
                        html.Img(
                            className="logo", src=app.get_asset_url("dash-logo-new.png")
                        ),
                        html.H2("CRISIS TWEET MAP"),
                        html.P("Get tweets from the last x minutes:"),
                        # Change to side-by-side for mobile layout
                        html.Div(
                            className="row",
                            children=[
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        # Dropdown for locations on map
                                        dcc.Slider(
                                            id="minute-slider",
                                            min=0,
                                            max=240,
                                            marks={i: str(i) for i in range(0, 240, 60)},
                                            value=60,
                                        )
                                    ],
                                ),
                                html.P("Get tweets by category:"),
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        # Dropdown to select labels
                                        dcc.Dropdown(
                                            id="label-selector",
                                            options=[
                                                {
                                                    "label": l,
                                                    "value": l,
                                                }
                                                for l in LABELS
                                            ],
                                            multi=True,
                                            value=LABELS,
                                        ),
                                        # Dropdown to select coordinates
                                        dcc.Dropdown(
                                            id="coords-selector",
                                            options=[
                                                {
                                                    "label": c,
                                                    "value": c,
                                                }
                                                for c in COORDS
                                            ],
                                            multi=True,
                                            value=COORDS[:4],
                                        )
                                    ],
                                ),
                            ],
                        ),
                        dcc.Graph(id="histogram"),
                    ],
                ),
                # Column for app graphs and plots
                html.Div(
                    className="eight columns div-for-charts bg-grey",
                    children=[
                        dcc.Graph(id="map-graph"),
                        dcc.Interval(
                            id='interval-component',
                            interval=3 * 1000,  # in milliseconds
                            n_intervals=0,
                        ),
                    ],
                ),
            ],
        )
    ]
)


# Update Histogram Figure based on Month, Day and Times Chosen
@app.callback(
    Output("histogram", "figure"),
    [
        Input("minute-slider", "value"),
        Input("label-selector", "value"),
        Input("interval-component", "n_intervals"),
     ],
)
def update_histogram(minutes, selection, n_intervals):
    num_hours = minutes // 60
    num_minutes = int(minutes % 60)
    df = get_data(DB, TABLE, num_hours, num_minutes)
    counts = df.prediction.value_counts()
    xVal = LABELS
    yVal = [counts[l] if l in counts and l in selection else 0 for l in xVal]
    yVal2 = [counts[l] if l in counts and l not in selection else 0 for l in xVal]
    colorVal = [LABEL_COLORS[l] for l in LABELS]

    layout = go.Layout(
        bargap=0.01,
        bargroupgap=0,
        barmode="group",
        margin=go.layout.Margin(l=10, r=0, t=0, b=50),
        showlegend=False,
        plot_bgcolor="#323130",
        paper_bgcolor="#323130",
        dragmode="select",
        font=dict(color="white"),
        xaxis=dict(
            showgrid=False,
            nticks=8,
            fixedrange=True,
        ),
        yaxis=dict(
            showticklabels=True,
            showgrid=False,
            fixedrange=True,
            rangemode="nonnegative",
            zeroline=True,
        ),
    )

    return go.Figure(
        data=[
            go.Bar(x=xVal, y=yVal, marker=dict(color=colorVal), hoverinfo="x"),
            go.Bar(x=xVal, y=yVal2, marker=dict(color=colorVal, opacity=0.1), hoverinfo="x"),

        ],
        layout=layout,
    )



# Update Map Graph based on date-picker, selected data on histogram and location dropdown
@app.callback(
    Output("map-graph", "figure"),
    [
        Input("minute-slider", "value"),
        Input("label-selector", "value"),
        Input("coords-selector", "value"),
        Input("interval-component", "n_intervals"),
    ],
    [State("map-graph", "relayoutData")],
)
def update_graph(minutes, selection, coordinates, n_intervals, relayoutData):
    num_hours = minutes // 60
    num_minutes = int(minutes % 60)
    df = get_data(DB, TABLE, num_hours, num_minutes)
    try:
        latInitial = (relayoutData['mapbox.center']['lat'])
        lonInitial = (relayoutData['mapbox.center']['lon'])
        zoom = (relayoutData['mapbox.zoom'])
    except:
        latInitial = 40.7272
        lonInitial = -73.991251
        zoom = 0
    bearing = 0
    df = df.loc[df.prediction.isin(selection)]
    df["colors"] = [LABEL_COLORS[l] for l in df.prediction]
    df["opacity"] = [i / len(df) for i in range(1, len(df)+1)]
    data = []
    for c in coordinates:
        col = f"{c}_coordinates"
        df[col] = df[col].apply(lambda x: json.loads(x))
        df_c = df.loc[(df[col].astype(bool))]
        df_c = df_c.explode(col)
        coords = df_c[col]
        # introduce small amount of noise to distinguish overlapping coordinates
        lat = [x[0] + (np.random.random() - 0.5)/100 for x in coords]
        lon = [x[1] + (np.random.random() - 0.5)/100 for x in coords]
        data += [
            go.Scattermapbox(
                lat=lat,
                lon=lon,
                mode="markers",
                hoverinfo="text",
                text=df_c.text,
                marker=dict(
                    showscale=True,
                    color=df_c.colors,
                    opacity=df_c.opacity,
                    size=5,
                ),
            )
        ]

    return go.Figure(
        data=data,
        layout=go.Layout(
            autosize=True,
            margin=go.layout.Margin(l=0, r=35, t=0, b=0),
            showlegend=False,
            mapbox=dict(
                accesstoken=mapbox_access_token,
                center=dict(lat=latInitial, lon=lonInitial),  # 40.7272  # -73.991251
                style="dark",
                bearing=bearing,
                zoom=zoom,
            ),
            updatemenus=[
                dict(
                    buttons=(
                        [
                            dict(
                                args=[
                                    {
                                        "mapbox.zoom": 12,
                                        "mapbox.center.lon": "-73.991251",
                                        "mapbox.center.lat": "40.7272",
                                        "mapbox.bearing": 0,
                                        "mapbox.style": "dark",
                                    }
                                ],
                                label="Reset Zoom",
                                method="relayout",
                            )
                        ]
                    ),
                    direction="left",
                    pad={"r": 0, "t": 0, "b": 0, "l": 0},
                    showactive=False,
                    type="buttons",
                    x=0.45,
                    y=0.02,
                    xanchor="left",
                    yanchor="bottom",
                    bgcolor="#323130",
                    borderwidth=1,
                    bordercolor="#6d6d6d",
                    font=dict(color="#FFFFFF"),
                )
            ],
        ),
    )


if __name__ == "__main__":
    app.run_server(debug=True, port=9001, host='0.0.0.0')