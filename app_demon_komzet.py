import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from flask import Flask, Response
import cv2

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        image = cv2.resize(image, (600, 600))
        ret, jpeg = cv2.imencode('.jpg', image)
        #return jpeg.tobytes()

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


df = pd.read_csv(r"C:\Users\u500646\Desktop\FastTestsDatenkost\20200218_testfile.csv")

trace_close = go.Scatter(x=list(df.distance[0:1000]),
                         y=list(df.z_acc_reell[0:1000]),
                         name="Z-Acc",
                         line=dict(color='#AE3326'))

trace_close = go.Scatter(x=list(df.distance[0:1000]),
                         y=list(df.z_acc_reell[0:1000]),
                         name="Z-Acc",
                         line=dict(color='#AE3326'))


server = Flask(__name__)
app = dash.Dash(__name__, server=server,
                external_stylesheets=[
                    "https://codepen.io/chriddyp/pen/bWLwgP.css"
                ])

@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.layout = html.Div(
    children=[
        html.Div(children=[
            html.Img(src="/assets/images/KomZ.png",
                     id="komzet"
                     ),

            html.Img(src="/assets/images/Mittelstand-Digital.png",
                     id="mittelstand_digital"
                     ),

            html.Img(src="/assets/images/BMWi.png",
                     id="bmwi"
                     ),

            html.Button(id='button',
                        className='play_button',
                        n_clicks=2)
        ], className='banner'),

        html.Div(children=[
                    html.Div(children=[
                        html.P("Acceleration Plot",
                                id="title_left")
                    ], className="plot_title_left"),

                    html.Div([
                        dcc.Graph(
                            id="graph_left",
                            figure={
                                    "data": [trace_close],
                                },
                        )
                    ], className="plot_left"),


                    html.Div(children=[
                        html.Div([
                            html.P("Video Stream",
                                    id="title_right")
                        ], className="plot_title_right"),

                        html.Div([
                            html.Img(src="/video_feed",
                                     id='graph_right')
                        ], className="plot_right"),
                    ])

        ])
    ])

@app.callback(dash.dependencies.Output('button', 'className'),
              [dash.dependencies.Input('button', 'n_clicks')])
def update_id(n_clicks):
    if n_clicks % 2 == 0:
        return 'play_button'
    else:
        return 'stop_button'

if __name__=="__main__":
    app.run_server(debug=True)