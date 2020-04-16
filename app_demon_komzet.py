import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State
import pandas as pd
import plotly.graph_objs as go
from flask import Flask, Response
import cv2
from mpu6050 import mpu6050
import time
from collections import deque

import edgetpu.detection.engine
import cv2
from PIL import Image
import os
import argparse



X = deque(maxlen=20)
X.append(0)

Y = deque(maxlen=20)



class VideoCamera(object):
    def __init__(self):
        #self.video = cv2.VideoCapture('http://192.168.2.110:8888/')
        self.video = cv2.VideoCapture(0)

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            '--model', help='File path of Tflite model.', required=False)
        self.parser.add_argument(
            '--label', help='File path of label file.', required=False)
        self.args = self.parser.parse_args()

        self.args.model = 'assets/models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
        self.args.label = 'assets/models/coco_labels.txt'

        with open(self.args.label, 'r') as f:
            self.pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
            self.labels = dict((int(k), v) for k, v in self.pairs)
            
        self.IM_WIDTH = 640
        self.IM_HEIGHT = 480
        self.ret = self.video.set(3, self.IM_WIDTH)
        self.ret = self.video.set(4, self.IM_HEIGHT)

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText = (10, self.IM_HEIGHT - 10)
        self.fontScale = 1
        self.fontColor = (255, 255, 255)  # white
        self.boxColor = (0, 0, 255)  # red
        self.boxLineWidth = 1
        self.lineType = 2

        self.annotate_text = ""
        self.annotate_text_time = time.time()
        self.time_to_show_prediction = 1.0  # ms
        self.min_confidence = 0.20

        # initial classification engine
        self.engine = edgetpu.detection.engine.DetectionEngine(self.args.model)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        image = cv2.resize(image, None, fx=0.9, fy=1)
        #image = image[0:640, 0:480]
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    
    def get_obj(self):

        # initialize open cv
        
        elapsed_ms = 0

        start_ms = time.time()
        ret, img = self.video.read()
        input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB color space
        img_pil = Image.fromarray(input)
        # input = cv2.resize(input, (width,height))
        start_tf_ms = time.time()
        results = self.engine.DetectWithImage(img_pil, threshold=self.min_confidence, keep_aspect_ratio=True,
                                     relative_coord=False, top_k=5)
        end_tf_ms = time.time()
        elapsed_tf_ms = end_tf_ms - start_ms

        if results:
            for obj in results:
                box = obj.bounding_box
                coord_top_left = (int(box[0][0]), int(box[0][1]))
                coord_bottom_right = (int(box[1][0]), int(box[1][1]))
                cv2.rectangle(img, coord_top_left, coord_bottom_right, self.boxColor, self.boxLineWidth)
                self.annotate_text = "%s, %.0f%%" % (self.labels[obj.label_id], obj.score * 100)
                coord_top_left = (coord_top_left[0], coord_top_left[1] + 15)
                cv2.putText(img, self.annotate_text, coord_top_left, self.font, self.fontScale, self.boxColor, self.lineType)
            img = cv2.resize(img, None, fx=0.9, fy=1)
            ret, jpeg = cv2.imencode('.jpg', img)
            return jpeg.tobytes()
        else:
            img = cv2.resize(img, None, fx=0.9, fy=1)
            ret, jpeg = cv2.imencode('.jpg', img)
            return jpeg.tobytes()
    

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def oen(camera):
    while True:
        frame = camera.get_obj()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


server = Flask(__name__)
app = dash.Dash(__name__, server=server,
                external_stylesheets=[
                    "https://codepen.io/chriddyp/pen/bWLwgP.css"
                ])

@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@server.route('/obj_feed')
def obj_feed():
    return Response(oen(VideoCamera()),
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
                        html.Div([
                            dcc.Graph(
                                id="graph_left",
                                animate=True
                            ),
                            dcc.Interval(
                                id='graph-update',
                                interval=1000,
                                n_intervals = 0
                                )
                            ], id='plot_left_inner', style={'display': 'none'})
                    ], className="plot_left"),


                    html.Div(children=[
                        html.Div([
                            html.P("Video Stream",
                                    id="title_right")
                        ], className="plot_title_right"),

                        html.Div([
                            html.Img(src="",
                                     id='graph_right')
                        ], className="plot_right"),
                    ])

        ])
    ])


@app.callback(Output('button', 'className'),
              [Input('button', 'n_clicks')])
def update_id(n_clicks):
    if n_clicks % 2 == 0:
        return 'play_button'
    else:
        return 'stop_button'
    
@app.callback(Output('plot_left_inner', 'style'),
              [Input('button', 'n_clicks')])
def update_left_style(n_clicks):
    if n_clicks % 2 == 0:
        return {'display': 'none'}
    else:
        return {'display': 'block'}
    
@app.callback(Output('graph_right', 'src'),
              [Input('button', 'n_clicks')])
def start_video(n_clicks):
    if n_clicks % 2 != 0:
        return '/obj_feed'
        #return '/video_feed'
    else:
        return ''

@app.callback(Output('graph_left', 'figure'),
              #[Input('button','n_clicks')],
              [Input('graph-update', 'n_intervals'), Input('button', 'n_clicks')])
def update_graph(n_intervals, n_clicks):
    global X
    global Y
    mpu = mpu6050(0x68)
    acc = mpu.get_accel_data()
    Y.append(acc['z'])
    X.append(X[-1]+1)
    
    while n_clicks % 2 != 0:
        data = go.Scatter(
            x = list(X),
            y = list(Y),
            name='Scatter',
            mode = 'lines+markers',
            line=dict(color='#AE3326')
                )
        return {'data': [data],
                'layout': go.Layout(xaxis = dict(showgrid=False,
                                                 ticks='',
                                                 showticklabels=False,
                                                 range=[min(X), max(X)]),
                                    yaxis = dict(range=[min(Y)-2, max(Y)+2]))}
    else:
        data = go.Scatter(
            x = list(),
            y = list(),
            name='Scatter',
            mode = 'lines+markers',
            line=dict(color='#AE3326')
            )
        
        return {'data': [],
                'layout': go.Layout(xaxis = dict(showgrid=False,
                                                 ticks='',
                                                 showticklabels=False,
                                                 range=[min(X), max(X)]),
                                    yaxis = dict(range=[min(Y)-2, max(Y)+2]))}
    

if __name__=="__main__":
    app.run_server(debug=False, port=8080, host='0.0.0.0')