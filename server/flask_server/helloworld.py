from flask import Flask
from flask import render_template
from flask import request
import socket

app = Flask(__name__)

port_1 = 7000

@app.route('/helloworld')
def hello_world():
    return 'Hello World'

@app.route('/get.html')
def get_html():
    return render_template('get.html')

@app.route('/post.html')
def post_html():
    return render_template('post.html')

@app.route('/deal_request', methods=['GET', 'POST'])
def deal_request():
    if request.method == 'GET':
        get_q = request.args.get("q", "")
        ret = query(get_q)
        return render_template('result.html', result=ret)
    elif request.method == 'POST':
        post_q = request.form["q"]
        ret = query(post_q)
        return render_template("result.html", result=ret)

def query(string):
    string = string.replace(';', '\n')
    s = socket.socket()
    host = socket.gethostname()
    s.connect((host, port_1))
    s.send(string.encode(encoding="utf-8"))
    s.send(bytes("done\n", 'utf-8'))
    ret = s.recv(102400)
    return ret


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
