from flask import Flask,render_template,request

app=Flask(__name__)

@app.route('/', methods=['GET','POST'])
def h():
    if request.method=='GET':
        return render_template('index.html', value="ddi")
    if request.method=='POST':
        return render_template('result.html',flower="ddff")


if __name__ == "__main__":
    #app.run(host='0.0.0.0' , debug=True)
    app.run(debug=True)
