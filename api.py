from flask import Flask,render_template,request

app=Flask(__name__)

from common import get_tensor
#from inference import get_name,get_type
from inference import get_type

@app.route('/', methods=['GET','POST'])
def h():
    if request.method=='GET':
        return render_template('index.html', value="ddi")
    if request.method=='POST':
        if 'canbe anything' not in request.files:
            print("file not uploaded")
        file = request.files['canbe anything']
        image = file.read()
        #tensor=get_tensor(image_bytes=image)
        #category,fname=get_name(image_bytes=image)
        category=get_type(image_bytes=image)
        #file.save('my_file.jpeg')
        #return render_template('result.html',flower=category,cl=fname)
        return render_template('result.html',flower=category)


if __name__ == "__main__":
    #app.run(host='0.0.0.0' , debug=True)
    app.run(debug=True)
    #app.run(debug=True, port=os.getenv('PORT',5000))
