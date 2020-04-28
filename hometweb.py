# html entype속성을 multipart/form-data로 지정해야 파이썬 리퀘스트를 이요한 전달을 사용가능하다



from flask import Flask, render_template,  send_file
#파일 업로드
from flask import request
#파일이름 보호
from werkzeug import secure_filename
import os
app = Flask(__name__)
#파일 업로드 용량 제한 단위:바이트
#최대 용량 16mb
#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.errorhandler(404)
def page_not_found(error):
	app.logger.error(error)
	return render_template('page_not_found.html'), 404

#HTML 렌더링
@app.route('/')
def home_page():
	return render_template('home.html')
#파일 리스트
# uploads 폴더에 파일 저장한다
@app.route('/list')
def file():
    file_list = os.listdir("./uploads")
    html= """<center><a href="/">홈페이지</a><br><br>"""
    html+="file_list:"{}".format(file_list)+"</center>"
    return html

#업로드 HTML 렌더링
@app.route('/upload')
def upload_page():
	return render_template('upload.html')

#파일 업로드 처리
@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
        # 업로드페이지에서 post 형식으로 넘어오면 requests객체에 file이름으로 전송된
        #파일을 가져온다
		f = request.files['file']
		#저장할 경로 + 파일명  secure는 파일 이름 보호하는 효과
		f.save('./uploads/' + secure_filename(f.filename))
		files = os.listdir("./uploads")
		return render_template('check.html')

#다운로드 HTML 렌더링
@app.route('/downfile')
def down_page():
	files = os.listdir("./uploads")
	return render_template('filedown.html',files=files)

#파일 다운로드 처리
@app.route('/fileDown', methods = ['GET', 'POST'])
def down_file():
	if request.method == 'POST':
		sw=0
		files = os.listdir("./uploads")
		for x in files:
			if(x==request.form['file']):
				sw=1

		path = "./uploads/"
		return send_file(path + request.form['file'],
				attachment_filename = request.form['file'],
				as_attachment=True)

if __name__ == '__main__':
	#서버 실행
	app.run(host='0.0.0.0', debug = True)
