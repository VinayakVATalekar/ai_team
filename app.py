from distutils.log import debug
from fileinput import filename
import pandas as pd
from flask import *
import os
from werkzeug.utils import secure_filename
from mongo import *
from additional import *
from Attrition_Module import main
#import papermill as pm

UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')

# Define allowed files
ALLOWED_EXTENSIONS = {'csv','xlsx'}

app = Flask(__name__,template_folder='template')

# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = 'you shall know pain'

dict={}


@app.route('/', methods=['GET', 'POST'])
def uploadFile():
	if request.method == 'POST':
	# upload file flask
		f = request.files.get('file')

		# Extracting uploaded file name
		data_filename = secure_filename(f.filename)
		
        
		f.save(os.path.join(app.config['UPLOAD_FOLDER'],data_filename))
		

		session['uploaded_data_file_path'] =os.path.join(app.config['UPLOAD_FOLDER'],data_filename)
		

		return render_template('index2.html')
	
	return render_template("index.html")


@app.route('/show_data')
def showData():    
    data_file_path= session.get('uploaded_data_file_path', None)
    ext=extension_reader(data_file_path)
    if ext==0:
        uploaded_df=pd.read_csv(data_file_path)
    else:
        uploaded_df=pd.read_excel(data_file_path)

    uploaded_df_html = uploaded_df.to_html()
    return render_template('show_csv_data.html',data_var=uploaded_df_html)

@app.route('/col/')
def dropdown():

    data_file_path = session.get('uploaded_data_file_path', None)
    ext=extension_reader(data_file_path)
    if ext==0:
        read=pd.read_csv(data_file_path)
    else:
        read=pd.read_excel(data_file_path)
    # read=pd.read_excel(r"D:\api team\staticFiles\uploads\HR Dummy Dataset.xlsx", sheet_name='Copy of Sheet1')
    columns=read.columns
    return render_template('test.html', columns=columns)

@app.route('/collect', methods=['POST']) ###----1
def collect():
    output = request.get_json()#result=_store(output)   
    result = json.loads(output)  
    x=order_store(result)  
    return x

@app.route('/collect1', methods=['POST']) ###----2
def collect1():
    output = request.get_json()
    result = json.loads(output) 
    x=drop_store(result)
    return x

@app.route('/collect2', methods=['POST']) ###----3
def collect2():
    output = request.get_json()
   
    result = json.loads(output) 
    x=target_store(result)
    return x

@app.route("/result/",methods=["GET"])
def result():
      data_file_path = session.get('uploaded_data_file_path', None)
      res=main(data_file_path)
      file_name = 'MarksData.xlsx'
      res.to_excel(file_name)
      result=pd.read_excel(file_name)
      uploaded_df_html = result.to_html()
      return render_template('show_csv_data.html',data_var=uploaded_df_html)
      
      

#

if __name__ == '__main__':
	app.run(debug=True)
