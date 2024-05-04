from flask import Flask, request, redirect, url_for, flash, render_template
import os
from werkzeug.utils import secure_filename
import subprocess  # Import the subprocess module
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'super secret key'
UPLOAD_FOLDER = 'upload_folder'  # Set this to your actual folder path
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'local_weather.csv')
        file.save(filename)  # Save the file

        # Now run the train.py script
        try:
            # Run the script and wait for it to complete
            result = subprocess.run(['python', 'train.py', filename], capture_output=True, text=True, check=True)
            output = result.stdout
            formatted_output = "{:.2f}".format(float(output.strip()))
        except subprocess.CalledProcessError as e:
            formatted_output = "An error occurred: {}".format(e.output)
        except ValueError:
            formatted_output = "Invalid output format"
        
        # Redirect to predict page with formatted output
        return redirect(url_for('predict', result=formatted_output))

@app.route('/predict')
def predict():
    result = request.args.get('result', 'No prediction available.')
    # Calculate tomorrow's date
    tomorrow = datetime.now() + timedelta(days=1)
    tomorrow_date = tomorrow.strftime('%B %d, %Y')  # Format the date as "Month Day, Year"
    return render_template('predict.html', prediction=result, tomorrow_date=tomorrow_date)


if __name__ == '__main__':
    app.run(debug=True)
