from flask import Flask, render_template
app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clusterEvaluation')
def clusterEvaluation():
    return render_template('clusterEvaluation.html')

@app.route('/machineLearning')
def machineLearning():
    return render_template('mlRecommender.html')

if __name__=="__main__":
    app.run(debug=True)
    