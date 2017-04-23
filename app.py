from flask import Flask, render_template
app=Flask(__name__)

@app.route("/")
def index():
    f=open("Non_NumericDataClustering.html", "r")
    nonNumericClustering=f.read()
    return render_template('index.html', nonNumericClustering=nonNumericClustering)


if __name__=="__main__":
    app.run(debug=True)
    