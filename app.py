from Querying import PrepareData, Querying, ComputeResults
from returnAllMovies import AllMovies
from returnAllUsers import AllUsers
# from Querying import returnAllUsers, returnAllMovies

# the data has to be prepared
obj1=ComputeResults()
dataObject=obj1.prepare()

from flask import Flask, render_template
app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clusterEvaluation/')
def clusterEvaluation():
    obj2=AllUsers()
    allUsers=obj2.users()
    return render_template('clusterEvaluation.html', users=allUsers)

@app.route('/machineLearning/')
def machineLearning():
    return render_template('mlRecommender.html')

@app.route('/clusterEvaluation/showRecommendation/<userid>')
def showRecommendation(userid):
    userid=int(userid)
    # this will show the recommendation to the current user and render the same template with the results
    recommendation=obj1.computeAverageRatings(userid, dataObject)
    # here recommendation is a tuple that stores the following data: (averageRatings, recMovieNames, ratedMovieNames) 
    # averageRatings is a list of 3 ratings, recMovieNames are the moviesRecommended and ratedMovieNames are the movie names of the movies that the user had watched.
    for i in range(len(recommendation[0])):
        recommendation[0][i]=round(recommendation[0][i],2)
    return render_template('clusterEvaluationResults.html', averageRatings=recommendation[0], recMovieNames=recommendation[1], ratedMovieNames=recommendation[2])

if __name__=="__main__":
    app.run(debug=True)
    