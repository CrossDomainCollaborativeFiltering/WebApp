from Querying import PrepareData, Querying, ComputeResults
from returnAllMovies import MovieDataSet
from returnAllUsers import UserDataSet
from RecommendationLogReg import AllUsers, User, Document, TfVectorizer, SampleDocument, SampleTfVectorizer, Recommend, PredictMovies
# from Querying import returnAllUsers, returnAllMovies

# the data has to be prepared
obj0=ComputeResults()
dataObject=obj0.prepare()

obj1=PredictMovies()
obj1.prepareDataModelBased()

movieNames=obj1.getMovieNames()

from flask import Flask, render_template
app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clusterEvaluation/')
def clusterEvaluation():
    obj2=UserDataSet()
    # print (type(obj2))
    allUsers=obj2.users()
    return render_template('clusterEvaluation.html', users=allUsers)

@app.route('/machineLearning/')
def machineLearning():
    obj2=UserDataSet()
    allUsers=obj2.users()
    # obj3=AllMovies()
    # allMovies=obj3.movies()
    return render_template('mlRecommender.html', users=allUsers)

@app.route('/machineLearning/trainLogRegModel/<userid>')
def trainModel(userid):
    userid=int(userid)
    obj3=MovieDataSet()
    allMovies=obj3.movies()
    a=obj1.trainModel(userid)
    if a==-1:
        # it means that the evaluator has entered a bogus user id through the system
        return render_template('/errorPage.html')
    return render_template('/predictMovieRating.html', movieNames=movieNames, allMovies=allMovies)

@app.route('/clusterEvaluation/showRecommendation/<userid>')
def showRecommendation(userid):
    userid=int(userid)
    # this will show the recommendation to the current user and render the same template with the results
    recommendation=obj0.computeAverageRatings(userid, dataObject)
    if recommendation==-1:
        return render_template('/errorPage.html')
    # here recommendation is a tuple that stores the following data: (averageRatings, recMovieNames, ratedMovieNames) 
    # averageRatings is a list of 3 ratings, recMovieNames are the moviesRecommended and ratedMovieNames are the movie names of the movies that the user had watched.
    for i in range(len(recommendation[0])):
        recommendation[0][i]=round(recommendation[0][i],2)
    return render_template('clusterEvaluationResults.html', averageRatings=recommendation[0], recMovieNames=recommendation[1], ratedMovieNames=recommendation[2])

@app.route('/machineLearning/trainLogRegModel/predictRating/<movieid>')
def predictMovieRating(movieid):
    movieid=int(movieid)
    prediction=obj1.predict(movieid)
    if prediction==-1:
        return render_template('/errorPage.html')
    prediction=int(prediction[0])
    return render_template('showPrediction.html', prediction=prediction)

# TODO for this app:
# Add loading icon when all movies are loaded onto screen for the route '/machineLearning/trainLogRegModel/<userid>'. 
# Make sure the system is fail safe. That is the user should not be able to enter any weird values into the system.

if __name__=="__main__":
    app.run(debug=True)
    