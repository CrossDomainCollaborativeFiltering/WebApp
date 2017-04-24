import numpy as np
import csv

# This is the querying part of the system after pre-processing
NUM_RATINGS_TO_QUERY=3
CLUSTER_FILENAME="../NonNumericClustering/LabelsNumreicClusters100Op1.csv"
# CLUSTER_FILENAME="../NonNumericClustering/LabelsClusters200NoIdfOp2ActorDirectorNoCountry.csv"
USER_MOVIE_FILENAME="../../../dataSets/movieLens/user_ratedmovies.csv"

class PrepareData:
    
    def __init__(self):
        
        self.users=[]
        self.ratings=[]
        self.movieID=[]
        self.allocatedMovieIDs=[]
        self.trueMovieIds=[]
        self.clusters=[]
        self.movieNames=[]

    def openFile(self, fileName):
    
        csvFile=open(fileName, newline="")
        reader=csv.reader(csvFile)
        return reader

    def stringToList(self, s): # this is a hack: avoid as far as possible
         
        s=s.strip('[')
        s=s.strip(']')
        # print(s)
        s=s.strip("'")
        # print(s)
        return s.split("', '")


    def getAllClusters(self):
        csvReader=self.openFile(CLUSTER_FILENAME)
        
        for row in csvReader:

            if row[0][0]=='[':
                # means it is a list
                for cluster in row:
                    temp=self.stringToList(cluster)
                    temp=temp[0]
                    temp=temp.split(', ')
                    # print(type(temp))
                    # print("list now: "+str(temp))
                    indices=[]
                    for index in temp:
                        # print("index now"+index)
                        # print(type(index))
                        indices.append(int(index))
                    self.clusters.append(indices)
                    del indices
            else:
                temp=[]
                for index in row:
                    temp.append(int(index))
                self.clusters.append(temp)     
                del temp
    

    def getUserDetails(self):
    
        csvReader=self.openFile(USER_MOVIE_FILENAME)
        rowIter=iter(csvReader)
        next(rowIter) # to get rid of the first row
        # users=[]
        while True:
            try:
                
                row=next(rowIter)
                self.users.append(int(row[0]))
                self.movieID.append(int(row[1]))
                self.ratings.append(float(row[2]))

            except StopIteration:   
                # StopIteration exception is reached when the iterator has iterated thorugh all the rows
                # of the csvFile
                break
    
    def getTrueMovieIds(self):
        csvReader=self.openFile("../MovieLens/ResultMovieDataSetClone.csv") # this file stores the true movieIds
        next(csvReader)
        for row in csvReader:
            self.allocatedMovieIDs.append(int(row[1]))
            self.trueMovieIds.append(int(row[0])-1) # this -1 is essential
            self.movieNames.append(row[2])

class Querying:

    def __init__(self, userFileName, users, ratings, movieID, trueMovieIds, allocatedMovieIDs, clusters):
        
        self.userFileName=userFileName

        self.users=users
        self.ratings=ratings
        self.movieID=movieID

        self.trueMovieIds=trueMovieIds
        self.index=-1
        self.clusters=clusters
        self.userStartIndex=0
        self.userEndIndex=0
        self.userRatedMovies=[]
        self.allocatedMovieIDs=allocatedMovieIDs
        self.userRatings=[] # ratings assigned by the specific user


    def openFile(self, fileName):

        csvFile=open(fileName, newline="")
        reader=csv.reader(csvFile)
        return reader

    # getLargestRatings gets the largest num ratings from the list ratings 
    def getLargestRated(self, ratings, num):

        large=0
        position=0
        highestRatedIndices=[]
        highestRatedValues=[]
        iterations=0
        while iterations < num:
            large=0    
            for i in range(len(ratings)):
                
                if (ratings[i] > large) and (i not in highestRatedIndices):
                    # print(ratings[i])
                    large=ratings[i]
                    position=i
            
            highestRatedIndices.append(position)
            highestRatedValues.append(large)
            # del ratings[position]
            iterations+=1

        return (highestRatedIndices, highestRatedValues)

    def getUserApproxIndex(self, userID):
        
        low=0
        high=len(self.users)-1
        self.index=-1
        # index=-1
        while low <= high:
            mid=int((low+high)/2)
            if self.users[mid]==userID:
                # print("found index:"+str(mid))
                self.index=mid
                break
            elif self.users[mid]>userID:
                high=mid-1
            elif self.users[mid]<userID:
                low=mid+1

    def getUserStartIndex(self, userID):

        if self.index==-1:
            return -1
        else:
            # startIndex of the userID to be found
            i=self.index
            while i>=0:
                if userID!=self.users[i]:
                    self.userStartIndex=i+1
                    return i+1
                i-=1
            return 0

    def getUserEndIndex(self, userID):
        
        if self.index==-1:
            return -1
        else:
            # startIndex of the userID to be found
            i=self.index
            while i<len(self.users):
                if userID!=self.users[i]:
                    self.userEndIndex=i
                    return i
                i+=1
            return len(self.users)-1

            

    def getTopKMovies(self, userID):
        
        self.getUserApproxIndex(userID)
        if self.index==-1:
            raise Exception('Unknown user')
        startIndex=self.getUserStartIndex(userID)
        endIndex=self.getUserEndIndex(userID)
        # print("start index is:"+str(startIndex)+"\n end index:"+str(endIndex))
        userRatings=[]
        for i in range(startIndex, endIndex):
            self.userRatedMovies.append(self.movieID[i])
            self.userRatings.append(self.ratings[i])

        # print ("userRatedMovies:"+str(self.userRatedMovies))
        # print ("userRatings:"+str(self.userRatings))

        val=self.getLargestRated(self.userRatings, NUM_RATINGS_TO_QUERY)
        indices=val[0]
        ratings=val[1]

        # print ("largest rated movie indices:"+str(indices))
        # print("largest ratings:"+str(ratings))
        # now indices has the positions where the highest ratings are stored

        # once we have the indices we need to get the movie id of these indices
        movies=[]
        for i in indices:
            movies.append(self.userRatedMovies[i])
    
        # movies is a list that stores the highest rated movies

        # print("movies are :"+str(movies))

        topKTrueIds=[]
        for i in movies:
            topKTrueIds.append(self.checkTrueId(i))

        # print("true movie ids are:"+str(topKTrueIds))

        return (topKTrueIds, ratings)

    """
    The following function returns the true movie ids of the movies. As observed the MovieLens MovieIds are different from the clusters and we need to find the cluster movieIds to check the movie in the clusters
    """
    def checkTrueId(self, movieId):
        
        # count=0
        # print("length odf allocated movieId:"+str(len(self.allocatedMovieIDs)))
        # print ("value at 8661:"+str(self.allocatedMovieIDs[8661]))
        index=self.allocatedMovieIDs.index(movieId)
        # print("index:"+str(index))
        return self.trueMovieIds[index]

    def stringToList(self, s): # this is a hack: avoid as far as possible
         
        s=s.strip('[')
        s=s.strip(']')
        # print(s)
        s=s.strip("'")
        # print(s)
        return s.split("', '")

    
    def findCluster(self, movieID):
        
        for index in range(len(self.clusters)):
            if movieID in self.clusters[index]:
                break
        # i now is the cluster number where the movie is present

        return index

    def recommend(self, movieID, userRating):
        
        recommendedMovies=[]
        clusterNum=self.findCluster(movieID)
        for i in self.clusters[clusterNum]:
            # remember userRated movies is the list that contains the movieIDs of the movies that the user
            # had rated
            # i is the true id of the first movie that is present in the cluster
            if i != movieID:
                allocatedID=self.allocatedMovieIDs[i]
                if allocatedID in self.userRatedMovies:
                    recommendedMovies.append(i)
        # so naturally the ultimate recommendation is the true movie ids 
        return recommendedMovies            

class ComputeResults:
    
    def __init__(self):
        self.dummy=0
        self.dataObj=None

    def prepare(self):
        # check()
        self.dataObj=PrepareData()
        
        self.dataObj.getUserDetails()
        print("acquired user details")
        self.dataObj.getTrueMovieIds()
        print("acquired true movie ids")
        self.dataObj.getAllClusters()
        print("all clusters cached")
        return self.dataObj

    def computeAverageRatings(self, userid, obj):# here obj of PrepareData()
        sumRating=0
        averageRatings=[]
        ratedMovieNames=[]
        recMovieNames=[]
        if userid not in self.dataObj.users:
            return -1
        # while True:
        # print("Enter a userid, -1 to quit-------------------------------------------------------------")
        # userid=int(input())
        q=Querying(USER_MOVIE_FILENAME, obj.users, obj.ratings, obj.movieID, obj.trueMovieIds, obj.allocatedMovieIDs, obj.clusters)
        if userid==-1:
            return -1
        # elif userid==71535:
        #     return 0
        # uid+=1
        v=q.getTopKMovies(userid)
        
        topkmovies=v[0]
        #-------------------------------------preparing ratedMovieNames----------------------------
        for movieid in topkmovies:
            ratedMovieNames.append(obj.movieNames[movieid])

        topkratings=v[1]
        emptyList=0
        topkAllocatedMovieIDs=[]
        for i in topkmovies:
            topkAllocatedMovieIDs.append(obj.allocatedMovieIDs[obj.trueMovieIds.index(i)])
        
        # print ("topkTrueMovieIDs: "+str(topkmovies))
        # print ("topkAllocatedMovieIDs: "+str(topkAllocatedMovieIDs))
        # print ("topkratings: "+str(topkratings))
        del topkAllocatedMovieIDs
        # print("checking valid user:"+str(uid-1))
        for i in range(NUM_RATINGS_TO_QUERY):
            # print("recommendation:---")
            rec=q.recommend(topkmovies[i], topkratings[i])
            # rec is a list of the recommendations of true movie ids
            # print (rec)

            # --------------------------------------preparing recMovieNames--------------------------------
            for movie in rec:
                recMovieNames.append(obj.movieNames[movie])

            # since we are getting trueMovieids here we can get their corresponding movie names from obj.movieNames[trueMovieID]
            if rec!=[]:
                for i in rec:
                    index=obj.trueMovieIds.index(i)
                    allocatedMovieID=obj.allocatedMovieIDs[index]
                    rating=obj.ratings[obj.movieID.index(allocatedMovieID)]
                    sumRating+=rating
                # print ("sum of ratings:"+str(sumRating))
                # print("recommended movies average rating: "+str(sumRating/len(rec)))
                # --------------------------preparing average ratings-------------------------------------
                averageRatings.append(sumRating/len(rec))
                
            sumRating=0
        del q
        return (averageRatings, recMovieNames, ratedMovieNames) 

# if __name__=="__main__":
#     main()
