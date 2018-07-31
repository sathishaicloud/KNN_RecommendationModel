import pandas as pd
import numpy as np

BRdataFile = 'C:/dev/MachineLearning/RecommendationModels/dataset/BX-Book-Ratings.csv'
BRdata = pd.read_csv(BRdataFile,sep=";",header=0,names=["user","isbn","ratings"], encoding = "ISO-8859-1", error_bad_lines=False)
print(BRdata.head())

BdataFile = 'C:/dev/MachineLearning/RecommendationModels/dataset/BX-Books.csv'
Bookdata = pd.read_csv(BdataFile,sep=";", usecols=[0,1,2], index_col=0,header=0,names=["isbn","title","Author"], encoding = "ISO-8859-1", error_bad_lines=False)
Bookdata.head()

#UdataFile = 'C:/dev/MachineLearning/RecommendationModels/dataset/BX-Users.csv'
#Userdata = pd.read_csv(UdataFile,sep=";", usecols=[0,1,2], index_col=0,header=0,names=["isbn","title","Author"], encoding = "ISO-8859-1", error_bad_lines=False)
#Userdata.head()

#######################################################################
#   Return book Metadata such as Title and Author for given ISBN 
#######################################################################

def bookMeta(isbn):
    title = Bookdata.at[isbn,"title"]
    author = Bookdata.at[isbn,"Author"]
    return title, author
#bookMeta("0195153448")

#####################################################
#   Pick fav books
#####################################################

def favBooks(user, N):
    userRatings = BRdata[BRdata["user"]==user]
    sortedRatings = pd.DataFrame.sort_values(userRatings,["ratings"],ascending=[0])[:N]
    sortedRatings["title"] = sortedRatings["isbn"].apply(bookMeta)
    return sortedRatings

#favBooks(276729,5)

#####################################################
#   Compute userItemRatingMatrix
#####################################################

BRdata = BRdata[BRdata["isbn"].isin(Bookdata.index)]

usersPerISBN = BRdata.isbn.value_counts()
usersPerISBN.head()

ISBNPerUser = BRdata.user.value_counts()
ISBNPerUser.head()

BRdata = BRdata[BRdata["isbn"].isin(usersPerISBN[usersPerISBN > 10].index)]
BRdata = BRdata[BRdata["user"].isin(ISBNPerUser[ISBNPerUser > 10].index)]
BRdata.shape

userItemRatingMatrix = pd.pivot_table(BRdata, values="ratings", index=['user'],columns=['isbn'])

userItemRatingMatrix.head()

#####################################################
#   Compute distance between users in multiD space
#####################################################

user1 = 408
user2 = 446
from scipy.spatial.distance import hamming
import numpy as np
def computeDistance(user1, user2):
    try:
        user1Ratings = userItemRatingMatrix.transpose()[user1]
        user2Ratings = userItemRatingMatrix.transpose()[user2]
        distance  = hamming(user1Ratings, user2Ratings)
    except:
        distance = np.NAN
    return distance
#computeDistance(user1, user2)

#####################################################
#   Find K nearest neighbor of user 408
#####################################################

user = 408
def findNearestNeighbors(user,K=10):
    allusers = pd.DataFrame(userItemRatingMatrix.index)
    allusers = allusers[allusers.user!=user]
    allusers["distance"] = allusers["user"].apply(lambda x:computeDistance(user,x))
    KNearestNeighbors = allusers.sort_values(["distance"],ascending=True)["user"][:K]
    return KNearestNeighbors

KNearestNeighbors = findNearestNeighbors(user)
print(KNearestNeighbors)

#####################################################
#   Find Top N Recommendations
#####################################################

def topN(user, N=10):
    NNRatings = userItemRatingMatrix[userItemRatingMatrix.index.isin(KNearestNeighbors)]
    avgRatings = NNRatings.apply(np.nanmean).dropna()
    booksAlreadyRead = userItemRatingMatrix.transpose()[user].dropna().index
    avgRatings = avgRatings[~avgRatings.index.isin(booksAlreadyRead)]
    avgRatings.head()
    topNISBNs = avgRatings.sort_values(ascending=False).index[:N]
    topNISBNs.head()
    recommendedBooks = pd.Series(topNISBNs).apply(bookMeta)
    return recommendedBooks

recommendedBooks = topN(user)
print(recommendedBooks)