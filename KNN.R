# k-nearest neighbors from scratch ------------------------------------------------------------------

library(ggplot2)
library(R6)
# for reproducibility
set.seed(0)

KnnClassification <- R6Class("KNN", public = list(
    ## public access
  
    ## training part. Store predictors and target variables. 
    fit = function(X, y,dist){
      row.names(X) <- NULL
      private$trainData = X
      private$target = y
      private$dist = dist
    },
    
    ## Prediction part. Finds the k clostest points and return the mode of the their target value. 
    findK = function(X,k){
      y = 0
      for(i in 1:nrow(X)){
        dist = private$findDistance(X[i,])
        sorted = sort(dist)
        # find the k closets neighbors target value
        kthTargets  = as.numeric(private$target[which(dist %in% sorted[1:5])]) 
        # find mode and store the value
        uniqv  = unique(kthTargets)
        k[i] = uniqv[which.max(tabulate(match(kthTargets, uniqv)))]
      }
      # return the the predictions
      return(k)
    }
    
  ),private = list(
    ## private access. Helper functions and variables.
    trainData = NULL,
    target = NULL,
    dist = NULL,
    
    ## finds the manhattan distance between a point and the points in the training data
    manhattan = function(X,dim){
      nrow = nrow(private$trainData)
      x_temp = as.numeric(matrix(rep(X,nrow), nrow =nrow, ncol = dim,byrow = T))
      diffAbs = abs(private$trainData - x_temp)
      dist = sqrt(apply(diffAbs, 1, sum))
      return(dist)
    },
    
    ## finds distance between a point and the points in the training data. euclidean or manhattan. 
    findDistance = function(X, nrow, ncol){
      dim = length(X)
      if(private$dist=="euclidean" || is.null(private$dist)){
        dist = private$euclidean(X,dim)
      }else if(private$dist=="manhattan"){
        dist = private$manhattan(X,dim)
      }
      return(dist)
    },
    
    ## finds the euclidean distance between a point and the points in the training data
    euclidean = function(X,dim){
      nrow = nrow(private$trainData)
      x_temp = as.numeric(matrix(rep(X,nrow), nrow =nrow, ncol = dim,byrow = T))
      diffSqrd = (private$trainData - x_temp)^2
      dist = sqrt(apply(diffSqrd, 1, sum))
      return(dist)
    }
  )
    
  
)


# example 1 ------------------------------------------------------------------

## create data
#create predictors
X1 = rnorm(100,100,2)
X2 = X1*rnorm(100,10,3)
data = data.frame(X1,X2)
#create target
data$Y = ifelse(data$X1*data$X2 > 100000, 2,1 )
data$Y = as.factor(data$Y)

# visualize the data
ggplot(data, aes(x= X1, X2, group = Y ,color = Y))+
  geom_point()



# create the model
knn = KnnClassification$new()

# create data split. ind = training data index. -ind = test data index
ind = sample(1:nrow(data), size = nrow(data)*0.7,replace = F)

# train the model on the training data
knn$fit(data[ind,1:2],data$Y[ind],"euclidean")

# predicted target values based on test data
pred = knn$findK(data[-ind,1:2],5)
# the actual target values of the test data
actual = data$Y[-ind]

# calculate the accuracy
accuracy = function(pred,actual){
  t = table(pred,actual)
  sum(diag(t))/sum(table(pred,actual))
  
}
# model performance. 
accuracy(pred,actual)




# example 2 ------------------------------------------------------------------

# get data
data = iris
# convert encode target values. string to numeric. 
data$Species = as.numeric(as.factor(data$Species))


# function to normalize data between 0 and 1
normalize = function(X){
  (X - min(X))/(max(X) - min(X))
}

# normalize the predictor variables
data[,1:4] = apply(data[,1:4],2,normalize)

# create a model
knn = KnnClassification$new()

# create data split. ind = training data index. -ind = test data index
ind = sample(1:nrow(data), size = nrow(data)*0.7,replace = F)

# train the model on the training data
knn$fit(data[ind,1:4],data$Species[ind],"euclidean")

# the actual target values of the test data
pred = knn$findK(data[-ind,1:4],5)

# the actual target values of the test data
actual = data$Species[-ind]

# model performance. 
accuracy(pred,actual)



