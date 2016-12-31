from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import split,udf,col, lit
import os,math, re, sys
from datetime import datetime
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint


sc = SparkContext()
sqlContext = SQLContext(sc)
df = sqlContext.read.load('file:///home/cloudera/Downloads/training_Data.csv', format='com.databricks.spark.csv', header='true', inferSchema='true')
df = df.filter(df.lpep_pickup_datetime.isNotNull())

'''Function that converts the pickup time into hours format'''
def timeinhours(t):
    tArray = re.split(":", t)
    minutes = float(tArray[1])
    hour = float(tArray[0])
    if minutes != 0.0:
        tim = minutes/60.0 + hour
    else:
        tim = hour
    return tim


'''Load and parse the data for Logistic regression'''
def parsePoint(line, classSurge):
    values = [float(x) for x in line]
    if values[0] == classSurge:
        values[0] = 1
    else:
        values[0] = 0
    return LabeledPoint(values[0], values[1:])

'''To compute the cost for theta values'''
def costFunction(data, classSurge, theta):
    # print data
    theta1 = float(theta[0])
    theta2 = float(theta[1])
    theta3 = float(theta[2])
    x1 = float(data[0])
    x2 = float(data[1])
    x3 = float(data[2])
    surge = float(data[3])
    if surge == classSurge:
        y = 1.0
    else:
        y = 0.0
    thetaTransposeX =  (theta1 * x1) + (theta2 * x2) + (theta3 * x3)
    # print theta
    hThetaX = 1.0 / (1.0 + (math.exp(-thetaTransposeX)))
    sum = y * math.log(hThetaX)+(1.0 - y) * math.log(1.0-hThetaX)
    return sum

'''Gradient Descent function implementation from scratch'''
def gradientDescentFunction(data, classSurge, thetas, j):
    theta0 = float(thetas[0])
    theta1 = float(thetas[1])
    theta2 = float(thetas[2])
    theta3 = float(thetas[3])
    x1 = float(data[0])
    x2 = float(data[1])
    x3 = float(data[2])
    surge = float(data[3])
    if surge == classSurge:
        y = 1.0
    else:
        y = 0.0
    thetaTransposeX = theta0 + (theta1 * x1) + (theta2 * x2) + (theta3 * x3)
    hypothesis = 1.0/(1.0 + (math.exp(-thetaTransposeX)))
    if j == 1:
        JTheta = (hypothesis - y) * x1
    elif j == 2:
        JTheta = (hypothesis - y) * x2
    elif j == 3:
        JTheta = (hypothesis - y) * x3
    elif j == 0:
        JTheta = (hypothesis - y) * 1.0
    return JTheta

'''Training Model'''
def trainingModel(data,thetas, classSurge):
    theta1 = float(thetas[0])
    theta2 = float(thetas[1])
    theta3 = float(thetas[2])
    x1 = float(data[1])
    x2 = float(data[2])
    x3 = float(data[3])
    y = float(data[0])
    thetaTransposeX = (theta1 * x1) + (theta2 * x2) + (theta3 * x3)
    hypothesis = 1.0 / (1.0 + (math.exp(-thetaTransposeX)))
    # hypothesis=(theta1 * x1) + (theta2 * x2) + (theta3 * x3)
    # print hypothesis
    if hypothesis >= 0.5:
        return [y, classSurge]
    else:
        if y != classSurge:
            return [-1.0, -1.0]
        else:
            return [y, -1.0]


#------------------------------------------Data PreProcessing--------------------------------------------------------------------------------

'''Splitting the column "lpep_pickup_datetime " into two columns "PickupTime" and "PickupWeekday"'''
split_col = split(df['lpep_pickup_datetime'], ' ')
df = df.withColumn('PickupTime', split_col.getItem(1)).withColumn('PickupWeekday', split_col.getItem(0))
sp = udf(lambda x: datetime.strptime(x,'%M/%d/%Y').strftime('%A'))
Fdf = df.withColumn('Weekday', sp(col('PickupWeekday')))
m = Fdf.count()
m = float(m)
timeUDF = udf(timeinhours)
dframe = Fdf.withColumn("TimeInHours", timeUDF(col('PickupTime')))

'''create dataframe of required columns only (eliminating all unused columns)'''
requiredDataframe = dframe.select('improvement_surcharge', 'Pickup_longitude', 'Pickup_latitude', 'TimeInHours')

'''array of theta values if gradient descent algorithm is used'''
# thetaArrayClass1 = [3, 0.02, 0.02, 0.02]
# alpha = 0.0001*(1.0/m)
# tempThetas = [0, 0, 0, 0]

'''Implementation of LogisticRegression With LBFGS'''
parsedDataOne = requiredDataframe.rdd.map(lambda x: parsePoint(x, 0.3))
parsedDataTwo = requiredDataframe.rdd.map(lambda x: parsePoint(x, 0.0))
parsedDataThree = requiredDataframe.rdd.map(lambda x: parsePoint(x, -0.3))
'''Build the model'''
modelOne = LogisticRegressionWithLBFGS.train(parsedDataOne)
modelTwo = LogisticRegressionWithLBFGS.train(parsedDataTwo)
modelThree = LogisticRegressionWithLBFGS.train(parsedDataThree)

thetaArrayClass1 = modelOne.weights #tuned theta values
thetaArrayClass2 = modelTwo.weights #tuned theta values
thetaArrayClass3 = modelThree.weights #tuned theta values

'''If gradient descent implementation method is used'''
# costList = []
# costList.append(costValue)
# for i in range(10):
#     print i
#     for j in range(4):
#         hThetaSum = requiredDataframe.rdd.map(lambda x: gradientDescentFunction(x, classSurge, thetaArrayClass1, j)).sum()
#         theta = float(thetaArrayClass1[j])
#         tempThetas[j] = theta - (alpha * hThetaSum)
#     thetaArrayClass1 = tempThetas
#     print thetaArrayClass1
#     cost = requiredDataframe.rdd.map(lambda x: costFunction(x,classSurge,thetaArrayClass1)).sum()
#     costValue = -cost/m
#     costList.append(costValue)

'''Evaluationg the model'''
print thetaArrayClass1
print thetaArrayClass2
print thetaArrayClass3

trainingModelOne = requiredDataframe.rdd.map(lambda x: trainingModel(x, thetaArrayClass1, 0.3))
accuracyOne = trainingModelOne.filter(lambda (v, p): v == p).count() / m * 100.0
print accuracyOne
trainingModelTwo = requiredDataframe.rdd.map(lambda x: trainingModel(x, thetaArrayClass2, 0.0))
accuracyTwo = trainingModelTwo.filter(lambda (v, p): v == p).count() / m * 100.0
print accuracyTwo
trainingModelThree = requiredDataframe.rdd.map(lambda x: trainingModel(x, thetaArrayClass3, -0.3))
accuracyThree = trainingModelThree.filter(lambda (v, p): v == p).count() / m * 100.0
print accuracyThree
