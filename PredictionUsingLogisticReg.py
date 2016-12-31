from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import split,udf,col, lit
import os,math, re, sys
from datetime import datetime

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

'''Predict the class based on the training model (theta values)'''
def predictClass(data,thetasList):
    # theta1 = thetasList[0]
    # theta2 = thetasList[1]
    # theta3 = thetasList[2]
    x1 = float(data[1])
    x2 = float(data[2])
    x3 = float(data[3])
    y = float(data[0])
    maxValue = float("-inf")
    for i in range(3):
        thetas = thetasList[i]
        thetaTransposeX = (float(thetas[0]) * x1) + (float(thetas[1]) * x2) + (float(thetas[2]) * x3)
        hypothesis = 1.0 / (1.0 + (math.exp(-thetaTransposeX)))
        if hypothesis > maxValue:
            maxValue = hypothesis
            classPredicted = i
    if classPredicted == 0:
        return [y, 0.3]
    elif classPredicted == 1:
        return [y, 0.0]
    elif classPredicted == 2:
        return [y, -0.3]


if __name__ == "__main__":

    '''Initialize SparkContext and read the testing Data from .csv file'''
    sc = SparkContext()
    sqlContext = SQLContext(sc)
    inputFIle = sys.argv[1]
    outputFile = sys.argv[2]
    '''INPUT - file:///home/cloudera/Downloads/Testing_Dataset_taxi.csv'''
    '''OUTPUT - file: // / home / cloudera / Downloads / logisticOutput'''
    df = sqlContext.read.load('file:///home/cloudera/Downloads/Testing_Dataset_taxi.csv', format='com.databricks.spark.csv', header='true', inferSchema='true')
    df = df.filter(df.lpep_pickup_datetime.isNotNull())

    '''Data Pre-processing'''
    '''Split the column "lpep_pickup_datetime " into two columns: "PickupTime" and "PickupWeekday" and add to the dataframe'''
    split_col = split(df['lpep_pickup_datetime'], ' ')
    df = df.withColumn('PickupTime', split_col.getItem(1)).withColumn('PickupWeekday', split_col.getItem(0))
    sp = udf(lambda x: datetime.strptime(x, '%M/%d/%Y').strftime('%A'))
    Fdf = df.withColumn('Weekday', sp(col('PickupWeekday')))

    '''Convert the time format to hours format eg. 12:30 to 12.5'''
    timeUDF = udf(timeinhours)
    processedDataSet = Fdf.withColumn("TimeInHours", timeUDF(col('PickupTime')))

    '''Count the total number of rows in the dataset'''
    m = processedDataSet.count()
    m = float(m)

    '''create dataframe of required columns only (eliminating all unused columns from the dataset)'''
    requiredDataColumns = processedDataSet.select('improvement_surcharge', 'Pickup_longitude', 'Pickup_latitude', 'TimeInHours')
    requiredDataColumns.cache

    '''Theta from the Training Model'''
    thetaforClass1 = [-7.73013851557, -13.9339527707, 0.0232455380356]
    thetaforClass2 = [8.27277654675, 14.9158760986, -0.0249594323763]
    thetaforClass3 = [1.59030657926, 2.74938187234, -0.0633678342143]
    thetaList = [thetaforClass1, thetaforClass2, thetaforClass3]
    prediction = requiredDataColumns.rdd.map(lambda x: predictClass(x, thetaList))
    prediction.saveAsTextFile(outputFile)
    accuracy = prediction.filter(lambda (v, p): v == p).count() / m * 100.0
    print accuracy
    '''97.4554573328'''
