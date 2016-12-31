from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.sql.functions import split,udf,col,monotonically_increasing_id
import  os,math,re,array
from datetime import  datetime
import random

def timeinhours(t):
    tArray = re.split(":", t)
    minutes = float(tArray[1])
    hour = float(tArray[0])
    if minutes != 0.0:
        tim = minutes / 60.0 + hour
    else:
        tim = hour
    return tim

#Creating 12 time slots in a day. 2 per hour
def timeslot(f):
    time24 = datetime.strptime(f,'%H:%M:%S').strftime('%X')
    tim=int(time24[0:2])/2
    timeZone=tim+1
    return timeZone

def Kmeans(x):
    closest = float("inf")
    j=1
    la=x[1]
    lo=x[0]
    for i in centroidlist:
        dist=math.sqrt(math.pow((la-i[0]),2)+math.pow((lo-i[1]),2))
        if dist<closest:
            closest=dist
            cluster=j
        j+=1
    return [cluster, [la,lo,x[4]]]

def computecentroid(line):
    x=line[0]
    y=list(line[1])
    latsum=0.0
    longsum=0.0
    count=0.0
    for i in y:
        count+=1
        latsum=latsum+i[0]
        longsum=longsum+i[1]
    newx=latsum/count
    newy=longsum/count
    return [x,[newx,newy]]

def compareClasses(line):
    zone = line[0]
    classPred = line[1]
    for i in classPred:
        if zone == i:
            return 1
        else:
            return 0


if __name__ == "__main__":

    sc=SparkContext()
    sqlContext=SQLContext(sc)
    df1= sqlContext.read.load('file:///home/cloudera/Downloads/green_tripdata_2016-06.csv', format='com.databricks.spark.csv', header='true', inferSchema='true')
    #dftest=sqlContext.read.load('file:///home/cloudera/Downloads/Book1test.csv', format='com.databricks.spark.csv', header='true', inferSchema='true')
    df=df1.limit(1120000)
    dftest=df1.subtract(df).limit(28000)
    df = df.filter(df.lpep_pickup_datetime.isNotNull())
    dftest = dftest.filter(dftest.lpep_pickup_datetime.isNotNull())
    #alpha=2.0
    '''------------------------------------------Training Data PreProcessing--------------------------------------------------------------------------------'''

    '''Splitting the column "lpep_pickup_datetime " into two columns "PickupTime" and "PickupWeekday"'''
    split_col = split(df['lpep_pickup_datetime'], ' ')
    '''Adding those two columns to the dataframe'''
    df = df.withColumn('PickupTime', split_col.getItem(1)).withColumn('PickupWeekday', split_col.getItem(0))
    sp=udf(lambda x:datetime.strptime(x,'%Y-%M-%d').strftime('%A'))
    timeUDF = udf(timeinhours)
    timeZone = udf(timeslot)
    totaldata = df.withColumn('Weekday', sp(col('PickupWeekday'))).withColumn("TimeInHours", timeUDF(col('PickupTime'))).withColumn("id",monotonically_increasing_id()).withColumn("Time_Zone", timeZone(col("PickupTime")))
    kmeansDF=totaldata.select('Pickup_longitude','Pickup_latitude','TimeInHours','Weekday','id', 'Time_Zone')
    kmeansDF.registerTempTable('bayesdata')
    '''..............................................................Testing data preprocessing..........................................................'''

    split_coltest = split(dftest['lpep_pickup_datetime'], ' ')
    dftest = dftest.withColumn('PickupTime', split_coltest.getItem(1)).withColumn('PickupWeekday',split_coltest.getItem(0))
    totaldatatest = dftest.withColumn('Weekday', sp(col('PickupWeekday'))).withColumn("TimeInHours", timeUDF(col('PickupTime'))).withColumn("id", monotonically_increasing_id()).withColumn("Time_Zone",timeZone(col("PickupTime")))
    kmeansDFtest = totaldatatest.select('Pickup_longitude', 'Pickup_latitude', 'TimeInHours', 'Weekday', 'id','Time_Zone')
    kmeansDFtest.registerTempTable('bayesdatatest')

    '''-------------------------------------kmeans on training data---------------------------------------------'''
    ycentroid=sqlContext.sql("SELECT Pickup_longitude from bayesdata").collect()
    xcentroid=sqlContext.sql("SELECT Pickup_latitude from bayesdata").collect()
    # xcen=random.sample(xcentroid,4)
    # ycen=random.sample(ycentroid,4)
    xcen=xcentroid[:80]
    ycen=ycentroid[:80]
    centroids=zip(xcen,ycen)
    centroidlist=[]
    for c in centroids:
        temp=[]
        xx=c[0].Pickup_latitude
        yy=c[1].Pickup_longitude
        temp=[xx,yy]
        centroidlist.append(temp)

    for itr in range(30):
        counter=0
        ff=kmeansDF.rdd.map(lambda x:Kmeans(x))
        new=ff.groupByKey().map(computecentroid)
        dd=new.collect()
        for itt in dd:
            index=itt[0]
            val=itt[1]
            centroidlist[index-1][0]=val[0]
            centroidlist[index-1][1]=val[1]

    classrow=ff.map(lambda x: (Row(Zone=x[0],ids=x[1][2])))
    finaldf=sqlContext.createDataFrame(classrow)
    naivedf=kmeansDF.join(finaldf,(kmeansDF.id==finaldf.ids))

    '''.................................................Kmeans on testing data...............................................'''

    ycentroidtest = sqlContext.sql("SELECT Pickup_longitude from bayesdatatest").collect()
    xcentroidtest= sqlContext.sql("SELECT Pickup_latitude from bayesdatatest").collect()
    # xcen=random.sample(xcentroid,4)
    # ycen=random.sample(ycentroid,4)
    xcentest = xcentroidtest[:80]
    ycentest = ycentroidtest[:80]
    centroidstest = zip(xcentest, ycentest)
    centroidlisttest = []
    for c in centroidstest:
        temptest = []
        xxtest = c[0].Pickup_latitude
        yytest= c[1].Pickup_longitude
        temptest = [xxtest, yytest]
        centroidlisttest.append(temptest)
    #print " centroids are ", centroidlisttest
    for itr in range(30):
        countertest = 0
        fftest = kmeansDFtest.rdd.map(lambda xx: Kmeans(xx))
       # print "fftest ", fftest.collect()
        newtest = fftest.groupByKey().map(computecentroid)
        ddtest = newtest.collect()
       # print ddtest
        for itt in ddtest:
            indextest = itt[0]
            valtest = itt[1]
            centroidlisttest[indextest - 1][0] = valtest[0]
            centroidlisttest[indextest- 1][1] = valtest[1]

    classrowtest = fftest.map(lambda x: (Row(Zone=x[0], ids=x[1][2])))
    finaldftest = sqlContext.createDataFrame(classrowtest)
    naivedftest = kmeansDFtest.join(finaldftest, (kmeansDFtest.id == finaldftest.ids))
    '''------------------------------------------------------Naive Bayes Distributed--------------------------------------------'''
    classification = []
    classfinal = {}

    naivedf.registerTempTable('bayesDataTable')
    naivedftest.registerTempTable('bayesDataTabletest')
    sqlContext.cacheTable("bayesDataTable")
    sqlContext.cacheTable("bayesDataTabletest")
    timevocab=sqlContext.sql("SELECT count(distinct(Time_Zone)) from bayesDataTable").collect()[0][0]
    weekdayvocab=sqlContext.sql("SELECT count(distinct(Weekday)) from bayesDataTable").collect()[0][0]
    #zoneList = sqlContext.sql("SELECT distinct(Zone) from bayesDataTable").collect() #wrong.......!
    min=0.0
    timelist=sqlContext.sql("SELECT Time_Zone FROM bayesDataTabletest").collect()
    weeklist=sqlContext.sql("SELECT Weekday FROM bayesDataTabletest").collect()
    count=0
    idlist=[]
    aplha=[0.001,0.01,0.2,5.0,10.0,20.0,30.0,35.0]
    accuracies=[]
for al in aplha:
    for i in range(len(timelist)):
        for zone in range(80):
            zoneValue = zone+1
            zon=sqlContext.sql("SELECT Zone FROM bayesDataTable WHERE Zone='"+str(zoneValue)+"'").collect()
            priorprobZone=float(len(zon))/len(timelist)
            timely=sqlContext.sql("SELECT Time_Zone FROM bayesDataTable WHERE Zone='"+str(zoneValue)+"' AND Time_Zone='"+timelist[i].Time_Zone+"'").collect()
            timelikelyhood=(float(len(timely))+al)/float(len(zon)+al*timevocab)
            wkdy = sqlContext.sql("SELECT Weekday FROM bayesDataTable WHERE Zone='" + str(zoneValue) + "' AND Weekday='" + weeklist[i].Weekday + "'").collect()
            weekdaylikelyhood=(float(len(wkdy))+al)/float(len(zon)+al*weekdayvocab)
            total=priorprobZone*timelikelyhood*weekdaylikelyhood
            classfinal[zoneValue]=total
        keys=max(classfinal.values())
        temp=[x for x,y in classfinal.items() if y==keys]
        classification.append(temp)
        count+=1
        idlist.append(count)
        classid=zip(idlist,classification)
    # def PredictNaiveBayes(timezone,weekday):
    #     count = 0
    #     idlist = []
    #     for zone in range(3):
    #         zoneValue = zone+1
    #         zon = sqlContext.sql("SELECT Zone FROM bayesDataTable WHERE Zone='" + str(zoneValue) + "'").collect()
    #         priorprobZone = float(len(zon)) / len(timelist)
    #         timely = sqlContext.sql(
    #             "SELECT Time_Zone FROM bayesDataTable WHERE Zone='" + str(zoneValue) + "' AND Time_Zone='" + timezone + "'").collect()
    #         timelikelyhood = (float(len(timely)) + alpha) / float(len(zon) + alpha * timevocab)
    #         wkdy = sqlContext.sql(
    #             "SELECT Weekday FROM bayesDataTable WHERE Zone='" + str(zoneValue) + "' AND Weekday='" + weekday+ "'").collect()
    #         weekdaylikelyhood = (float(len(wkdy)) + alpha) / float(len(zon) + alpha * weekdayvocab)
    #         total = priorprobZone * timelikelyhood * weekdaylikelyhood
    #         classfinal[zoneValue] = total
    #     keys = max(classfinal.values())
    #     temp = [x for x, y in classfinal.items() if y == keys]
    #     classification.append(temp)
    #     count += 1
    #     idlist.append(count)
    #     return [idlist, classification]

    #classid=naivedftest.rdd.map(lambda x:PredictNaiveBayes(x[5],x[3]))
    predictedclass=sc.parallelize(classid).map(lambda x: (Row(rowid=x[0],predictedclas=x[1])))
    final = sqlContext.createDataFrame(predictedclass)
    naivedfFinal=naivedf.join(final, (naivedf.id == final.rowid))
    # naivedfFinal.show()

    '''Compute accuracy of Naive Bayes'''
    countTruth = naivedfFinal.select('Zone', 'predictedclas').rdd.map(compareClasses).filter(lambda x: x ==1).count()
    accuracyNaiveBayes = float(countTruth)/float(naivedfFinal.rdd.count()) * 100
    accuracies.append(accuracyNaiveBayes)
print accuracies



#-----------------------------------------------Naive bayes--------------------------------------------------------------------
#list of zones(Areas)


#Naive Bayes algorithm implementation

# def NaiveBayes(weeday,Pickuptime):
#     Bussiestprob=0
#     Bussiestarea=""
#     for area in areas:
#         timevocab=nbdf.select('TimeInHours').distinct().count()
#         weedayvocab=nbdf.select('Weekday').distinct().count()
#         countlikely=nbdf.filter(nbdf[4]==area)
#         arealikelyhood=float(countlikely.count())/nbdf.count()
#         probWeekday=float(countlikely.filter(countlikely[3]==str(weeday)).count()+alpha)/(float(countlikely.count())+alpha*timevocab)
#         Probtime=float(countlikely.filter(countlikely[2]==Pickuptime).count()+alpha)/(float(countlikely.count())+alpha*weedayvocab)
#         probarea=math.log(arealikelyhood)
#         probarea=probarea+math.log(probWeekday)+math.log(Probtime)
#        # print "prob area", probarea
#         posteriorprob=math.exp(probarea)
#        # print "prob" , posteriorprob , "area ", area
#         if(posteriorprob>=Bussiestprob):
#            Bussiestprob=posteriorprob
#            Bussiestarea=area
#     return Bussiestarea


# #------------------------------------------------------logistic regression.................cost function---------------------
#
# print coun
#
# logistic=Fdf.select('improvement_surcharge','Pickup_longitude','Pickup_latitude','TimeInHours')
# logistic.show()
# print logistic.rdd.zipWithUniqueId().collect()
#
# logistic.collect()
# def cost(x,cls,theeta):
#     ttx=theeta[0]*float(x[1])+theeta[1]*float(x[2])+theeta[2]*float(x[3])
#     hxi=1/(1+math.exp(-ttx))
#     if cls==float(x[0]):
#         yi=1
#     else:
#         yi=0
#     sum=yi*math.log(hxi)+(1-yi)*math.log(1-hxi)
#     return  sum
# cls=0.3
#
# #gradient descent using spark ml lib for logistic regression
# def parsePoint(line):
#     values = [float(x) for x in line]
#     if values[0]==0.3:
#         values[0]=1
#     else:
#         values[0]=0
#     return LabeledPoint(values[0], values[1:])
#
# #data = sc.textFile("data/mllib/sample_svm_data.txt")
# parsedData = logistic.rdd.map(parsePoint)
# parsedData.collect()
#
# # Build the model
# model = LogisticRegressionWithLBFGS.train(parsedData)
# print model.weights
# mo=model.weights
#
# jtheeta=logistic.rdd.map(lambda  x: cost(x,cls,mo)).sum()
# m=Fdf.count()
# cost=-jtheeta/m
# print cost

# Evaluating the model on training data
# labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
# trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
# print("Training Error = " + str(trainErr))

#------------------------------------------------K Means----------------------------------------------------------------------------------------
# kmeandata=Fdf.select('Pickup_longitude','Pickup_latitude','TimeInHours','Weekday')
# kmeandata=kmeandata.withColumn("id",monotonically_increasing_id())
