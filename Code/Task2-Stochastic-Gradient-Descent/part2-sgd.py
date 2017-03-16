from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import *
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.linalg import Vectors
	

# Get the SparkContext
conf = SparkConf().setMaster("local[*]").setAppName("CTR Predictor")
sc= SparkContext(conf=conf)
#sc=SparkContext(appname="CTR Predictor")

# Load the train.csv
train="hdfs:///user/sdeshpa2/input/train.csv"
train_data=sc.textFile(train)

# skip header
header=train_data.first()
train_data=train_data.filter(lambda x: x!=header)

# get the SparkSQL context
sqlContext = SQLContext(sc)

# Split the data and create RDD with rows as instances of all columns, excluding attributes "id" and "hour"
ctr_split=train_data.map(lambda l: l.split(",", -1))
ctr_tuples=ctr_split.map(lambda c: Row(click=(c[1]), C1=(c[3]), banner_pc=(c[4]), site_id=(c[5]), site_domain=(c[6]), site_category=(c[7]), 
	app_id=(c[8]), app_domain=(c[9]), app_category=(c[10]), device_id=(c[11]), device_ip=(c[12]), device_model=(c[13]), device_type=(c[14]),
	device_conn_type=(c[15]), C14=(c[16]), C15=(c[17]), C16=(c[18]), C17=(c[19]), C18=(c[20]), C19=(c[21]), C20=(c[22]), C21=(c[23]) ))

#Count distinct values from each categorical column
categories={}
idxCategories= [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

#for i in idxCategories: ##idxCategories contains indexes of rows that contains categorical data
#    distinctValues = ctr_tuples.map(lambda x : x[i]).distinct().count()
#    categories[i] = distinctValues

#print categories

#Removing columns with more than 100  distinct values
# Create RDD
ctrSplit=train_data.map(lambda l: l.split(",", -1))
ctrTuples=ctrSplit.map(lambda c: Row(click=(c[1]), C1=(c[3]), banner_pc=(c[4]), site_category=(c[7]), app_category=(c[10]), device_type=(c[14]), 
	device_conn_type=(c[15]), C15=(c[17]), C16=(c[18]), C18=(c[20]), C19=(c[21]), C21=(c[23]) ))

#Create DataFrame from the RDD
schemaClick=sqlContext.createDataFrame(ctrTuples)

#Register the data frame as table
schemaClick.registerTempTable("CTRPrediction")
schemaClick.printSchema()

#Cast "click" column from String to double
schemaClick.withColumn("clickTmp", schemaClick.click.cast('double')).drop("click").withColumnRenamed("clickTmp", "click")

#Replace null values in column C1 with NA
schemaClick=schemaClick.na.replace('-1', 'NA', 'C1')

#Drop rows containing null values for numerical variables
schemaClick=schemaClick.dropna()

schemaClick.printSchema()
schemaClick.show(2)

# Use StringIndexer and OneHotEncoder to convert categorical variables to numerical
# Use StringIndexer
C1Indexer = StringIndexer(inputCol="C1", outputCol="indexedC1", handleInvalid="skip")
#C1Indexer_features=C1Indexer.fit(schemaClick).transform(schemaClick)
#C1Indexer_features.show()

# Use One-hot Encoder
C1Encoder = OneHotEncoder(dropLast=False, inputCol="indexedC1", outputCol="VectorC1")
#C1Encoder_features = C1Encoder.transform(C1Indexer_features)
#C1Encoder_features.select("C1", "VectorC1").show()

#For banner_pc
BannerPcIndexer = StringIndexer(inputCol="banner_pc", outputCol="indexed_banner_pc", handleInvalid="skip")
BannerPcEncoder = OneHotEncoder(dropLast=False, inputCol="indexed_banner_pc", outputCol="Vector_banner_pc")

#For site_category
SiteCategoryIndexer = StringIndexer(inputCol="site_category", outputCol="indexed_site_category", handleInvalid="skip")
SiteCategoryEncoder = OneHotEncoder(dropLast=False, inputCol="indexed_site_category", outputCol="Vector_site_category")

#For app_category
AppCategoryIndexer = StringIndexer(inputCol="app_category", outputCol="indexed_app_category", handleInvalid="skip")
AppCategoryEncoder = OneHotEncoder(dropLast=False, inputCol="indexed_app_category", outputCol="Vector_app_category")

#For device_type
DeviceTypeIndexer = StringIndexer(inputCol="device_type", outputCol="indexed_device_type", handleInvalid="skip")
DeviceTypeEncoder = OneHotEncoder(dropLast=False, inputCol="indexed_device_type", outputCol="Vector_device_type")

#For device_conn_type
DeviceConnTypeIndexer = StringIndexer(inputCol="device_conn_type", outputCol="indexed_device_conn_type", handleInvalid="skip")
DeviceConnTypeEncoder = OneHotEncoder(dropLast=False, inputCol="indexed_device_conn_type", outputCol="Vector_device_conn_type")

#For C15
C15Indexer = StringIndexer(inputCol="C15", outputCol="indexedC15", handleInvalid="skip")
C15Encoder = OneHotEncoder(dropLast=False, inputCol="indexedC15", outputCol="VectorC15")

#For C16
C16Indexer = StringIndexer(inputCol="C16", outputCol="indexedC16", handleInvalid="skip")
C16Encoder = OneHotEncoder(dropLast=False, inputCol="indexedC16", outputCol="VectorC16")

#For C18
C18Indexer = StringIndexer(inputCol="C18", outputCol="indexedC18", handleInvalid="skip")
C18Encoder = OneHotEncoder(dropLast=False, inputCol="indexedC18", outputCol="VectorC18")

#For C19
C19Indexer = StringIndexer(inputCol="C19", outputCol="indexedC19", handleInvalid="skip")
C19Encoder = OneHotEncoder(dropLast=False, inputCol="indexedC19", outputCol="VectorC19")

#For C21
C21Indexer = StringIndexer(inputCol="C21", outputCol="indexedC21", handleInvalid="skip")
C21Encoder = OneHotEncoder(dropLast=False, inputCol="indexedC21", outputCol="VectorC21")

#Vector Assembler
FeatureAssembler = VectorAssembler(inputCols=["VectorC1", "Vector_banner_pc", "Vector_site_category", "Vector_app_category", "Vector_device_type", 
	"Vector_device_conn_type", "VectorC15", "VectorC16", "VectorC18", "VectorC19", "VectorC21"], outputCol="VectoredFeatures")

# Using pipeline
pipelineTmp = Pipeline(stages=[C1Indexer, BannerPcIndexer, SiteCategoryIndexer, AppCategoryIndexer, DeviceTypeIndexer, DeviceConnTypeIndexer, C15Indexer, C16Indexer, C18Indexer, C19Indexer, C21Indexer, 
	C1Encoder, BannerPcEncoder, SiteCategoryEncoder, AppCategoryEncoder, DeviceTypeEncoder, DeviceConnTypeEncoder, C15Encoder, C16Encoder, C18Encoder, C19Encoder, C21Encoder, FeatureAssembler])
modelTmp = pipelineTmp.fit(schemaClick)
tmp = modelTmp.transform(schemaClick).select("click", "VectoredFeatures")
tmp.registerTempTable("CLICK")

# Selecting click and VectoredFeatures from Table "CLICK" and creating new dataFrame as results
results=sqlContext.sql("SELECT click, VectoredFeatures from CLICK")
results.show()

# Creating label points for attributes click and VectoredFeatures
click_transformed=results.select('click', 'VectoredFeatures').rdd.map(lambda row: LabeledPoint(float(row.click), Vectors.dense((row.VectoredFeatures).toArray())))
click_transformed.take(2)

#Divide the data into training and test sets
weights = [.6, .4]
seed = 15L

ClickTrain, ClickTest = click_transformed.randomSplit(weights, seed)

# Train the training data set for the Stochastic Grdient Decent
modelSGD = LogisticRegressionWithSGD.train(ClickTrain, iterations=15, step=0.01, miniBatchFraction=0.01, regType=None, validateData="False")

# Perform Testing on the Test Dataset and do prediction on the label
clickAndFeatures = ClickTest.map(lambda p: (float(modelSGD.predict(p.features)), p.label))

#Calculating TrueNegative(tn), FalsePositive(fp)

tn=clickAndFeatures.filter(lambda (v,p): v==0.0 and p==0.0)
fp=clickAndFeatures.filter(lambda (v,p): v==1.0 and p==0.0)

#Area Under Curve
metrics=BinaryClassificationMetrics(clickAndFeatures)
print("Area under ROC = %s" % metrics.areaUnderROC)

#Accuracy
accuracy=clickAndFeatures.filter(lambda (v, p): v == p).count() / float(ClickTest.count())
print("Accuracy = %s" % accuracy)

#False Positive rate
fpr=float(fp.count())/(float(fp.count())+float(tn.count()))
print("False Positive Rate = %s" % fpr)
