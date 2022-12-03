from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pickle
import pymongo
import firebase_admin
from firebase_admin import credentials, storage
import environ

# Initialise environment variables
env = environ.Env()
environ.Env.read_env()
MONGODB_URI = env('MONGODB_URI')
CROPS_COLLECTION_NAME = env('CROPS_COLLECTION_NAME')
DB_NAME = env('DB_NAME')

client = pymongo.MongoClient(MONGODB_URI)
#Define Db Name
dbname = client[DB_NAME]
#Define Collection
cropsCollection = dbname[CROPS_COLLECTION_NAME]

def getCropsList():
    similarCrops = cropsCollection.find()
    list_cur = list(similarCrops)
    #list_of_crops = [each_crop for each_crop in similarCrops]
    jsonResult = json.dumps(list_cur, default=str)
    response = HttpResponse(jsonResult)
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response["Access-Control-Max-Age"] = "1000"
    response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
    return response