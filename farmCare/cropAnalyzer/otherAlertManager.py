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
OTHER_ALERTS_COLLECTION_NAME = env('OTHER_ALERTS_COLLECTION_NAME')
DB_NAME = env('DB_NAME')

client = pymongo.MongoClient(MONGODB_URI)
#Define Db Name
dbname = client[DB_NAME]
#Define Collection
otherAlertsCollection = dbname[OTHER_ALERTS_COLLECTION_NAME]

def addOtherAlert(data):
    newAlert={
        "alertTitle": data['data']['alertTitle'],
        "alertDescription" : data['data']['alertDescription'],
        "ImageList" : data['data']['imageList'],
    }
    #Modifying Database
    otherAlertsCollection.insert_one(newAlert) 
    response = HttpResponse(status=200)
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response["Access-Control-Max-Age"] = "1000"
    response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
    return response

def getOtherAlertsList():
    activeAlerts = otherAlertsCollection.find()
    list_cur = list(activeAlerts)
    jsonResult = json.dumps(list_cur, default=str)
    response = HttpResponse(jsonResult)
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response["Access-Control-Max-Age"] = "1000"
    response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
    return response