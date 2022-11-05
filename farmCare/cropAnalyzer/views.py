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



#Define Db Name
dbname = client['myFirstDatabase']
#Define Collection
cropsCollection = dbname['crops']

cred = credentials.Certificate("cropAnalyzer/credentials.json")
firebase_admin.initialize_app(cred,{'storageBucket': 'farmcare-8b04f.appspot.com'}) # connecting to firebase


def index(request):
    return HttpResponse("Hello, world.")

@csrf_exempt
def predict(request):
    if (request.method == 'POST'):
        json_data = json.loads(request.body)
        family = json_data['family']
        temperature = json_data['temperature']
        ph = json_data['ph']
        zone = json_data['zone']
        season = json_data['season']

    inputChars = np.array([family, temperature, ph, zone, season])
    inputMatrix = inputChars.reshape(1, -1)
    #trainModel()
    stringPred = str(predictByModel(inputMatrix)[0]) 
    similarCropsJson = getSimilarCropsByPrediction(int(stringPred))

    return HttpResponse(similarCropsJson)

def trainModel():
    # Data preprocessing - loading initial dataset and treat null values
    data = pd.read_excel('https://storage.googleapis.com/farmcare-8b04f.appspot.com/cropAnalyzer/dataset.xlsx')
    print(data.head())
    data.drop(['RainFall - mm'], axis=1, inplace=True)
    data = data.dropna()

    # Data preprocessing - Replasing values with numbers
    data["Season"].replace({"yala": 0, "yala, maha": 2, "yala,maha": 2 ,"maha": 1}, inplace=True)
    data["Zone"].replace({"dry": 0, "wet": 1, "intermediate": 2, "dry,wet": 3,"wet,dry":3, "dry, intermediate": 4, "dry,intermediate": 4,
                        "wet, dry": 3, "dry,wet,intermediate": 6, "dry,intermediate,wet": 6, "intermediate,wet":5,
                        "wet,intermediate":5, "wet,intermediate,dry":6,"wet, intermediate":5,}, inplace=True)
    labels = data['Family'].astype('category').cat.categories.tolist()
    data['Family'] = data['Family'].astype('category').cat.codes
    replaced_data = {'Family' : {k: v for k,v in zip(labels,list(range(0,len(labels))))}} 
    data.rename(columns = {'Temp - C':'temp_C', 'PH val':'pH_val'}, inplace = True)
    labels = data['temp_C'].astype('category').cat.categories.tolist()
    data['temp_C'] = data['temp_C'].astype('category').cat.codes
    replaced_data = {'Temp_C' : {k: v for k,v in zip(labels,list(range(0,len(labels))))}}

    labels = data['pH_val'].astype('category').cat.categories.tolist()
    data['pH_val'] = data['pH_val'].astype('category').cat.codes
    replaced_data = {'pH_val' : {k: v for k,v in zip(labels,list(range(0,len(labels))))}}

    data.drop(['Crop'], axis=1, inplace=True)

    # Fit data into K-Means model
    X=data
    scaler = MinMaxScaler()
    scaler.fit(X)
    X=scaler.transform(X)
    inertia = []
    kmeans = KMeans(
        n_clusters=6, init="k-means++",
        n_init=10,
        tol=1e-04, random_state=42
    )
    kmeans.fit(X)
    clusters=pd.DataFrame(X,columns=data.columns)
    clusters['label']=kmeans.labels_
    polar=clusters.groupby("label").mean().reset_index()
    polar=pd.melt(polar,id_vars=["label"])

    cols = data.columns
    ms = MinMaxScaler()
    X = ms.fit_transform(data)
    X = pd.DataFrame(X, columns=[cols])

    pickle.dump(kmeans, open("cropAnalyzer/model.pkl", "wb"))



def predictByModel(npArray):
    model = pickle.load(open("cropAnalyzer/model.pkl", "rb"))
    predicton = model.predict(npArray)
    return predicton


def getSimilarCrops(request):
    if (request.method == 'GET'):
        json_data = json.loads(request.body)
        cropName = json_data['CropName']
    
    if(cropName is not None and cropName != ""):
        affevtedCrop = cropsCollection.find_one({"Crop":cropName})
        if(affevtedCrop is not None):
            affevtedCropCategory = affevtedCrop['Predicted_category']
            similarCrops = cropsCollection.find({"Predicted_category":affevtedCropCategory})
            similarCropNameList = []
            for crop in similarCrops:
                similarCropNameList.append(crop['Crop'])
            jsonResult = json.dumps(similarCropNameList)
            return HttpResponse(jsonResult , status=200)
        else:
            return HttpResponse(status=404)
    else:
        return HttpResponse(status=204)

@csrf_exempt
def addnewCrop(request):
    if (request.method == 'POST'):
        json_data = json.loads(request.body)
        family = json_data['family']
        temperature = json_data['temperature']
        ph = json_data['ph']
        zone = json_data['zone']
        season = json_data['season']
        familyText = json_data['familyText']
        temperatureText = json_data['temperatureText']
        phText = json_data['phText']
        zoneText = json_data['zoneText']
        seasonText = json_data['seasonText']
        cropName = json_data['cropName']
    
    newCrop={
        "Crop": cropName,
        "Family" : familyText,
        "Temp - C" : temperatureText,
        "PH val" : phText,
        "Zone" : zoneText,
        "Season" : seasonText,
    }
    #Modifying Excel
    originalDf = pd.read_excel('https://storage.googleapis.com/farmcare-8b04f.appspot.com/cropAnalyzer/dataset.xlsx')
    originalDf = originalDf.append(newCrop, ignore_index=True)
    originalDf.to_excel("cropAnalyzer/dataset.xlsx")
    #Upload file to firebase
    file_path = "cropAnalyzer/dataset.xlsx"
    bucket = storage.bucket() # storage bucket
    blob = bucket.blob(file_path)
    blob.upload_from_filename(file_path)
    blob.make_public()
    print(blob.public_url)

    trainModel()

    inputChars = np.array([family, temperature, ph, zone, season])
    inputMatrix = inputChars.reshape(1, -1)
    stringPred = str(predictByModel(inputMatrix)[0])

    newCrop={
        "Crop": cropName,
        "Family" : familyText,
        "Temp - C" : temperatureText,
        "PH val" : phText,
        "Zone" : zoneText,
        "Season" : seasonText,
        "Predicted_category" : int(stringPred),
    }
    #Modifying Database
    cropsCollection.insert_one(newCrop) 
    
    similarCropsJson = getSimilarCropsByPrediction(int(stringPred))
    return HttpResponse(similarCropsJson)
    
    


def getSimilarCropsByPrediction(predictedCategory):
    similarCrops = cropsCollection.find({"Predicted_category":predictedCategory})
    similarCropNameList = []
    for crop in similarCrops:
        similarCropNameList.append(crop['Crop'])
        jsonResult = json.dumps(similarCropNameList)
    return jsonResult


def getSimilarCropsByName(CropName):
    if (request.method == 'GET'):
        json_data = json.loads(request.body)
        cropName = json_data['CropName']
    
    if(cropName is not None and cropName != ""):
        affevtedCrop = cropsCollection.find_one({"Crop":cropName})
        if(affevtedCrop is not None):
            affevtedCropCategory = affevtedCrop['Predicted_category']
            similarCrops = cropsCollection.find({"Predicted_category":affevtedCropCategory})
            similarCropNameList = []
            for crop in similarCrops:
                similarCropNameList.append(crop['Crop'])
            jsonResult = json.dumps(similarCropNameList)
            return HttpResponse(jsonResult , status=200)
        else:
            return HttpResponse(status=404)
    else:
        return HttpResponse(status=204)
    
