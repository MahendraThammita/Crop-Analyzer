from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pickle

def index(request):
    return HttpResponse("Hello, world.")

@csrf_exempt
def predict(request):
    testvar = "mahendra"
    if (request.method == 'POST'):
        json_data = json.loads(request.body)
        family = json_data['family']
        temperature = json_data['temperature']
        ph = json_data['ph']
        zone = json_data['zone']
        season = json_data['season']

    inputChars = np.array([family, temperature, ph, zone, season])
    inputMatrix = inputChars.reshape(1, -1)
    stringVal = str(predictByModel(inputMatrix)[0])
    #trainModel(request)
    testOut = 'Sample'

    return HttpResponse("Hello, world. You're inside predict. output --> " + stringVal)

def trainModel(request):

    # Data preprocessing - loading initial dataset and treat null values
    data = pd.read_excel('E:/Ekanayaka/FarmCare/Crop-Analyzer/farmCare/cropAnalyzer/dataset.xlsx')
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

    pickle.dump(kmeans, open("E:/Ekanayaka/FarmCare/Crop-Analyzer/farmCare/cropAnalyzer/model.pkl", "wb"))


def predictByModel(npArray):
    model = pickle.load(open("E:/Ekanayaka/FarmCare/Crop-Analyzer/farmCare/cropAnalyzer/model.pkl", "rb"))
    predicton = model.predict(npArray)
    return predicton