from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import sys
import pymongo
import environ

# Initialise environment variables
env = environ.Env()
environ.Env.read_env()
MONGODB_URI = env('MONGODB_URI')
USER_COLLECTION_NAME = env('USER_COLLECTION_NAME')
DB_NAME = env('DB_NAME')

client = pymongo.MongoClient(MONGODB_URI)
#Define Db Name
dbname = client[DB_NAME]
#Define Collection
usersCollection = dbname[USER_COLLECTION_NAME]


def userLogin(userData):
    user = usersCollection.find_one({"userName":userData['data']['userName']})
    if(user != None):
        if(user['password'] == userData['data']['password']):
            jsonResult = json.dumps(user, default=str)
            return HttpResponse(jsonResult)
        else:
            return HttpResponse("Invalid Password")
    else:
        return HttpResponse("Invalid User")

