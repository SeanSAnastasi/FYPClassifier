import requests
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

response = requests.get("https://fyp-sensors.web.app/getdata")
data = json.loads(response.content.decode("utf-8"))

idKeys = [] 
hrmData = []
dateData = []


for x in data:
    
    exists = -1
    tempHrm = []
    tempDate = []
    if x["id"] in idKeys:
        
        exists = idKeys.index(x["id"])
        
    else:
        
        idKeys.append(x["id"])
   
    for y in x["data"]:
        date =datetime.strptime(y["date"], '%a %b %d %Y %H:%M:%S %Z%z')
        
        # secondsPassed = (date-initDate).total_seconds()
        if exists == -1:
            
            
            tempHrm.append(y["heartRate"])
            tempDate.append(date)


            
        else:
            
            hrmData[exists].append(y["heartRate"])
            dateData[exists].append(date)
    if len(tempHrm)>0 and len(tempDate)>0:
        hrmData.append(tempHrm)
        dateData.append(tempDate)
        tempDate = []
        tempHrm = []

for x in range(len(idKeys)):
    print(idKeys)
    np.savetxt('watch/S'+str(x)+'.csv', hrmData[x], fmt='%i', delimiter=',')
    # plt.scatter(dateData[x], hrmData[x])
    # plt.xlabel("Time (D H:M:S)")
    # plt.ylabel("Heart Rate (BPM)")
    # plt.title(idKeys[x])
    # plt.show()

