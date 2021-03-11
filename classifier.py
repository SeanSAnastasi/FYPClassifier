import requests
import json
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


            hrmData.append(tempHrm)
            dateData.append(tempDate)
        else:
            
            hrmData[exists].append(y["heartRate"])
            dateData[exists].append(date)

for x in range(len(idKeys)):
    plt.scatter(dateData[x], hrmData[x])
    plt.xlabel("Time (D H:M:S)")
    plt.ylabel("Heart Rate (BPM)")
    plt.title(idKeys[x])
    plt.show()

