import requests
url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data?datasetid=GHCNDMS&location=ZIP:28801&startdate=2000-01-01&enddata=2010-01-01"
headers = { "token" : "AeGIKqAEEmqExNuRYcEwuhJFJsuyacbF" }
response = requests.get( url, headers = headers )
data = response.json()
