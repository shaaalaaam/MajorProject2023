from fastapi import FastAPI
# from DjangoProject.trafficLight.traffic_light_app.traffic_light_app.views import VEHICLE_COUNT
from DjangoProject.trafficLight.traffic_light_app.traffic_light_app.constant import VEHICLE_COUNT

app = FastAPI()

@app.get('/get-count0')
def get_count0():
    # read the numeric value from the file
    with open('DjangoProject/trafficLight/traffic_light_app/my_file0.txt', 'r') as f:
        value = int(f.read())
        
    # Write your logic to fetch count
    # return {'count': value}
    return value


@app.get('/get-count1')
def get_count1():
    # read the numeric value from the file
    with open('DjangoProject/trafficLight/traffic_light_app/my_file1.txt', 'r') as f:
        value = int(f.read())
        
    # Write your logic to fetch count
    # return {'count': value}
    return value

@app.get('/get-count2')
def get_count2():
    # read the numeric value from the file
    with open('DjangoProject/trafficLight/traffic_light_app/my_file2.txt', 'r') as f:
        value = int(f.read())
        
    # Write your logic to fetch count
    # return {'count': value}
    return value

@app.get('/get-count3')
def get_count3():
    # read the numeric value from the file
    with open('DjangoProject/trafficLight/traffic_light_app/my_file3.txt', 'r') as f:
        value = int(f.read())
        
    # Write your logic to fetch count
    # return {'count': value}
    return value