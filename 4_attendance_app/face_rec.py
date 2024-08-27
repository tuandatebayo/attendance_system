import numpy as np
import pandas as pd
import cv2
import time
from datetime import datetime
import redis
import os
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

#connect to redis
hostname = 'redis-14584.c1.asia-northeast1-1.gce.redns.redis-cloud.com'
port = 14584
password = 'dcZEr1OivCI6rRb27Eqw8mdCC9Lez0wJ'

r = redis.Redis(host=hostname, port=port, password=password)

#retrieve data from database
def retrieve_data(name):
    retrive_dict = r.hgetall(name)
    retrive_series = pd.Series(retrive_dict).apply(lambda x: np.frombuffer(x, dtype=np.float32))
    index = retrive_series.index
    index = list(map(lambda x: x.decode(), index))
    retrive_series.index = index
    retrive_df = retrive_series.to_frame().reset_index()
    retrive_df.columns = ['Name_Role', 'Facial_Features']
    retrive_df[['Name', 'Role']] = retrive_df['Name_Role'].apply(lambda x: x.split('@')).apply(pd.Series)
    return retrive_df[['Name', 'Role', 'Facial_Features']]

#configure face analysis
faceapp = FaceAnalysis(name='buffalo_sc')
faceapp.prepare(ctx_id=0, det_thresh=0.5, det_size=(640,640))

#ML search algorithm
def ml_search_algorithm(dataframe, feature_column, test_vector, name_role=['Name','Role' ],thresh=0.5):
    '''
    cosine similarity base search algorithm
    '''
    # step-1: take the dataframe (collection of data)
    dataframe = dataframe.copy()
    # step-2: Index face embeding from the dataframe and convert into array
    X_list = dataframe[feature_column].tolist()
    X = np.asarray(X_list)
    
    # step-3: Cal. cosine similarity
    similar = pairwise.cosine_similarity(X, test_vector.reshape(1,-1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine' ] = similar_arr
    
    # step-4: filter the data
    data_filter = dataframe.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:
        # step-5: get the person name
        data_filter.reset_index(drop=True,inplace=True)
        argmax = data_filter['cosine' ].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]
    
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'
    
    return person_name, person_role

#Real time prediction
#save log every 1 minute
class RealTimePrediction:
    def __init__(self):
        self.logs = dict(name = [], role = [], time = [])
        
    def reset_dict(self):
        self.logs = dict(name = [], role = [], time = [])
    
    def save_log_redis(self):
        #create log dataframe
        dataframe = pd.DataFrame(self.logs)
        #drop duplicates
        dataframe.drop_duplicates('name',inplace=True)
        #save to redis (list)
        #encode the data
        name_list = dataframe['name'].tolist()
        role_list = dataframe['role'].tolist()
        time_list = dataframe['time'].tolist()
        encode_data = []
        for name, role, time in zip(name_list, role_list, time_list):
            if name != 'Unknown':
                concat_str = f'{name}@{role}@{time}'
                encode_data.append(concat_str)
        if len(encode_data) > 0:
            r.lpush('attendance:logs', *encode_data)
        self.reset_dict()
            
    def face_prediction(self, test_image, dataframe, feature_column, test_vector, name_role=['Name','Role' ],thresh=0.5):
        #find time
        current_time = str(datetime.now())
        # step-1: take the test image and apply to insight face
        results = faceapp.get(test_image)
        test_copy = test_image.copy()
        # step-2: use for loop and extract each embedding and pass to ml_search_algorithm

        for res in results:
            x1, y1,x2,y2=res['bbox'].astype(int)
            embeddings = res['embedding' ]
            person_name, person_role = ml_search_algorithm(dataframe,
                                                            feature_column,
                                                            test_vector=embeddings,
                                                            name_role=name_role,
                                                            thresh=thresh)
                                                            
            if person_name == 'Unknown':
                color =(0,0,255) # bgr
            else:
                color = (0,255,0)
            
            cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)
            
            text_gen = person_name
            cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
            cv2.putText(test_copy,current_time,(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
            #save log
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['time'].append(current_time)
            
        return test_copy
    
#registation form
class RegistrationForm:
    def __init__(self):
        self.sample = 0
    def reset(self):
        self.sample = 0
        
    def get_embedding(self, frame):
        results = faceapp.get(frame,max_num=1)
        embeddings = None
        for res in results:
            self.sample += 1
            x1, y1, x2,y2=res['bbox'].astype(int)
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),1)
            #put text sample info
            text = f"Sample: {self.sample}" 
            cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,(0,255,0),2)
            # facial features
            embeddings = res['embedding']
        
        return frame, embeddings
    
    def save_data_redis(self, name, role):
        #validate sample
        if name is None:
            if name.strip() != '':
                key = f'{name}@{role}'
            else:
                return 'Name cannot be empty'
        else:
            return 'Name cannot be empty'
        #if face_embedding.txt exist
        if 'face_embedding.txt' not in os.listdir():
            return 'No data to save'
        
        #load data from txt
        X_array = np.loadtxt('face_embedding.txt', dtype=np.float32) #flatten array
        #convert into array
        receive_samples = int(X_array.size/512) 
        X_array = X_array.reshape(receive_samples,512)
        X_array = np.asarray(X_array)
        #calculate mean
        X_mean = np.mean(X_array, axis=1)
        X_mean_bytes = X_mean.tobytes()
        #save to redis 
        r.hset(name='academy:register',key = key, value = X_mean_bytes)
        os.remove('face_embedding.txt')
        self.reset()
        
        return 'Data saved successfully'