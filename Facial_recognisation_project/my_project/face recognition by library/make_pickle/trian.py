import pickle
import os
import face_recognition as fr
from tqdm import tqdm

input_file_path=r'C:\Users\chemi\Desktop\Data_For_face_recognistion\sociometrik_employes'
file_names=os.listdir(input_file_path)
full_paths = [os.path.join(input_file_path, file_name) for file_name in file_names]
full_paths


image_paths = full_paths


file_names
known_face_names=[]
for name in file_names:
    x=name.replace('.jpg','').replace('.png','')
    known_face_names.append(x)


known_face_encodings = []


for image_path in tqdm(image_paths):
    image = fr.load_image_file(image_path)
    face_encoding = fr.face_encodings(image)[0]  
    known_face_encodings.append(face_encoding)



data_to_save = {
    'known_face_encodings': known_face_encodings,
    'known_face_names': known_face_names
}

with open('pickle_files/trained_faces.pkl', 'wb') as file:
    pickle.dump(data_to_save, file)
    print('trained successful')