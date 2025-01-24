import os
import numpy as np
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


mobilenet = MobileNet(weights='imagenet', include_top=False, pooling='avg')


def vectorize_image(image_path, model):
    img = load_img(image_path, target_size=(224, 224)) 
    img_array = img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = preprocess_input(img_array)  
    feature_vector = model.predict(img_array)  
    return feature_vector.flatten() 


def vectorize_dataset(dataset_path, model):
    image_vectors = {}
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                image_vectors[image_path] = vectorize_image(image_path, model)
    return image_vectors


def search_similar_images(query_image_path, dataset_vectors, model, top_k=5):
    query_vector = vectorize_image(query_image_path, model) 
    similarities = []
    for image_path, vector in dataset_vectors.items():
        sim = cosine_similarity([query_vector], [vector])[0][0]  
        similarities.append((image_path, sim))
    
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def show_similar_images(query_image_path, similar_images):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(similar_images) + 1, 1)
    plt.imshow(load_img(query_image_path))
    plt.title("Query Image")
    plt.axis("off")
    
    for i, (image_path, sim) in enumerate(similar_images):
        plt.subplot(1, len(similar_images) + 1, i + 2)
        plt.imshow(load_img(image_path))
        plt.title(f"Sim: {sim:.2f}")
        plt.axis("off")
    plt.show()


#dataset_path = 'IMAGE/Image_Animal'  
#query_image_path = 'IMAGE/fox1.jpg'  




def find_similar_images(query_image_path, dataset_path):
    print("开始矢量化数据集...")
    dataset_vectors = vectorize_dataset(dataset_path, mobilenet)
    
    print("搜索相似图像...")
    similar_images = search_similar_images(query_image_path, dataset_vectors, mobilenet, top_k=5)
    
    print("相似图像：")
    for img_path, similarity in similar_images:
        print(f"{img_path}: {similarity:.2f}")
    
    
    show_similar_images(query_image_path, similar_images)

