# Start with image transformationimport torch
# import torchvision
import numpy
import cv2
import albumentations as A
import matplotlib.pyplot as plt

image_path = r'E:\All Practices TukiTaki\resources\cat.jpg'
image = cv2.imread(image_path)


if image is None:
    print("Error: Image not found or could not be read")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image_rgb.shape)  # Check the image size

    # here is the transformer
    transformer = A.Compose([
        A.CenterCrop(width=255, height=255)
    ], seed=42, strict=True, additional_targets={'image': 'image'})
    
    # print(f"Transformer: {transformer}")
    
    # apply transformation 
    cropped_image = transformer(image=image_rgb)['image']
    # print(f"Transformed: {cropped_image}")
    
    brightness_transformer = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p=1.0),
    ], seed=42, strict=True, additional_targets={'image': 'image'})
    transformed_image1 = brightness_transformer(image=cropped_image)['image']
    transformed_image2 = brightness_transformer(image=cropped_image)['image']
    transformed_image3 = brightness_transformer(image=cropped_image)['image']
    
    # Plotting both images: original and cropped
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cropped_image)
    plt.axis('off')
    plt.title('cropped_image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(transformed_image)
    plt.axis('off')
    plt.title('Transformed (Random Brightness) Image')
    
    plt.show()
