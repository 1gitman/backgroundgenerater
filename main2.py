from imageai.Prediction import ImagePrediction
import os

try:
    # Get the current working directory
    execution_path = os.getcwd()

    # Initialize the ImagePrediction class
    prediction = ImagePrediction()

    # Set the model type to SqueezeNet
    prediction.setModelTypeAsSqueezeNet()

    # Set the path to the SqueezeNet model file
    model_path = os.path.join(execution_path, "squeezenet_weights_tf_dim_ordering_tf_kernels.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    prediction.setModelPath(model_path)

    # Load the model
    prediction.loadModel()

    # Path to the image you want to predict
    image_path = os.path.join(execution_path, "j.jpg")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")

    # Perform prediction
    predictions, probabilities = prediction.predictImage(image_path, result_count=5)

    # Print the results
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(f"{eachPrediction} : {eachProbability}%")

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")