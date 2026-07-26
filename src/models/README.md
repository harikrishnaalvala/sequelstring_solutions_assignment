# models/

Place your trained TensorFlow model files here. Examples:

- models/classifier.h5  (Keras HDF5 model)
- models/saved_model/ (SavedModel directory)

The FastAPI app will attempt to load models from these locations if present and use them to classify uploaded documents during processing. The starter scaffold does not include a trained model by default.
