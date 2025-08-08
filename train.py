from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Paths
train = r"D:\Waste management\DATASET\TRAIN"
test = r"D:\Waste management\DATASET\TEST"

# Data generators
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(
    rescale=1./255
)

# Train data
train_data = train_gen.flow_from_directory(
    train,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# Test data
test_data = test_gen.flow_from_directory(
    test,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# Example model (replace with your own)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

model = Sequential([
    Flatten(input_shape=(224, 224, 3)),
    Dense(128, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Example callback
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=3)

# Training (fixed dot to comma)
model.fit(
    train_data,
    validation_data=test_data,
    epochs=1,
    callbacks=[es]
)

# Your fine-tuning part (fixed syntax)
model.trainable = True
for layer in model.layers[:-31]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data,
          validation_data=test_data,
          epochs=2,
          callbacks=[es])

model.save("waste_management.h5")
