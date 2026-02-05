import time
import json # ADDED
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping  # ADDED

if __name__ == '__main__':
    start_time = time.time()

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        'C:/Users/jimmy/Linux/beetle_improved/mushroom_dataset',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        'C:/Users/jimmy/Linux/beetle_improved/mushroom_dataset',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # TRANSFER LEARNING
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze for feature extraction

    # PREVENT OVERFITTING
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)  
    x = layers.Dropout(0.5)(x)  
    output = layers.Dense(len(train_generator.class_indices), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # EARLY STOPPING
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=20,
        callbacks=[early_stop]  # ✅ ADDED
    )

    model.save('mushroom_model.h5')

    with open('class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes).")

    #LIMITATIONS - not using GPU because tensorflow does not work with CUDA Version 12.7+
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
