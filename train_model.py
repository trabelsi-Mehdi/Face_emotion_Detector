"""
Train Custom Emotion Recognition Model
Uses FER2013 dataset or custom dataset for training
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.layers import BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import os

# Fix for newer Keras/TensorFlow versions
try:
    from keras.preprocessing.image import ImageDataGenerator
except ImportError:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

class EmotionModelTrainer:
    def __init__(self, img_size=48, num_classes=7):
        """
        Initialize the model trainer
        
        Args:
            img_size: Size of input images (48x48 for FER2013)
            num_classes: Number of emotion classes
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.model = None
        
    def load_fer2013_data(self, csv_path='fer2013.csv'):
        """
        Load and preprocess FER2013 dataset
        
        Args:
            csv_path: Path to FER2013 CSV file
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("📊 Loading FER2013 dataset...")
        
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"❌ Could not find {csv_path}")
            print("\nTo download FER2013:")
            print("1. Go to https://www.kaggle.com/datasets/msambare/fer2013")
            print("2. Download fer2013.csv")
            print("3. Place it in the project directory")
            return None, None, None, None
        
        # Parse pixel data
        pixels = df['pixels'].tolist()
        images = []
        
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.array(face).reshape(self.img_size, self.img_size)
            images.append(face)
        
        images = np.array(images)
        emotions = pd.get_dummies(df['emotion']).values
        
        # Normalize pixel values
        images = images.astype('float32') / 255.0
        images = images.reshape(images.shape[0], self.img_size, self.img_size, 1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            images, emotions, test_size=0.2, random_state=42
        )
        
        print(f"✓ Training samples: {X_train.shape[0]}")
        print(f"✓ Testing samples: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, architecture='custom'):
        """
        Build CNN model for emotion recognition
        
        Args:
            architecture: 'custom', 'simple', or 'deep'
        """
        if architecture == 'simple':
            self.model = self._build_simple_model()
        elif architecture == 'deep':
            self.model = self._build_deep_model()
        else:
            self.model = self._build_custom_model()
        
        print(f"\n🏗️  Built {architecture} model architecture")
        self.model.summary()
        
    def _build_simple_model(self):
        """Simple CNN for quick training"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', 
                   input_shape=(self.img_size, self.img_size, 1)),
            MaxPooling2D((2, 2)),
            
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def _build_custom_model(self):
        """Custom CNN with batch normalization"""
        model = Sequential()
        
        # Block 1
        model.add(Conv2D(64, (3, 3), padding='same', 
                        input_shape=(self.img_size, self.img_size, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Block 2
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Block 3
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Block 4
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Fully connected layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        
        model.add(Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def _build_deep_model(self):
        """Deeper CNN for better accuracy"""
        model = Sequential()
        
        # Block 1
        model.add(Conv2D(64, (3, 3), padding='same', 
                        input_shape=(self.img_size, self.img_size, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Block 2
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Block 3
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Fully connected
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        
        model.add(Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def train(self, X_train, X_test, y_train, y_test, 
              epochs=50, batch_size=64, save_path='models/emotion_model.h5'):
        """
        Train the model with data augmentation
        
        Args:
            X_train, X_test, y_train, y_test: Training and testing data
            epochs: Number of training epochs
            batch_size: Batch size for training
            save_path: Path to save the best model
        """
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Callbacks
        checkpoint = ModelCheckpoint(
            save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        print(f"\n🚀 Starting training for {epochs} epochs...")
        
        # Train model
        history = self.model.fit(
            train_datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_test, y_test),
            epochs=epochs,
            callbacks=[checkpoint, early_stop, reduce_lr],
            verbose=1
        )
        
        print(f"\n✓ Training complete! Model saved to {save_path}")
        
        return history
    
    def plot_history(self, history, save_path='training_history.png'):
        """
        Plot training history
        
        Args:
            history: Keras training history object
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Val Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history plot saved to {save_path}")
        plt.show()

def main():
    """Main training function"""
    # Initialize trainer
    trainer = EmotionModelTrainer()
    
    # Load data
    X_train, X_test, y_train, y_test = trainer.load_fer2013_data('fer2013.csv')
    
    if X_train is None:
        print("\n⚠️  Please download FER2013 dataset first!")
        return
    
    # Build model (try 'simple', 'custom', or 'deep')
    trainer.build_model(architecture='custom')
    
    # Train model
    history = trainer.train(
        X_train, X_test, y_train, y_test,
        epochs=50,
        batch_size=64,
        save_path='models/emotion_model.h5'
    )
    
    # Plot results
    trainer.plot_history(history)

if __name__ == "__main__":
    main()