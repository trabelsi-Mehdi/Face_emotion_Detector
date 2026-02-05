"""
Model Evaluation and Testing Script
Evaluate trained models and generate performance metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from keras.models import load_model
import pandas as pd

class ModelEvaluator:
    def __init__(self, model_path='models/emotion_model.h5'):
        """
        Initialize model evaluator
        
        Args:
            model_path: Path to trained model
        """
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.model = load_model(model_path)
        print(f"✓ Loaded model from {model_path}")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data
        
        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n📊 Evaluating model...")
        
        # Get predictions
        y_pred = self.model.predict(X_test, verbose=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\n✓ Test Loss: {test_loss:.4f}")
        print(f"✓ Test Accuracy: {test_accuracy:.4f}")
        
        # Classification report
        print("\n📈 Classification Report:")
        print(classification_report(
            y_true_classes, 
            y_pred_classes, 
            target_names=self.emotion_labels
        ))
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'y_pred': y_pred_classes,
            'y_true': y_true_classes
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.emotion_labels,
            yticklabels=self.emotion_labels
        )
        plt.title('Confusion Matrix - Emotion Recognition')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
        plt.show()
    
    def plot_per_class_accuracy(self, y_true, y_pred, save_path='per_class_accuracy.png'):
        """
        Plot per-class accuracy
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.emotion_labels, per_class_accuracy, color='skyblue', edgecolor='navy')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{per_class_accuracy[i]:.2%}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Emotion Class', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Per-class accuracy plot saved to {save_path}")
        plt.show()
    
    def plot_sample_predictions(self, X_test, y_test, num_samples=16, 
                               save_path='sample_predictions.png'):
        """
        Plot sample predictions
        
        Args:
            X_test: Test images
            y_test: Test labels
            num_samples: Number of samples to display
            save_path: Path to save the plot
        """
        # Get random samples
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        predictions = self.model.predict(X_test[indices], verbose=0)
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            # Get image
            img = X_test[idx].squeeze()
            
            # Get labels
            true_label = self.emotion_labels[np.argmax(y_test[idx])]
            pred_label = self.emotion_labels[np.argmax(predictions[i])]
            confidence = np.max(predictions[i])
            
            # Plot
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
            
            # Color: green if correct, red if wrong
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(
                f'True: {true_label}\nPred: {pred_label} ({confidence:.2f})',
                color=color,
                fontsize=9
            )
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Sample predictions saved to {save_path}")
        plt.show()
    
    def analyze_misclassifications(self, X_test, y_test, save_path='misclassifications.png'):
        """
        Analyze and visualize common misclassifications
        
        Args:
            X_test: Test images
            y_test: Test labels
            save_path: Path to save the plot
        """
        predictions = self.model.predict(X_test, verbose=1)
        y_pred_classes = np.argmax(predictions, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Find misclassifications
        misclassified_indices = np.where(y_pred_classes != y_true_classes)[0]
        
        print(f"\n🔍 Total misclassifications: {len(misclassified_indices)}")
        print(f"   Accuracy: {1 - len(misclassified_indices)/len(X_test):.2%}")
        
        # Sample 16 misclassifications
        if len(misclassified_indices) > 0:
            sample_size = min(16, len(misclassified_indices))
            sample_indices = np.random.choice(misclassified_indices, sample_size, replace=False)
            
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            axes = axes.ravel()
            
            for i, idx in enumerate(sample_indices):
                img = X_test[idx].squeeze()
                true_label = self.emotion_labels[y_true_classes[idx]]
                pred_label = self.emotion_labels[y_pred_classes[idx]]
                confidence = np.max(predictions[idx])
                
                axes[i].imshow(img, cmap='gray')
                axes[i].axis('off')
                axes[i].set_title(
                    f'True: {true_label}\nPred: {pred_label} ({confidence:.2f})',
                    color='red',
                    fontsize=9
                )
            
            # Hide empty subplots
            for i in range(sample_size, 16):
                axes[i].axis('off')
            
            plt.suptitle('Sample Misclassifications', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Misclassification analysis saved to {save_path}")
            plt.show()

def main():
    """Main evaluation function"""
    from train_model import EmotionModelTrainer
    
    # Load test data
    print("Loading test data...")
    trainer = EmotionModelTrainer()
    X_train, X_test, y_train, y_test = trainer.load_fer2013_data('fer2013.csv')
    
    if X_test is None:
        print("⚠️  Could not load test data!")
        return
    
    # Evaluate model
    evaluator = ModelEvaluator('models/emotion_model.h5')
    results = evaluator.evaluate(X_test, y_test)
    
    # Generate visualizations
    evaluator.plot_confusion_matrix(results['y_true'], results['y_pred'])
    evaluator.plot_per_class_accuracy(results['y_true'], results['y_pred'])
    evaluator.plot_sample_predictions(X_test, y_test)
    evaluator.analyze_misclassifications(X_test, y_test)
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()