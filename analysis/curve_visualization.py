import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def load_tensorboard_data(log_dir, tag):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

log_dirs = ['.\\results\\012-traning', 
            '.\\results\\025-traning', 
            '.\\results\\026-traning']
labels = ['MLP', 'ResNet18', 'ResNet34']

# Plot Loss
plt.figure(figsize=(10, 6))
for log_dir, label in zip(log_dirs, labels):
    steps, losses = load_tensorboard_data(log_dir, 'loss')
    plt.plot(steps, losses, label=label)
plt.title('Loss over Epochs')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('combined_loss.png')
plt.close()

# Plot Accuracy
plt.figure(figsize=(10, 6))
for log_dir, label in zip(log_dirs, labels):
    steps, accs = load_tensorboard_data(log_dir, 'accuracy')
    plt.plot(steps, accs, label=label)
plt.title('Accuracy over Epochs')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('combined_accuracy.png')
plt.close()
