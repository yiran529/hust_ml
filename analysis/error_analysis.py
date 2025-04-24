import matplotlib.pyplot as plt

# 根据results文件夹下predicting中的target_and_pred.txt填写
error_counts = [
    {2:14, 3:10, 6:78, 8:1},
    {0:1, 2:1, 3:1, 6:1, 8:1},
    {0:17, 3:10, 4:8, 6:31},
    {0:7, 1:4, 2:8, 4:13, 6:15},
    {0:2, 2:21, 3:11, 6:35},
    {7:5, 8:1, 9:4},
    {0:57, 1:1, 2:34, 3:20, 4:30, 8:1},
    {5:6, 8:1, 9:22},
    {0:4, 1:1, 5:1, 6:2, 9:1},
    {5:3, 7:18, 8:2}
]

num_classes = len(error_counts)
bar_width = 0.5

bottom = [0] * num_classes
x = list(range(num_classes))

plt.figure(figsize=(10, 6))
for pred_label in range(num_classes):
    heights = [error_counts[true_label].get(pred_label, 0) for true_label in range(num_classes)]
    plt.bar(x, heights, bottom=bottom, label=f"Pred {pred_label}")
    bottom = [bottom[i] + heights[i] for i in range(num_classes)]

plt.xticks(x, [f"True {i}" for i in x])
plt.xlabel("True Label")
plt.ylabel("Error Count")
plt.title("Error Breakdown per True Label")
plt.legend()
plt.tight_layout()
plt.savefig("error_stacked_bar.png")
plt.show()
