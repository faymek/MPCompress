import matplotlib.pyplot as plt
import cv2

# 可视化结果
def plot_image_with_boxes(img, boxes, file_name='output.png'):
    # plt.imshow(np.transpose(img.squeeze().numpy(), (1, 2, 0)))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for box in boxes:
        plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red', linewidth=2))
    # plt.show()
    plt.savefig(file_name, bbox_inches='tight', dpi=300)