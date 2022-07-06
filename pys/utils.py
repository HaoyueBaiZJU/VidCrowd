import matplotlib.pyplot as plt
import cv2


def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(12, 12))

        ax[0].imshow(image)
        ax[1].imshow(mask)
        print(mask.sum())
    else:
        f, ax = plt.subplots(2, 2, figsize=(12, 12))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
        #print(mask.sum())
        
        
def plot_data(img,points,fig_size=(18,12)):
    p_img = img.copy()
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    for point in points:
        cv2.circle(p_img, tuple(point), radius=0,color=(0, 1, 0), thickness=5)
    ax.imshow(p_img)