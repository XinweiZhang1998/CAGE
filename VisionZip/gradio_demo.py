import gradio as gr
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPVisionModel
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import io

# Load CLIP model and processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
model.eval()

def load_image(image):
    return image.convert("RGB")

def resize_image(image, target_size):
    original_size = image.size
    resized_image = image.resize(target_size, Image.BILINEAR)
    scale_x = target_size[0] / original_size[0]
    scale_y = target_size[1] / original_size[1]
    scale_back_x = original_size[0] / target_size[0]
    scale_back_y = original_size[1] / target_size[1]
    return resized_image, original_size, (scale_x, scale_y), (scale_back_x, scale_back_y)

def preprocess_image(processor, image):
    return processor(images=image, return_tensors="pt")

def get_attention_map(model, inputs, layer_num=-2):
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attention_map = outputs.attentions[layer_num]
    return attention_map.squeeze(0)

def smooth_heatmap(grid_attention, grid_size, image_size):
    zoom_factors = (image_size[1] / grid_size, image_size[0] / grid_size)
    heatmap = zoom(grid_attention, zoom_factors, order=1)
    return heatmap[:image_size[1], :image_size[0]]

def pixel_to_grid_index(x, y, resized_size, grid_size):
    patch_width = resized_size[0] / grid_size
    patch_height = resized_size[1] / grid_size
    col = min(int(x / patch_width), grid_size - 1)
    row = min(int(y / patch_height), grid_size - 1)
    return row * grid_size + col


def compute_patch_attention(attention_map, grid_size, image_size):
    
    patch_attention = attention_map[:, 1:, 1:]
    avg_patch_attention = patch_attention.mean(dim=1) 
    avg_patch_attention = avg_patch_attention.mean(dim=0).detach().cpu().numpy() 
    grid_patch_attention = avg_patch_attention.reshape(grid_size, grid_size)
    heatmap = np.kron(grid_patch_attention, np.ones((image_size[1]//grid_size, image_size[0]//grid_size)))
    return heatmap[:image_size[1], :image_size[0]]

def overlay_heatmap_with_point_on_image(original_image, heatmap_resized, scale_factors_back, original_size, x_orig, y_orig, alpha=0.5, cmap='plasma'):

    plt.figure(figsize=(original_size[0] / 100, original_size[1] / 100), dpi=100)
    ax = plt.gca()
    ax.imshow(original_image)
    
    scale_back_x, scale_back_y = scale_factors_back
    
    heatmap_original = zoom(heatmap_resized, (scale_back_y, scale_back_x), order=1)
    heatmap_original = heatmap_original[:original_size[1], :original_size[0]]
    
    if heatmap_original.shape != (original_size[1], original_size[0]):
        heatmap_original = np.array(Image.fromarray(heatmap_original).resize(original_size, Image.BILINEAR))
    
    im = ax.imshow(heatmap_original, cmap=cmap, alpha=alpha)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention', rotation=270, labelpad=15)
    
    ax.plot(x_orig, y_orig, 'ro', markersize=8)
    
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return Image.open(buf)



def overlay_heatmap_with_point_on_image_patch(original_image, heatmap_resized, scale_factors_back, original_size, x_orig, y_orig, alpha=0.5, cmap='plasma'):
    """
    Overlay heatmap on original image with a specific point marked.
    """
    plt.figure(figsize=(original_size[0] / 100, original_size[1] / 100), dpi=100)
    ax = plt.gca()
    ax.imshow(original_image)
    
    scale_back_x, scale_back_y = scale_factors_back
    
    heatmap_original = zoom(heatmap_resized, (scale_back_y, scale_back_x), order=1)
    heatmap_original = heatmap_original[:original_size[1], :original_size[0]]
    
    if heatmap_original.shape != (original_size[1], original_size[0]):
        heatmap_original = np.array(Image.fromarray(heatmap_original).resize(original_size, Image.BILINEAR))
    
    im = ax.imshow(heatmap_original, cmap=cmap, alpha=alpha)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention', rotation=270, labelpad=15)
    
    ax.plot(x_orig, y_orig, 'ro', markersize=8)
    
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return Image.open(buf)



def process_click(image, evt: gr.SelectData):
    target_size = (336, 336)
    grid_size = 24
    layer_num = -2

    original_image = load_image(image)
    resized_image, original_size, scale_factors, scale_back_factors = resize_image(original_image, target_size)
    inputs = preprocess_image(processor, resized_image)

    attention_map = get_attention_map(model, inputs, layer_num)
    x_resized = evt.index[0] * scale_factors[0]
    y_resized = evt.index[1] * scale_factors[1]
    grid_index = min(pixel_to_grid_index(x_resized, y_resized, target_size, grid_size), grid_size * grid_size - 1)

    cls_attention = attention_map[:, grid_index, 1:]
    avg_cls_attention = cls_attention.mean(dim=0).cpu().numpy().reshape(grid_size, grid_size)
    smooth_cls_heatmap = smooth_heatmap(avg_cls_attention, grid_size, target_size)


    patch_attention_heatmap = compute_patch_attention(attention_map, grid_size, target_size)
    smooth_patch_heatmap = (patch_attention_heatmap - patch_attention_heatmap.min()) / (patch_attention_heatmap.max() - patch_attention_heatmap.min())

    smooth_cls_heatmap = (smooth_cls_heatmap - smooth_cls_heatmap.min()) / (smooth_cls_heatmap.max() - smooth_cls_heatmap.min())

    cls_heatmap_image = overlay_heatmap_with_point_on_image(
        original_image, 
        smooth_cls_heatmap, 
        scale_factors_back=scale_back_factors,
        original_size=original_size,
        x_orig=evt.index[0], 
        y_orig=evt.index[1]
    )
    patch_heatmap_image = overlay_heatmap_with_point_on_image_patch(
        original_image, 
        smooth_patch_heatmap, 
        scale_factors_back=scale_back_factors,
        original_size=original_size,
        x_orig=evt.index[0], 
        y_orig=evt.index[1]
    )

    return cls_heatmap_image, patch_heatmap_image




with gr.Blocks() as demo:
    gr.Markdown("""
    # VisionZip: Longer is Better but Not Necessary in Vision Language Models
    ## Redundancy and Feature Misalignment Visualizer
    This tool enables the visualization of attention mechanisms in CLIP by analyzing redundancy and feature misalignment in token attention.

    ## Features
    - **Attention to the Selected Token**: Displays the attention heatmap of the selected token across all patches.
    - **Patch Attention Heatmap**: Visualizes the relationships and redundancy between visual patches.

    ## Insights
    - The **first heatmap** shows that the selected token's attention focuses on dominant tokens rather than semantically related tokens.
    - The **second heatmap** shows attention concentrated on a few tokens, emphasizing the redundancy in visual tokens.
    """)
    with gr.Row():
        with gr.Column(scale=1):  
            image_input = gr.Image(type="pil", label="Step 1: Upload an Image")
        with gr.Column(scale=2):  
            instructions = gr.Markdown("""
            ### Instructions
            1. **Upload** an image using the left panel.
            2. **Click** on a specific point in the image to analyze.
            3. **View** the generated heatmaps below for insights.
            """)

    with gr.Row():
        with gr.Column():
            cls_image_output = gr.Image(type="pil", label="Attention to the Selected Point")
        with gr.Column():
            patch_image_output = gr.Image(type="pil", label="Patch Attention Heatmap")

    image_input.select(
        process_click,
        inputs=[image_input],
        outputs=[cls_image_output, patch_image_output]
    )

demo.launch()



