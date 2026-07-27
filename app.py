import gradio as gr
import tensorflow as tf
import numpy as np

# Load your trained generator
generator = tf.keras.models.load_model("face_generator.keras")
LATENT_DIM = 100

def generate_faces(num_faces, seed):
    if seed == 0:
        noise = tf.random.normal([int(num_faces), LATENT_DIM])
    else:
        tf.random.set_seed(int(seed))
        noise = tf.random.normal([int(num_faces), LATENT_DIM])

    faces = generator(noise, training=False)
    faces = ((faces + 1) / 2.0 * 255).numpy().astype(np.uint8)
    return [face for face in faces]

custom_css = """
#title { text-align: center; }
#subtitle { text-align: center; color: gray; }
"""

with gr.Blocks(title="AI Face Generator", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🧑 Realistic Human Face Generator", elem_id="title")
    gr.Markdown(
        "A GAN (Generative Adversarial Network) trained from scratch on CelebA — "
        "generates entirely new, non-existent human faces from random noise.",
        elem_id="subtitle"
    )

    with gr.Row():
        with gr.Column(scale=1):
            num_faces = gr.Slider(minimum=1, maximum=16, value=9, step=1, label="Number of faces")
            seed = gr.Number(value=0, label="Seed (0 = random each time)", precision=0)
            generate_btn = gr.Button("✨ Generate Faces", variant="primary", size="lg")
            gr.Markdown(
                "**How it works:** A Generator network takes random noise vectors "
                "and turns them into face-like images, trained adversarially against "
                "a Discriminator that learns to tell real faces from fake ones."
            )
        with gr.Column(scale=2):
            gallery = gr.Gallery(label="Generated Faces", columns=3, height=500, object_fit="contain")

    generate_btn.click(fn=generate_faces, inputs=[num_faces, seed], outputs=gallery)

    gr.Markdown("---")
    gr.Markdown("Built with TensorFlow + Gradio · DCGAN architecture · Trained on CelebA")

demo.launch()