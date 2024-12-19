import gradio as gr
import dffnn
import cv2
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image
import matplotlib.colors as colors
import plotly.figure_factory as ff

import os
import shutil
dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ['GRADIO_TEMP_DIR'] = os.path.join(dir_path, "tmp")
if not os.path.isdir(os.environ['GRADIO_TEMP_DIR']):
    os.makedirs(os.environ['GRADIO_TEMP_DIR'])
rng_agent = np.random.default_rng(0)

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def plotField(X, Y, U, title):
    fig = plt.figure(figsize=(10,10*U.shape[0]/U.shape[1]+2))
    ax = fig.subplots(1,1)
    # ax.set_title(title)
    col = ax.contourf(X, Y, U, levels=21, cmap="RdBu", norm=colors.CenteredNorm(0))
    ax.axis("scaled")
    fig.colorbar(col, ax=ax, orientation="horizontal")
    return fig


def predict(img1, img2, sigma, epochs, random_seed, B_size, S_layers, N_layers, batchs,
            progress=gr.Progress(track_tqdm=True)):
    
    if random_seed == -1:
        random_seed = rng_agent.choice(int(1e8))

    _, _ ,vel = dffnn.fastVel(img1, img2, 
                              scaler=sigma, its=epochs,
                              random_seed=random_seed,
                              nB=B_size,
                              sizelayer=S_layers,
                              nlayers=N_layers,
                              batch_size=batchs
                              )
    
    Ux = vel[:,0].reshape(img1.shape)[::-1]
    Uy = -vel[:,1].reshape(img1.shape)[::-1]
    
    x = np.arange(img1.shape[0])
    y = np.arange(img1.shape[1])
    X, Y = np.meshgrid(y, x[::-1])
    print("saving")
    np.savez_compressed(os.path.join(os.environ['GRADIO_TEMP_DIR'],"flow_result.npz"), X=X, Y=Y, U=Ux, V=Uy)
    print("saved")
    Xr, Yr = np.meshgrid(y, x)

    scale = 10
    step = 10
    
    tr = np.quantile(img1.flatten(), 0.75)
    
    fig = plt.figure(figsize=(10,10*img1.shape[0]/img1.shape[1]+1))
    ax = fig.subplots(1,1)
    ax.imshow(img1[::-1], cmap="Greys_r", alpha=0.1)
    # ax.contourf(X, Y, np.sqrt(Uy**2+Ux**2), levels=21, cmap="magma")
    
    # ax.streamplot(Xr, Yr, Ux, Uy, color="red"
    #         )
    Uxn = Ux/np.sqrt(Uy**2+Ux**2)
    Uyn = Uy/np.sqrt(Uy**2+Ux**2)
    ax.quiver(X[::step,::step], Y[::step,::step], 
            Uxn[::step,::step], Uyn[::step,::step],
            np.sqrt(Uy**2+Ux**2)[::step,::step],
            cmap="magma",
            pivot='tail',
            angles='xy', scale_units='xy', scale=1/scale
            )
    ax.axis("scaled")
    ax.set_xlim(0,X.max())
    ax.set_ylim(0,Y.max())
    
    fig0 = plt.figure(figsize=(10,10*img1.shape[0]/img1.shape[1]+1))
    ax = fig0.subplots(1,1)
    ax.contour(X+Ux, Y+Uy, img1, levels=[tr], colors="blue")
    ax.contour(X, Y, img2, levels=[tr], colors="red")
    ax.axis("scaled")
    
    oimg0 = fig2img(fig0)

    
    fig0 = plt.figure(figsize=(10,10*img1.shape[0]/img1.shape[1]+1))
    ax = fig0.subplots(1,1)
    ax.contour(X, Y, img1, levels=[tr], colors="blue")
    ax.contour(X, Y, img2, levels=[tr], colors="red")
    ax.axis("scaled")
    
    oimg01 = fig2img(fig0)

    # oimg1 = fig2img(fig)
    oimg2 = fig2img(plotField(X, Y, Ux, "Ux"))
    oimg3 = fig2img(plotField(X, Y, Uy, "Uy"))
    oimg4 = fig2img(fig)
    # fig = ff.create_quiver(X, Y, Ux, Uy)
    # print(type(fig))
    return oimg01, oimg0, oimg2, oimg3, oimg4, os.path.join(os.environ['GRADIO_TEMP_DIR'],"flow_result.npz")#,  fig.to_html()

def pre_procces(img1, img2, 
                clahe, blur, reduce_by, 
                kernel_s, flipy, flipx,
                cLimit, rest_mean):
    
    if rest_mean:
        mean = np.minimum(img1, img2)
        mean = mean.astype(img1.dtype)
        img1 -=mean
        img2 -=mean

    def pipeline(img):
    
        if img.shape[-1]>=3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if img.shape[-1]<3:
            img = img[:,:,0]
        img = img[::reduce_by, ::reduce_by]
        
        if blur:
            kernel = (kernel_s, kernel_s)
            img = cv2.GaussianBlur(img, kernel, 0)

        if clahe:
            clahe_op = cv2.createCLAHE(clipLimit=cLimit)
            img = clahe_op.apply(img)
        
        div = (256.*(img.max()>1) + 1.*(img.max()<=1))
            
        img = img/div
        
        if flipy:
            img = img[::-1]
        if flipx:
            img = img[:,::-1]
        return img
    img1 = pipeline(img1)
    img2 = pipeline(img2)

    return img1, img2

if __name__ == "__main__":
    
    with gr.Blocks() as demo:
        gr.Markdown(
        """
        # Direct Displacement Field estimation using Neural Networks
        Code Author: Efraín Magaña (emmagana at uc.cl)
        """)
        gr.Markdown(
        """
        ## Load images & preprocessing
        """)
        with gr.Row():
            img1 = gr.Image()#eraser=False, brush=False, layers=False)
            img2 = gr.Image()#eraser=False, brush=False, layers=False)
    
        with gr.Accordion():
            with gr.Row():
                with gr.Column():
                    clahe = gr.Checkbox(label="CLAHE")
                    clipLimit  = gr.Slider(1, 100, 20, step=2, label="Clip Limit")
                    with gr.Row():
                        flipy = gr.Checkbox(label="Flip Y")
                        flipx = gr.Checkbox(label="Flip X")
                with gr.Column():
                    blur = gr.Checkbox(label="Gaussian Blur")
                    kernel = gr.Slider(1, 50, 7,step=2, label="Kernel Size")
                with gr.Column():
                    skip = gr.Number(1, label="skip", minimum=1)
                    rest_mean = gr.Checkbox(label="Rest min")
                
            with gr.Row():
                img1_pre = gr.Image(image_mode="L")
                img2_pre = gr.Image(image_mode="L")
            inputs_pre = [img1, img2, clahe, blur, skip, kernel, flipy, flipx, clipLimit, rest_mean]
            img1.change(pre_procces, inputs=inputs_pre, 
                        outputs=[img1_pre, img2_pre])
            img2.change(pre_procces, inputs=inputs_pre, 
                        outputs=[img1_pre, img2_pre])
            clahe.change(pre_procces, inputs=inputs_pre, 
                         outputs=[img1_pre, img2_pre])
            clipLimit.change(pre_procces, inputs=inputs_pre, 
                         outputs=[img1_pre, img2_pre])
            blur.change(pre_procces, inputs=inputs_pre,
                         outputs=[img1_pre, img2_pre])
            kernel.change(pre_procces, inputs=inputs_pre, 
                          outputs=[img1_pre, img2_pre])
            skip.change(pre_procces, inputs=inputs_pre, 
                        outputs=[img1_pre, img2_pre])
            flipy.change(pre_procces, inputs=inputs_pre, 
                        outputs=[img1_pre, img2_pre])
            flipx.change(pre_procces, inputs=inputs_pre, 
                        outputs=[img1_pre, img2_pre])
            rest_mean.change(pre_procces, inputs=inputs_pre, 
                        outputs=[img1_pre, img2_pre])
        gr.Markdown(
        """
        ## Hyperparameters configuration
        """)
            
        with gr.Row():
            with gr.Column():
                sigma = gr.Number(50, label="beta", minimum=1)
                epochs = gr.Number(100, label="epochs", minimum=1)
                batchs = gr.Number(100000, label="batchs", minimum=32, step=100)
            with gr.Column():
                bsize = gr.Number(200, label="B size", minimum=1)
                nlayers = gr.Number(1, label="N layers", minimum=0)
                slayers = gr.Number(100, label="S layers")
            with gr.Column():
                random = gr.Number(-1, label="seed")
                with gr.Row():
                    fixed_bttn = gr.Button("Fixed Seed")
                    random_bttn = gr.Button("Random Seed")
        gr.Markdown(
        """
        ## Run & Results
        """)
        with gr.Column():
            run = gr.Button()        
            down = gr.File(label="Output files")

        def setRnd():
            return gr.Number(rng_agent.choice(int(1e8)))
        fixed_bttn.click(setRnd, inputs=None, outputs=[random])
        def changeRnd():
            return gr.Number(-1)
        random_bttn.click(changeRnd, inputs=None, outputs=[random])

        with gr.Row():
            output = gr.Image(label="Original Superpotition", format="png")
            output2 = gr.Image(label="Deformed Superpotition", format="png")
        # output_quiver = gr.HTML(label="Quiver")

        with gr.Row():
            output3 = gr.Image(label="Flow on x", format="png")
            output4 = gr.Image(label="Flow on y", format="png")

        output_vel = gr.Image(label="Velocity", format="png")
        # def pri():print(img2.t, type(img2.transforms[0]))
        # img2.apply(pri, inputs=None, outputs=None)

        event = run.click(predict, 
                          [img1_pre, img2_pre, sigma, epochs, random, bsize, slayers, nlayers, batchs], 
                          [output, output2, output3, output4, output_vel, down]
                          )
        

    # demo.launch(share=True)
    demo.launch()
    print("Deleting temp files")
    # tmp_files = os.removedirs
    shutil.rmtree(os.environ['GRADIO_TEMP_DIR'])