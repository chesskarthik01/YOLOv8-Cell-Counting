from tkinter import Tk, Button, filedialog, Label, Frame
from PIL import Image, ImageTk
import pandas as pd
from ultralytics import YOLO
import numpy as np


def run_object_detection():
    # Load model
    model = YOLO('sample_cilia_counter_100.pt')

    # Get file name for source
    source = filedialog.askopenfilename()

    # Run model predictions
    results = model.predict(source=source, save=False, show=False)

    for r in results:
        # Display the first image from predictions
        im_array = r.plot(labels=False, probs=False)
        im = Image.fromarray(im_array[..., ::-1])
        im = im.resize((650, 650))  # Resize image to fit the frame
        img = ImageTk.PhotoImage(im)

        img_label.config(image=img)
        img_label.image = img

        # Extract classes and calculate cell counts
        classes = pd.Series(r.boxes.cls)
        class_list = list(classes)
        total_cells = sum([class_list.count(0), class_list.count(1)])
        num_ciliated_cells = class_list.count(1)
        ciliated_cell_percentage = num_ciliated_cells / total_cells * 100

        result_text = f'Total cell count: {total_cells}\nCiliated cell count: {num_ciliated_cells}\nCiliated cell percentage: {ciliated_cell_percentage:.2f}%'
        result_label.config(text=result_text)


def initialize_gui():
    root = Tk()
    root.title("AUTOMATIC CELL DETECTION")
    root.iconbitmap("logo.ico")

    # Black frame creation
    frame = Frame(root, width=650, height=650, bg="gray")  # Gray frame
    # Adding padding for better visibility
    frame.grid(row=0, column=0, padx=10, pady=10)

    global img_label
    img_label = Label(frame)
    img_label.grid(row=0, column=0)

    # Initialize the black frame with an empty black image
    empty_image = Image.fromarray(
        np.zeros((650, 650, 3), dtype=np.uint8), 'RGB')
    img = ImageTk.PhotoImage(empty_image)
    img_label.config(image=img)
    img_label.image = img  # Keep a reference to avoid garbage collection

    # Initialize result text placeholders
    initial_result_text = "Total cell count: \nCiliated cell count: \nCiliated cell percentage: "
    global result_label
    result_label = Label(root, text=initial_result_text, justify="left")
    # Adding padding above the text
    result_label.grid(row=2, column=0)

    run_button = Button(root, text="Select file for analysis",
                        command=run_object_detection, bg='#0D4F8B', fg='white')
    # Adding padding below the button
    run_button.grid(row=1, column=0, pady=10)

    root.mainloop()


if __name__ == "__main__":
    initialize_gui()
