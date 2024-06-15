import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class PhotoViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Photo Viewer")
        self.root.geometry("800x600")
        
        self.image_label = tk.Label(root)
        self.image_label.pack(expand=True)
        
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(fill=tk.X)
        
        self.dir_label = tk.Label(self.control_frame, text="Directory:")
        self.dir_label.pack(side=tk.LEFT, padx=5)
        
        self.dir_entry = tk.Entry(self.control_frame, width=50)
        self.dir_entry.pack(side=tk.LEFT, padx=5)
        
        self.browse_button = tk.Button(self.control_frame, text="Browse", command=self.browse_directory)
        self.browse_button.pack(side=tk.LEFT, padx=5)
        
        self.time_label = tk.Label(self.control_frame, text="Display time (seconds):")
        self.time_label.pack(side=tk.LEFT, padx=5)
        
        self.time_entry = tk.Entry(self.control_frame, width=5)
        self.time_entry.pack(side=tk.LEFT, padx=5)
        
        self.start_button = tk.Button(self.control_frame, text="Start", command=self.start_slideshow)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(self.control_frame, text="Stop", command=self.stop_slideshow)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.image_files = []
        self.current_image_index = 0
        self.display_time = 3.0
        self.slideshow_running = False
        
    def browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, directory)
    
    def load_images(self):
        directory = self.dir_entry.get()
        if os.path.isdir(directory):
            self.image_files = []
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp')):
                        self.image_files.append(os.path.join(root, file))
    
    def start_slideshow(self):
        try:
            self.display_time = float(self.time_entry.get())
        except ValueError:
            self.display_time = 3.0
        
        self.load_images()
        if self.image_files:
            self.slideshow_running = True
            self.current_image_index = 0
            self.show_image()
    
    def stop_slideshow(self):
        self.slideshow_running = False
    
    def show_image(self):
        if self.slideshow_running and self.image_files:
            image_path = self.image_files[self.current_image_index]
            img = Image.open(image_path)
            img = img.resize((800, 600), Image.Resampling.BICUBIC)
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk
            self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
            self.root.after(int(self.display_time * 1000), self.show_image)

def main():
    root = tk.Tk()
    app = PhotoViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
