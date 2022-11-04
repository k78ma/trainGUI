from cgi import print_arguments
import tkinter as tk
import numpy as np
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk
import time
#from main_bert_tcn import NRC_Revenue_Prediction_Training
from gui_runner import *
#from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)

import os

class GUI:

    def __init__(self, root):

        self.root = root
        root.title('Training GUI')
        root.geometry('600x450')
        #root.configure(bg='#DAE0E6')
        root.resizable(False, False)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True)
        self.main_frame = ttk.Frame(self.notebook, width=600, height=450)
        self.param_frame = ttk.Frame(self.notebook, width=600, height=450)
        self.visual_frame = ttk.Frame(self.notebook, width=600, height=450)
        self.notebook.add(self.main_frame, text='Main')
        self.notebook.add(self.param_frame, text='Parameters')
        self.notebook.add(self.visual_frame, text='Visualization')


        self.param_dict = {}

        waterloo_logo = Image.open('images/university-of-waterloo-vertical-logo.png')
        resized_waterloo_logo = waterloo_logo.resize((212, 138))
        self.tk_waterloo_logo = ImageTk.PhotoImage(resized_waterloo_logo)
        self.waterloo_label = tk.Label(self.main_frame, image = self.tk_waterloo_logo) #bg='#DAE0E6'
        self.waterloo_label.place(x = 340, y = 0)

        nrc_logo = Image.open('images/NRC-new-logo.png')
        resized_nrc_logo = nrc_logo.resize((212, 62))
        self.tk_nrc_logo = ImageTk.PhotoImage(resized_nrc_logo)
        self.nrc_label = tk.Label(self.main_frame, image = self.tk_nrc_logo) #bg='#DAE0E6'
        self.nrc_label.place(x = 340, y = 130)

        # Session name
        self.sessionNameInput = tk.StringVar()
        self.sessionName = tk.Entry(self.main_frame, textvariable=self.sessionNameInput)
        self.sessionName.insert(0, "Session name")
        #sessionName.insert(tk.END, 'Session name')
        self.sessionName.place(x=30, y = 20)
        self.sessionName.focus()

        self.saveSessionNameButton = tk.Button(self.main_frame, text = 'Save name', command=lambda: self.button_action('Save name'), bg='#283B5B', fg='white')
        self.saveSessionNameButton.place(x = 210, y = 17.5, anchor = 'nw')

        # Model path upload
        self.uploadButton = tk.Button(self.main_frame, text = 'Browse', command=lambda: self.button_action('Upload'), bg='#283B5B', fg='white')
        self.uploadButton.place(x = 210, y = 65, anchor = 'nw')
        self.uploadLabel = tk.Label(self.main_frame, text='Upload training data:          ', bg='#64778D', fg='white') #CSV file
        self.uploadLabel.place(x = 30, y = 67.5, anchor = 'nw')

        # Choose output directory
        self.outdirButton = tk.Button(self.main_frame, text = 'Browse', command=lambda: self.button_action('Choose directory'), bg='#283B5B', fg='white')
        self.outdirButton.place(x = 210, y = 107.5, anchor = 'nw')
        self.outdirLabel = tk.Label(self.main_frame, text='Log/model output directory: ', bg='#64778D', fg='white')
        self.outdirLabel.place(x = 30, y = 110, anchor = 'nw')

        # Parameter configuration button
        #self.paramButton = tk.Button(self.main_frame, text = 'Configure', command=lambda: self.button_action('Configure'), bg='#283B5B', fg='white')
        #self.paramButton.place(x = 210, y = 150, anchor = 'nw')
        #self.paramLabel = tk.Label(self.main_frame, text='Training parameter settings: ', bg='#64778D', fg='white')
        #self.paramLabel.place(x = 30, y = 152.5, anchor = 'nw')

        # Help button
        self.helpButton = tk.Button(self.main_frame, text = 'Help', command=lambda: self.button_action('Help'), bg='#283B5B', fg='white')
        self.helpButton.place(x = 20, y = 360, anchor = 'nw')

        # Quit button
        self.quitButton = tk.Button(self.main_frame, text = 'Quit', command=root.destroy, bg='#283B5B', fg='white')
        self.quitButton.place(x = 530, y = 360, anchor = 'nw')
        #stop_button.place(x = 300, y = 290, anchor = 'ne')

        # Start training
        self.startTrainButton = tk.Button(self.main_frame, text = 'Start training',command=lambda:self.button_action("Train"), bg='#283B5B', fg='white')
        self.startTrainButton.place(x = 47.5, y = 200, anchor = 'nw')
        
        # Restart training
        restartTrainButton = tk.Button(self.main_frame, text = 'Restart training', command = lambda:self.button_action("Restart"), bg='#283B5B', fg='white')
        restartTrainButton.place(x = 147.5, y = 200, anchor = 'nw')
        #stop_button.place(x = 300, y = 290, anchor = 'nw')

        # General bar (total # of epochs)
        self.gen_bar = ttk.Progressbar(
            self.main_frame,
            orient='horizontal',
            mode='determinate',
            length=500
        )
        self.gen_bar.place(x = 300, y = 255, anchor = 'center')

        self.gen_label = tk.Label(self.main_frame, text=self.update_progress_label('General')) #bg='#DAE0E6'
        self.gen_label.place(x = 300, y = 285, anchor = 'center')
        
        # Epoch bar (current epoch)
        self.epoch_bar = ttk.Progressbar(
            self.main_frame,
            orient='horizontal',
            mode='determinate',
            length=500
        )
        self.epoch_bar.place(x = 300, y = 320, anchor = 'center')

        self.epoch_bar_label = tk.Label(self.main_frame, text=self.update_progress_label('Epoch')) #bg='#DAE0E6'
        self.epoch_bar_label.place(x = 300, y = 350, anchor = 'center')

        ######## PARAMETER TAB #########

        self.epoch_value = tk.IntVar()
        self.epoch_Label = tk.Label(self.param_frame, text='Epochs:          ', bg='#64778D', fg='white')
        self.epoch_Label.place(x = 30, y = 20, anchor = 'nw')
        self.epoch_SpinBox = ttk.Spinbox(self.param_frame, from_="0", to="50", values=(0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 75, 80, 85, 90, 95, 100), textvariable=self.epoch_value, wrap = True)
        self.epoch_SpinBox.place(x = 130, y = 20)

        # Learning rate
        self.lr_value = tk.DoubleVar()
        self.lr_Label = tk.Label(self.param_frame, text='Learning rate:', bg='#64778D', fg='white')
        self.lr_Label.place(x = 30, y = 60, anchor = 'nw')
        self.lr_Entry = ttk.Entry(self.param_frame, textvariable=self.lr_value)
        self.lr_Entry.place(x = 130, y = 60)

        # Batch size
        self.batch_value = tk.IntVar()
        self.batch_Label = tk.Label(self.param_frame, text='Batch size:      ', bg='#64778D', fg='white')
        self.batch_Label.place(x = 30, y = 100, anchor = 'nw')
        self.batch_SpinBox = ttk.Spinbox(self.param_frame, values=(8, 32, 64, 128), textvariable=self.batch_value, wrap = True)
        self.batch_SpinBox.place(x = 130, y = 100)
        
        # Optimizer
        self.val_ratio_value = tk.IntVar()
        self.val_ratio_Label = tk.Label(self.param_frame, text='Validation Ratio:      ', bg='#64778D', fg='white')
        self.val_ratio_Label.place(x = 30, y = 140, anchor = 'nw')
        self.val_ratio_Entry = ttk.Entry(self.param_frame, textvariable=self.val_ratio_value)
        self.val_ratio_Entry.place(x = 130, y = 140)

        # Start year
        self.start_year_value = tk.IntVar()
        self.start_year = tk.Label(self.param_frame, text='Start Year:       ', bg='#64778D', fg='white')
        self.start_year.place(x = 30, y = 180, anchor = 'nw')
        self.start_year_Entry = ttk.Entry(self.param_frame, textvariable=self.start_year_value)
        self.start_year_Entry.place(x = 130, y = 180)

        # End year
        self.end_year_value = tk.IntVar()
        self.end_year = tk.Label(self.param_frame, text='End Year:        ', bg='#64778D', fg='white')
        self.end_year.place(x = 30, y = 220, anchor = 'nw')
        self.end_year_Entry = ttk.Entry(self.param_frame, textvariable=self.end_year_value)
        self.end_year_Entry.place(x = 130, y = 220)

        # Save parameters button
        self.defParamButton = tk.Button(self.param_frame, text = "Use default", command = lambda:self.paramButtonFunction('Use default'), bg='#283B5B', fg='white')
        self.defParamButton.place(x = 300, y = 70)
        self.setParamButton = tk.Button(self.param_frame, text = "Set parameters", command = lambda:self.paramButtonFunction('Set params'), bg='#283B5B', fg='white')
        self.setParamButton.place(x = 300, y = 110)
        self.importParamButton = tk.Button(self.param_frame, text = "Import parameters", command = lambda:self.paramButtonFunction('Import params'), bg='#283B5B', fg='white')
        self.importParamButton.place(x = 300, y = 150)

        ####### VISUALIZATION TAB ########
        #self.plot_button = tk.Button(self.visual_frame,
                    #command = self.plot(),
        #             height = 2,
        #             width = 10,
        #            text = "Plot")
        #self.plot_button.pack()

  
        # the figure that will contain the plot
        self.fig = plt.figure(figsize = (6, 3.5),
                    dpi = 100)
    
        # list of squares
        y = [i**2 for i in range(5)]
    
        # adding the subplot
        self.train_loss_plot = self.fig.add_subplot(211)
        self.train_loss_plot.title.set_text("Training Loss vs. Epochs")
        self.train_loss_plot.set_xlabel("Epoch")
        self.train_loss_plot.set_ylabel("Training Loss")
        
        self.val_loss_plot = self.fig.add_subplot(212)
        self.val_loss_plot.title.set_text("Validation Loss vs. Epochs")
        self.val_loss_plot.set_xlabel("Epoch")
        self.val_loss_plot.set_ylabel("Validation Loss")
        
        #self.l1_loss_plot = self.fig.add_subplot(212)
        #self.l1_loss_plot.title.set_text("Minimum L1 Loss vs. Epochs")
        #self.l1_loss_plot.set_xlabel("Epoch")
        #self.l1_loss_plot.set_ylabel("L1 Loss")
    
        self.fig.tight_layout()
    
        # plotting the graph
    
        # creating the Tkinter canvas
        # containing the Matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, self.visual_frame)  
        self.canvas.draw()
    
        # placing the canvas on the Tkinter window
        self.canvas.get_tk_widget().pack()
    
        # creating the Matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.visual_frame)
        self.toolbar.update()
    
        # placing the toolbar on the Tkinter window
        self.canvas.get_tk_widget().pack()
        
        #plt.xticks(np.arange(min(x), max(x)+1, 1.0))

    def select_file(self):
        filetypes = (
        ('text files', '*.csv'),
        ('All files', '*.*')
        )

        self.filename = fd.askopenfilename(
            title='Open a file',
            initialdir='/Users/kaixy/vscode/localGUI',
            filetypes=filetypes)

        showinfo(
            title='Selected File',
            message=self.filename
        )

    def select_dir(self):

        self.dirname = fd.askdirectory(
            title='Open a file',
            initialdir='/Users/kaixy/vscode/localGUI')

        showinfo(
            title='Selected File',
            message=self.dirname
        )


    def paramButtonFunction(self, args):

        self.param_dict = {}

        if args == 'Set params':
            #param.title('Training GUI - ' + str(epoch_value.get()))

            self.param_dict["epochs"] = self.epoch_SpinBox.get()
            self.param_dict["lr"] = self.lr_Entry.get()
            self.param_dict["batch"] = self.batch_SpinBox.get()
            self.param_dict["val_ratio"] = self.val_ratio_Entry.get()
            self.param_dict["start_year"] = self.start_year_Entry.get()
            self.param_dict["end_year"] = self.end_year_Entry.get()
            
            print(self.param_dict)
            return self.param_dict

        if args == 'Import params':
            self.select_file()

        if args == "Use default":
            self.epoch_SpinBox.delete(0, 'end')
            self.lr_Entry.delete(0, 'end')
            self.batch_SpinBox.delete(0, 'end')
            self.val_ratio_Entry.delete(0, 'end')
            self.start_year_Entry.delete(0, 'end')
            self.end_year_Entry.delete(0, 'end')
            
            self.epoch_SpinBox.insert(0, 3)
            self.lr_Entry.insert(0, "0.001")
            self.batch_SpinBox.insert(0, 8)
            self.val_ratio_Entry.insert(0, "0.32")
            self.start_year_Entry.insert(0, "2017")
            self.end_year_Entry.insert(0, "2022")
            
            #self.param.update_idletasks()
            self.param_dict["epochs"] = 3
            self.param_dict["lr"] = 0.001
            self.param_dict["batch"] = 8
            self.param_dict["val_ratio"] = 0.32
            self.param_dict["start_year"] = 2017
            self.param_dict["end_year"] = 2020
        
        print(args)


    def help_window(self):
        help = tk.Tk()
        help.resizable = (False, False)
        help.title('Help!')

        text = tk.Text(help, height = 16)

        text.insert('2.0', 'This is a Text widget demo \n')
        text.insert('2.2', 'TODO: \n')
        text.insert('3.6', '  - Save configs to .txt file \n')
        text.insert('4.6', '  - Tracking configs/file saving information --- global variables\n')
        text.insert('5.6', '  - Saving finished model path to .pth or .ckpt \n')
        text.insert('6.6', '  - Integrate progress bars with config information \n')
        text.insert('7.6', '  - Try to integrate with other model for now? \n')
        text.insert('7.6', '  - Make things prettier \n')
        text.insert('7.6', '  - Put stuff on GitHub/GitLab for version control \n')
        text.pack()

        help.mainloop()

    def update_progress_label(self, args):
        #if args == "General":
        #    return f"Overall Progress: {gen_bar['value']}%"
        if args == "General":
            epoch = int(self.gen_bar['value']/20)
            #self.epoch_label.destroy()
            return f"Epochs: " + str(epoch) + "/" + str(self.param_dict.get('epochs'))
        if args == "Epoch":
            return f"Epoch Progress: {self.epoch_bar['value']}%"


    def progress(self):
        while self.gen_bar['value'] < 100:

            for step in range(5):

                self.root.update_idletasks()

                self.epoch_bar['value'] = 0

                self.gen_bar['value'] += 20
                self.gen_label['text'] = self.update_progress_label('General')

                time.sleep(1)

                for step in range(20):
                    while self.epoch_bar['value'] < 100:
                        self.root.update_idletasks()
                        self.epoch_bar['value'] += 5
                        self.epoch_label['text'] = self.update_progress_label('Epoch')
                        time.sleep(0.05)

        showinfo(message='Training completed!')

    def button_action(self,args):
        if args == 'Save name':
            self.root.title('Training GUI - ' + self.sessionNameInput.get())
        if args == 'Configure':
            self.param_window()
        if args == 'Upload':
            self.select_file()
        if args == 'Choose directory':
            self.select_dir()
        if args == 'Help':
            self.help_window()
        if args == 'Train':
            gui_runner(self)
        if args == "Restart":
            self.progress()
        print(args)

        
   
if __name__ == "__main__":
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()




