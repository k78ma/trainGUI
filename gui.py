from cgi import print_arguments
import tkinter as tk
import numpy as np
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk
import time
#from main_bert_tcn import NRC_Revenue_Prediction_Training
from gui_runner import *
#from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
import sys

import os

class GUI:

    def __init__(self, root):

        self.root = root
        root.title('Training GUI')
        root.geometry('605x450')
        root.configure(bg='#DAE0E6')
        root.resizable(True, True)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True)
        self.main_frame = tk.Frame(self.notebook, width=605, height=450, bg='#DAE0E6')
        #self.main_frame.configure
        self.param_frame = tk.Frame(self.notebook, width=605, height=450, bg='#DAE0E6')
        self.loss_frame = tk.Frame(self.notebook, width=605, height=450, bg='#DAE0E6')
        self.fp_frame = tk.Frame(self.notebook, width=605, height=450, bg='#DAE0E6')
        self.fn_frame = tk.Frame(self.notebook, width=605, height=450, bg='#DAE0E6')
        self.pgi_frame = tk.Frame(self.notebook, width=605, height=450, bg='#DAE0E6')
        self.notebook.add(self.main_frame, text='Main')
        self.notebook.add(self.param_frame, text='Parameters')
        self.notebook.add(self.loss_frame, text='Loss')
        self.notebook.add(self.fp_frame, text='FP Rate')
        self.notebook.add(self.fn_frame, text='FN Rate')
        self.notebook.add(self.pgi_frame, text='PGI')


        self.param_dict = {}

        self.title_label = tk.Label(self.main_frame, text='IRAP Portfolio Growth Training', bg='#283B5B', fg='white', font=('Arial', 18)) #CSV file
        self.title_label.place(x = 20, y = 20, anchor = 'nw')

        #waterloo_logo = Image.open('images/university-of-waterloo-vertical-logo.png')
        #resized_waterloo_logo = waterloo_logo.resize((212, 138))
        #self.tk_waterloo_logo = ImageTk.PhotoImage(resized_waterloo_logo)
        #self.waterloo_label = tk.Label(self.main_frame, image = self.tk_waterloo_logo, bg='#DAE0E6') #bg='#DAE0E6'
        #self.waterloo_label.place(x = 340, y = 0)

        nrc_logo = Image.open('images/nrc-logo-white.png')
        resized_nrc_logo = nrc_logo.resize((215, 16))
        self.tk_nrc_logo = ImageTk.PhotoImage(resized_nrc_logo)
        self.nrc_label = tk.Label(self.main_frame, image = self.tk_nrc_logo, bg='#DAE0E6') #bg='#DAE0E6'
        self.nrc_label.place(x = 375, y = 25)

        # Session name
        self.sessionNameLabel = tk.Label(self.main_frame, text='Session name:', bg='#64778D', fg='white', width=16, anchor = 'w')
        self.sessionNameLabel.place(x = 20, y = 80, anchor = 'nw')

        self.sessionNameInput = tk.StringVar()
        self.sessionName = tk.Entry(self.main_frame, textvariable=self.sessionNameInput, width=51)
        self.sessionName.place(x=150, y = 80)
        self.sessionName.focus()

        self.saveSessionNameButton = tk.Button(self.main_frame, text = 'Save', command=lambda: self.button_action('Save name'), bg='#283B5B', fg='white')
        self.saveSessionNameButton.place(x = 530, y = 77.5, anchor = 'nw', width=55)

        # Model path upload
        self.uploadLabel = tk.Label(self.main_frame, text='Input training data:', bg='#64778D', fg='white', width=16, anchor = 'w') #CSV file
        self.uploadLabel.place(x = 20, y = 117.5, anchor = 'nw')
      
        self.uploadInput = tk.StringVar()
        self.uploadInputBox = tk.Entry(self.main_frame, textvariable=self.uploadInput, width=51)
        self.uploadInputBox.place(x=150, y = 120)

        self.uploadButton = tk.Button(self.main_frame, text = 'Browse', command=lambda: self.button_action('Upload'), bg='#283B5B', fg='white')
        self.uploadButton.place(x = 530, y = 120, anchor = 'nw', width=55)

        # Choose output directory
        self.outdirLabel = tk.Label(self.main_frame, text='Output directory:', bg='#64778D', fg='white', width=16, anchor = 'w')
        self.outdirLabel.place(x = 20, y = 160, anchor = 'nw')

        self.outdirInput = tk.StringVar()
        self.outdirInputBox = ttk.Entry(self.main_frame, textvariable=self.outdirInput, width=51)
        self.outdirInputBox.place(x=150, y = 160)

        self.outdirButton = tk.Button(self.main_frame, text = 'Browse', command=lambda: self.button_action('Choose directory'), bg='#283B5B', fg='white')
        self.outdirButton.place(x = 530, y = 157.5, anchor = 'nw', width=55)
        

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

        self.gen_label = tk.Label(self.main_frame, text=self.update_progress_label('General'), bg='#DAE0E6') #bg='#DAE0E6'
        self.gen_label.place(x = 300, y = 285, anchor = 'center')
        
        # Epoch bar (current epoch)
        self.epoch_bar = ttk.Progressbar(
            self.main_frame,
            orient='horizontal',
            mode='determinate',
            length=500
        )
        self.epoch_bar.place(x = 300, y = 320, anchor = 'center')

        self.epoch_bar_label = tk.Label(self.main_frame, text=self.update_progress_label('Epoch'), bg='#DAE0E6') #bg='#DAE0E6'
        self.epoch_bar_label.place(x = 300, y = 350, anchor = 'center')

        ######## PARAMETER TAB #########
        
        self.param_title_label = tk.Label(self.param_frame, text='Parameter Settings', bg='#283B5B', fg='white', font=('Arial', 18)) #CSV file
        self.param_title_label.place(x = 20, y = 20, anchor = 'nw')
        
        param_nrc_logo = Image.open('images/nrc-logo-white.png')
        param_resized_nrc_logo = param_nrc_logo.resize((215, 16))
        self.param_tk_nrc_logo = ImageTk.PhotoImage(param_resized_nrc_logo)
        self.param_nrc_label = tk.Label(self.param_frame, image = self.param_tk_nrc_logo, bg='#DAE0E6') #bg='#DAE0E6'
        self.param_nrc_label.place(x = 340, y = 25)

        self.epoch_value = tk.IntVar()
        self.epoch_Label = tk.Label(self.param_frame, text='Epochs:', bg='#64778D', fg='white', width=14, anchor = 'w')
        self.epoch_Label.place(x = 30, y = 80, anchor = 'nw')
        self.epoch_SpinBox = ttk.Spinbox(self.param_frame, from_="0", to="50", values=(0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 75, 80, 85, 90, 95, 100), textvariable=self.epoch_value, wrap = True, width=47)
        self.epoch_SpinBox.place(x = 140, y = 80)

        # Learning rate
        self.lr_value = tk.DoubleVar()
        self.lr_Label = tk.Label(self.param_frame, text='Learning rate:', bg='#64778D', fg='white', width=14, anchor = 'w')
        self.lr_Label.place(x = 30, y = 120, anchor = 'nw')
        self.lr_Entry = ttk.Entry(self.param_frame, textvariable=self.lr_value, width=50)
        self.lr_Entry.place(x = 140, y = 120)

        # Batch size
        self.batch_value = tk.IntVar()
        self.batch_Label = tk.Label(self.param_frame, text='Batch size:', bg='#64778D', fg='white', width=14, anchor = 'w')
        self.batch_Label.place(x = 30, y = 160, anchor = 'nw')
        self.batch_SpinBox = ttk.Spinbox(self.param_frame, values=(8, 32, 64, 128), textvariable=self.batch_value, wrap = True, width=47)
        self.batch_SpinBox.place(x = 140, y = 160)
        
        # Optimizer
        self.val_ratio_value = tk.IntVar()
        self.val_ratio_Label = tk.Label(self.param_frame, text='Validation Ratio:', bg='#64778D', fg='white', width=14, anchor = 'w')
        self.val_ratio_Label.place(x = 30, y = 200, anchor = 'nw')
        self.val_ratio_Entry = ttk.Entry(self.param_frame, textvariable=self.val_ratio_value, width=50)
        self.val_ratio_Entry.place(x = 140, y = 200)

        # Start year
        self.start_year_value = tk.IntVar()
        self.start_year = tk.Label(self.param_frame, text='Start Year:', bg='#64778D', fg='white', width=14, anchor = 'w')
        self.start_year.place(x = 30, y = 240, anchor = 'nw')
        self.start_year_Entry = ttk.Entry(self.param_frame, textvariable=self.start_year_value, width=50)
        self.start_year_Entry.place(x = 140, y = 240)

        # End year
        self.end_year_value = tk.IntVar()
        self.end_year = tk.Label(self.param_frame, text='End Year:', bg='#64778D', fg='white', width=14, anchor = 'w')
        self.end_year.place(x = 30, y = 280, anchor = 'nw')
        self.end_year_Entry = ttk.Entry(self.param_frame, textvariable=self.end_year_value, width=50)
        self.end_year_Entry.place(x = 140, y = 280)

        # Save parameters button
        self.defParamButton = tk.Button(self.param_frame, text = "Use default", command = lambda:self.paramButtonFunction('Use default'), bg='#283B5B', fg='white')
        self.defParamButton.place(x = 150, y = 345)
        self.setParamButton = tk.Button(self.param_frame, text = "Set parameters", command = lambda:self.paramButtonFunction('Set params'), bg='#283B5B', fg='white')
        self.setParamButton.place(x = 320, y = 345)
        #self.importParamButton = tk.Button(self.param_frame, text = "Import parameters", command = lambda:self.paramButtonFunction('Import params'), bg='#283B5B', fg='white')
        #self.importParamButton.place(x = 400, y = 345)

        ####### VISUALIZATION TAB ########
        #self.plot_button = tk.Button(self.visual_frame,
                    #command = self.plot(),
        #             height = 2,
        #             width = 10,
        #            text = "Plot")
        #self.plot_button.pack()

  
        # the figure that will contain the plot
        self.loss_fig = plt.figure(figsize = (6, 3.8),
                    dpi = 100)
    
        self.fp_fig = plt.figure(figsize = (6, 3.8),
                    dpi = 100)
        
        self.fn_fig = plt.figure(figsize = (6, 3.8),
            dpi = 100)
            
        self.pgi_fig = plt.figure(figsize = (6, 3.4),
            dpi = 100)
        
        # list of squares
        y = [i**2 for i in range(5)]
    
        ################################## adding the loss plots ############################################
        self.train_loss_plot = self.loss_fig.add_subplot(211)
        self.train_loss_plot.title.set_text("Training Loss vs. Epochs")
        self.train_loss_plot.set_xlabel("Epoch")
        self.train_loss_plot.set_ylabel("Training Loss")
        
        self.val_loss_plot = self.loss_fig.add_subplot(212)
        self.val_loss_plot.title.set_text("Validation Loss vs. Epochs")
        self.val_loss_plot.set_xlabel("Epoch")
        self.val_loss_plot.set_ylabel("Validation Loss")
        
        #self.l1_loss_plot = self.fig.add_subplot(212)
        #self.l1_loss_plot.title.set_text("Minimum L1 Loss vs. Epochs")
        #self.l1_loss_plot.set_xlabel("Epoch")
        #self.l1_loss_plot.set_ylabel("L1 Loss")
    
        self.loss_fig.tight_layout()
    
        # plotting the graph
    
        # creating the Tkinter canvas
        # containing the Matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.loss_fig, self.loss_frame)  
        self.canvas.draw()
    
        # placing the canvas on the Tkinter window
        self.canvas.get_tk_widget().pack()
    
        # creating the Matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.loss_frame)
        self.toolbar.update()
    
        # placing the toolbar on the Tkinter window
        self.canvas.get_tk_widget().pack()
        
        
        ################################## adding the fp plots ############################################
        self.fp_plot = self.fp_fig.add_subplot(111)
        self.fp_plot.title.set_text("False Positive Rate vs. Epochs")
        self.fp_plot.set_xlabel("Epoch")
        self.fp_plot.set_ylabel("False Positive Rate")
        
        self.fp_fig.tight_layout()
    
        # plotting the graph
    
        # creating the Tkinter canvas
        # containing the Matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fp_fig, self.fp_frame)  
        self.canvas.draw()
    
        # placing the canvas on the Tkinter window
        self.canvas.get_tk_widget().pack()
    
        # creating the Matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.fp_frame)
        self.toolbar.update()
    
        # placing the toolbar on the Tkinter window
        self.canvas.get_tk_widget().pack()
        
        #plt.xticks(np.arange(min(x), max(x)+1, 1.0))
        
        ################################## adding the fn plots ############################################
        self.fn_plot = self.fn_fig.add_subplot(111)
        self.fn_plot.title.set_text("False Negative Rate vs. Epochs")
        self.fn_plot.set_xlabel("Epoch")
        self.fn_plot.set_ylabel("False Negative Rate")
        
        self.fn_fig.tight_layout()
    
        # plotting the graph
    
        # creating the Tkinter canvas
        # containing the Matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fn_fig, self.fn_frame)  
        self.canvas.draw()
    
        # placing the canvas on the Tkinter window
        self.canvas.get_tk_widget().pack()
    
        # creating the Matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.fn_frame)
        self.toolbar.update()
    
        # placing the toolbar on the Tkinter window
        self.canvas.get_tk_widget().pack()
        
        #plt.xticks(np.arange(min(x), max(x)+1, 1.0))
        
        ################################## adding the pgi plots ############################################
        self.pgi_plot = self.pgi_fig.add_subplot(111)
        self.pgi_plot.title.set_text("Portfolio Growth Improvement vs. Epochs")
        self.pgi_plot.set_xlabel("Epoch")
        self.pgi_plot.set_ylabel("Portfolio Growth Improvement")
        
        self.pgi_fig.tight_layout()
    
        # plotting the graph
    
        # creating the Tkinter canvas
        # containing the Matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.pgi_fig, self.pgi_frame)  
        self.canvas.draw()
    
        # placing the canvas on the Tkinter window
        self.canvas.get_tk_widget().pack()
    
        # creating the Matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.pgi_frame)
        self.toolbar.update()
    
        # placing the toolbar on the Tkinter window
        self.canvas.get_tk_widget().pack()
        
        self.pgiButton = tk.Button(self.pgi_frame, text = 'PGI Explanation', command=lambda: self.button_action('PGI'), bg='#283B5B', fg='white')
        self.pgiButton.place(x = 230, y = 350, anchor = 'nw')
        
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
        
        self.uploadInputBox.insert(0, str(self.filename))

        showinfo(
            title='Selected File',
            message=self.filename
        )

    def select_dir(self):

        self.dirname = fd.askdirectory(
            title='Open a file',
            initialdir='/Users/kaixy/vscode/localGUI')
            
        self.outdirInputBox.insert(0, str(self.dirname))

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

        if args == "Use default":
            self.epoch_SpinBox.delete(0, 'end')
            self.lr_Entry.delete(0, 'end')
            self.batch_SpinBox.delete(0, 'end')
            self.val_ratio_Entry.delete(0, 'end')
            self.start_year_Entry.delete(0, 'end')
            self.end_year_Entry.delete(0, 'end')
            
            self.epoch_SpinBox.insert(0, 3)
            self.lr_Entry.insert(0, "0.001")
            self.batch_SpinBox.insert(0, 2)
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

        help_text = tk.Text(help, height = 16)

        help_text.insert('2.0', 'This is a Text widget demo \n')
        help_text.insert('2.2', 'TODO: \n')
        help_text.insert('3.6', '  - Save configs to .txt file \n')
        help_text.insert('4.6', '  - Tracking configs/file saving information --- global variables\n')
        help_text.insert('5.6', '  - Saving finished model path to .pth or .ckpt \n')
        help_text.insert('6.6', '  - Integrate progress bars with config information \n')
        help_text.insert('7.6', '  - Try to integrate with other model for now? \n')
        help_text.insert('7.6', '  - Make things prettier \n')
        help_text.insert('7.6', '  - Put stuff on GitHub/GitLab for version control \n')
        help_text.pack()

        help.mainloop()
        
    def pgi_window(self):
        pgi = tk.Toplevel(self.root)
        pgi.geometry('470x128')
        pgi.resizable = (False, False)
        pgi.title('Portfolio Growth Improvement Definition')
        
        pgi_img = Image.open('images/PGI.png')
        resized_pgi = pgi_img.resize((469, 126), Image.ANTIALIAS)
        tk_pgi = ImageTk.PhotoImage(resized_pgi)
        pgi_label = ttk.Label(pgi, image = tk_pgi) #bg='#DAE0E6'
        pgi_label.pack()
        
        pgi.mainloop()

    def update_progress_label(self, args):
        #if args == "General":
        #    return f"Overall Progress: {gen_bar['value']}%"
        if args == "General":
            epoch = int(self.gen_bar['value']/20)
            #self.epoch_label.destroy()
            return f"Epochs: " + str(epoch) + "/" + str(self.param_dict.get('epochs'))
        if args == "Epoch":
            return f"Epoch Progress: {self.epoch_bar['value']}%"


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
        if args == 'PGI':
            self.pgi_window()
        if args == 'Train':
            gui_runner(self)
        if args == "Restart":
            self.restart_program()
        print(args)

  
    def restart_program(self):
      """Restarts the current program.
      Note: this function does not return. Any cleanup action (like
      saving data) must be done before calling this function."""
      python = sys.executable
      os.execl(python, python, * sys.argv)
        
   
if __name__ == "__main__":
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()




