import customtkinter as ct
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from customtkinter import CTk, CTkFrame
from tkinter import PhotoImage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

import dataVisualizer
import fileProcessor
import dataProcessor
import modelProcessor
import Forecaster
import pandas as pd

ct.set_appearance_mode('Dark')
ct.set_default_color_theme("green")


class App(CTk):
    def __init__(self):
        super().__init__()
        self.title('Electric Vehicles Demand Forecasting')
        self.geometry('1050x500')
        self.df = pd.DataFrame(data={'Date': [], 'BEV Sales': []})

        self.tabs = ct.CTkTabview(self, width=1000, height=550)
        self.tabs.pack()

        self.tab1 = self.tabs.add('Configurations')
        self.tab2 = self.tabs.add('Table View')
        self.tab3 = self.tabs.add('Graphic View')

        # self.frame1 = CTkFrame(self.tab1)
        self.step1_label = ct.CTkLabel(self.tab1, text="Step 1: Open CSV file and load data from it.")
        self.step1_label.pack()
        self.open_btn = ct.CTkButton(self.tab1, text="Open file",
                                     command=lambda: [self.load_file(), self.widgets(), self.change_file_label(),
                                                      self.load_train_menu()])
        self.open_btn.pack()
        self.file_label = ct.CTkLabel(self.tab1, text="Data is not loaded!")
        self.file_label.pack()

        self.tv1 = ttk.Treeview(self.tab2)
        self.tv1.place(relheight=1, relwidth=1)
        self.treescrolly = tk.Scrollbar(self.tab2, orient='vertical', command=self.tv1.yview)
        self.tv1.configure(yscrollcommand=self.treescrolly.set)
        self.treescrolly.pack(side='right', fill="y")
        self.style = ttk.Style()
        self.style.configure("Treeview.Heading", font=("Cambria", 14), background="green", foreground="black",
                             fieldbackground="green")
        self.style.configure("Treeview", background="black", foreground="white", rowheight=25, fieldbackground="black",
                             font=("Cambria", 14))

    def load_file(self):
        filename = filedialog.askopenfilename(title='Select a file', filetypes=[("CSV files", "*.csv")])
        fp = fileProcessor.FileProcessor()
        self.df = fp.load_file(filename)

    def change_file_label(self):
        self.file_label.configure(text="Data is loaded!")

    def change_train_label(self):
        self.file_label.configure(text="Model is trained!")

    def sliding(self, val):
        self.sliding_label.configure(text=f"Forecast horizon: {int(val)}")

    def load_train_menu(self):
        self.step2_label = ct.CTkLabel(self.tab1, text="Step 2: Train a model.")
        self.step2_label.pack()
        self.train_btn = ct.CTkButton(self.tab1, text="Train model",
                                      command=lambda: [self.train_model(), self.change_train_label(),
                                                       self.load_forecast_menu()])
        self.train_btn.pack()
        self.train_label = ct.CTkLabel(self.tab1, text="Model is not trained!")
        self.train_label.pack()

    def train_model(self):
        self.mp = modelProcessor.ModelProcessor(self.df)
        self.mp.getTrainedModel()

    def load_forecast_menu(self):
        self.step3_label = ct.CTkLabel(self.tab1, text="Step 3: Set a forecast horizon (number of months for the future demand values).")
        self.step3_label.pack()
        self.forecast_slider = ct.CTkSlider(self.tab1, from_=1, to=24, command=self.sliding)
        self.forecast_slider.pack()
        self.forecast_slider.set(1)

        self.sliding_label = ct.CTkLabel(self.tab1, text='')
        self.sliding_label.pack()
        self.step4_label = ct.CTkLabel(self.tab1,
                                       text="Step 4: Make forecast for next n month (n = number of months, set in step 3).")
        self.step4_label.pack()
        self.predict_btn = ct.CTkButton(self.tab1, text="Make forecast",
                                        command=lambda: [self.forecast(), self.load_save_menu()])
        self.predict_btn.pack()

    def forecast(self):
        self.fc = Forecaster.Forecaster(self.df)
        predictions = self.fc.makePrediction(int(self.forecast_slider.get()))
        dv = dataVisualizer.DataVisualizer(self.df)
        self.predictions = pd.DataFrame(predictions)
        print(self.predictions)
        fig, ax = dv.showForecast(self.predictions)
        FigureCanvasTkAgg(fig, master=self.tab3).get_tk_widget().pack(expand=True, fill='both')

    def load_save_menu(self):
        self.step5_label = ct.CTkLabel(self.tab1,
                                       text="(Optional) Step 5: Save your forecast to CSV file.")
        self.step5_label.pack()
        self.save_btn = ct.CTkButton(self.tab1, text="Save forecast",
                                        command=lambda: [self.save_forecast(), self.show_message()])
        self.save_btn.pack()

    def save_forecast(self):
        self.predictions.to_csv("forecast.csv")

    def show_message(self):
        messagebox.showinfo("Success!", "Forecast saved!")

    def widgets(self):
        dp = dataProcessor.DataProcessor(self.df)
        df = dp.getPreparedData()
        x = df.index

        y = df["BEV Sales"]
        self.fig1, self.ax1 = plt.subplots(dpi=80, facecolor='#000000')
        self.ax1.set_facecolor('#000000')
        self.line = self.ax1.plot(x, y, 'green', marker='o', linewidth=2)

        self.ax1.grid(alpha=0.2)
        self.ax1.set_xlabel("Date", color='white', family='Cambria', size=15)
        self.ax1.set_ylabel("Electric Vehicles Sales", color='white', family='Cambria', size=15)
        self.ax1.tick_params(color='white', labelcolor='white')
        self.ax1.spines['bottom'].set_color('white')
        self.ax1.spines['left'].set_color('white')
        FigureCanvasTkAgg(self.fig1, master=self.tab3).get_tk_widget().pack(expand=True, fill='both')
        self.clear_data()
        self.tv1["column"] = list(self.df.columns)
        self.tv1["show"] = "headings"
        for column in self.tv1["columns"]:
            self.tv1.heading(column, text=column)
        df_rows = self.df.to_numpy().tolist()
        for row in df_rows:
            self.tv1.insert("", "end", values=row)

    def clear_data(self):
        self.tv1.delete(*self.tv1.get_children())


if __name__ == '__main__':
    app = App()
    app.mainloop()
