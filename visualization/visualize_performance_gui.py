#!/usr/bin/env python3
"""
Performance Data Visualizer GUI for RealVision-ObjectUnderstandingAI
===================================================================

Easy-to-use GUI application for visualizing performance data.

Author: RealVision-ObjectUnderstandingAI Team
Date: July 2025
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import os
import glob
from datetime import datetime
import webbrowser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading

class PerformanceVisualizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RealVision Performance Data Visualizer")
        self.root.geometry("1200x800")
        
        # Data storage
        self.data = None
        self.df = None
        self.current_file = None
        
        # Setup GUI
        self.setup_gui()
        self.load_available_files()
        
    def setup_gui(self):
        """Setup the GUI interface"""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🤖 RealVision Performance Data Visualizer", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="📁 Data File Selection", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        # File list
        ttk.Label(file_frame, text="Available Files:").grid(row=0, column=0, sticky=tk.W)
        
        self.file_var = tk.StringVar()
        self.file_combo = ttk.Combobox(file_frame, textvariable=self.file_var, 
                                      state="readonly", width=50)
        self.file_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        self.file_combo.bind('<<ComboboxSelected>>', self.on_file_selected)
        
        # Buttons frame
        button_frame = ttk.Frame(file_frame)
        button_frame.grid(row=0, column=2, padx=(10, 0))
        
        ttk.Button(button_frame, text="📂 Browse", 
                  command=self.browse_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="🔄 Refresh", 
                  command=self.load_available_files).pack(side=tk.LEFT)
        
        # Data info frame
        info_frame = ttk.LabelFrame(main_frame, text="📊 Data Information", padding="10")
        info_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(1, weight=1)
        
        # Info text
        self.info_text = tk.Text(info_frame, height=8, width=40, wrap=tk.WORD)
        info_scroll = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scroll.set)
        
        self.info_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_scroll.grid(row=1, column=1, sticky=(tk.N, tk.S))
        # Visualization frame
        viz_frame = ttk.LabelFrame(main_frame, text="📈 Visualization Options", padding="10")
        viz_frame.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # Visualization buttons
        button_configs = [
            ("📊 Performance Dashboard", self.create_dashboard, "Create comprehensive performance dashboard"),
            ("📈 Model Comparison", self.create_model_comparison, "Compare different model performances"),
            ("📉 Time Series Analysis", self.create_time_series, "Analyze performance over time"),
            ("📋 Generate Report", self.generate_report, "Generate detailed performance report"),
            ("📁 Open Data Folder", self.open_data_folder, "Open the data folder in file explorer")
        ]
        
        for i, (text, command, tooltip) in enumerate(button_configs):
            btn = ttk.Button(viz_frame, text=text, command=command, width=25)
            btn.grid(row=i, column=0, pady=5, sticky=tk.W)
            self.create_tooltip(btn, tooltip)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Select a data file to begin")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def create_tooltip(self, widget, text):
        """Create tooltip for widget"""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = tk.Label(tooltip, text=text, background="lightyellow", 
                           relief=tk.SOLID, borderwidth=1, font=("Arial", 9))
            label.pack()
            widget.tooltip = tooltip
            
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
                
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)
        
    def load_available_files(self):
        """Load available data files into combobox"""
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        
        if os.path.exists(data_dir):
            json_files = glob.glob(os.path.join(data_dir, "*.json"))
            file_names = [os.path.basename(f) for f in json_files]
            self.file_combo['values'] = file_names
            
            if file_names:
                self.status_var.set(f"Found {len(file_names)} data files")
            else:
                self.status_var.set("No data files found in data folder")
        else:
            self.status_var.set("Data folder not found")
            
    def browse_file(self):
        """Browse for data file"""
        file_path = filedialog.askopenfilename(
            title="Select Performance Data File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            self.load_data_file(file_path)
            
    def on_file_selected(self, event):
        """Handle file selection from combobox"""
        selected_file = self.file_var.get()
        if selected_file:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            file_path = os.path.join(data_dir, selected_file)
            self.load_data_file(file_path)
            
    def load_data_file(self, file_path):
        """Load performance data from file"""
        try:
            self.status_var.set("Loading data...")
            self.root.update()
            
            with open(file_path, 'r') as f:
                self.data = json.load(f)
            
            self.df = pd.DataFrame(self.data)
            self.current_file = file_path
            
            # Convert timestamp
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df['time_elapsed'] = (self.df['timestamp'] - self.df['timestamp'].iloc[0]).dt.total_seconds()
            
            # Update info display
            self.update_data_info()
            self.status_var.set(f"Loaded: {os.path.basename(file_path)} ({len(self.df)} records)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data file:\\n{str(e)}")
            self.status_var.set("Error loading file")
            
    def update_data_info(self):
        """Update data information display"""
        if self.df is None:
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, "No data loaded")
            return
            
        # Calculate summary statistics
        summary = {
            'File': os.path.basename(self.current_file),
            'Total Frames': len(self.df),
            'Duration': f"{self.df['time_elapsed'].max():.1f} seconds",
            'Models Used': ', '.join(self.df['model'].unique()),
            'Avg FPS': f"{self.df['fps'].mean():.2f}",
            'Avg Inference Time': f"{self.df['inference_time_ms'].mean():.2f} ms",
            'Total Detections': self.df['detection_count'].sum(),
            'Date Range': f"{self.df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {self.df['timestamp'].max().strftime('%H:%M')}"
        }
        
        # Update info text
        self.info_text.delete(1.0, tk.END)
        info_text = "📊 Dataset Summary\\n" + "="*30 + "\\n"
        
        for key, value in summary.items():
            info_text += f"{key}: {value}\\n"
            
        # Model breakdown
        info_text += "\\n📈 Model Performance\\n" + "-"*25 + "\\n"
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            info_text += f"{model.upper()}:\\n"
            info_text += f"  Frames: {len(model_data)}\\n"
            info_text += f"  Avg FPS: {model_data['fps'].mean():.2f}\\n"
            info_text += f"  Avg Inference: {model_data['inference_time_ms'].mean():.2f} ms\\n\\n"
            
        self.info_text.insert(tk.END, info_text)
        
    def create_dashboard(self):
        """Create performance dashboard"""
        if self.df is None:
            messagebox.showwarning("No Data", "Please load a data file first")
            return
            
        self.status_var.set("Creating dashboard...")
        self.root.update()
        
        try:
            # Run in thread to prevent GUI freezing
            threading.Thread(target=self._create_dashboard_thread, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create dashboard:\\n{str(e)}")
            self.status_var.set("Error creating dashboard")
            
    def _create_dashboard_thread(self):
        """Create dashboard in separate thread"""
        try:
            # Create dashboard using plotly
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['FPS Over Time', 'Inference Time Distribution', 
                              'Detection Count Over Time', 'Model Performance'],
                specs=[[{"secondary_y": False}, {"type": "histogram"}],
                       [{"secondary_y": True}, {"type": "box"}]]
            )
            
            # FPS over time
            for model in self.df['model'].unique():
                model_data = self.df[self.df['model'] == model]
                fig.add_trace(
                    go.Scatter(x=model_data['time_elapsed'], y=model_data['fps'],
                             name=f'{model.upper()} FPS', mode='lines+markers'),
                    row=1, col=1
                )
            
            # Inference time histogram
            fig.add_trace(
                go.Histogram(x=self.df['inference_time_ms'], name='Inference Time', nbinsx=30),
                row=1, col=2
            )
            
            # Detection count over time
            fig.add_trace(
                go.Scatter(x=self.df['time_elapsed'], y=self.df['detection_count'],
                         name='Detections', mode='markers', opacity=0.6),
                row=2, col=1
            )
            
            # Model performance box plot
            for model in self.df['model'].unique():
                model_data = self.df[self.df['model'] == model]
                fig.add_trace(
                    go.Box(y=model_data['inference_time_ms'], name=f'{model.upper()}'),
                    row=2, col=2
                )
            
            fig.update_layout(height=800, title_text="Performance Dashboard")
            
            # Save and open
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"dashboard_{timestamp}.html"
            fig.write_html(output_file)
            
            # Update status on main thread
            self.root.after(0, lambda: self.status_var.set(f"Dashboard saved: {output_file}"))
            webbrowser.open(f"file://{os.path.abspath(output_file)}")
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Dashboard creation failed:\\n{str(e)}"))
            
    def create_model_comparison(self):
        """Create model comparison charts"""
        if self.df is None:
            messagebox.showwarning("No Data", "Please load a data file first")
            return
            
        models = self.df['model'].unique()
        if len(models) < 2:
            messagebox.showwarning("Insufficient Data", "Need at least 2 different models for comparison")
            return
            
        self.status_var.set("Creating model comparison...")
        self.root.update()
        
        try:
            # Create matplotlib figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
            
            # FPS comparison
            sns.boxplot(data=self.df, x='model', y='fps', ax=axes[0,0])
            axes[0,0].set_title('FPS Distribution')
            
            # Inference time comparison
            sns.boxplot(data=self.df, x='model', y='inference_time_ms', ax=axes[0,1])
            axes[0,1].set_title('Inference Time Distribution')
            
            # Detection count comparison
            sns.boxplot(data=self.df, x='model', y='detection_count', ax=axes[1,0])
            axes[1,0].set_title('Detection Count Distribution')
            
            # Performance vs time
            for model in models:
                model_data = self.df[self.df['model'] == model]
                axes[1,1].plot(model_data['time_elapsed'], model_data['fps'], 
                             label=f'{model.upper()}', alpha=0.7)
            axes[1,1].set_title('FPS Over Time')
            axes[1,1].set_xlabel('Time (seconds)')
            axes[1,1].legend()
            
            plt.tight_layout()
            
            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"model_comparison_{timestamp}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.show()
            
            self.status_var.set(f"Model comparison saved: {output_file}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Model comparison failed:\\n{str(e)}")
            
    def create_time_series(self):
        """Create time series analysis"""
        if self.df is None:
            messagebox.showwarning("No Data", "Please load a data file first")
            return
            
        self.status_var.set("Creating time series analysis...")
        self.root.update()
        
        try:
            threading.Thread(target=self._create_time_series_thread, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Time series creation failed:\\n{str(e)}")
            
    def _create_time_series_thread(self):
        """Create time series in separate thread"""
        try:
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=['FPS Over Time', 'Inference Time Over Time', 'Detection Count Over Time'],
                shared_xaxes=True
            )
            
            # FPS over time
            for model in self.df['model'].unique():
                model_data = self.df[self.df['model'] == model]
                fig.add_trace(
                    go.Scatter(x=model_data['timestamp'], y=model_data['fps'],
                             name=f'{model.upper()} FPS', mode='lines+markers'),
                    row=1, col=1
                )
            
            # Inference time over time
            for model in self.df['model'].unique():
                model_data = self.df[self.df['model'] == model]
                fig.add_trace(
                    go.Scatter(x=model_data['timestamp'], y=model_data['inference_time_ms'],
                             name=f'{model.upper()} Inference', mode='lines+markers'),
                    row=2, col=1
                )
            
            # Detection count over time
            fig.add_trace(
                go.Scatter(x=self.df['timestamp'], y=self.df['detection_count'],
                         name='Detection Count', mode='lines+markers'),
                row=3, col=1
            )
            
            fig.update_layout(height=900, title_text="Time Series Analysis")
            
            # Save and open
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"time_series_{timestamp}.html"
            fig.write_html(output_file)
            
            self.root.after(0, lambda: self.status_var.set(f"Time series saved: {output_file}"))
            webbrowser.open(f"file://{os.path.abspath(output_file)}")
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Time series creation failed:\\n{str(e)}"))
            
    def generate_report(self):
        """Generate performance report"""
        if self.df is None:
            messagebox.showwarning("No Data", "Please load a data file first")
            return
            
        try:
            # Generate report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"performance_report_{timestamp}.md"
            
            with open(report_file, 'w') as f:
                f.write(f"""# Performance Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Data File:** {os.path.basename(self.current_file)}
**Total Frames:** {len(self.df):,}
**Duration:** {self.df['time_elapsed'].max():.1f} seconds

## Summary Statistics
- **Average FPS:** {self.df['fps'].mean():.2f}
- **Average Inference Time:** {self.df['inference_time_ms'].mean():.2f} ms
- **Total Detections:** {self.df['detection_count'].sum():,}
- **Models Used:** {', '.join(self.df['model'].unique())}

## Model Performance
""")
                
                for model in self.df['model'].unique():
                    model_data = self.df[self.df['model'] == model]
                    f.write(f"""
### {model.upper()} Model
- **Frames:** {len(model_data):,}
- **Average FPS:** {model_data['fps'].mean():.2f}
- **Average Inference Time:** {model_data['inference_time_ms'].mean():.2f} ms
- **Detection Rate:** {model_data['detection_count'].mean():.1f} objects/frame
""")
            
            self.status_var.set(f"Report saved: {report_file}")
            messagebox.showinfo("Report Generated", f"Performance report saved as:\\n{report_file}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Report generation failed:\\n{str(e)}")
            
    def open_data_folder(self):
        """Open data folder in file explorer"""
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        
        if os.path.exists(data_dir):
            os.startfile(data_dir)  # Windows
        else:
            messagebox.showwarning("Not Found", "Data folder not found")

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = PerformanceVisualizerGUI(root)
    
    # Set icon and additional properties
    try:
        root.iconbitmap(default="icon.ico")  # Add if you have an icon
    except:
        pass
        
    root.mainloop()

if __name__ == "__main__":
    main()
