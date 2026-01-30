import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
from datetime import datetime, timezone
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Check for skyfield
try:
    from skyfield.api import load
    SKYFIELD_AVAILABLE = True
except ImportError:
    SKYFIELD_AVAILABLE = False

# --- PHYSICS CONSTANTS ---
G_SOLAR = 0.00029591220828 

class SolarSystemLoader:
    def __init__(self):
        self.bodies_data = []
        self.masses = {
            'Mercury': 1.6601e-7, 'Venus': 2.4478e-6, 'Earth': 3.0035e-6,
            'Mars': 3.2272e-7, 'Jupiter': 9.5479e-4, 'Saturn': 2.8588e-4,
            'Uranus': 4.3662e-5, 'Neptune': 5.1514e-5
        }
        self.colors = {
            'Sun': '#FFFF00', 'Mercury': '#A5A5A5', 'Venus': '#E3BB76',
            'Earth': '#4444FF', 'Mars': '#FF4444', 'Jupiter': '#D9A066',
            'Saturn': '#EDDCA1', 'Uranus': '#A6E7FF', 'Neptune': '#4B70DD'
        }

    def fetch_data(self, date_str="NOW"):
        if not SKYFIELD_AVAILABLE:
            raise ImportError("Skyfield library not found. Run: pip install skyfield")

        eph = load('de421.bsp')
        ts = load.timescale()
        
        if date_str.upper().strip() == "NOW" or date_str.strip() == "":
            t = ts.now()
        else:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                dt = dt.replace(tzinfo=timezone.utc)
                t = ts.from_datetime(dt)
            except ValueError:
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                    dt = dt.replace(tzinfo=timezone.utc)
                    t = ts.from_datetime(dt)
                except ValueError:
                    print("Date format invalid. Using NOW.")
                    t = ts.now()

        sun = eph['sun']
        
        data = [{
            'name': 'Sun', 'm': 1.0, 'color': self.colors['Sun'],
            'p': np.array([0.0, 0.0, 0.0]), 'v': np.array([0.0, 0.0, 0.0])
        }]

        planet_keys = [
            ('Mercury', eph['mercury']), ('Venus', eph['venus']),
            ('Earth', eph['earth']), ('Mars', eph['mars']),
            ('Jupiter', eph['jupiter barycenter']), ('Saturn', eph['saturn barycenter']),
            ('Uranus', eph['uranus barycenter']), ('Neptune', eph['neptune barycenter'])
        ]

        for name, body_obj in planet_keys:
            astrometric = sun.at(t).observe(body_obj)
            p = astrometric.position.au
            v = astrometric.velocity.au_per_d
            data.append({
                'name': name, 'm': self.masses[name], 'color': self.colors[name],
                'p': p, 'v': v
            })
            
        self.bodies_data = data
        return data, t.utc_strftime('%Y-%m-%d')

class NBodySolver:
    def equations(self, t, state, masses):
        n = len(masses)
        r = state[:3*n].reshape((n, 3))
        v = state[3*n:].reshape((n, 3))
        dr_dt = v
        dv_dt = np.zeros((n, 3))
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = r[j] - r[i]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-5: dist = 1e-5
                    dv_dt[i] += G_SOLAR * masses[j] * diff / (dist**3)
        return np.concatenate((dr_dt.flatten(), dv_dt.flatten()))

    def solve(self, masses, initial_state, duration_days, steps, progress_callback):
        t_eval = np.linspace(0, duration_days, steps)
        sol = solve_ivp(
            fun=lambda t, y: self.equations(t, y, masses),
            t_span=(0, duration_days), y0=initial_state, method='DOP853',
            t_eval=t_eval, rtol=1e-6, atol=1e-6
        )
        progress_callback(100)
        return sol.t, sol.y

class SolarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real Solar System Explorer")
        
        try:
            self.root.state('zoomed')
        except:
            self.root.attributes('-zoomed', True)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(".", background="#222", foreground="white")
        style.configure("TLabel", background="#222", foreground="white")
        style.configure("TButton", background="#444", foreground="white", bordercolor="#555")
        style.map("TButton", background=[('active', '#555')])
        self.root.configure(bg="#222")

        self.solver = NBodySolver()
        self.loader = SolarSystemLoader()
        
        self.sim_data = None
        self.anim = None
        self.is_paused = False
        self.current_frame = 0
        self.loaded_bodies = []

        self.setup_ui()
        
        self.log("Initializing...")
        self.start_loading_data("NOW")

    def setup_ui(self):
        left_panel = ttk.Frame(self.root, padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(left_panel, text="SOLAR CONTROL", font=("Arial", 14, "bold")).pack(pady=10)
        
        data_frame = ttk.LabelFrame(left_panel, text="Date Selection", padding=10)
        data_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(data_frame, text="Start Date (YYYY-MM-DD):").pack(anchor="w")
        self.date_entry = tk.Entry(data_frame, bg="#333", fg="white", insertbackground="white")
        self.date_entry.pack(fill=tk.X, pady=5)
        self.date_entry.insert(0, "NOW")
        
        self.load_btn = ttk.Button(data_frame, text="LOAD POSITIONS", command=lambda: self.start_loading_data(self.date_entry.get()))
        self.load_btn.pack(fill=tk.X, pady=5)

        self.status_label = ttk.Label(data_frame, text="Status: Waiting...", foreground="#AAA", font=("Arial", 9))
        self.status_label.pack(anchor="w")

        sets_frame = ttk.LabelFrame(left_panel, text="Simulation Params", padding=10)
        sets_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(sets_frame, text="Duration (Years):").pack(anchor="w")
        self.years_entry = tk.Entry(sets_frame, bg="#333", fg="white", insertbackground="white")
        self.years_entry.pack(fill=tk.X, pady=2)
        self.years_entry.insert(0, "2.0")
        
        ttk.Label(sets_frame, text="Steps (Resolution):").pack(anchor="w", pady=(5,0))
        self.steps_entry = tk.Entry(sets_frame, bg="#333", fg="white", insertbackground="white")
        self.steps_entry.pack(fill=tk.X, pady=2)
        self.steps_entry.insert(0, "2000")

        self.run_btn = ttk.Button(left_panel, text="CALCULATE ORBITS", command=self.start_sim, state=tk.DISABLED)
        self.run_btn.pack(fill=tk.X, pady=15)
        
        self.progress = ttk.Progressbar(left_panel, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)

        vis_frame = ttk.LabelFrame(left_panel, text="View Settings", padding=10)
        vis_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(vis_frame, text="Planet Size:").pack(anchor="w")
        self.scale_var = tk.DoubleVar(value=5.0)
        tk.Scale(vis_frame, from_=1, to=15, orient=tk.HORIZONTAL, variable=self.scale_var, 
                 bg="#222", fg="white", highlightthickness=0).pack(fill=tk.X)

        self.log_lbl = ttk.Label(left_panel, text="Ready.", font=("Courier", 9), wraplength=200)
        self.log_lbl.pack(side=tk.BOTTOM, anchor="w", pady=10)

        right_panel = tk.Frame(self.root, bg="#151515")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(dpi=100)
        self.fig.patch.set_facecolor('#151515')
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('black')
        self.ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
        self.ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
        self.ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
        self.ax.grid(False)
        self.ax.set_axis_off()

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

        toolbar_frame = tk.Frame(right_panel, bg="#111")
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.config(background="#222")
        self.toolbar._message_label.config(background="#222", foreground="white")
        for button in self.toolbar.winfo_children():
            button.config(background="#222")
        self.toolbar.update()
        
        controls = tk.Frame(right_panel, bg="#111")
        controls.pack(side=tk.BOTTOM, fill=tk.X, pady=0)
        
        self.play_btn = ttk.Button(controls, text="Pause", command=self.toggle_pause)
        self.play_btn.pack(side=tk.LEFT, padx=10, pady=10)
        
        ttk.Label(controls, text="Speed:").pack(side=tk.LEFT, padx=(20,5))
        self.speed_scale = tk.Scale(controls, from_=1, to=20, orient=tk.HORIZONTAL, showvalue=0, length=200, bg="#111", fg="white", highlightthickness=0)
        self.speed_scale.set(5)
        self.speed_scale.pack(side=tk.LEFT)

    def on_scroll(self, event):
        if event.inaxes != self.ax: return
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zlim = self.ax.get_zlim()
        if event.button == 'up': scale = 0.9
        else: scale = 1.1
        self.ax.set_xlim([x * scale for x in xlim])
        self.ax.set_ylim([y * scale for y in ylim])
        self.ax.set_zlim([z * scale for z in zlim])
        self.canvas.draw_idle()

    def log(self, msg):
        self.log_lbl.config(text=f"> {msg}")

    def start_loading_data(self, date_str):
        self.load_btn.config(state=tk.DISABLED)
        self.log(f"Fetching NASA data for {date_str}...")
        threading.Thread(target=self.load_thread, args=(date_str,), daemon=True).start()

    def load_thread(self, date_str):
        try:
            data, actual_date = self.loader.fetch_data(date_str)
            self.loaded_bodies = data
            self.root.after(0, lambda: self.finish_load(actual_date))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.load_btn.config(state=tk.NORMAL))

    def finish_load(self, date_str):
        self.status_label.config(text=f"Loaded: {date_str}")
        self.run_btn.config(state=tk.NORMAL)
        self.load_btn.config(state=tk.NORMAL)
        self.log("Data Ready. Click Calculate.")
        if self.sim_data is None: 
            self.start_sim()

    def start_sim(self):
        if not self.loaded_bodies: return
        self.run_btn.config(state=tk.DISABLED)
        self.log("Computing orbits...")
        
        masses = np.array([b['m'] for b in self.loaded_bodies])
        initial_state = []
        for b in self.loaded_bodies:
            initial_state.extend(b['p'])
        for b in self.loaded_bodies:
            initial_state.extend(b['v'])
        initial_state = np.array(initial_state)
        
        try:
            yrs = float(self.years_entry.get())
            stps = int(self.steps_entry.get())
        except:
            yrs = 2.0
            stps = 2000
            
        duration = yrs * 365.25
        threading.Thread(target=self.calc_thread, args=(masses, initial_state, duration, stps)).start()

    def calc_thread(self, masses, state, duration, steps):
        try:
            t, y = self.solver.solve(masses, state, duration, steps, 
                                     lambda p: self.root.after(0, lambda: self.progress.config(value=p)))
            self.sim_data = {'t': t, 'y': y}
            self.root.after(0, self.start_animation)
        except Exception as e:
            print(e)

    def start_animation(self):
        if self.anim: self.anim.event_source.stop()
        
        t = self.sim_data['t']
        y = self.sim_data['y']
        n_bodies = len(self.loaded_bodies)
        
        self.ax.clear()
        self.ax.set_axis_off()
        limit = 35 
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_zlim(-limit, limit)
        
        self.lines = []
        self.points = []
        self.labels = []
        
        for i in range(n_bodies):
            color = self.loaded_bodies[i]['color']
            name = self.loaded_bodies[i]['name']
            
            line, = self.ax.plot([], [], [], '-', color=color, lw=1, alpha=0.6) 
            self.lines.append(line)
            
            pt, = self.ax.plot([], [], [], 'o', color=color, markeredgecolor='white', markeredgewidth=0.5)
            self.points.append(pt)
            
            # Initial text creation. 
            # Note: We use zdir=None to ensure it's "screen aligned" (billboard style)
            txt = self.ax.text(0, 0, 0, name, color='white', fontsize=9, 
                               ha='center', va='bottom', clip_on=True, weight='bold', zdir=None)
            self.labels.append(txt)

        self.current_frame = 0
        self.run_btn.config(state=tk.NORMAL)
        self.log("Simulation Running.")

        def update(frame_idx):
            if self.is_paused: return
            
            speed = int(self.speed_scale.get())
            self.current_frame += speed
            if self.current_frame >= len(t): self.current_frame = 0
            
            idx = self.current_frame
            scale = self.scale_var.get()
            
            for i in range(n_bodies):
                xi = y[i*3, 0:idx]
                yi = y[i*3+1, 0:idx]
                zi = y[i*3+2, 0:idx]
                self.lines[i].set_data(xi, yi)
                self.lines[i].set_3d_properties(zi)
                
                px, py, pz = y[i*3, idx], y[i*3+1, idx], y[i*3+2, idx]
                self.points[i].set_data([px], [py])
                self.points[i].set_3d_properties([pz])
                
                # CRITICAL FIX: Update position but explicitly force zdir=None
                # This ensures the text stays 2D facing the screen
                self.labels[i].set_position((px, py))
                self.labels[i].set_3d_properties(pz, zdir=None)
                
                ms = scale * 2 if i == 0 else scale
                self.points[i].set_markersize(ms)
                
            self.log_lbl.config(text=f"Year: {t[idx]/365.25:.2f}")

        self.anim = FuncAnimation(self.fig, update, frames=None, interval=30, blit=False)
        self.canvas.draw()

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.play_btn.config(text="Resume" if self.is_paused else "Pause")

if __name__ == "__main__":
    root = tk.Tk()
    app = SolarApp(root)
    root.mainloop()