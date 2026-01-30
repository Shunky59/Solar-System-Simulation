# Solar-System-Simulation

A high-precision, interactive 3D simulation of the Solar System built with Python. This application fetches real-time astronomical data from NASA JPL (via Skyfield), calculates orbits using N-Body physics (Scipy), and visualizes the results in an interactive 3D environment.

## 🚀 Features

* **Real-Time Data:** Fetches accurate positions and velocities of planets for any given date using the NASA DE421 ephemeris.
* **N-Body Physics Engine:** Simulates gravitational interactions between the Sun and all 8 planets using `scipy.integrate.solve_ivp`.
* **Interactive 3D View:**
    * Full 360° rotation and zoom.
    * "Billboard" text labels that always face the user.
    * Dynamic sizing and speed controls.
* **Zero-Margin Interface:** Maximized plotting area with a modern dark-mode GUI.

## 🛠️ Installation

### 1. Prerequisites
Ensure you have Python 3.7 or newer installed.

### 2. Install Dependencies
This project requires `tkinter` (usually built-in), `numpy`, `scipy`, `matplotlib`, and `skyfield`.

```bash
pip install numpy scipy matplotlib skyfield
