# **Mittag-Lefller Function graph**

## **Execution Environment**
- **Python**: 3.14.0  
- **scipy**: 1.16.2
- **numpy**: 2.3.4
- **matplotlib**: 3.10.7

---

## **overview**
- This project implements a **Mittag-Leffler Function graph** using the `matplotlib` library in Python. 
- Enter a number greater than or equal to 1 in m to generate a graph.
- Showing graphs for alphas of 0.30, 0.64, and 1.0.

### **Features**
- Graph the following two Mittag-Leffler (ML) related functions:
  
  1.  $f_1(t) = E_{m\alpha, 1}(-(\omega t)^{m\alpha})$
  2.  $f_2(t) = (\omega t)^\alpha E_{m\alpha, 1+\alpha}(-(\omega t)^{m\alpha})$

- The value of the parameter $m$ can be entered interactively when the script is run.
- The ML function is approximated by a series expansion (up to $K=70$ terms).
- Handles known special cases such as $E_{2,1}(-z^2) = \cos(z)$ and $E_{1,1}(-z) = e^{-z}$ when $m=2.0$.
- Uses `matplotlib` to generate dynamic, high-quality graphs.
- Graph titles and legends are dynamically updated to reflect the entered value of $m$ (using the f-string `fr'...'`).
- The resulting plot is saved as a PNG file (e.g., `MittagLeffler_F1_m=2.0.png`).

## **Code Structure**
### **Main Functionalities**
- **`MittagLefflerPlotter` (class)**: The main class that encapsulates all parameters and methods related to plotting graphs.
  - **`__init__(self, m, ...)`**: Initializes parameters such as $m$, $\omega$ (frequency), $t$ (time vector), and $K$ (number of terms in the series).
  - **`_approx_f1(...)` / `_approx_f2(...)`**: Private methods that perform an approximation of the series expansion using `scipy.special.gamma`.
  - **`calculate_f1(...)` / `calculate_f2(...)`**: Public methods that select whether to use the approximation or the optimized special case (when $m=2$).
  - **`plot_function(...)`**: The main plotting method that sets up the `matplotlib` figure, axes, and labels, and saves the graph to a file.
- **`if __name__ == '__main__':`**:
  - This is the execution block of the script.
  - This prompts the user for the value of $m$, creates a `plotter` instance, and generates graphs of both f1 and f2.

---

## **How to Use**
1. Make sure the required libraries are installed:
```bash
pip install numpy matplotlib scipy
```
2. Save the code as a Python file (e.g., `plot_ml.py`).
3. Run the script from a terminal:
```bash
python plot_ml.py
```
4. When prompted, enter a value for $m$ and press Enter:
```
input m (e.g., 2.0, 3.0, ...): 2.0
```
5. The script will run and save two PNG files (e.g., `MittagLeffler_F1_m=2.0.png` and `MittagLeffler_F2_m=2.0.png`) in the same directory.

## **ToDo**
### **Completed**
- Dynamically reflect the value of $m$ in plot labels.

### **Pending**
- Implement more robust error checking on user input (values ​​of $m$).

## **Licence**
- This project is licensed under the MIT License. See the `LICENSE` file for details.


---
