import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# 1. المعاملات الفيزيائية والميكانيكية للمحاكاة
# ==========================================
m = 0.045      # الكتلة الفعالة (kg) [المغناطيس + Tip Mass]
f_res = 25.0   # التردد الطبيعي المستهدف للرنين (Hz)
omega_n = 2 * np.pi * f_res
k = m * (omega_n ** 2) # الصلابة الميكانيكية للهيكل المخروطي (N/m)
zeta = 0.03    # نسبة التخميد الميكانيكي (Mechanical Damping Ratio)
c = 2 * zeta * m * omega_n

# المعاملات الكهربائية
theta = 0.0025  # معامل الإزدواج الكهروضغطي للـ PZT (N/V)
alpha_em = 0.8  # معامل الحث الكهرومغناطيسي للوشيعة (V·s/m)
Cp = 15e-9      # سعة شريحة الـ PZT الكهربائية (Farad)
RL = 100e3      # مقاومة الحمل (Ohm) [LTC3588 Equivalent Input]
c_e = (alpha_em**2) / 10.0  # التخميد الكهرومغناطيسي المضاف

# معطيات الإثارة الميكانيكية (الاهتزاز)
A_g = 1.5 * 9.81 # تسارع الاهتزاز الخارجي (1.5g)
F0 = m * A_g     # القوة الميكانيكية المؤثرة
f_ext = 25.0     # تردد اهتزاز الآلة (Hz)

# ==========================================
# 2. المعادلة التفاضلية للنظام الهجين
# ==========================================
def hybrid_harvester(t, y):
    x, v, V_pzt = y
    
    # القوة الخارجية المترددة
    F_ext = F0 * np.sin(2 * np.pi * f_ext * t)
    
    # 1. معادلة الحركة الميكانيكية (Mechanical DOF)
    dxdt = v
    dvdt = (F_ext - (c + c_e) * v - k * x - theta * V_pzt) / m
    
    # 2. المعادلة الكهربائية للـ PZT (Piezoelectric DOF)
    dV_pzt_dt = (theta * v - V_pzt / RL) / Cp
    
    return [dxdt, dvdt, dV_pzt_dt]

# ==========================================
# 3. حل المعادلة زمنياً (Time-Domain Simulation)
# ==========================================
t_span = (0, 0.5)  # زمن المحاكاة (0.5 ثانية)
t_eval = np.linspace(t_span[0], t_span[1], 2000)
y0 = [0.0, 0.0, 0.0] # الشروط الابتدائية

sol = solve_ivp(hybrid_harvester, t_span, y0, t_eval=t_eval, method='RK45')

# حساب الجهد الكهرومغناطيسي وتيار الوشيعة
V_em = alpha_em * sol.y[1] # V = B*L*v
P_pzt = (sol.y[2]**2) / RL
P_em = (V_em**2) / 10.0
P_total_mW = (P_pzt + P_em) * 1000 # القدرة بالمللي واط

# ==========================================
# 4. رسم النتائج والمخططات البيانية
# ==========================================
plt.figure(figsize=(12, 8))

# الرسم الأول: الإزاحة الميكانيكية للهيكل
plt.subplot(3, 1, 1)
plt.plot(sol.t, sol.y[0] * 1000, 'b-', label='Mechanical Displacement (mm)')
plt.ylabel('Displacement [mm]')
plt.title('VibraCharge AI - Hybrid Energy Harvester Dynamic Simulation')
plt.grid(True)
plt.legend()

# الرسم الثاني: الجهد الكهربائي المتولد (PZT + EM)
plt.subplot(3, 3, 4)
plt.plot(sol.t, sol.y[2], 'r-', label='PZT Voltage (V)')
plt.ylabel('Voltage [V]')
plt.grid(True)
plt.legend()

plt.subplot(3, 3, 5)
plt.plot(sol.t, V_em, 'g-', label='Electromagnetic Voltage (V)')
plt.ylabel('Voltage [V]')
plt.grid(True)
plt.legend()

# الرسم الثالث: القدرة الإجمالية المحصودة (mW)
plt.subplot(3, 1, 3)
plt.plot(sol.t, P_total_mW, 'm-', label='Total Harvested Power (mW)')
plt.xlabel('Time [s]')
plt.ylabel('Power [mW]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
