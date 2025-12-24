import matplotlib.pyplot as plt
import numpy as np

# --- SETUP DATA ---
# Metric 1: Cost (The Control Variable)
cost_data = [62919, 62863]

# Metric 2: Reliability (The Improvement)
crash_data = [30, 14]

# Metric 3: User Experience (The 'Kill Shot')
latency_data = [5000, 25.65]

labels = ['Reactive\n(Baseline)', 'Predictive\n(LSTM)']
colors = ['#95a5a6', '#27ae60'] # Grey for Baseline, Green for Success

# --- CREATE PLOT ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(wspace=0.4) # Add space between charts

# --- FUNCTION TO ADD LABELS ---
def add_labels(ax, rects, format_str="{:.0f}"):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(format_str.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=11)

# --- CHART 1: TOTAL COST ---
rects1 = ax1.bar(labels, cost_data, color=colors, width=0.6)
ax1.set_title("Operational Cost", fontsize=14, fontweight='bold')
ax1.set_ylabel("Total Cost ($)", fontsize=12)
ax1.set_ylim(0, 75000) # Give some headroom
add_labels(ax1, rects1, "${:,.0f}")
# Annotation for "Equal"
ax1.text(0.5, 68000, "Budgets are Equal", ha='center', va='center', 
         fontsize=10, style='italic', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# --- CHART 2: CRASH EVENTS ---
rects2 = ax2.bar(labels, crash_data, color=colors, width=0.6)
ax2.set_title("Reliability (Crash Events)", fontsize=14, fontweight='bold')
ax2.set_ylabel("Count of Outages", fontsize=12)
ax2.set_ylim(0, 35)
add_labels(ax2, rects2)
# Arrow annotation
ax2.annotate("-53% Failures", xy=(1, 14), xytext=(1, 25),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             ha='center', fontsize=11, color='#c0392b', fontweight='bold')

# --- CHART 3: P99 LATENCY ---
rects3 = ax3.bar(labels, latency_data, color=colors, width=0.6)
ax3.set_title("User Experience (P99 Latency)", fontsize=14, fontweight='bold')
ax3.set_ylabel("Latency (ms)", fontsize=12)
ax3.set_ylim(0, 5800)
add_labels(ax3, rects3, "{:.1f} ms")

# Highlight the failure
ax3.text(0, 5200, "Service Timeout", ha='center', color='#c0392b', fontweight='bold')
ax3.annotate("200x Improvement", xy=(1, 25), xytext=(1, 2000),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             ha='center', fontsize=11, color='#27ae60', fontweight='bold')

# --- SAVE ---
plt.suptitle("Simulation Results: 1-Week Load Test", fontsize=16, y=1.05)
plt.savefig('final_metrics_comparison.png', bbox_inches='tight', dpi=300)
print("Figure saved as final_metrics_comparison.png")
plt.show()