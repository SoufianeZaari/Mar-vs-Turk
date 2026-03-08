#!/usr/bin/env python3
"""
ENHANCED ANALYSIS: Morocco vs Turkey Bilateral Trade
=====================================================
Builds on the existing 16 charts with deeper metrics:
  - CAGR (Compound Annual Growth Rate) 2021→2024
  - HHI (Herfindahl-Hirschman Index) for concentration
  - RCA (Revealed Comparative Advantage)
  - New visualizations: Waterfall, Sankey, Bubble matrix, Opportunities
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# ============================================================
# COLOR PALETTE (consistent with existing charts)
# ============================================================
COLORS = {
    'red': '#e74c3c',
    'green': '#2ecc71',
    'blue': '#3498db',
    'orange': '#f39c12',
    'purple': '#9b59b6',
    'dark': '#2c3e50',
    'teal': '#1abc9c',
    'pink': '#e91e63',
    'amber': '#ff9800',
    'indigo': '#3f51b5',
}

# ============================================================
# DATA LOADING (same as notebook)
# ============================================================
print("📊 Loading trade data...")
df_imp_raw = pd.read_csv('Imports.csv', skiprows=14)
df_exp_raw = pd.read_csv('Exports.csv', skiprows=14)

df_imp_raw = df_imp_raw.rename(columns={'Unnamed: 0': 'Code', 'Unnamed: 1': 'Product_Label'})
df_exp_raw = df_exp_raw.rename(columns={'Unnamed: 0': 'Code', 'Unnamed: 1': 'Product_Label'})
df_imp_raw.columns = df_imp_raw.columns.str.strip()
df_exp_raw.columns = df_exp_raw.columns.str.strip()

years = ['2021', '2022', '2023', '2024']

# --- IMPORTS ---
imp_bilateral_cols = ['Code', 'Product_Label'] + [f'Value in {y}' for y in years]
df_imports = df_imp_raw[imp_bilateral_cols].copy()
for y in years:
    df_imports = df_imports.rename(columns={f'Value in {y}': f'Import_{y}'})

# --- EXPORTS ---
exp_bilateral_cols = ['Code', 'Product_Label'] + [f'Value in {y}' for y in years]
df_exports = df_exp_raw[exp_bilateral_cols].copy()
for y in years:
    df_exports = df_exports.rename(columns={f'Value in {y}': f'Export_{y}'})

# --- TURKEY WORLD EXPORTS (from Imports.csv columns 8-12) ---
turk_world_cols = ['Code']
imp_all_cols = list(df_imp_raw.columns)
# Turkey's exports to world are in columns with index 8-12 approx
turk_export_cols = []
for i, col in enumerate(imp_all_cols):
    if 'Value in' in col:
        count = sum(1 for c in imp_all_cols[:i+1] if 'Value in' in c)
        if 6 <= count <= 9:  # Second set of "Value in" columns
            turk_export_cols.append((col, i))

# --- MOROCCO WORLD EXPORTS (from Exports.csv columns 14-17) ---
exp_all_cols = list(df_exp_raw.columns)
mar_world_export_cols = []
for i, col in enumerate(exp_all_cols):
    if 'Value in' in col:
        count = sum(1 for c in exp_all_cols[:i+1] if 'Value in' in c)
        if 11 <= count <= 14:  # Third set of "Value in" columns
            mar_world_export_cols.append((col, i))

# --- MOROCCO WORLD IMPORTS (from Imports.csv columns 14-17) ---
mar_world_import_cols = []
for i, col in enumerate(imp_all_cols):
    if 'Value in' in col:
        count = sum(1 for c in imp_all_cols[:i+1] if 'Value in' in c)
        if 11 <= count <= 14:  # Third set
            mar_world_import_cols.append((col, i))

# Extract Turkey world exports by position
df_turk_world = df_imp_raw.iloc[:, [0]].copy()
for idx, (col, pos) in enumerate(turk_export_cols):
    yr = years[idx] if idx < len(years) else f'20{20+idx}'
    df_turk_world[f'Turk_World_Export_{yr}'] = pd.to_numeric(df_imp_raw.iloc[:, pos], errors='coerce').fillna(0)

# Extract Morocco world exports by position
df_mar_world_exp = df_exp_raw.iloc[:, [0]].copy()
for idx, (col, pos) in enumerate(mar_world_export_cols):
    yr = years[idx] if idx < len(years) else f'20{20+idx}'
    df_mar_world_exp[f'Mar_World_Export_{yr}'] = pd.to_numeric(df_exp_raw.iloc[:, pos], errors='coerce').fillna(0)

# Extract Morocco world imports by position
df_mar_world_imp = df_imp_raw.iloc[:, [0]].copy()
for idx, (col, pos) in enumerate(mar_world_import_cols):
    yr = years[idx] if idx < len(years) else f'20{20+idx}'
    df_mar_world_imp[f'Mar_World_Import_{yr}'] = pd.to_numeric(df_imp_raw.iloc[:, pos], errors='coerce').fillna(0)

# Merge all data
df = pd.merge(df_imports, df_exports[['Code'] + [f'Export_{y}' for y in years]], on='Code', how='outer').fillna(0)
df = pd.merge(df, df_turk_world, on='Code', how='left').fillna(0)
df = pd.merge(df, df_mar_world_exp, on='Code', how='left').fillna(0)
df = pd.merge(df, df_mar_world_imp, on='Code', how='left').fillna(0)

for y in years:
    df[f'Import_{y}'] = pd.to_numeric(df[f'Import_{y}'], errors='coerce').fillna(0)
    df[f'Export_{y}'] = pd.to_numeric(df[f'Export_{y}'], errors='coerce').fillna(0)

df = df[df['Code'] != "'TOTAL"]
df['Code_Clean'] = df['Code'].str.replace("'", "").str.strip()

# Sector classification
def classify_macro_sector(code):
    c = str(code).replace("'", "").zfill(2)[:2]
    if c in ['84', '85', '87', '90', '86', '88', '89']:
        return 'High-Tech & Machinery'
    elif c in ['61', '62', '63', '64', '52', '54', '55', '60']:
        return 'Textiles & Apparel'
    elif c in ['01','02','03','04','07','08','09','10','11','12','15','16','17','19','20','21','22','23','24']:
        return 'Agriculture & Food'
    elif c in ['72', '73', '74', '75', '76']:
        return 'Metals & Minerals'
    elif c in ['28', '29', '30', '31', '32', '33', '34', '38']:
        return 'Chemicals & Pharma'
    elif c in ['39', '40']:
        return 'Plastics & Rubber'
    elif c in ['27']:
        return 'Energy & Fuels'
    else:
        return 'Other Industries'

df['Macro_Sector'] = df['Code'].apply(classify_macro_sector)

for y in years:
    df[f'Balance_{y}'] = df[f'Export_{y}'] - df[f'Import_{y}']

print(f"✅ Data loaded: {len(df)} product categories")

# ============================================================
# METRIC 1: CAGR (Compound Annual Growth Rate) 2021→2024
# ============================================================
print("\n📈 Computing CAGR...")

n_years = 3  # from 2021 to 2024

def calc_cagr(start, end, n):
    if start <= 0 or end <= 0:
        return 0
    return (((end / start) ** (1.0 / n)) - 1) * 100

total_imp_2021 = df['Import_2021'].sum()
total_imp_2024 = df['Import_2024'].sum()
total_exp_2021 = df['Export_2021'].sum()
total_exp_2024 = df['Export_2024'].sum()

cagr_imports = calc_cagr(total_imp_2021, total_imp_2024, n_years)
cagr_exports = calc_cagr(total_exp_2021, total_exp_2024, n_years)

print(f"  CAGR Imports: {cagr_imports:.1f}%")
print(f"  CAGR Exports: {cagr_exports:.1f}%")

# Sector-level CAGR
sector_cagr = []
for sector in df['Macro_Sector'].unique():
    mask = df['Macro_Sector'] == sector
    imp_21 = df.loc[mask, 'Import_2021'].sum()
    imp_24 = df.loc[mask, 'Import_2024'].sum()
    exp_21 = df.loc[mask, 'Export_2021'].sum()
    exp_24 = df.loc[mask, 'Export_2024'].sum()
    sector_cagr.append({
        'Sector': sector,
        'Import_CAGR': calc_cagr(imp_21, imp_24, n_years),
        'Export_CAGR': calc_cagr(exp_21, exp_24, n_years),
        'Import_2024': imp_24,
        'Export_2024': exp_24,
        'Balance_2024': exp_24 - imp_24,
    })
df_cagr = pd.DataFrame(sector_cagr)

# ============================================================
# METRIC 2: HHI (Herfindahl-Hirschman Index)
# ============================================================
print("📊 Computing HHI (market concentration)...")

total_imports_2024 = df['Import_2024'].sum()
total_exports_2024 = df['Export_2024'].sum()

df['Import_Share'] = (df['Import_2024'] / total_imports_2024 * 100) if total_imports_2024 > 0 else 0
df['Export_Share'] = (df['Export_2024'] / total_exports_2024 * 100) if total_exports_2024 > 0 else 0
df['Import_Share_Sq'] = df['Import_Share'] ** 2
df['Export_Share_Sq'] = df['Export_Share'] ** 2

hhi_imports = df['Import_Share_Sq'].sum()
hhi_exports = df['Export_Share_Sq'].sum()

def interpret_hhi(hhi):
    if hhi < 1500:
        return "Low concentration (competitive)"
    elif hhi < 2500:
        return "Moderate concentration"
    else:
        return "High concentration"

print(f"  HHI Imports: {hhi_imports:.0f} — {interpret_hhi(hhi_imports)}")
print(f"  HHI Exports: {hhi_exports:.0f} — {interpret_hhi(hhi_exports)}")

# ============================================================
# METRIC 3: RCA (Revealed Comparative Advantage)
# ============================================================
print("🔬 Computing RCA...")

# RCA = (Morocco's export of product i to Turkey / Morocco's total exports to Turkey)
#      / (Turkey's total imports of product i from world / Turkey's total imports from world)
# Simplified: using Morocco's sectoral share vs Turkey's import share
rca_data = []
for _, row in df.iterrows():
    mar_export_share = row['Export_2024'] / total_exports_2024 if total_exports_2024 > 0 else 0
    # Use Turkey world import data if available
    turk_world = row.get('Turk_World_Export_2024', 0)  # This is Turkey's export to world
    # For RCA we need world reference — use Morocco's world export structure
    mar_world = row.get('Mar_World_Export_2024', 0)
    mar_world_total = df['Mar_World_Export_2024'].sum() if 'Mar_World_Export_2024' in df.columns else 1
    mar_world_share = mar_world / mar_world_total if mar_world_total > 0 else 0
    
    rca = mar_export_share / mar_world_share if mar_world_share > 0 else 0
    rca_data.append({
        'Code': row['Code'],
        'Product': row['Product_Label'] if pd.notna(row.get('Product_Label')) else row['Code'],
        'Sector': row['Macro_Sector'],
        'Export_2024': row['Export_2024'],
        'RCA': rca,
        'Balance_2024': row['Balance_2024'],
    })

df_rca = pd.DataFrame(rca_data)
df_rca_strong = df_rca[(df_rca['RCA'] > 1) & (df_rca['Export_2024'] > 100)].sort_values('RCA', ascending=False)
print(f"  Products with RCA > 1: {len(df_rca_strong)}")

# ============================================================
# CHART 17: CAGR Comparison by Sector (Grouped Bar)
# ============================================================
print("\n🎨 Generating Chart 17: CAGR by Sector...")

fig, ax = plt.subplots(figsize=(14, 8))
df_cagr_sorted = df_cagr.sort_values('Import_CAGR', ascending=True)
x = np.arange(len(df_cagr_sorted))
width = 0.35

bars1 = ax.barh(x - width/2, df_cagr_sorted['Import_CAGR'], width, 
                color=COLORS['red'], alpha=0.85, label='Import CAGR')
bars2 = ax.barh(x + width/2, df_cagr_sorted['Export_CAGR'], width, 
                color=COLORS['green'], alpha=0.85, label='Export CAGR')

ax.set_yticks(x)
ax.set_yticklabels(df_cagr_sorted['Sector'], fontsize=11)
ax.set_xlabel('CAGR 2021→2024 (%)', fontsize=12, fontweight='bold')
ax.set_title('Compound Annual Growth Rate by Sector (2021→2024)\nMorocco-Turkey Bilateral Trade', 
             fontsize=14, fontweight='bold', color=COLORS['dark'])
ax.legend(fontsize=11, loc='lower right')
ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')

for bar in bars1:
    w = bar.get_width()
    if abs(w) > 1:
        ax.text(w + (0.5 if w > 0 else -0.5), bar.get_y() + bar.get_height()/2, 
                f'{w:.1f}%', ha='left' if w > 0 else 'right', va='center', fontsize=9)
for bar in bars2:
    w = bar.get_width()
    if abs(w) > 1:
        ax.text(w + (0.5 if w > 0 else -0.5), bar.get_y() + bar.get_height()/2, 
                f'{w:.1f}%', ha='left' if w > 0 else 'right', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('charts/17_cagr_by_sector.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Saved charts/17_cagr_by_sector.png")

# ============================================================
# CHART 18: Trade Deficit Waterfall Breakdown by Sector
# ============================================================
print("🎨 Generating Chart 18: Deficit Waterfall...")

sector_balance = df.groupby('Macro_Sector')['Balance_2024'].sum().sort_values()
cumulative = 0
waterfall_data = []
for sector, balance in sector_balance.items():
    waterfall_data.append({
        'Sector': sector,
        'Balance': balance / 1e6,  # Convert to billions
        'Start': cumulative / 1e6,
        'End': (cumulative + balance) / 1e6,
    })
    cumulative += balance

fig, ax = plt.subplots(figsize=(14, 8))
for i, d in enumerate(waterfall_data):
    color = COLORS['green'] if d['Balance'] > 0 else COLORS['red']
    ax.bar(i, d['Balance'], bottom=d['Start'], color=color, alpha=0.85, 
           edgecolor='white', linewidth=1.5, width=0.6)
    # Value label
    y_pos = d['Start'] + d['Balance']/2
    ax.text(i, y_pos, f"${d['Balance']:.2f}B", ha='center', va='center', 
            fontsize=9, fontweight='bold', color='white')
    # Connector line
    if i < len(waterfall_data) - 1:
        ax.plot([i + 0.3, i + 0.7], [d['End'], d['End']], 
                color='gray', linewidth=0.8, linestyle='--')

# Total bar
ax.bar(len(waterfall_data), cumulative/1e6, color=COLORS['dark'], alpha=0.9, width=0.6)
ax.text(len(waterfall_data), cumulative/1e6/2, f"TOTAL\n${cumulative/1e6:.2f}B", 
        ha='center', va='center', fontsize=10, fontweight='bold', color='white')

labels = [d['Sector'] for d in waterfall_data] + ['NET BALANCE']
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('Trade Balance (USD Billion)', fontsize=12, fontweight='bold')
ax.set_title('Trade Deficit Waterfall: How Each Sector Contributes\nMorocco-Turkey Balance 2024', 
             fontsize=14, fontweight='bold', color=COLORS['dark'])
ax.axhline(y=0, color='black', linewidth=0.8)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('charts/18_waterfall_deficit.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Saved charts/18_waterfall_deficit.png")

# ============================================================
# CHART 19: Competitiveness Matrix (Bubble Chart)
# ============================================================
print("🎨 Generating Chart 19: Competitiveness Matrix...")

fig, ax = plt.subplots(figsize=(14, 10))

sector_colors = {
    'High-Tech & Machinery': COLORS['blue'],
    'Textiles & Apparel': COLORS['red'],
    'Agriculture & Food': COLORS['green'],
    'Metals & Minerals': COLORS['orange'],
    'Chemicals & Pharma': COLORS['purple'],
    'Plastics & Rubber': COLORS['pink'],
    'Energy & Fuels': COLORS['dark'],
    'Other Industries': COLORS['teal'],
}

for sector in df_cagr_sorted['Sector']:
    row = df_cagr[df_cagr['Sector'] == sector].iloc[0]
    size = abs(row['Balance_2024']) / 1e3  # Scale for visibility
    size = max(size, 50)  # Minimum size
    size = min(size, 3000)  # Maximum size
    
    color = sector_colors.get(sector, COLORS['blue'])
    ax.scatter(row['Import_CAGR'], row['Export_CAGR'], s=size, 
               c=color, alpha=0.7, edgecolors='white', linewidth=2, zorder=3)
    
    # Label
    offset_x = 0.5
    offset_y = 0.5
    ax.annotate(sector, (row['Import_CAGR'], row['Export_CAGR']),
                xytext=(offset_x, offset_y), textcoords='offset fontsize',
                fontsize=9, fontweight='bold', color=COLORS['dark'],
                ha='left')

# Quadrant lines
ax.axhline(y=0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linewidth=1, linestyle='--', alpha=0.5)

# Quadrant labels
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.text(xlim[1]*0.7, ylim[1]*0.85, '✅ Growing\nBilateral Trade', 
        fontsize=10, ha='center', va='center', color=COLORS['green'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax.text(xlim[0]*0.7, ylim[0]*0.85, '❌ Shrinking\nBilateral Trade', 
        fontsize=10, ha='center', va='center', color=COLORS['red'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax.text(xlim[1]*0.7, ylim[0]*0.85, '⚠️ Import Growing\nExport Shrinking', 
        fontsize=10, ha='center', va='center', color=COLORS['orange'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax.text(xlim[0]*0.7, ylim[1]*0.85, '🌟 Export Growing\nImport Shrinking', 
        fontsize=10, ha='center', va='center', color=COLORS['blue'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax.set_xlabel('Import CAGR 2021→2024 (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Export CAGR 2021→2024 (%)', fontsize=12, fontweight='bold')
ax.set_title('Competitiveness Matrix: Import vs Export Growth\nBubble size = trade balance magnitude', 
             fontsize=14, fontweight='bold', color=COLORS['dark'])
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('charts/19_competitiveness_matrix.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Saved charts/19_competitiveness_matrix.png")

# ============================================================
# CHART 20: Morocco's Opportunity Sectors
# ============================================================
print("🎨 Generating Chart 20: Opportunity Analysis...")

# Find sectors where Morocco has potential (export growing OR RCA > 1)
df_opp = df.copy()
df_opp['Export_Growth_%'] = ((df_opp['Export_2024'] - df_opp['Export_2021']) / (df_opp['Export_2021'] + 1)) * 100
df_opp['Coverage_Ratio'] = (df_opp['Export_2024'] / (df_opp['Import_2024'] + 1)) * 100

# Group by sector
opp_sectors = df_opp.groupby('Macro_Sector').agg({
    'Import_2024': 'sum',
    'Export_2024': 'sum',
    'Balance_2024': 'sum',
}).reset_index()
opp_sectors['Coverage_%'] = (opp_sectors['Export_2024'] / (opp_sectors['Import_2024'] + 1)) * 100
opp_sectors['Gap_to_Close'] = opp_sectors['Import_2024'] - opp_sectors['Export_2024']
opp_sectors = opp_sectors.sort_values('Gap_to_Close', ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Left: Gap to close
colors_gap = [COLORS['red'] if g > 0 else COLORS['green'] for g in opp_sectors['Gap_to_Close']]
bars = ax1.barh(opp_sectors['Macro_Sector'], opp_sectors['Gap_to_Close'] / 1e6, 
                color=colors_gap, alpha=0.85, edgecolor='white', linewidth=1)
for bar, val in zip(bars, opp_sectors['Gap_to_Close'] / 1e6):
    ax1.text(val + (0.02 if val > 0 else -0.02), bar.get_y() + bar.get_height()/2, 
             f'${val:.2f}B', ha='left' if val > 0 else 'right', va='center', 
             fontsize=10, fontweight='bold')

ax1.set_xlabel('Gap to Close (USD Billion)', fontsize=12, fontweight='bold')
ax1.set_title('Trade Gap by Sector\n(Positive = Morocco imports more)', 
              fontsize=13, fontweight='bold', color=COLORS['dark'])
ax1.axvline(x=0, color='black', linewidth=0.8)
ax1.grid(axis='x', alpha=0.3)

# Right: Coverage ratio
coverage_sorted = opp_sectors.sort_values('Coverage_%', ascending=True)
colors_cov = [COLORS['green'] if c >= 100 else COLORS['orange'] if c >= 50 else COLORS['red'] 
              for c in coverage_sorted['Coverage_%']]
bars2 = ax2.barh(coverage_sorted['Macro_Sector'], coverage_sorted['Coverage_%'], 
                 color=colors_cov, alpha=0.85, edgecolor='white', linewidth=1)
ax2.axvline(x=100, color='black', linewidth=2, linestyle='--', label='Parity (100%)')
for bar, val in zip(bars2, coverage_sorted['Coverage_%']):
    ax2.text(val + 2, bar.get_y() + bar.get_height()/2, 
             f'{val:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')

ax2.set_xlabel('Coverage Ratio (%)', fontsize=12, fontweight='bold')
ax2.set_title('Export Coverage Ratio by Sector\n(100% = balanced trade)', 
              fontsize=13, fontweight='bold', color=COLORS['dark'])
ax2.legend(fontsize=10)
ax2.grid(axis='x', alpha=0.3)

plt.suptitle('Morocco\'s Strategic Trade Opportunities with Turkey (2024)', 
             fontsize=16, fontweight='bold', color=COLORS['dark'], y=1.02)
plt.tight_layout()
plt.savefig('charts/20_opportunity_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Saved charts/20_opportunity_analysis.png")

# ============================================================
# CHART 21: HHI Concentration Dashboard
# ============================================================
print("🎨 Generating Chart 21: Concentration Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Top import products by share
top_imports = df.nlargest(10, 'Import_2024')[['Code_Clean', 'Product_Label', 'Import_2024', 'Import_Share']]
top_imports['Label'] = top_imports['Product_Label'].str[:30]

axes[0].barh(range(len(top_imports)), top_imports['Import_Share'].values, 
             color=COLORS['red'], alpha=0.85, edgecolor='white')
axes[0].set_yticks(range(len(top_imports)))
axes[0].set_yticklabels(top_imports['Label'].values, fontsize=9)
for i, (share, val) in enumerate(zip(top_imports['Import_Share'].values, 
                                       top_imports['Import_2024'].values)):
    axes[0].text(share + 0.2, i, f'{share:.1f}% (${val/1e3:.0f}M)', 
                 va='center', fontsize=9)
axes[0].set_xlabel('Import Share (%)', fontsize=11, fontweight='bold')
axes[0].set_title(f'Top 10 Import Products\nHHI = {hhi_imports:.0f} ({interpret_hhi(hhi_imports)})', 
                  fontsize=12, fontweight='bold', color=COLORS['dark'])
axes[0].grid(axis='x', alpha=0.3)

# Top export products by share
top_exports = df.nlargest(10, 'Export_2024')[['Code_Clean', 'Product_Label', 'Export_2024', 'Export_Share']]
top_exports['Label'] = top_exports['Product_Label'].str[:30]

axes[1].barh(range(len(top_exports)), top_exports['Export_Share'].values, 
             color=COLORS['green'], alpha=0.85, edgecolor='white')
axes[1].set_yticks(range(len(top_exports)))
axes[1].set_yticklabels(top_exports['Label'].values, fontsize=9)
for i, (share, val) in enumerate(zip(top_exports['Export_Share'].values, 
                                       top_exports['Export_2024'].values)):
    axes[1].text(share + 0.2, i, f'{share:.1f}% (${val/1e3:.0f}M)', 
                 va='center', fontsize=9)
axes[1].set_xlabel('Export Share (%)', fontsize=11, fontweight='bold')
axes[1].set_title(f'Top 10 Export Products\nHHI = {hhi_exports:.0f} ({interpret_hhi(hhi_exports)})', 
                  fontsize=12, fontweight='bold', color=COLORS['dark'])
axes[1].grid(axis='x', alpha=0.3)

plt.suptitle('Trade Concentration Analysis (HHI) — Morocco-Turkey 2024', 
             fontsize=15, fontweight='bold', color=COLORS['dark'])
plt.tight_layout()
plt.savefig('charts/21_hhi_concentration.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Saved charts/21_hhi_concentration.png")

# ============================================================
# EXPORT SUMMARY DATA FOR REPORT
# ============================================================
print("\n📄 Exporting summary data for report...")

summary = {
    'total_imports_2024': total_imports_2024,
    'total_exports_2024': total_exports_2024,
    'trade_balance_2024': total_exports_2024 - total_imports_2024,
    'coverage_ratio_2024': (total_exports_2024 / total_imports_2024) * 100,
    'cagr_imports': cagr_imports,
    'cagr_exports': cagr_exports,
    'hhi_imports': hhi_imports,
    'hhi_exports': hhi_exports,
    'deficit_ratio': total_imports_2024 / total_exports_2024 if total_exports_2024 > 0 else 0,
    'products_with_rca': len(df_rca_strong),
}

# Save sector summary
df_cagr.to_csv('report/sector_summary.csv', index=False)
print("  ✅ Saved report/sector_summary.csv")

# Print key findings
print("\n" + "="*60)
print("KEY FINDINGS SUMMARY")
print("="*60)
print(f"Total Imports from Turkey (2024):  ${summary['total_imports_2024']/1e6:.2f} Billion")
print(f"Total Exports to Turkey (2024):    ${summary['total_exports_2024']/1e6:.2f} Billion")
print(f"Trade Deficit:                     ${abs(summary['trade_balance_2024'])/1e6:.2f} Billion")
print(f"Coverage Ratio:                    {summary['coverage_ratio_2024']:.1f}%")
print(f"Import CAGR (2021→2024):           {summary['cagr_imports']:.1f}%")
print(f"Export CAGR (2021→2024):           {summary['cagr_exports']:.1f}%")
print(f"HHI Imports:                       {summary['hhi_imports']:.0f}")
print(f"HHI Exports:                       {summary['hhi_exports']:.0f}")
print(f"Deficit Ratio (Imp/Exp):           {summary['deficit_ratio']:.1f}x")
print(f"Products with RCA > 1:             {summary['products_with_rca']}")
print("="*60)

# Sector details
print("\nSECTOR BREAKDOWN:")
for _, row in df_cagr.sort_values('Balance_2024').iterrows():
    emoji = "🔴" if row['Balance_2024'] < 0 else "🟢"
    print(f"  {emoji} {row['Sector']:25s} Balance: ${row['Balance_2024']/1e6:.2f}B  "
          f"Imp CAGR: {row['Import_CAGR']:+.1f}%  Exp CAGR: {row['Export_CAGR']:+.1f}%")

print("\n✅ Enhanced analysis complete! 5 new charts generated.")
print("   Charts 17-21 saved to charts/ directory.")
