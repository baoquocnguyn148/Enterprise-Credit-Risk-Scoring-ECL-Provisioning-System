import sys

with open('app/streamlit_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find Insight Strip block
start_insight = -1
end_insight = -1
for i, line in enumerate(lines):
    if '# INSIGHT STRIP' in line:
        start_insight = i - 1
    if '<div class="insight-strip">' in line and start_insight != -1:
        pass
    if 'unsafe_allow_html=True)' in line and start_insight != -1 and end_insight == -1 and i > start_insight + 10:
        end_insight = i
        break

insight_block = lines[start_insight:end_insight+1]
del lines[start_insight:end_insight+1]

# Find Rows
row1_idx = -1
row3_idx = -1
for i, line in enumerate(lines):
    if 'ROW 1 — Who defaults' in line:
        row1_idx = i - 1
    if 'ROW 3 — IFRS 9 Stage Detail Cards' in line:
        row3_idx = i - 1

# Tabs injection
tabs_code = [
    '\n',
    '# ═══════════════════════════════════════════════════════════════════\n',
    '# TABS\n',
    '# ═══════════════════════════════════════════════════════════════════\n',
    'main_tab, insights_tab = st.tabs(["Credit Risk Analyst", "Insights"])\n\n',
    'with main_tab:\n'
]

# Indent Row 1 and Row 2
row1_2_indented = []
for line in lines[row1_idx:row3_idx]:
    if line.strip() == '':
        row1_2_indented.append('\n')
    else:
        row1_2_indented.append('    ' + line)

# Indent Insights Strip and Row 3
tab2_code = ['\n', 'with insights_tab:\n']
for line in insight_block:
    if line.strip() == '':
        tab2_code.append('\n')
    else:
        tab2_code.append('    ' + line)

row3_indented = []
for line in lines[row3_idx:]:
    if line.strip() == '':
        row3_indented.append('\n')
    else:
        row3_indented.append('    ' + line)

new_lines = lines[:row1_idx] + tabs_code + row1_2_indented + tab2_code + row3_indented

with open('app/streamlit_app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Restructured successfully!")
