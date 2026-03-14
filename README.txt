Audit AI merged v5.5

Files:
- audit_app.py : merged Streamlit app
- requirements.txt : Python dependencies
- tab_descriptions.py : enhanced Mongolian explanations
- dansnii_jagsaalt.xlsx : account master reference used for auto-detection

Run:
1) pip install -r requirements.txt
2) streamlit run audit_app.py

Notes:
- Menu structure follows the newer separated app design.
- TB / Part1 dashboards and charts were improved using the richer v4-style visuals.
- Account names can be auto-detected from the included account master workbook, or you can upload another account list inside page 1.
