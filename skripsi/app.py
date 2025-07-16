import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ========================
# Load model dan encoder
# ========================
model_bulan = joblib.load('compress_model_rf_bulan.pkl')
model_tahun = joblib.load('compress_model_rf_tahun.pkl')
fitur_bulan = joblib.load('fitur_bulan.pkl')
fitur_tahun = joblib.load('fitur_tahun.pkl')
encoder_bulan = joblib.load('encoder_bulan.pkl')
encoder_tahun = joblib.load('encoder_tahun.pkl')

# ========================
# Load dataset asli
# ========================
sewa_df = pd.read_excel("sewa_df.xlsx")
sewa_df = sewa_df.dropna(subset=['city', 'apartment_name', 'agent_name'])

# ========================
# Ekstrak nilai unik
# ========================
list_kota = sorted(sewa_df['city'].unique())
list_condition = sorted(sewa_df['condition'].dropna().unique())

# ========================
# Streamlit UI
# ========================
st.title("Prediksi Harga Sewa Apartemen")

periode = st.selectbox("Pilih Periode Sewa:", ["Per Bulan", "Per Tahun"])
city = st.selectbox("Kota:", list_kota)

# Filter data sesuai kota
filtered_df = sewa_df[sewa_df['city'] == city]

# Inisialisasi Session State
if 'apartemen' not in st.session_state:
    st.session_state.apartemen = "(Kosongkan jika tidak ada)"
if 'agen' not in st.session_state:
    st.session_state.agen = "(Kosongkan jika tidak ada)"
if 'apartemen_changed' not in st.session_state:
    st.session_state.apartemen_changed = False
if 'agen_changed' not in st.session_state:
    st.session_state.agen_changed = False

# Fungsi update flag perubahan
def update_apartemen():
    st.session_state.apartemen_changed = True
    st.session_state.agen_changed = False

def update_agen():
    st.session_state.agen_changed = True
    st.session_state.apartemen_changed = False

# Fungsi untuk generate list berdasarkan perubahan terakhir
def get_apt_agen_lists():
    apt_selected = st.session_state.apartemen
    agen_selected = st.session_state.agen

    apt_list = sorted(filtered_df['apartment_name'].unique())
    agen_list = sorted(filtered_df['agent_name'].unique())

    if st.session_state.apartemen_changed and apt_selected != "(Kosongkan jika tidak ada)":
        agen_list = sorted(filtered_df[filtered_df['apartment_name'] == apt_selected]['agent_name'].unique())

    elif st.session_state.agen_changed and agen_selected != "(Kosongkan jika tidak ada)":
        apt_list = sorted(filtered_df[filtered_df['agent_name'] == agen_selected]['apartment_name'].unique())

    elif apt_selected != "(Kosongkan jika tidak ada)" and agen_selected != "(Kosongkan jika tidak ada)":
        filtered = filtered_df[
            (filtered_df['apartment_name'] == apt_selected) &
            (filtered_df['agent_name'] == agen_selected)
        ]
        apt_list = sorted(filtered['apartment_name'].unique())
        agen_list = sorted(filtered['agent_name'].unique())

    # Tambahkan nilai saat ini jika tidak ada dalam list
    if apt_selected not in apt_list:
        apt_list.append(apt_selected)
    if agen_selected not in agen_list:
        agen_list.append(agen_selected)

    apt_list = ["(Kosongkan jika tidak ada)"] + [a for a in apt_list if a != "(Kosongkan jika tidak ada)"]
    agen_list = ["(Kosongkan jika tidak ada)"] + [a for a in agen_list if a != "(Kosongkan jika tidak ada)"]

    return apt_list, agen_list

# Ambil list dinamis
apt_list, agen_list = get_apt_agen_lists()

# Selectbox dengan on_change trigger
st.selectbox(
    "Nama Apartemen (opsional):",
    apt_list,
    index=apt_list.index(st.session_state.apartemen),
    key="apartemen",
    on_change=update_apartemen
)

st.selectbox(
    "Nama Agen (opsional):",
    agen_list,
    index=agen_list.index(st.session_state.agen),
    key="agen",
    on_change=update_agen
)


# Input lainnya
bedroom = st.number_input("Jumlah Kamar Tidur", min_value=0, value=2)
bathroom = st.number_input("Jumlah Kamar Mandi", min_value=0, value=1)
building_size = st.number_input("Luas Bangunan (m²)", min_value=0.0, value=40.0)
condition = st.selectbox("Furnish:", list_condition)

# ========================
# Prediksi
# ========================
if st.button("Prediksi Harga Sewa"):
    input_data = {
        'bedroom_count': bedroom,
        'bathroom_count': bathroom,
        'building_size': building_size,
    }

    # One-hot encoding
    for c in list_kota:
        input_data[f'city_{c}'] = 1 if c == city else 0
    for cond in list_condition:
        input_data[f'condition_{cond}'] = 1 if cond == condition else 0

    # Target Encoding (kondisional)
    encoder = encoder_bulan if periode == "Per Bulan" else encoder_tahun
    apartemen_val = st.session_state.apartemen if not st.session_state.apartemen.startswith("(Kosongkan") else None
    agen_val = st.session_state.agen if not st.session_state.agen.startswith("(Kosongkan") else None

    if apartemen_val or agen_val:
        encode_df = pd.DataFrame([{
            'apartment_name': apartemen_val if apartemen_val else None,
            'agent_name': agen_val if agen_val else None
        }])
        encoded = encoder.transform(encode_df)
        input_data['apartment_name'] = encoded['apartment_name'].values[0]
        input_data['agent_name'] = encoded['agent_name'].values[0]
    else:
        input_data['apartment_name'] = 0
        input_data['agent_name'] = 0

    df = pd.DataFrame([input_data])

    model = model_bulan if periode == "Per Bulan" else model_tahun
    fitur = fitur_bulan if periode == "Per Bulan" else fitur_tahun

    for col in fitur:
        if col not in df.columns:
            df[col] = 0
    df = df[fitur]

    prediksi = model.predict(df)[0]

    # Output
    st.success("Perkiraan Harga Sewa:")
    st.write(f"Kota: {city}")
    if apartemen_val:
        st.write(f"Apartemen: {apartemen_val}")
    if agen_val:
        st.write(f"Agen: {agen_val}")
    st.info(f"Rp {int(prediksi):,} {periode.lower()}")
