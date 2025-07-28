import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ========================
# Load model dan encoder
# ========================
model_bulan = joblib.load('my-apartment-app/compress_model_rf_bulan.pkl')
model_tahun = joblib.load('my-apartment-app/compress_model_rf_tahun.pkl')
fitur_bulan = joblib.load('my-apartment-app/fitur_bulan.pkl')
fitur_tahun = joblib.load('my-apartment-app/fitur_tahun.pkl')
encoder_bulan = joblib.load('my-apartment-app/encoder_bulan.pkl')
encoder_tahun = joblib.load('my-apartment-app/encoder_tahun.pkl')
scaler_bulan = joblib.load('my-apartment-app/scaler_bulan.pkl')
scaler_tahun = joblib.load('my-apartment-app/scaler_tahun.pkl')

# ========================
# Load dataset asli
# ========================
data_bulan = pd.read_excel("my-apartment-app/data_bulan.xlsx")
data_tahun = pd.read_excel("my-apartment-app/data_tahun.xlsx")

# ========================
# Streamlit UI
# ========================
st.title("Prediksi Harga Sewa Apartemen")

periode = st.selectbox("Pilih Periode Sewa:", ["Per Bulan", "Per Tahun"])
sewa_df = data_bulan if periode == "Per Bulan" else data_tahun
sewa_df = sewa_df.dropna(subset=['city', 'apartment_name', 'agent_name'])

list_kota = sorted(sewa_df['city'].unique())
list_condition = sorted(sewa_df['condition'].dropna().unique())

# Inisialisasi session state
if 'apartemen' not in st.session_state:
    st.session_state.apartemen = "(Kosongkan jika tidak ada)"
if 'agen' not in st.session_state:
    st.session_state.agen = "(Kosongkan jika tidak ada)"
if 'apartemen_changed' not in st.session_state:
    st.session_state.apartemen_changed = False
if 'agen_changed' not in st.session_state:
    st.session_state.agen_changed = False
if 'last_city' not in st.session_state:
    st.session_state.last_city = None

# Selectbox kota
city = st.selectbox("Kota:", list_kota)

# Reset jika kota berubah
if st.session_state.last_city != city:
    st.session_state.apartemen = "(Kosongkan jika tidak ada)"
    st.session_state.agen = "(Kosongkan jika tidak ada)"
    st.session_state.apartemen_changed = False
    st.session_state.agen_changed = False
    st.session_state.last_city = city

# Filter data berdasarkan kota
filtered_df = sewa_df[sewa_df['city'] == city]

# Fungsi update perubahan
def update_apartemen():
    st.session_state.apartemen_changed = True
    st.session_state.agen_changed = False

def update_agen():
    st.session_state.agen_changed = True
    st.session_state.apartemen_changed = False

# Fungsi buat list apartemen & agen
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

    if apt_selected not in apt_list:
        apt_list.append(apt_selected)
    if agen_selected not in agen_list:
        agen_list.append(agen_selected)

    apt_list = ["(Kosongkan jika tidak ada)"] + [a for a in apt_list if a != "(Kosongkan jika tidak ada)"]
    agen_list = ["(Kosongkan jika tidak ada)"] + [a for a in agen_list if a != "(Kosongkan jika tidak ada)"]

    return apt_list, agen_list

# Dapatkan list apartemen & agen
apt_list, agen_list = get_apt_agen_lists()

# Input apartemen dan agen
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

# Input numerik & kondisi
bedroom_count = st.number_input("Jumlah Kamar Tidur", min_value=1, value=1)
bathroom_count = st.number_input("Jumlah Kamar Mandi", min_value=1, value=1)
building_size = st.number_input("Luas Bangunan (m²)", min_value=15.0, value=25.0)
facility_count = st.number_input("Jumlah Fasilitas", min_value=0, max_value=20, value=0)
condition = st.selectbox("Kondisi:", list_condition)

# ========================
# Prediksi
# ========================
if st.button("Prediksi Harga Sewa"):
    model = model_bulan if periode == "Per Bulan" else model_tahun
    fitur = fitur_bulan if periode == "Per Bulan" else fitur_tahun
    scaler = scaler_bulan if periode == "Per Bulan" else scaler_tahun
    encoder = encoder_bulan if periode == "Per Bulan" else encoder_tahun

    # Input numerik
    input_numerik = {
        'bedroom_count': bedroom_count,
        'bathroom_count': bathroom_count,
        'building_size': building_size,
        'facility_count': facility_count,
    }

    num_df = pd.DataFrame([input_numerik])
    num_df = num_df[scaler.feature_names_in_]  # pastikan urutan sama
    num_scaled = scaler.transform(num_df)
    num_scaled_df = pd.DataFrame(num_scaled, columns=num_df.columns)

    # One-hot encoding kota & kondisi
    onehot_data = {}
    for c in list_kota:
        onehot_data[f'city_{c}'] = 1 if c == city else 0
    for cond in list_condition:
        onehot_data[f'condition_{cond}'] = 1 if cond == condition else 0

    # Target encoding
    apartemen_val = st.session_state.apartemen if not st.session_state.apartemen.startswith("(Kosongkan") else None
    agen_val = st.session_state.agen if not st.session_state.agen.startswith("(Kosongkan") else None

    if apartemen_val or agen_val:
        encode_df = pd.DataFrame([{
            'apartment_name': apartemen_val,
            'agent_name': agen_val
        }])
        encoded = encoder.transform(encode_df)
        target_enc_data = {
            'apartment_name': encoded['apartment_name'].values[0],
            'agent_name': encoded['agent_name'].values[0]
        }
    else:
        target_enc_data = {
            'apartment_name': 0,
            'agent_name': 0
        }

    # Gabungkan semua fitur
    final_input = pd.concat([
        num_scaled_df,
        pd.DataFrame([onehot_data]),
        pd.DataFrame([target_enc_data])
    ], axis=1)

    # Tambah kolom kosong jika perlu
    for col in fitur:
        if col not in final_input.columns:
            final_input[col] = 0

    final_input = final_input[fitur]  # pastikan urutan fitur sesuai

    # Prediksi
    prediksi = model.predict(final_input)[0]

    # Output
    st.success(f"Perkiraan Harga Sewa di {city}:")
    if apartemen_val:
        st.write(f"Apartemen: {apartemen_val}")
    if agen_val:
        st.write(f"Agen: {agen_val}")
    st.info(f"Rp {int(prediksi):,} {periode.lower()}")
