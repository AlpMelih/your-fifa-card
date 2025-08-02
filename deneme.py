from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

df = pd.read_csv('players_info.csv')

features = ['pac', 'sho', 'pas', 'dri', 'def', 'phy']

print("Aşağıdaki özellikleri 0-100 arasında giriniz:")
user_values = []
for feature in features:
    while True:
        try:
            value = float(input(f"{feature.upper()}: "))
            if 0 <= value <= 100:
                user_values.append(value)
                break
            else:
                print("0 ile 100 arasında bir değer girin.")
        except ValueError:
            print("Geçerli bir sayı girin.")

user_input = np.array([user_values])

X = df[features].values

#Calculate cosine similarity
cos_sim = cosine_similarity(X, user_input).flatten()

#sımular first 4000 players
top_n = 4000
top_indices = np.argsort(cos_sim)[-top_n:]

# Calculate Euclidean distances for the top N similar players
distances = np.linalg.norm(X[top_indices] - user_input, axis=1)

final_idx_in_top = np.argmin(distances)
idx = top_indices[final_idx_in_top]
closest_player = df.iloc[idx]

info_cols = ['player_name', 'age', 'gender', 'club', 'league', 'nationality',
             'position', 'preffered_foot', 'height_(in cm)', 'weight_(in kg)', 'ovr']

print("\n🎯 En yakın oyuncu bilgileri:\n")
print(closest_player[info_cols])
print("\n📊 Oyuncu istatistikleri:")
print(closest_player[features])
print(f"\n🔎 Benzerlik skoru (Kosinüs): {cos_sim[idx]:.4f}")
print(f"📏 Sayısal yakınlık (Öklidyen mesafe): {distances[final_idx_in_top]:.4f}")
