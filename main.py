from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Kleines deutsches Embedding-Modell laden
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 2. Beispieldokumente (z. B. aus einer behördlichen Satzung)
documents = [
    "Einfriedungen wie Hecken und Zäune dürfen eine Höhe von 1,80 Metern nicht überschreiten.",
    "Für das Halten von Hunden im Gemeindegebiet wird eine jährliche Hundesteuer erhoben.",
    "Bauanträge sind schriftlich in dreifacher Ausfertigung bei der Gemeinde einzureichen.",
    "Ruhestörender Lärm im Wohngebiet ist ab 22:00 Uhr zu unterlassen."
]

# 3. Dokumente in mathematische Vektoren umwandeln
doc_embeddings = model.encode(documents)

# 4. Suchanfrage in Alltagssprache eingeben
query = "Wie hoch darf der Gartenzaun sein?"
query_embedding = model.encode([query])

# 5. Ähnlichkeit berechnen (Cosine Similarity via Skalarprodukt)
scores = np.dot(doc_embeddings, query_embedding.T).flatten()
best_match_idx = np.argmax(scores)

print(f"Suchanfrage: {query}")
print(f"Gefundener Treffer: {documents[best_match_idx]}")
