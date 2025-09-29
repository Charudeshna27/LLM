# LLM
Ayurvedic Medicine Recommendation System

# Install required packages
!pip install -q transformers torch sentence-transformers faiss-cpu

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google.colab import files
from typing import List, Dict

# Function to upload the JSON file
def upload_json_file():
    uploaded = files.upload()
    for filename in uploaded.keys():
        line_count = uploaded[filename].decode("utf-8").count('\n') + 1
        print(f'Uploaded {filename} ({line_count} lines)')
        return filename
    return None

# Upload your dataset
print("Please upload your ayurveda_medicine_dataset.json file")
dataset_path = upload_json_file()

if not dataset_path:
    raise ValueError("No file was uploaded. Please try again.")

# Load the dataset
with open(dataset_path, 'r') as f:
    dataset = json.load(f)

print(f"\nLoaded dataset with {len(dataset)} entries")

# Initialize the sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Preprocess the dataset and create embeddings
print("\nCreating embeddings for diseases and symptoms...")
disease_embeddings = []
symptom_embeddings = []
processed_data = []

for entry in dataset:
    # Create disease embedding
    disease_text = entry["disease"]
    disease_embedding = model.encode(disease_text)
    disease_embeddings.append(disease_embedding)

    # Create symptoms embedding (combine all symptoms)
    symptoms_text = " ".join(entry["symptoms"])
    symptom_embedding = model.encode(symptoms_text)
    symptom_embeddings.append(symptom_embedding)

    # Store processed data with embeddings
    processed_entry = {
        "original_data": entry,
        "disease_embedding": disease_embedding,
        "symptom_embedding": symptom_embedding
    }
    processed_data.append(processed_entry)

# Convert to numpy arrays
disease_embeddings = np.array(disease_embeddings)
symptom_embeddings = np.array(symptom_embeddings)

# Create FAISS indices for fast similarity search
print("Building search indices...")
disease_index = faiss.IndexFlatL2(disease_embeddings.shape[1])
disease_index.add(disease_embeddings)

symptom_index = faiss.IndexFlatL2(symptom_embeddings.shape[1])
symptom_index.add(symptom_embeddings)

print("ayurveda Medicine Recommendation System ready!\n")

def recommend_ayurveda(input_text: str, search_by: str = "both", top_k: int = 3) -> List[Dict]:
    """Get ayurveda recommendations based on input text"""
    input_embedding = model.encode(input_text)
    results = []

    if search_by in ["disease", "both"]:
        D, I = disease_index.search(np.array([input_embedding]), top_k)
        for dist, idx in zip(D[0], I[0]):
            entry = processed_data[idx]["original_data"]
            results.append({
                "disease": entry["disease"],
                "symptoms": entry["symptoms"],
                "medicine": entry["medicine"],
                "dosage": entry["dosage"],
                "food_to_prefer": entry["food_to_prefer"],
                "food_to_avoid": entry["food_to_avoid"],
                "score": float(1 - dist/10),  # Convert distance to similarity score
                "match_type": "disease"
            })

    if search_by in ["symptoms", "both"]:
        D, I = symptom_index.search(np.array([input_embedding]), top_k)
        for dist, idx in zip(D[0], I[0]):
            entry = processed_data[idx]["original_data"]
            results.append({
                "disease": entry["disease"],
                "symptoms": entry["symptoms"],
                "medicine": entry["medicine"],
                "dosage": entry["dosage"],
                "food_to_prefer": entry["food_to_prefer"],
                "food_to_avoid": entry["food_to_avoid"],
                "score": float(1 - dist/10),  # Convert distance to similarity score
                "match_type": "symptoms"
            })

    # Deduplicate and sort results
    results.sort(key=lambda x: x["score"], reverse=True)
    seen = set()
    return [x for x in results if not (x["disease"] in seen or seen.add(x["disease"]))][:top_k]

def format_recommendation(result: Dict) -> str:
    """Format recommendation into readable string"""
    return (
        f"\n**Disease:** {result['disease']}\n"
        f"**Matched by:** {result['match_type']} (confidence: {result['score']:.2f})\n"
        f"**Symptoms:** {', '.join(result['symptoms'])}\n"
        f"**Medicine:** {result['medicine']}\n"
        f"**Dosage:** {result['dosage']}\n"
        f"**Foods to prefer:** {', '.join(result['food_to_prefer'])}\n"
        f"**Foods to avoid:** {', '.join(result['food_to_avoid'])}\n"
    )

def get_ayurveda_recommendation():
    print(" ayuervedic Medicine Recommendation System ")
    print("------------------------------------------")
    print("Type symptoms or disease name for recommendations")
    print("Type 'exit' or 'quit' to end session\n")

    while True:
        user_input = input("\nDescribe your condition: ").strip()

        if user_input.lower() in ['exit', 'quit']:
            print("\nThank you for using the ayurveda Medicine system! May you have good health! ")
            return

        if not user_input:
            print("Please enter symptoms or a disease name.")
            continue

        print("\nAnalyzing your condition with ayurveda medicine principles...")
        recommendations = recommend_ayurveda(user_input)

        if not recommendations:
            print("\nNo specific recommendations found. Please consult a Hakim (ayurveda practitioner).")
            continue

        print(f"\nHere are your ayurveda medicine recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f" Recommendation {i}:")
            print(format_recommendation(rec))

# Run the system
get_ayurveda_recommendation()
