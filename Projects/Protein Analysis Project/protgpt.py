import os
import shutil
import tkinter as tk
from tkinter import messagebox

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModel,
    T5Tokenizer, T5ForConditionalGeneration
)
from sentence_transformers import util
from colabfold.batch import run

# === Model Initialization ===
print("Loading models...")

protgpt2_tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
protgpt2_model = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2")

protbert_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
protbert_model = AutoModel.from_pretrained("Rostlab/prot_bert")

t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# === Global Variables ===
protein_a, protein_b = "", ""
output_dir = "protein_structures"
fasta_dir = "fasta_seqs"

# === Functional Components ===
def generate_protein_sequence():
    inputs = protgpt2_tokenizer("<|startoftext|>", return_tensors="pt")
    outputs = protgpt2_model.generate(
        inputs["input_ids"],
        do_sample=True,
        top_k=950,
        temperature=1.0,
        max_length=200,
        repetition_penalty=1.2
    )
    return protgpt2_tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ", "")

def compute_embedding(sequence):
    spaced_seq = " ".join(list(sequence))
    inputs = protbert_tokenizer(spaced_seq, return_tensors="pt")
    with torch.no_grad():
        embedding = protbert_model(**inputs).last_hidden_state.mean(dim=1)
    return embedding

def save_as_fasta(name, sequence):
    os.makedirs(fasta_dir, exist_ok=True)
    path = os.path.join(fasta_dir, f"{name}.fasta")
    with open(path, "w") as file:
        file.write(f">{name}\n{sequence}")
    return path

def predict_structure_from_fasta():
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    run(fasta_dir, output_dir, use_templates=False, use_amber=False, model_type="alphafold2", is_complex=False)

def compare_embeddings_and_explain():
    global protein_a, protein_b

    if not protein_a or not protein_b:
        messagebox.showwarning("Input Error", "Please generate both protein sequences first.")
        return

    emb1, emb2 = compute_embedding(protein_a), compute_embedding(protein_b)
    similarity = util.cos_sim(emb1, emb2).item()

    prompt = f"Explain how this protein differs:\nProtein A: {protein_a}\nProtein B: {protein_b}"
    input_ids = t5_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids
    output_ids = t5_model.generate(input_ids, max_length=100)
    explanation = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Display Results
    result_text.config(state="normal")
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, f"Cosine Similarity: {similarity:.4f}\n\n")
    result_text.insert(tk.END, "Explanation:\n" + explanation + "\n\n")
    result_text.insert(tk.END, "Running AlphaFold2 prediction...\n")
    result_text.config(state="disabled")
    app.update()

    save_as_fasta("ProteinA", protein_a)
    save_as_fasta("ProteinB", protein_b)
    predict_structure_from_fasta()

    result_text.config(state="normal")
    result_text.insert(tk.END, "Structure prediction complete.\nCheck 'protein_structures' folder.\n")
    result_text.config(state="disabled")

# === GUI Functions ===
def set_protein_a():
    global protein_a
    protein_a = generate_protein_sequence()
    entry_a.delete("1.0", tk.END)
    entry_a.insert(tk.END, protein_a)

def set_protein_b():
    global protein_b
    protein_b = generate_protein_sequence()
    entry_b.delete("1.0", tk.END)
    entry_b.insert(tk.END, protein_b)

# === GUI Layout ===
app = tk.Tk()
app.title("Protein Generator + Structure Predictor")
app.geometry("920x720")

main_frame = tk.Frame(app)
main_frame.pack(pady=10)

tk.Label(main_frame, text="Protein A:", font=("Arial", 12)).grid(row=0, column=0, sticky="w", padx=5)
tk.Label(main_frame, text="Protein B:", font=("Arial", 12)).grid(row=1, column=0, sticky="w", padx=5)

entry_a = tk.Text(main_frame, width=85, height=4)
entry_b = tk.Text(main_frame, width=85, height=4)
entry_a.grid(row=0, column=1, padx=5)
entry_b.grid(row=1, column=1, padx=5)

btn_frame = tk.Frame(app)
btn_frame.pack(pady=15)

tk.Button(btn_frame, text="Generate Protein A", command=set_protein_a, width=20, bg="#ADD8E6").grid(row=0, column=0, padx=10)
tk.Button(btn_frame, text="Generate Protein B", command=set_protein_b, width=20, bg="#90EE90").grid(row=0, column=1, padx=10)
tk.Button(btn_frame, text="Compare + Predict", command=compare_embeddings_and_explain, width=25, bg="#FFA500").grid(row=0, column=2, padx=10)

tk.Label(app, text="Comparison Results", font=("Arial", 14)).pack(pady=5)
result_text = tk.Text(app, width=110, height=18, font=("Courier", 10), state="disabled")
result_text.pack()

# === Run the App ===
app.mainloop()
