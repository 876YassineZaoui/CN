from cProfile import label
import tkinter as tk
from tkinter import Entry, messagebox
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.pyplot as plt
import random
import customtkinter  

def afficher_signal():
    sequence = entree_sequence.get()
    if not sequence:
        messagebox.showerror("Erreur", "Veuillez entrer une séquence binaire.")
        return
    
    fig, ax = plt.subplots()
    ax.step(range(len(sequence)), [int(bit) for bit in sequence], where='post')
    ax.set_ylim(-0.5, 1.5)
    ax.set_title("Signal Binaire")
    ax.set_xlabel("Temps")
    ax.set_ylabel("Amplitude")
    
    popup = tk.Toplevel()
    popup.title("Signal Binaire")
    
    canvas = FigureCanvasTkAgg(fig, master=popup)
    canvas.draw()
    canvas.get_tk_widget().pack()


# les variable global pour reserver les signal (kynin F 300 o 220)
last_noisy_signal = None
last_filtered_signal = None
last_decoded_bits = None

# fonction des codeur 
def coder_rz(sequence):
    encoded = []
    for bit in sequence:
        if bit == '1':
            encoded.extend([1, 0])
        else:
            encoded.extend([0, 0])
    return encoded

def coder_nrz(sequence):
    return [1 if bit == '1' else -1 for bit in sequence]

def coder_miller(sequence):
    encoded = []
    last_level = 1
    for i, bit in enumerate(sequence):
        if bit == '1':
            if i % 2 == 0:
                encoded.append(last_level)
            else:
                last_level = -last_level
                encoded.append(last_level)
        else:
            encoded.append(encoded[-1] if encoded else last_level)
    return encoded

def coder_manchester(sequence):
    encoded = []
    for bit in sequence:
        if bit == '1':
            encoded.extend([1, -1])
        else:
            encoded.extend([-1, 1])
    return encoded

def coder_bipolaire(sequence):
    encoded = []
    last_level = 1
    for bit in sequence:
        if bit == '1':
            encoded.append(last_level)
            last_level = -last_level
        else:
            encoded.append(0)
    return encoded

def coder_hdb3(sequence):
    encoded = []
    violation = 0
    last_polarity = 1
    count_zeros = 0
    for bit in sequence:
        if bit == '1':
            if count_zeros == 4:
                encoded[-4] = last_polarity
                if violation % 2 == 0:
                    encoded.append(-last_polarity)
                else:
                    encoded[-1] = 0
                    encoded.append(last_polarity)
                violation += 1
                last_polarity = -last_polarity
            encoded.append(last_polarity)
            last_polarity = -last_polarity
            count_zeros = 0
        else:
            encoded.append(0)
            count_zeros += 1
    return encoded

def appliquer_codeur():
    sequence = entree_sequence.get()
    if not sequence:
        messagebox.showerror("Erreur", "Veuillez entrer une séquence binaire.")
        return
    
    codeur = codeur_selection.get()
    if not codeur:
        messagebox.showerror("Erreur", "Veuillez sélectionner un codeur.")
        return
    
    if codeur == "RZ":
        encoded_sequence = coder_rz(sequence)
    elif codeur == "NRZ":
        encoded_sequence = coder_nrz(sequence)
    elif codeur == "Miller":
        encoded_sequence = coder_miller(sequence)
    elif codeur == "Manchester":
        encoded_sequence = coder_manchester(sequence)
    elif codeur == "Bipolaire":
        encoded_sequence = coder_bipolaire(sequence)
    elif codeur == "HDB3":
        encoded_sequence = coder_hdb3(sequence)
    else:
        messagebox.showerror("Erreur", "Codeur inconnu.")
        return

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(encoded_sequence, drawstyle='steps-pre')
    plt.title(f'Signal codé ({codeur})')
    plt.xlabel('Temps')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    if codeur in ["NRZ", "RZ", "Manchester"]:
        f = np.linspace(-10, 10, 400)
        V = 1
        Ts = 1
        
        plt.subplot(2, 1, 2)
        if codeur == "NRZ":
            S_nrz = V**2 * Ts * (np.sinc(Ts * f))**2
            plt.plot(f, S_nrz, label="NRZ")
        elif codeur == "RZ":
            S_rz = (V**2 * Ts / 4) * (np.sinc(np.pi * Ts / 2 * f))**2
            plt.plot(f, S_rz, label="RZ")
        elif codeur == "Manchester":
            S_manchester = V**2 * Ts * (np.sin(np.pi * f * Ts / 2)**2) * (np.sinc(f * Ts)**2)
            plt.plot(f, S_manchester, label="Manchester")
        plt.title('Densité Spectrale de Puissance')
        plt.xlabel('Fréquence')
        plt.ylabel('Densité Spectrale')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def nyquist_filter(t, Ts, alpha):
    numerator = np.sin(np.pi * t / Ts) * np.cos(np.pi * alpha * t / Ts)
    denominator = (np.pi * t / Ts) * (1 - (2 * alpha * t / Ts)**2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        g_t = np.where(t == 0, 1, numerator / denominator)
        g_t[np.isnan(g_t)] = 1
    
    return g_t

def appliquer_filtre_emission():
    global last_filtered_signal

    sequence = entree_sequence.get()
    if not sequence:
        messagebox.showerror("Erreur", "Veuillez entrer une séquence binaire.")
        return
    
    codeur = codeur_selection.get()
    if not codeur:
        messagebox.showerror("Erreur", "Veuillez sélectionner un codeur.")
        return
    
    if codeur == "RZ":
        encoded_sequence = coder_rz(sequence)
    elif codeur == "NRZ":
        encoded_sequence = coder_nrz(sequence)
    elif codeur == "Manchester":
        encoded_sequence = coder_manchester(sequence)
    else:
        messagebox.showerror("Erreur", "Filtrage uniquement disponible pour NRZ, RZ et Manchester.")
        return
    
    bit_duration = 2
    sampling_rate = 0.01
    time = np.arange(0, len(encoded_sequence) * bit_duration, sampling_rate)
    signal_filtre = np.zeros_like(time)
    
    for index, bit in enumerate(encoded_sequence):
        time_index = int(index * bit_duration / sampling_rate)
        signal_filtre[time_index] = bit
    
    Ts = 2
    alpha = 0.5
    time_filter = np.arange(-6, 6, sampling_rate)
    g_t = nyquist_filter(time_filter, Ts, alpha)
    filtered_signal = np.convolve(signal_filtre, g_t, mode='same')
    
    time_convolved = np.arange(0, len(filtered_signal) * sampling_rate, sampling_rate)
    
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.stem(time, signal_filtre, linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.title('Apres filtre blanchissant ')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True, which='both', color='gray', alpha=0.5)

    plt.subplot(2, 2, 2)
    plt.plot(time_filter, g_t, 'g-')
    plt.title('Filter de Nyquist')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True, which='both', color='gray', alpha=0.5)
     
    plt.subplot(2 , 2, 3 )
    plt.plot(time_convolved[:len(filtered_signal)], filtered_signal, 'b-', marker='o', markersize=2)
    plt.title('Filtered Signal with Nyquist Filter')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True, which='both', color='gray', alpha=0.5)
    plt.ylim(-3, 3)
    
    plt.tight_layout()
    plt.show()

    last_filtered_signal = filtered_signal

def ajouter_bruit():
    global last_noisy_signal
    global last_filtered_signal

    sequence = entree_sequence.get()
    if not sequence:
        messagebox.showerror("Erreur", "Veuillez entrer une séquence binaire.")
        return
    
    codeur = codeur_selection.get()
    if not codeur:
        messagebox.showerror("Erreur", "Veuillez sélectionner un codeur.")
        return
    
    if codeur == "RZ":
        encoded_sequence = coder_rz(sequence)
    elif codeur == "NRZ":
        encoded_sequence = coder_nrz(sequence)
    elif codeur == "Manchester":
        encoded_sequence = coder_manchester(sequence)
    else:
        messagebox.showerror("Erreur", "Ajout de bruit uniquement disponible pour NRZ, RZ et Manchester.")
        return
    
    bit_duration = 2
    sampling_rate = 0.01
    time = np.arange(0, len(encoded_sequence) * bit_duration, sampling_rate)
    signal_filtre = np.zeros_like(time)
    
    for index, bit in enumerate(encoded_sequence):
        time_index = int(index * bit_duration / sampling_rate)
        signal_filtre[time_index] = bit
    
    Ts = 2
    alpha = 0.5
    time_filter = np.arange(-6, 6, sampling_rate)
    g_t = nyquist_filter(time_filter, Ts, alpha)
    filtered_signal = np.convolve(signal_filtre, g_t, mode='same')
    
    noise = np.random.normal(0, 0.25, len(filtered_signal))
    noisy_signal = filtered_signal + noise
    
    time_convolved = np.arange(0, len(noisy_signal) * sampling_rate, sampling_rate)
    
    plt.figure(figsize=(12, 8))
    plt.plot(time_convolved[:len(noisy_signal)], noisy_signal, 'b-', marker='o', markersize=2)
    plt.title('Noisy Filtered Signal with Nyquist Filter')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True, which='both', color='gray', alpha=0.5)
    plt.ylim(-3, 3)
    
    plt.tight_layout()
    plt.show()

    last_noisy_signal = noisy_signal
    last_filtered_signal = filtered_signal

def filtrer_bruit():
    global last_noisy_signal
    global last_filtered_signal

    if last_noisy_signal is None:
        messagebox.showerror("Erreur", "Veuillez d'abord ajouter du bruit.")
        return
    
    sampling_rate = 0.01
    Ts = 2
    alpha = 0.5
    time_filter = np.arange(-6, 6, sampling_rate)
    g_t = nyquist_filter(time_filter, Ts, alpha)
    filtered_signal = np.convolve(last_noisy_signal, g_t, mode='same')
    
    time_convolved = np.arange(0, len(filtered_signal) * sampling_rate, sampling_rate)
    
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(time_convolved[:len(last_filtered_signal)], last_filtered_signal, 'b-', marker='o', markersize=2)
    plt.title('Filtered Signal with Nyquist Filter (Original)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True, which='both', color='gray', alpha=0.5)
    plt.ylim(-3, 3)

    plt.subplot(2, 1, 2)
    plt.plot(time_convolved[:len(filtered_signal)], filtered_signal, 'b-', marker='o', markersize=2)
    plt.title('Filtered Noisy Signal')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True, which='both', color='gray', alpha=0.5)
    plt.ylim(-3, 3)
    
    plt.tight_layout()
    plt.show()

    last_filtered_signal = filtered_signal

def generer_sequence_aleatoire():
    sequence = ''.join(random.choice('01') for _ in range(15))
    entree_sequence.delete(0, tk.END)
    entree_sequence.insert(0, sequence)

def recuperer_horloge_binaire():
    if last_filtered_signal is None:
        messagebox.showerror("Erreur", "Veuillez d'abord appliquer le filtre d'émission.")
        return

    Ts = 2
    sampling_rate = 0.01
    time = np.arange(0, len(last_filtered_signal) * sampling_rate, sampling_rate)
    clock_signal = np.zeros_like(last_filtered_signal)

    for k in range(0, len(last_filtered_signal), int(Ts / sampling_rate)):
        clock_signal[k] = 1

    plt.figure(figsize=(10, 6))
    plt.plot(time, clock_signal, 'r-', label='Horloge Binaire')
    plt.title('Récupération de l\'Horloge Binaire')
    plt.xlabel('Temps (ms)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, which='both', color='gray', alpha=0.5)
    plt.show()

def decision():
    global last_decoded_bits

    if last_filtered_signal is None:
        messagebox.showerror("Erreur", "Veuillez d'abord appliquer le filtre de réception.")
        return

    codeur = codeur_selection.get()
    if codeur == "NRZ":
        seuil = 0.15
        Ts = 2
        sampling_rate = 0.01
    elif codeur == "RZ":
        seuil = 0.3145
        Ts = 4
        sampling_rate = 0.01
    elif codeur == "Manchester":
        seuil = -0.385
        Ts = 4
        sampling_rate = 0.01
    else:
        messagebox.showerror("Erreur", "Décision uniquement disponible pour NRZ, RZ et Manchester.")
        return

    
    bits = []
    for k in range(0, len(last_filtered_signal), int(Ts / sampling_rate)):
        sample = last_filtered_signal[k]
        if sample > seuil:
            bits.append('1')
        else:
            bits.append('0')

    last_decoded_bits = ''.join(bits)

def afficher_bits():
    if last_decoded_bits is None:
        messagebox.showerror("Erreur", "Aucun bit reçu. Veuillez effectuer la décision d'abord.")
        return

    messagebox.showinfo("Bits Reçus", f"Bits reçus : {last_decoded_bits}")

# GUI 
root = customtkinter.CTk()
root.title("Communication Numérique")
root.geometry("800x700")

tk.Label(root, text="CHAINE DE TRANSMISSION BINAIRE", font = ('Arial' , 20)).pack(padx=20 , pady= 20 )


# les sequence
tk.Label(root, text="Séquence Binaire:" , font=('italica' , 13)).pack(padx=20 , pady= 20 )
entree_sequence =  Entry(root , width=30,font=('arial' , 12) )
entree_sequence.pack()

# l aleatoire
button_generer_sequence = customtkinter.CTkButton(root, text="Générer Séquence Aléatoire", command=generer_sequence_aleatoire)
button_generer_sequence.pack(pady=10 )

#  signal d'entrer 
btn_signal = customtkinter.CTkButton(root , text="Signal" , command=afficher_signal).pack(padx = 20 , pady =20)


# selection du codeur
tk.Label(root, text="Selectionner un codeur:" , font=('italica' , 13)).pack(padx=20 , pady= 20 )
codeur_selection = tk.StringVar()
codeur_selection.set("NRZ")
codeurs = ["NRZ", "RZ", "Manchester", "Miller", "Bipolaire", "HDB3"]
customtkinter.CTkOptionMenu(root, variable=codeur_selection, values=codeurs).pack()

# button codeur
button_encoder = customtkinter.CTkButton(root, text="Appliquer le Codeur", command=appliquer_codeur)
button_encoder.pack(pady=10)

# Filter emission button
button_filtre_emission = customtkinter.CTkButton(root, text="Filtre d'Emission", command=appliquer_filtre_emission)
button_filtre_emission.pack(pady=10)

# ajouter le bruit  button
button_ajouter_bruit = customtkinter.CTkButton(root, text="Ajouter du Bruit", command=ajouter_bruit)
button_ajouter_bruit.pack(pady=10)

# Filter noise button
button_filtrer_bruit = customtkinter.CTkButton(root, text="Filtrer le Bruit", command=filtrer_bruit)
button_filtrer_bruit.pack(pady=10)

# horloge button
button_recuperer_horloge = customtkinter.CTkButton(root, text="Récupération d'Horloge", command=recuperer_horloge_binaire)
button_recuperer_horloge.pack(pady=10)

# Decision button
button_decision = customtkinter.CTkButton(root, text="Décision", command=decision)
button_decision.pack(pady=10)

#  bits reçu button
button_afficher_bits = customtkinter.CTkButton(root, text="Afficher les Bits Reçus", command=afficher_bits)
button_afficher_bits.pack(pady=10)

root.mainloop()
