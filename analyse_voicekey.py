import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy.io import wavfile
from scipy.signal.windows import hann
from scipy.interpolate import interp1d
import os
import json
import threading  # CHANGE 3: needed for audio thread
import voice_onset_core as core
import sv_ttk
from ui_widgets import add_tooltip, CollapsibleFrame

SESSION_FILENAME = "voicekey_session.json"


class NumpyJSONEncoder(json.JSONEncoder):
    """Allows json.dump to handle the numpy scalar types that flow into
    file_specific_params/results (onset/offset times are numpy float64,
    since they come from indexing a numpy array in voice_onset_core)."""

    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    pyaudio = None
    PYAUDIO_AVAILABLE = False

PARAM_TOOLTIPS = {
    'nSD': "Combien de fois l'écart-type du bruit de fond le signal doit dépasser pour déclencher la détection. "
           "Plus élevé = moins de fausses détections mais risque de manquer un début de parole faible.",
    'minTH': "Seuil plancher exprimé en pourcentage de l'amplitude maximale du fichier. Évite les faux déclenchements "
             "dans un enregistrement très calme où l'écart-type du bruit est quasi nul.",
    'min_duration': "Durée pendant laquelle le signal doit rester au-dessus (onset) / en dessous (offset) du seuil "
                    "pour confirmer la détection. Évite qu'un simple clic ou bruit parasite soit pris pour un début de parole.",
    'noise_window': "Portion du début de l'enregistrement utilisée pour estimer le niveau de bruit de fond. "
                    "Doit correspondre à une zone réellement silencieuse.",
    'search_delay': "Temps ignoré avant de commencer la recherche d'un onset — utile pour éviter de détecter un bip "
                    "de consigne ou un bruit de démarrage.",
    'frame_hop': "Paramètres techniques de calcul de l'enveloppe d'amplitude (taille de fenêtre d'analyse et pas entre "
                 "fenêtres). Les valeurs par défaut conviennent à la plupart des enregistrements de parole.",
    'highpass': "Supprime les basses fréquences (bourdonnement, bruit de manipulation du micro) sous la fréquence "
                "choisie, pour améliorer la détection.",
    'manual_threshold': "Amplitude fixe au-dessus/en dessous de laquelle l'onset/offset est déclenché en mode manuel.",
}


class VoiceKeyAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Analyseur Voice Key - PsychoPy (Improved)")
        self.root.geometry("1400x900")
        
        # Variables
        self.directory = None
        self.wav_files = []
        self.current_index = 0
        self.audio_data = None
        self.original_audio_data = None  # Store original before cut
        self.sample_rate = None
        self.filtered_data = None
        self.is_cut_applied = False  # Track if current file is cut
        
        # Résultats
        self.results = []
        
        # Variables de lecture audio
        self.is_playing = False
        self.audio_thread = None

        # CHANGE 3: Playback position tracking
        self.playback_position_sec = 0.0
        self._playback_line1 = None
        self._playback_line2 = None

        # CHANGE 1: Dragging state for manual markers
        self._dragging = None  # 'onset', 'offset', or None
        self._onset_vline = None   # reference to onset axvline on ax2
        self._offset_vline = None  # reference to offset axvline on ax2
        
        # Paramètres par défaut de l'algorithme VOAT
        self.default_params = {
            'nSD': 4.0,
            'minTH': 0.20,
            'min_duration_ms': 20,
            'noise_window_ms': 50,
            'search_delay_ms': 100,
            'frame_length_ms': 25,
            'hop_length_ms': 10,
            'manual_onset_threshold': 0.02,
            'manual_offset_threshold': 0.01,
            'noise_reduction': True,
            'highpass_freq': 80,
            'offset_min_duration_ms': 20
        }
        
        # Paramètres de l'algorithme VOAT
        self.nSD = tk.DoubleVar(value=self.default_params['nSD'])
        self.minTH = tk.DoubleVar(value=self.default_params['minTH'])
        self.min_duration_ms = tk.DoubleVar(value=self.default_params['min_duration_ms'])
        self.noise_window_ms = tk.DoubleVar(value=self.default_params['noise_window_ms'])
        self.search_delay_ms = tk.DoubleVar(value=self.default_params['search_delay_ms'])
        
        # Frame parameters for RMS envelope
        self.frame_length_ms = tk.DoubleVar(value=self.default_params['frame_length_ms'])
        self.hop_length_ms = tk.DoubleVar(value=self.default_params['hop_length_ms'])
        
        self.offset_min_duration_ms = tk.DoubleVar(value=self.default_params['offset_min_duration_ms'])
        
        # Mode adaptatif
        self.adaptive_threshold = tk.BooleanVar(value=True)
        
        # Seuils manuels
        self.manual_onset_threshold = tk.DoubleVar(value=self.default_params['manual_onset_threshold'])
        self.manual_offset_threshold = tk.DoubleVar(value=self.default_params['manual_offset_threshold'])
        
        # Manual mode search delay
        self.manual_search_delay_ms = tk.DoubleVar(value=self.default_params['search_delay_ms'])
        
        # Stockage des paramètres par fichier
        self.file_specific_params = {}
        
        # Filtrage
        self.noise_reduction = tk.BooleanVar(value=self.default_params['noise_reduction'])
        self.highpass_freq = tk.DoubleVar(value=self.default_params['highpass_freq'])
        
        # Variables pour la découpe
        self.cut_start_ms = tk.DoubleVar(value=0)
        self.cut_end_ms = tk.DoubleVar(value=0)
        self.cut_markers = {'start': None, 'end': None}
        
        # Résultats de détection
        self.detection_result = None
        self.onset_time = None
        self.offset_time = None
        self.onset_sample = None
        self.offset_sample = None
        self.rms_envelope = None
        self.envelope_times = None

        # Correction manuelle (glisser-déposer), indépendante du mode
        self.manual_onset_override = None
        self.manual_offset_override = None

        # Cache for performance
        self.cached_envelope = None
        self.cached_envelope_params = None

        # CHANGE 9: explicit generation counter instead of id(self.audio_data) for cache keys
        self._audio_generation = 0

        # Batch-processing state
        self._batch_mode = False
        self._batch_warnings = []
        self._suppress_batch_warnings = False

        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        """Enregistre la session avant de fermer l'application."""
        self.save_session()
        self.root.destroy()

    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # column 0 = fixed left panel, column 1 = expanding plot
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=0)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # === LEFT SCROLLABLE PANEL ===
        left_outer = ttk.Frame(main_frame)
        left_outer.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E), padx=(0, 5))
        left_outer.rowconfigure(0, weight=1)
        left_outer.columnconfigure(0, weight=1)

        # Canvas that enables vertical scrolling of the whole left column
        self._left_canvas = tk.Canvas(left_outer, width=520, highlightthickness=0)
        self._left_canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))

        left_scrollbar = ttk.Scrollbar(left_outer, orient=tk.VERTICAL,
                                       command=self._left_canvas.yview)
        left_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self._left_canvas.configure(yscrollcommand=left_scrollbar.set)

        # Inner frame — all existing widgets go in here exactly as before
        left_inner = ttk.Frame(self._left_canvas)
        self._left_canvas_window = self._left_canvas.create_window(
            (0, 0), window=left_inner, anchor='nw')

        # Update scrollregion when inner frame resizes
        def _on_left_inner_configure(event):
            self._left_canvas.configure(
                scrollregion=self._left_canvas.bbox("all"))
        left_inner.bind('<Configure>', _on_left_inner_configure)

        # Keep inner frame width matched to canvas width
        def _on_left_canvas_configure(event):
            self._left_canvas.itemconfig(
                self._left_canvas_window, width=event.width)
        self._left_canvas.bind('<Configure>', _on_left_canvas_configure)

        # Mouse-wheel scrolling (Windows & Linux)
        def _on_mousewheel(event):
            self._left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _on_mousewheel_up(event):
            self._left_canvas.yview_scroll(-1, "units")
        def _on_mousewheel_down(event):
            self._left_canvas.yview_scroll(1, "units")
        self._left_canvas.bind_all('<MouseWheel>', _on_mousewheel)
        self._left_canvas.bind_all('<Button-4>', _on_mousewheel_up)
        self._left_canvas.bind_all('<Button-5>', _on_mousewheel_down)

        # === SECTION 1: Fichiers ===
        files_frame = ttk.LabelFrame(left_inner, text="Fichiers", padding="5")
        files_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        files_frame.columnconfigure(0, weight=1)

        # Bouton sur sa propre ligne: toujours à une position fixe et cliquable,
        # même si le chemin du répertoire choisi est long.
        ttk.Button(files_frame, text="Parcourir...", command=self.select_directory).grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=(0, 2))

        self.dir_label = ttk.Label(files_frame, text="Aucun répertoire sélectionné", wraplength=380)
        self.dir_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=(0, 5))

        tree_frame = ttk.Frame(files_frame)
        tree_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        tree_frame.columnconfigure(0, weight=1)

        self.file_tree = ttk.Treeview(tree_frame, columns=("status",), show="tree headings", height=10)
        self.file_tree.heading("#0", text="Fichier")
        self.file_tree.heading("status", text="Statut")
        self.file_tree.column("#0", width=140)
        self.file_tree.column("status", width=90, anchor=tk.CENTER)
        self.file_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        file_tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.file_tree.yview)
        file_tree_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.file_tree.configure(yscrollcommand=file_tree_scroll.set)

        self.file_tree.tag_configure('reviewed', foreground='#1a7f37')
        self.file_tree.tag_configure('customized', foreground='#b8860b')
        self.file_tree.tag_configure('unreviewed', foreground='')

        self._syncing_tree = False
        self.file_tree.bind('<<TreeviewSelect>>', self._on_file_tree_select)

        # === SECTION 2: Contrôles (all existing widgets unchanged) ===
        control_frame = ttk.LabelFrame(left_inner, text="Contrôles", padding="5")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N), padx=5, pady=5)
        
        # Navigation et lecture
        nav_frame = ttk.Frame(control_frame)
        nav_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        ttk.Button(nav_frame, text="◄◄ Précédent", command=self.previous_file).pack(
            side=tk.LEFT, padx=2)
        self.play_button = ttk.Button(nav_frame, text="▶ Jouer", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=2)
        if not PYAUDIO_AVAILABLE:
            self.play_button.config(state='disabled', text="▶ Jouer (indisponible: PortAudio manquant)")
        ttk.Button(nav_frame, text="Suivant ►►", command=self.next_file).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="🔄 Reset", command=self.reset_to_default).pack(
            side=tk.LEFT, padx=2)
        
        self.file_label = ttk.Label(control_frame, text="Fichier: -")
        self.file_label.grid(row=1, column=0, columnspan=2, pady=5)

        self.channel_warning_label = ttk.Label(control_frame, text="", foreground='#cc6600', wraplength=380)
        self.channel_warning_label.grid(row=2, column=0, columnspan=2, pady=(0, 5), sticky=tk.W)

        # Cut section
        cut_frame = ttk.LabelFrame(control_frame, text="Découpage du fichier", padding="5")
        cut_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(cut_frame, text="Début (ms):").grid(row=0, column=0, sticky=tk.W, pady=2)
        cut_start_entry = ttk.Entry(cut_frame, textvariable=self.cut_start_ms, width=10)
        cut_start_entry.grid(row=0, column=1, sticky=tk.W, pady=2, padx=5)
        
        ttk.Label(cut_frame, text="Fin (ms, 0=jusqu'à la fin):").grid(row=0, column=2, sticky=tk.W, pady=2, padx=(10,0))
        cut_end_entry = ttk.Entry(cut_frame, textvariable=self.cut_end_ms, width=10)
        cut_end_entry.grid(row=0, column=3, sticky=tk.W, pady=2, padx=5)
        
        ttk.Button(cut_frame, text="Appliquer la découpe", command=self.apply_cut).grid(
            row=1, column=0, columnspan=2, pady=5, sticky=tk.W)
        ttk.Button(cut_frame, text="Annuler la découpe", command=self.undo_cut).grid(
            row=1, column=2, columnspan=1, pady=5, sticky=tk.W)
        ttk.Button(cut_frame, text="Sauvegarder fichier découpé", command=self.save_cut_file).grid(
            row=2, column=0, columnspan=4, pady=5, sticky=tk.W)
        
        self.cut_status_label = ttk.Label(cut_frame, text="", foreground='blue')
        self.cut_status_label.grid(row=3, column=0, columnspan=4, sticky=tk.W)
        
        # Mode de détection
        mode_frame = ttk.LabelFrame(control_frame, text="Mode de détection", padding="5")
        mode_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Checkbutton(mode_frame, text="Seuils adaptatifs (VOAT recommandé)", 
                       variable=self.adaptive_threshold,
                       command=self.toggle_adaptive_mode).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # Paramètres adaptatifs (VOAT)
        adaptive_params = ttk.LabelFrame(control_frame, text="Paramètres adaptatifs (VOAT)", padding="5")
        adaptive_params.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        lbl = ttk.Label(adaptive_params, text="nSD (écarts-types au-dessus du bruit):")
        lbl.grid(row=0, column=0, sticky=tk.W, pady=2)
        add_tooltip(lbl, PARAM_TOOLTIPS['nSD'])
        nsd_frame = ttk.Frame(adaptive_params)
        nsd_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        self.nsd_scale = ttk.Scale(nsd_frame, from_=1.5, to=8.0,
                 variable=self.nSD, orient=tk.HORIZONTAL,
                 command=lambda x: self.update_analysis())
        self.nsd_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.nsd_label = ttk.Label(nsd_frame, text=f"{self.nSD.get():.1f}")
        self.nsd_label.pack(side=tk.LEFT, padx=5)
        self.nSD.trace_add('write', lambda *args: self.nsd_label.config(text=f"{self.nSD.get():.1f}"))

        lbl = ttk.Label(adaptive_params, text="minTH (% du pic):")
        lbl.grid(row=1, column=0, sticky=tk.W, pady=2)
        add_tooltip(lbl, PARAM_TOOLTIPS['minTH'])
        minth_frame = ttk.Frame(adaptive_params)
        minth_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        self.minth_scale = ttk.Scale(minth_frame, from_=0.05, to=0.40,
                 variable=self.minTH, orient=tk.HORIZONTAL,
                 command=lambda x: self.update_analysis())
        self.minth_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.minth_label = ttk.Label(minth_frame, text=f"{self.minTH.get():.2f}")
        self.minth_label.pack(side=tk.LEFT, padx=5)
        self.minTH.trace_add('write', lambda *args: self.minth_label.config(text=f"{self.minTH.get():.2f}"))

        lbl = ttk.Label(adaptive_params, text="Durée minimale onset (ms):")
        lbl.grid(row=2, column=0, sticky=tk.W, pady=2)
        add_tooltip(lbl, PARAM_TOOLTIPS['min_duration'])
        mindur_frame = ttk.Frame(adaptive_params)
        mindur_frame.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=2)
        self.mindur_scale = ttk.Scale(mindur_frame, from_=10, to=100,
                 variable=self.min_duration_ms, orient=tk.HORIZONTAL,
                 command=lambda x: self.update_analysis())
        self.mindur_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.mindur_label = ttk.Label(mindur_frame, text=f"{self.min_duration_ms.get():.0f}")
        self.mindur_label.pack(side=tk.LEFT, padx=5)
        self.min_duration_ms.trace_add('write', lambda *args: self.mindur_label.config(text=f"{self.min_duration_ms.get():.0f}"))

        lbl = ttk.Label(adaptive_params, text="Durée minimale offset (ms):")
        lbl.grid(row=3, column=0, sticky=tk.W, pady=2)
        add_tooltip(lbl, PARAM_TOOLTIPS['min_duration'])
        offset_dur_frame = ttk.Frame(adaptive_params)
        offset_dur_frame.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=2)
        self.offset_dur_scale = ttk.Scale(offset_dur_frame, from_=10, to=100,
                 variable=self.offset_min_duration_ms, orient=tk.HORIZONTAL,
                 command=lambda x: self.update_analysis())
        self.offset_dur_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.offset_dur_label = ttk.Label(offset_dur_frame, text=f"{self.offset_min_duration_ms.get():.0f}")
        self.offset_dur_label.pack(side=tk.LEFT, padx=5)
        self.offset_min_duration_ms.trace_add('write', lambda *args: self.offset_dur_label.config(text=f"{self.offset_min_duration_ms.get():.0f}"))

        adaptive_advanced = CollapsibleFrame(adaptive_params, "Paramètres avancés")
        adaptive_advanced.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

        lbl = ttk.Label(adaptive_advanced.body, text="Fenêtre bruit initial (ms):")
        lbl.grid(row=0, column=0, sticky=tk.W, pady=2)
        add_tooltip(lbl, PARAM_TOOLTIPS['noise_window'])
        noise_frame = ttk.Frame(adaptive_advanced.body)
        noise_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        self.noise_scale = ttk.Scale(noise_frame, from_=20, to=200,
                 variable=self.noise_window_ms, orient=tk.HORIZONTAL,
                 command=lambda x: self.update_analysis())
        self.noise_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.noise_win_label = ttk.Label(noise_frame, text=f"{self.noise_window_ms.get():.0f}")
        self.noise_win_label.pack(side=tk.LEFT, padx=5)
        self.noise_window_ms.trace_add('write', lambda *args: self.noise_win_label.config(text=f"{self.noise_window_ms.get():.0f}"))

        lbl = ttk.Label(adaptive_advanced.body, text="Délai avant recherche (ms):")
        lbl.grid(row=1, column=0, sticky=tk.W, pady=2)
        add_tooltip(lbl, PARAM_TOOLTIPS['search_delay'])
        delay_frame = ttk.Frame(adaptive_advanced.body)
        delay_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        self.delay_scale = ttk.Scale(delay_frame, from_=0, to=300,
                 variable=self.search_delay_ms, orient=tk.HORIZONTAL,
                 command=lambda x: self.update_analysis())
        self.delay_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.delay_label = ttk.Label(delay_frame, text=f"{self.search_delay_ms.get():.0f}")
        self.delay_label.pack(side=tk.LEFT, padx=5)
        self.search_delay_ms.trace_add('write', lambda *args: self.delay_label.config(text=f"{self.search_delay_ms.get():.0f}"))

        self.threshold_info_label = ttk.Label(adaptive_params, text="", foreground='blue')
        self.threshold_info_label.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Paramètres manuels
        manual_frame = ttk.LabelFrame(control_frame, text="Seuils manuels (amplitude)", padding="5")
        manual_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        lbl = ttk.Label(manual_frame, text="Seuil onset:")
        lbl.grid(row=0, column=0, sticky=tk.W, pady=2)
        add_tooltip(lbl, PARAM_TOOLTIPS['manual_threshold'])
        manual_onset_frame = ttk.Frame(manual_frame)
        manual_onset_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        self.manual_onset_scale = ttk.Scale(manual_onset_frame, from_=0.001, to=0.1,
                               variable=self.manual_onset_threshold, orient=tk.HORIZONTAL,
                               command=lambda x: self.update_analysis())
        self.manual_onset_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.manual_onset_label = ttk.Label(manual_onset_frame, text=f"{self.manual_onset_threshold.get():.3f}")
        self.manual_onset_label.pack(side=tk.LEFT, padx=5)
        self.manual_onset_threshold.trace_add('write', lambda *args: self.manual_onset_label.config(text=f"{self.manual_onset_threshold.get():.3f}"))

        lbl = ttk.Label(manual_frame, text="Seuil offset:")
        lbl.grid(row=1, column=0, sticky=tk.W, pady=2)
        add_tooltip(lbl, PARAM_TOOLTIPS['manual_threshold'])
        manual_offset_frame = ttk.Frame(manual_frame)
        manual_offset_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        self.manual_offset_scale = ttk.Scale(manual_offset_frame, from_=0.001, to=0.1,
                                variable=self.manual_offset_threshold, orient=tk.HORIZONTAL,
                                command=lambda x: self.update_analysis())
        self.manual_offset_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.manual_offset_label = ttk.Label(manual_offset_frame, text=f"{self.manual_offset_threshold.get():.3f}")
        self.manual_offset_label.pack(side=tk.LEFT, padx=5)
        self.manual_offset_threshold.trace_add('write', lambda *args: self.manual_offset_label.config(text=f"{self.manual_offset_threshold.get():.3f}"))

        lbl = ttk.Label(manual_frame, text="Durée minimale onset (ms):")
        lbl.grid(row=2, column=0, sticky=tk.W, pady=2)
        add_tooltip(lbl, PARAM_TOOLTIPS['min_duration'])
        manual_onset_dur_frame = ttk.Frame(manual_frame)
        manual_onset_dur_frame.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=2)
        self.manual_onset_dur_scale = ttk.Scale(manual_onset_dur_frame, from_=10, to=100,
                 variable=self.min_duration_ms, orient=tk.HORIZONTAL,
                 command=lambda x: self.update_analysis())
        self.manual_onset_dur_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.manual_onset_dur_label = ttk.Label(manual_onset_dur_frame, text=f"{self.min_duration_ms.get():.0f}")
        self.manual_onset_dur_label.pack(side=tk.LEFT, padx=5)
        self.min_duration_ms.trace_add('write', lambda *args: self.manual_onset_dur_label.config(text=f"{self.min_duration_ms.get():.0f}"))

        lbl = ttk.Label(manual_frame, text="Durée minimale offset (ms):")
        lbl.grid(row=3, column=0, sticky=tk.W, pady=2)
        add_tooltip(lbl, PARAM_TOOLTIPS['min_duration'])
        manual_offset_dur_frame = ttk.Frame(manual_frame)
        manual_offset_dur_frame.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=2)
        self.manual_offset_dur_scale = ttk.Scale(manual_offset_dur_frame, from_=10, to=100,
                 variable=self.offset_min_duration_ms, orient=tk.HORIZONTAL,
                 command=lambda x: self.update_analysis())
        self.manual_offset_dur_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.manual_offset_dur_label = ttk.Label(manual_offset_dur_frame, text=f"{self.offset_min_duration_ms.get():.0f}")
        self.manual_offset_dur_label.pack(side=tk.LEFT, padx=5)
        self.offset_min_duration_ms.trace_add('write', lambda *args: self.manual_offset_dur_label.config(text=f"{self.offset_min_duration_ms.get():.0f}"))

        manual_advanced = CollapsibleFrame(manual_frame, "Paramètres avancés")
        manual_advanced.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

        lbl = ttk.Label(manual_advanced.body, text="Délai avant recherche (ms):")
        lbl.grid(row=0, column=0, sticky=tk.W, pady=2)
        add_tooltip(lbl, PARAM_TOOLTIPS['search_delay'])
        manual_delay_frame = ttk.Frame(manual_advanced.body)
        manual_delay_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        self.manual_delay_scale = ttk.Scale(manual_delay_frame, from_=0, to=300,
                 variable=self.manual_search_delay_ms, orient=tk.HORIZONTAL,
                 command=lambda x: self.update_analysis())
        self.manual_delay_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.manual_delay_label = ttk.Label(manual_delay_frame, text=f"{self.manual_search_delay_ms.get():.0f}")
        self.manual_delay_label.pack(side=tk.LEFT, padx=5)
        self.manual_search_delay_ms.trace_add('write', lambda *args: self.manual_delay_label.config(text=f"{self.manual_search_delay_ms.get():.0f}"))

        # Filtrage
        filter_frame = ttk.LabelFrame(control_frame, text="Filtrage (appliqué à tous les modes)", padding="5")
        filter_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Checkbutton(filter_frame, text="Filtre passe-haut",
                       variable=self.noise_reduction,
                       command=self.update_analysis).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=2)

        lbl = ttk.Label(filter_frame, text="Fréquence coupure (Hz):")
        lbl.grid(row=1, column=0, sticky=tk.W, pady=2)
        add_tooltip(lbl, PARAM_TOOLTIPS['highpass'])
        highpass_frame = ttk.Frame(filter_frame)
        highpass_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        self.highpass_scale = ttk.Scale(highpass_frame, from_=50, to=300,
                 variable=self.highpass_freq, orient=tk.HORIZONTAL,
                 command=lambda x: self.update_analysis())
        self.highpass_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.highpass_label = ttk.Label(highpass_frame, text=f"{self.highpass_freq.get():.0f}")
        self.highpass_label.pack(side=tk.LEFT, padx=5)
        self.highpass_freq.trace_add('write', lambda *args: self.highpass_label.config(text=f"{self.highpass_freq.get():.0f}"))

        filter_advanced = CollapsibleFrame(filter_frame, "Paramètres avancés")
        filter_advanced.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

        # CHANGE 4c: Add frame_length_ms and hop_length_ms sliders in filter section
        lbl = ttk.Label(filter_advanced.body, text="Longueur trame RMS (ms):")
        lbl.grid(row=0, column=0, sticky=tk.W, pady=2)
        add_tooltip(lbl, PARAM_TOOLTIPS['frame_hop'])
        frame_len_frame = ttk.Frame(filter_advanced.body)
        frame_len_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        self.frame_len_scale = ttk.Scale(frame_len_frame, from_=10, to=50,
                 variable=self.frame_length_ms, orient=tk.HORIZONTAL,
                 command=lambda x: [self.invalidate_cache(), self.update_analysis()])
        self.frame_len_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.frame_len_label = ttk.Label(frame_len_frame, text=f"{self.frame_length_ms.get():.0f}")
        self.frame_len_label.pack(side=tk.LEFT, padx=5)
        self.frame_length_ms.trace_add('write', lambda *args: self.frame_len_label.config(text=f"{self.frame_length_ms.get():.0f}"))

        lbl = ttk.Label(filter_advanced.body, text="Pas de trame RMS (ms):")
        lbl.grid(row=1, column=0, sticky=tk.W, pady=2)
        add_tooltip(lbl, PARAM_TOOLTIPS['frame_hop'])
        hop_len_frame = ttk.Frame(filter_advanced.body)
        hop_len_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        self.hop_len_scale = ttk.Scale(hop_len_frame, from_=5, to=20,
                 variable=self.hop_length_ms, orient=tk.HORIZONTAL,
                 command=lambda x: [self.invalidate_cache(), self.update_analysis()])
        self.hop_len_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.hop_len_label = ttk.Label(hop_len_frame, text=f"{self.hop_length_ms.get():.0f}")
        self.hop_len_label.pack(side=tk.LEFT, padx=5)
        self.hop_length_ms.trace_add('write', lambda *args: self.hop_len_label.config(text=f"{self.hop_length_ms.get():.0f}"))

        # Résultats
        results_frame = ttk.LabelFrame(control_frame, text="Résultats", padding="5")
        results_frame.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.onset_label = ttk.Label(results_frame, text="Onset: -")
        self.onset_label.grid(row=0, column=0, sticky=tk.W)
        
        self.offset_label = ttk.Label(results_frame, text="Offset: -")
        self.offset_label.grid(row=1, column=0, sticky=tk.W)
        
        self.duration_label = ttk.Label(results_frame, text="Durée: -")
        self.duration_label.grid(row=2, column=0, sticky=tk.W)

        self.manual_correction_label = ttk.Label(results_frame, text="", foreground='#1a7f37')
        self.manual_correction_label.grid(row=3, column=0, sticky=tk.W, pady=(4, 0))

        ttk.Button(results_frame, text="Annuler la correction manuelle",
                  command=self.undo_manual_correction).grid(row=4, column=0, sticky=tk.W, pady=(4, 0))

        # Boutons d'action
        action_frame = ttk.Frame(control_frame)
        action_frame.grid(row=9, column=0, columnspan=2, pady=10)
        
        ttk.Button(action_frame, text="Sauvegarder ce fichier", 
                  command=self.save_current).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Traiter tous", 
                  command=self.process_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Exporter CSV", 
                  command=self.export_csv).pack(side=tk.LEFT, padx=2)
        
        # === SECTION 3: Graphique (right column, full height) ===
        plot_frame = ttk.LabelFrame(main_frame, text="Forme d'onde et enveloppe RMS", padding="5")
        plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S),
                       padx=5, pady=5)
        
        # Matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 7))
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # CHANGE 1: Connect mouse events for draggable markers
        self.canvas.mpl_connect('button_press_event', self._on_marker_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_marker_motion)
        self.canvas.mpl_connect('button_release_event', self._on_marker_release)
        
        # Activer/désactiver les contrôles selon le mode
        self.toggle_adaptive_mode()

        # Navigation clavier — inactive quand le focus est sur un contrôle qui
        # utilise lui-même les flèches gauche/droite (champ texte, curseur, liste).
        self.root.bind('<Left>', lambda e: self._on_arrow_key(self.previous_file))
        self.root.bind('<Right>', lambda e: self._on_arrow_key(self.next_file))

    def _on_arrow_key(self, action):
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, ttk.Entry, ttk.Scale, ttk.Treeview)):
            return
        action()

    # =========================================================
    # CHANGE 1: Draggable marker methods
    # =========================================================

    def _on_marker_press(self, event):
        """Handle mouse press: start dragging onset/offset marker (both modes)."""
        if event.inaxes != self.ax2:
            return
        if event.xdata is None:
            return

        # Tolerance: 5ms in data coordinates
        tol = 0.005

        onset_dist = abs(event.xdata - self.onset_time) if self.onset_time is not None else np.inf
        offset_dist = abs(event.xdata - self.offset_time) if self.offset_time is not None else np.inf

        # Pick the closest marker within tolerance
        if onset_dist < tol and onset_dist <= offset_dist:
            self._dragging = 'onset'
        elif offset_dist < tol:
            self._dragging = 'offset'
        else:
            self._dragging = None

    def _on_marker_motion(self, event):
        """Handle mouse motion: update marker position and cursor affordance."""
        # Cursor affordance: change cursor near a draggable marker (skip when dragging — cursor already set)
        if self._dragging is None:
            if event.inaxes == self.ax2 and event.xdata is not None:
                tol = 0.005
                onset_dist = abs(event.xdata - self.onset_time) if self.onset_time is not None else np.inf
                offset_dist = abs(event.xdata - self.offset_time) if self.offset_time is not None else np.inf
                if onset_dist < tol or offset_dist < tol:
                    self.canvas.get_tk_widget().config(cursor='sb_h_double_arrow')
                else:
                    self.canvas.get_tk_widget().config(cursor='')
            else:
                self.canvas.get_tk_widget().config(cursor='')

        # Dragging logic
        if self._dragging is None:
            return
        if event.inaxes != self.ax2:
            return
        if event.xdata is None:
            return

        # Clamp to audio duration
        audio_duration = len(self.audio_data) / self.sample_rate
        new_time = max(0.0, min(float(event.xdata), audio_duration))
        new_sample = int(new_time * self.sample_rate)

        if self._dragging == 'onset':
            self.onset_time = new_time
            self.onset_sample = new_sample
            # Update vline xdata directly for performance
            if self._onset_vline is not None:
                self._onset_vline.set_xdata([new_time, new_time])
        elif self._dragging == 'offset':
            self.offset_time = new_time
            self.offset_sample = new_sample
            if self._offset_vline is not None:
                self._offset_vline.set_xdata([new_time, new_time])

        # Update labels without full replot
        self.update_result_labels()
        self.canvas.draw_idle()

    def _on_marker_release(self, event):
        """Handle mouse release: stop dragging, record the override, and persist."""
        if self._dragging is not None:
            if self._dragging == 'onset':
                self.manual_onset_override = self.onset_time
            elif self._dragging == 'offset':
                self.manual_offset_override = self.offset_time
            self._dragging = None
            self.update_result_labels()
            self.save_current_params()  # Persist the manual adjustment
            self.save_session()

    def undo_manual_correction(self):
        """Efface la correction manuelle et revient à la détection automatique/seuils actuels."""
        self.manual_onset_override = None
        self.manual_offset_override = None
        self.save_current_params()
        self.update_analysis()
        self.save_session()

    # =========================================================
    # CHANGE 3: Playback cursor methods
    # =========================================================

    def _update_playback_cursor(self):
        """Update the moving playback cursor on both axes at ~20fps."""
        if self.is_playing:
            pos = self.playback_position_sec
            # Create or update playback lines on ax1
            if self._playback_line1 is None:
                self._playback_line1 = self.ax1.axvline(
                    x=pos, color='red', linewidth=1, alpha=0.6, linestyle='-', zorder=10)
            else:
                self._playback_line1.set_xdata([pos, pos])
                self._playback_line1.set_visible(True)
            # Create or update playback line on ax2
            if self._playback_line2 is None:
                self._playback_line2 = self.ax2.axvline(
                    x=pos, color='red', linewidth=1, alpha=0.6, linestyle='-', zorder=10)
            else:
                self._playback_line2.set_xdata([pos, pos])
                self._playback_line2.set_visible(True)
            self.canvas.draw_idle()
            # Reschedule at 50ms (20fps)
            self.root.after(50, self._update_playback_cursor)
        else:
            # Hide lines when not playing
            if self._playback_line1 is not None:
                self._playback_line1.set_visible(False)
            if self._playback_line2 is not None:
                self._playback_line2.set_visible(False)
            self.canvas.draw_idle()

    # =========================================================
    # Existing methods (with targeted changes noted)
    # =========================================================

    def reset_to_default(self):
        """Reset current trial to default parameters"""
        if not self.wav_files:
            return
        
        filename = Path(self.wav_files[self.current_index]).stem
        
        if filename in self.file_specific_params:
            del self.file_specific_params[filename]
        
        # CHANGE 4e: Call undo_cut silently to avoid disruptive messagebox
        if self.is_cut_applied:
            self.undo_cut(silent=True)
        
        self.adaptive_threshold.set(True)
        self.nSD.set(self.default_params['nSD'])
        self.minTH.set(self.default_params['minTH'])
        self.min_duration_ms.set(self.default_params['min_duration_ms'])
        self.noise_window_ms.set(self.default_params['noise_window_ms'])
        self.search_delay_ms.set(self.default_params['search_delay_ms'])
        self.manual_onset_threshold.set(self.default_params['manual_onset_threshold'])
        self.manual_offset_threshold.set(self.default_params['manual_offset_threshold'])
        self.manual_search_delay_ms.set(self.default_params['search_delay_ms'])
        self.noise_reduction.set(self.default_params['noise_reduction'])
        self.highpass_freq.set(self.default_params['highpass_freq'])
        self.offset_min_duration_ms.set(self.default_params['offset_min_duration_ms'])
        
        self.cut_start_ms.set(0)
        self.cut_end_ms.set(0)

        # Effacer toute correction manuelle avant de relancer l'analyse, pour
        # que toggle_adaptive_mode() (qui appelle update_analysis()) ne la réapplique pas.
        self.manual_onset_override = None
        self.manual_offset_override = None

        self.toggle_adaptive_mode()
        # toggle_adaptive_mode already calls invalidate_cache() and update_analysis() internally
        self._refresh_file_list()
        self.save_session()

        messagebox.showinfo("Reset", "Paramètres réinitialisés aux valeurs par défaut pour ce fichier")
    
    def toggle_adaptive_mode(self):
        """Active/désactive les contrôles selon le mode"""
        if self.adaptive_threshold.get():
            self.nsd_scale.config(state='normal')
            self.minth_scale.config(state='normal')
            self.mindur_scale.config(state='normal')
            self.noise_scale.config(state='normal')
            self.delay_scale.config(state='normal')
            self.offset_dur_scale.config(state='normal')
            
            self.manual_onset_scale.config(state='disabled')
            self.manual_offset_scale.config(state='disabled')
            self.manual_delay_scale.config(state='disabled')
            self.manual_onset_dur_scale.config(state='disabled')
            self.manual_offset_dur_scale.config(state='disabled')
        else:
            self.nsd_scale.config(state='disabled')
            self.minth_scale.config(state='disabled')
            self.mindur_scale.config(state='disabled')
            self.noise_scale.config(state='disabled')
            self.delay_scale.config(state='disabled')
            self.offset_dur_scale.config(state='disabled')
            
            self.manual_onset_scale.config(state='normal')
            self.manual_offset_scale.config(state='normal')
            self.manual_delay_scale.config(state='normal')
            self.manual_onset_dur_scale.config(state='normal')
            self.manual_offset_dur_scale.config(state='normal')
        
        # CHANGE 4d: Invalidate cache when switching modes
        self.detection_result = None
        self.invalidate_cache()
        
        if hasattr(self, 'audio_data') and self.audio_data is not None:
            self.update_analysis()
    
    def apply_cut(self):
        """Apply cut to current audio data"""
        if self.audio_data is None:
            messagebox.showwarning("Attention", "Aucun fichier chargé")
            return
        
        try:
            start_ms = float(self.cut_start_ms.get())
            end_ms = float(self.cut_end_ms.get())
        except (tk.TclError, ValueError):
            messagebox.showwarning("Attention", "Veuillez entrer des valeurs numériques valides")
            return
        
        if start_ms < 0:
            messagebox.showwarning("Attention", "Le début de découpe doit être positif")
            return
        
        if end_ms < 0:
            messagebox.showwarning("Attention", "La fin de découpe doit être positive ou 0")
            return
        
        source_data = self.original_audio_data if self.is_cut_applied else self.audio_data
        total_duration_ms = len(source_data) / self.sample_rate * 1000
        
        if start_ms >= total_duration_ms:
            messagebox.showwarning("Attention", f"Le début de découpe ({start_ms:.0f}ms) dépasse la durée du fichier ({total_duration_ms:.0f}ms)")
            return
        
        if end_ms > 0 and end_ms <= start_ms:
            messagebox.showwarning("Attention", "La fin de découpe doit être après le début")
            return
        
        if end_ms > total_duration_ms:
            messagebox.showwarning("Attention", f"La fin de découpe ({end_ms:.0f}ms) dépasse la durée du fichier ({total_duration_ms:.0f}ms)")
            return
        
        if not self.is_cut_applied:
            self.original_audio_data = self.audio_data.copy()
        
        start_sample = int(start_ms / 1000 * self.sample_rate)
        
        if end_ms > 0:
            end_sample = int(end_ms / 1000 * self.sample_rate)
            self.audio_data = source_data[start_sample:end_sample].copy()
            cut_desc = f"Découpe appliquée: {start_ms:.0f}ms à {end_ms:.0f}ms"
        else:
            self.audio_data = source_data[start_sample:].copy()
            cut_desc = f"Découpe appliquée: {start_ms:.0f}ms à la fin"

        self._audio_generation += 1
        self.is_cut_applied = True
        self.cut_status_label.config(text=cut_desc)

        # Une découpe change l'axe des temps: une correction manuelle antérieure
        # ne correspondrait plus à la bonne position.
        self.manual_onset_override = None
        self.manual_offset_override = None

        self.invalidate_cache()
        self.update_analysis()
        self.save_session()

        messagebox.showinfo("Découpe appliquée", cut_desc)
    
    def undo_cut(self, silent=False):
        """Undo the cut and restore original audio.
        
        CHANGE 4e: Accept silent parameter to suppress messagebox when called from reset.
        """
        if not self.is_cut_applied:
            if not silent:
                messagebox.showinfo("Information", "Aucune découpe à annuler")
            return
        
        self.audio_data = self.original_audio_data.copy()
        self._audio_generation += 1
        self.is_cut_applied = False
        self.cut_status_label.config(text="")

        self.manual_onset_override = None
        self.manual_offset_override = None

        self.invalidate_cache()
        self.update_analysis()
        self.save_session()

        # Only show messagebox when not called silently
        if not silent:
            messagebox.showinfo("Annulation", "Découpe annulée, fichier original restauré")
    
    def save_cut_file(self):
        """Save the currently cut audio file"""
        if self.audio_data is None:
            messagebox.showwarning("Attention", "Aucun fichier chargé")
            return
        
        if not self.is_cut_applied:
            messagebox.showinfo("Information", "Aucune découpe appliquée. Le fichier actuel sera sauvegardé.")
        
        if not self.wav_files:
            return
        
        original_filename = Path(self.wav_files[self.current_index]).stem
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            initialfile=f"{original_filename}_cut.wav")
        
        if filepath:
            audio_int16 = (self.audio_data * 32767).astype(np.int16)
            wavfile.write(filepath, self.sample_rate, audio_int16)
            messagebox.showinfo("Sauvegarde", f"Fichier sauvegardé:\n{filepath}")
    
    def toggle_play(self):
        """Joue ou arrête la lecture audio"""
        if self.is_playing:
            self.stop_audio()
        else:
            self.play_audio()
    
    def play_audio(self):
        """CHANGE 3: Play audio on a daemon thread to avoid UI freeze."""
        if self.audio_data is None:
            return

        # CHANGE 7: defensive guard in case the button state gets out of sync
        if not PYAUDIO_AVAILABLE:
            messagebox.showerror("Erreur", "La lecture audio est indisponible (PortAudio/pyaudio non installé).")
            return

        self.is_playing = True
        self.playback_position_sec = 0.0
        self.play_button.config(text="⏸ Pause")

        # Start the playback cursor updater on the main thread
        self._update_playback_cursor()

        def _audio_thread():
            """Audio playback runs in background thread."""
            p = None
            stream = None
            try:
                p = pyaudio.PyAudio()
                audio_int16 = (self.audio_data * 32767).astype(np.int16)

                stream = p.open(format=pyaudio.paInt16,
                              channels=1,
                              rate=self.sample_rate,
                              output=True)

                try:
                    chunk_size = 1024
                    for i in range(0, len(audio_int16), chunk_size):
                        if not self.is_playing:
                            break
                        chunk = audio_int16[i:i + chunk_size]
                        stream.write(chunk.tobytes())
                        # Update playback position safely
                        self.playback_position_sec = (i + chunk_size) / self.sample_rate
                finally:
                    # CHANGE 6: guarantee stream/PyAudio teardown even if write() raises partway through
                    stream.stop_stream()
                    stream.close()

            except Exception as e:
                # CHANGE 4f: Report audio errors from thread via main thread
                self.root.after(0, lambda: messagebox.showerror(
                    "Erreur", f"Erreur de lecture audio: {str(e)}"))
            finally:
                if p is not None:
                    p.terminate()
                # Safely update UI from thread
                self.is_playing = False
                self.root.after(0, lambda: self.play_button.config(text="▶ Jouer"))

        # Launch as daemon thread so it doesn't block app exit
        t = threading.Thread(target=_audio_thread, daemon=True)
        t.start()
        self.audio_thread = t
    
    def stop_audio(self):
        """Arrête la lecture audio"""
        self.is_playing = False
        self.playback_position_sec = 0.0  # CHANGE 3: Reset position
        self.play_button.config(text="▶ Jouer")
    
    def _file_status(self, stem):
        """Statut d'un fichier pour la liste: revu (sauvegardé), personnalisé, ou non revu."""
        if any(r['filename'] == stem for r in self.results):
            return 'reviewed'
        if stem in self.file_specific_params:
            return 'customized'
        return 'unreviewed'

    def _sync_tree_selection(self):
        """Met en surbrillance le fichier courant dans la liste, sans reconstruire la liste."""
        if not hasattr(self, 'file_tree') or not self.wav_files:
            return
        stem = Path(self.wav_files[self.current_index]).stem
        if not self.file_tree.exists(stem):
            return
        self._syncing_tree = True
        try:
            self.file_tree.selection_set(stem)
            self.file_tree.see(stem)
        finally:
            self._syncing_tree = False

    def _refresh_file_list(self):
        """Reconstruit la liste de fichiers avec le statut à jour de chacun."""
        if not hasattr(self, 'file_tree'):
            return
        status_labels = {'reviewed': 'Revu', 'customized': 'Personnalisé', 'unreviewed': 'Non revu'}
        self._syncing_tree = True
        try:
            self.file_tree.delete(*self.file_tree.get_children())
            for f in self.wav_files:
                stem = Path(f).stem
                status = self._file_status(stem)
                self.file_tree.insert('', tk.END, iid=stem, text=f,
                                       values=(status_labels[status],), tags=(status,))
        finally:
            self._syncing_tree = False
        self._sync_tree_selection()

    def _on_file_tree_select(self, event):
        """Navigue vers le fichier sélectionné dans la liste."""
        if self._syncing_tree:
            return
        selection = self.file_tree.selection()
        if not selection:
            return
        stem = selection[0]
        for i, f in enumerate(self.wav_files):
            if Path(f).stem == stem:
                if i != self.current_index:
                    self.stop_audio()
                    self.current_index = i
                    self.load_current_file()
                return

    def _session_file_path(self):
        """Chemin du fichier de session pour le répertoire actuellement ouvert."""
        if not self.directory:
            return None
        return os.path.join(self.directory, SESSION_FILENAME)

    def save_session(self):
        """Enregistre silencieusement les paramètres/résultats dans le répertoire ouvert."""
        path = self._session_file_path()
        if path is None:
            return
        data = {
            'file_specific_params': self.file_specific_params,
            'results': self.results,
        }
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False)
        except OSError as e:
            print(f"Impossible d'enregistrer la session ({path}): {e}")

    def load_session(self):
        """Recharge les paramètres/résultats d'une session précédente, si présente."""
        path = self._session_file_path()
        if path is None or not os.path.exists(path):
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            messagebox.showwarning("Session", f"Impossible de charger la session précédente: {e}")
            return
        self.file_specific_params = data.get('file_specific_params', {})
        self.results = data.get('results', [])

    def select_directory(self):
        directory = filedialog.askdirectory(
            title="Sélectionner le répertoire contenant les fichiers WAV")

        if directory:
            self.directory = directory
            self.dir_label.config(text=directory)

            # wav_files contains only basenames (os.listdir output)
            self.wav_files = sorted([f for f in os.listdir(directory)
                                    if f.lower().endswith('.wav')])
            self.load_session()
            self._refresh_file_list()

            if self.wav_files:
                self.current_index = 0
                self.load_current_file()
                messagebox.showinfo("Succès", 
                    f"{len(self.wav_files)} fichiers WAV trouvés")
            else:
                messagebox.showwarning("Attention", 
                    "Aucun fichier WAV trouvé dans ce répertoire")
    
    def load_current_file(self):
        if not self.wav_files:
            return

        # 4a: wav_files contains basenames; os.path.join adds directory correctly
        filepath = os.path.join(self.directory, self.wav_files[self.current_index])
        filename = Path(self.wav_files[self.current_index]).stem

        try:
            self.sample_rate, audio_data, was_multichannel = core.load_wav(filepath)
        except ValueError as e:
            self._warn("Erreur de lecture", str(e))
            self.audio_data = None
            self.detection_result = None
            self.onset_time = None
            self.offset_time = None
            self.update_result_labels()
            self.file_label.config(
                text=f"Fichier: {self.wav_files[self.current_index]} (échec du chargement) "
                     f"({self.current_index + 1}/{len(self.wav_files)})")
            self.plot_waveform()
            self._sync_tree_selection()
            return

        self.audio_data = audio_data
        self._audio_generation += 1

        if was_multichannel:
            self.channel_warning_label.config(text="⚠ Fichier multicanal : seul le canal 1 est utilisé")
        else:
            self.channel_warning_label.config(text="")

        self.original_audio_data = None
        self.is_cut_applied = False
        self.cut_start_ms.set(0)
        self.cut_end_ms.set(0)
        self.cut_status_label.config(text="")

        file_duration_ms = len(self.audio_data) / self.sample_rate * 1000
        if file_duration_ms < self.noise_window_ms.get():
            self._warn("Attention",
                f"Le fichier ({file_duration_ms:.0f}ms) est plus court que la fenêtre de bruit ({self.noise_window_ms.get():.0f}ms). "
                "La détection adaptative pourrait échouer.")

        if filename in self.file_specific_params:
            params = self.file_specific_params[filename]
            self.adaptive_threshold.set(params['adaptive'])
            if params['adaptive']:
                self.nSD.set(params['nSD'])
                self.minTH.set(params['minTH'])
                self.min_duration_ms.set(params['min_duration_ms'])
                self.noise_window_ms.set(params['noise_window_ms'])
                self.search_delay_ms.set(params['search_delay_ms'])
                self.offset_min_duration_ms.set(params['offset_min_duration_ms'])
            else:
                self.manual_onset_threshold.set(params['manual_onset'])
                self.manual_offset_threshold.set(params['manual_offset'])
                self.manual_search_delay_ms.set(params.get('manual_search_delay_ms', self.default_params['search_delay_ms']))
                self.min_duration_ms.set(params['min_duration_ms'])
                self.offset_min_duration_ms.set(params['offset_min_duration_ms'])
            self.noise_reduction.set(params['noise_reduction'])
            self.highpass_freq.set(params['highpass_freq'])
            self.manual_onset_override = params.get('manual_onset_override')
            self.manual_offset_override = params.get('manual_offset_override')
            self.toggle_adaptive_mode()
        else:
            # CHANGE 2: no saved customization for this file — explicitly reset every
            # relevant Tk variable to defaults so it never inherits whatever was left
            # on-screen from the previously viewed file (mirrors reset_to_default()).
            self.adaptive_threshold.set(True)
            self.nSD.set(self.default_params['nSD'])
            self.minTH.set(self.default_params['minTH'])
            self.min_duration_ms.set(self.default_params['min_duration_ms'])
            self.noise_window_ms.set(self.default_params['noise_window_ms'])
            self.search_delay_ms.set(self.default_params['search_delay_ms'])
            self.manual_onset_threshold.set(self.default_params['manual_onset_threshold'])
            self.manual_offset_threshold.set(self.default_params['manual_offset_threshold'])
            self.manual_search_delay_ms.set(self.default_params['search_delay_ms'])
            self.noise_reduction.set(self.default_params['noise_reduction'])
            self.highpass_freq.set(self.default_params['highpass_freq'])
            self.offset_min_duration_ms.set(self.default_params['offset_min_duration_ms'])
            self.manual_onset_override = None
            self.manual_offset_override = None
            self.toggle_adaptive_mode()

        self.file_label.config(
            text=f"Fichier: {self.wav_files[self.current_index]} ({self.current_index + 1}/{len(self.wav_files)})")
        self._sync_tree_selection()

        self.invalidate_cache()
        self.detection_result = None

        self.update_analysis()
    
    def save_current_params(self):
        """Sauvegarde les paramètres actuels pour ce fichier"""
        if not self.wav_files:
            return
        
        filename = Path(self.wav_files[self.current_index]).stem
        
        self.file_specific_params[filename] = {
            'adaptive': self.adaptive_threshold.get(),
            'nSD': self.nSD.get(),
            'minTH': self.minTH.get(),
            'min_duration_ms': self.min_duration_ms.get(),
            'noise_window_ms': self.noise_window_ms.get(),
            'search_delay_ms': self.search_delay_ms.get(),
            'manual_onset': self.manual_onset_threshold.get(),
            'manual_offset': self.manual_offset_threshold.get(),
            'manual_search_delay_ms': self.manual_search_delay_ms.get(),
            'noise_reduction': self.noise_reduction.get(),
            'highpass_freq': self.highpass_freq.get(),
            'offset_min_duration_ms': self.offset_min_duration_ms.get(),
            'manual_onset_override': self.manual_onset_override,
            'manual_offset_override': self.manual_offset_override
        }
    
    def invalidate_cache(self):
        """Invalidate envelope cache"""
        self.cached_envelope = None
        self.cached_envelope_params = None

    def _warn(self, title, message):
        """Show a warning, batch-mode aware.

        Interactive (not batch): behaves exactly like messagebox.showwarning today.
        Batch mode: never blocks the whole run on a modal messagebox. If the user
        already chose to suppress remaining warnings, just record it silently;
        otherwise show a small non-blocking-to-the-batch Toplevel with an option
        to suppress the rest. Either way the warning is recorded in
        self._batch_warnings so the end-of-batch summary can report a count.
        """
        if not self._batch_mode:
            messagebox.showwarning(title, message)
            return

        self._batch_warnings.append((title, message))

        if self._suppress_batch_warnings:
            return

        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.transient(self.root)
        dialog.grab_set()

        msg_label = tk.Label(dialog, text=message, justify=tk.LEFT, wraplength=380, padx=10, pady=10)
        msg_label.pack()

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        def _on_ok():
            dialog.destroy()

        def _on_suppress():
            self._suppress_batch_warnings = True
            dialog.destroy()

        ttk.Button(button_frame, text="OK", command=_on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Ignorer les avertissements restants pour ce traitement",
                   command=_on_suppress).pack(side=tk.LEFT, padx=5)

        dialog.wait_window()

    def apply_noise_reduction(self, data):
        """Applique un filtre passe-haut pour réduire le bruit de fond"""
        return core.apply_highpass(data, self.sample_rate, self.highpass_freq.get(),
                                    enabled=self.noise_reduction.get())

    def get_envelope_cache_key(self):
        """Generate cache key for envelope computation"""
        return (
            self.noise_reduction.get(),
            self.highpass_freq.get(),
            self.frame_length_ms.get(),
            self.hop_length_ms.get(),
            len(self.audio_data),
            self._audio_generation
        )

    def compute_rms_envelope(self, data):
        """Compute RMS envelope with caching - shared by both modes.

        Thin caching wrapper: on a cache hit, returns the cached arrays; on a
        miss, delegates the actual computation to core.compute_rms_envelope.
        """
        cache_key = self.get_envelope_cache_key()
        if self.cached_envelope is not None and self.cached_envelope_params == cache_key:
            return self.cached_envelope, self.cached_envelope_times

        env_full, full_times = core.compute_rms_envelope(
            data, self.sample_rate, self.frame_length_ms.get(), self.hop_length_ms.get())

        self.cached_envelope = env_full
        self.cached_envelope_times = full_times
        self.cached_envelope_params = cache_key

        return env_full, full_times

    def compute_voice_onset_voat(self, data):
        """Algorithme VOAT pour la détection d'onset ET offset"""
        return core.compute_voice_onset_voat(
            data, self.sample_rate,
            nSD=self.nSD.get(),
            minTH=self.minTH.get(),
            min_duration_ms=self.min_duration_ms.get(),
            noise_window_ms=self.noise_window_ms.get(),
            search_delay_ms=self.search_delay_ms.get(),
            frame_length_ms=self.frame_length_ms.get(),
            hop_length_ms=self.hop_length_ms.get(),
            offset_min_duration_ms=self.offset_min_duration_ms.get()
        )

    def detect_speech_boundaries_manual(self, data):
        """Détection avec seuils manuels - symmetric with adaptive mode"""
        return core.detect_speech_boundaries_manual(
            data, self.sample_rate,
            manual_onset_threshold=self.manual_onset_threshold.get(),
            manual_offset_threshold=self.manual_offset_threshold.get(),
            manual_search_delay_ms=self.manual_search_delay_ms.get(),
            min_duration_ms=self.min_duration_ms.get(),
            offset_min_duration_ms=self.offset_min_duration_ms.get(),
            frame_length_ms=self.frame_length_ms.get(),
            hop_length_ms=self.hop_length_ms.get()
        )
    
    def update_analysis(self):
        """Met à jour l'analyse et le graphique"""
        if self.audio_data is None:
            return
        
        self.save_current_params()
        
        self.filtered_data = self.apply_noise_reduction(self.audio_data)
        
        current_cache_key = self.get_envelope_cache_key()
        if self.cached_envelope_params != current_cache_key:
            self.invalidate_cache()
        
        if self.adaptive_threshold.get():
            self.detection_result = self.compute_voice_onset_voat(self.filtered_data)
            
            if self.detection_result:
                self.onset_time = self.detection_result["t_onset"]
                self.offset_time = self.detection_result["t_offset"]
                self.onset_sample = self.detection_result["onset_sample"]
                self.offset_sample = self.detection_result["offset_sample"]
                self.rms_envelope = self.detection_result["envelope"]
                self.envelope_times = self.detection_result["envelope_times"]
                
                info_text = (f"Bruit: {self.detection_result['noise_mean']:.4f} ± {self.detection_result['noise_std']:.4f} | "
                           f"Seuil nSD: {self.detection_result['threshold_nSD']:.4f} | "
                           f"Seuil minTH: {self.detection_result['threshold_minTH']:.4f} | "
                           f"Final: {self.detection_result['threshold']:.4f}")
                self.threshold_info_label.config(text=info_text)
            else:
                self.onset_time = None
                self.offset_time = None
                self.onset_sample = None
                self.offset_sample = None
                self.rms_envelope = None
                self.envelope_times = None
                self.threshold_info_label.config(text="Aucun onset détecté")
        else:
            onset_idx, offset_idx, envelope, time_axis = self.detect_speech_boundaries_manual(self.filtered_data)
            
            if onset_idx is not None:
                self.onset_time = time_axis[onset_idx]
                self.onset_sample = onset_idx
            else:
                self.onset_time = None
                self.onset_sample = None
            
            if offset_idx is not None:
                self.offset_time = time_axis[offset_idx]
                self.offset_sample = offset_idx
            else:
                self.offset_time = None
                self.offset_sample = None
            
            self.rms_envelope = envelope
            self.envelope_times = time_axis
            self.detection_result = None
            
            info_text = f"Mode manuel | Onset: >{self.manual_onset_threshold.get():.3f} ({self.min_duration_ms.get():.0f}ms) | Offset: <{self.manual_offset_threshold.get():.3f} ({self.offset_min_duration_ms.get():.0f}ms)"
            self.threshold_info_label.config(text=info_text)

        # Une correction manuelle (glisser-déposer) prime toujours sur la détection
        # automatique, quel que soit le mode, jusqu'à annulation explicite.
        if self.manual_onset_override is not None:
            self.onset_time = self.manual_onset_override
            self.onset_sample = int(self.onset_time * self.sample_rate)
        if self.manual_offset_override is not None:
            self.offset_time = self.manual_offset_override
            self.offset_sample = int(self.offset_time * self.sample_rate)

        self.update_result_labels()
        self.plot_waveform()
    
    def update_result_labels(self):
        """Met à jour les labels de résultats"""
        if self.onset_time is not None:
            self.onset_label.config(text=f"Onset: {self.onset_time:.3f} sec ({self.onset_time*1000:.1f} ms)")
        else:
            self.onset_label.config(text="Onset: Non détecté")
        
        if self.offset_time is not None:
            self.offset_label.config(text=f"Offset: {self.offset_time:.3f} sec ({self.offset_time*1000:.1f} ms)")
        else:
            self.offset_label.config(text="Offset: Non détecté")
        
        if self.onset_time is not None and self.offset_time is not None:
            duration = self.offset_time - self.onset_time
            if duration > 0:
                self.duration_label.config(text=f"Durée: {duration:.3f} sec ({duration*1000:.1f} ms)")
            else:
                self.duration_label.config(text=f"Durée: ERREUR (offset avant onset)", foreground='red')
        else:
            self.duration_label.config(text="Durée: -", foreground='black')

        onset_corrected = self.manual_onset_override is not None
        offset_corrected = self.manual_offset_override is not None
        if onset_corrected and offset_corrected:
            self.manual_correction_label.config(text="✓ Onset et offset corrigés manuellement")
        elif onset_corrected:
            self.manual_correction_label.config(text="✓ Onset corrigé manuellement")
        elif offset_corrected:
            self.manual_correction_label.config(text="✓ Offset corrigé manuellement")
        else:
            self.manual_correction_label.config(text="")

    def plot_waveform(self):
        """Affiche la forme d'onde avec les marqueurs"""
        self.ax1.clear()
        self.ax2.clear()

        # CHANGE 3: Axes were cleared — invalidate stored line references so cursor loop recreates them
        self._playback_line1 = None
        self._playback_line2 = None

        # CHANGE 1: Reset vline references since axes are cleared
        self._onset_vline = None
        self._offset_vline = None
        
        if self.audio_data is None:
            self.canvas.draw_idle()
            return
        
        time_axis = np.arange(len(self.audio_data)) / self.sample_rate
        
        self.ax1.plot(time_axis, self.audio_data, linewidth=0.5, alpha=0.7, color='gray')
        title_suffix = " (DÉCOUPÉ)" if self.is_cut_applied else ""
        self.ax1.set_title(f"Signal audio{title_suffix}")
        self.ax1.set_ylabel("Amplitude")
        self.ax1.grid(True, alpha=0.3)
        
        if self.rms_envelope is not None and self.envelope_times is not None:
            self.ax2.plot(self.envelope_times, self.rms_envelope, linewidth=1.5, color='blue', label='Enveloppe RMS')
        
        if self.adaptive_threshold.get() and self.detection_result:
            self.ax2.axhline(y=self.detection_result['threshold'], color='orange', 
                           linestyle='--', linewidth=2, alpha=0.7, label='Seuil hybride')
            self.ax2.axhline(y=self.detection_result['threshold_nSD'], color='purple', 
                           linestyle=':', alpha=0.5, label=f'Seuil nSD ({self.nSD.get():.1f}σ)')
            self.ax2.axhline(y=self.detection_result['threshold_minTH'], color='brown', 
                           linestyle=':', alpha=0.5, label=f'Seuil minTH ({self.minTH.get():.0%})')
            self.ax2.axhline(y=self.detection_result['noise_mean'], color='red', 
                           linestyle=':', alpha=0.3, label='Niveau bruit')
            
            noise_time = self.noise_window_ms.get() / 1000
            self.ax2.axvspan(0, noise_time, alpha=0.15, color='red', label='Fenêtre bruit')
            
            search_delay = self.search_delay_ms.get() / 1000
            if search_delay > 0:
                self.ax2.axvline(x=search_delay, color='gray', linestyle='--', alpha=0.5, label='Début recherche')
        else:
            self.ax2.axhline(y=self.manual_onset_threshold.get(), color='red', 
                           linestyle='--', alpha=0.5, label='Seuil onset')
            self.ax2.axhline(y=self.manual_offset_threshold.get(), color='orange', 
                           linestyle='--', alpha=0.5, label='Seuil offset')
            
            search_delay = self.manual_search_delay_ms.get() / 1000
            if search_delay > 0:
                self.ax2.axvline(x=search_delay, color='gray', linestyle='--', alpha=0.5, label='Début recherche')
        
        # CHANGE 1: Store vline references for direct xdata updates during drag
        if self.onset_time is not None:
            self._onset_vline = self.ax2.axvline(x=self.onset_time, color='green', 
                           linestyle='-', linewidth=2.5, label=f'Onset ({self.onset_time*1000:.1f}ms)', alpha=0.8)
        
        if self.offset_time is not None:
            self._offset_vline = self.ax2.axvline(x=self.offset_time, color='darkblue', 
                           linestyle='-', linewidth=2.5, label=f'Offset ({self.offset_time*1000:.1f}ms)', alpha=0.8)
        
        # Rappel: les marqueurs onset/offset peuvent être glissés pour les corriger,
        # quel que soit le mode de détection.
        if self.onset_time is not None or self.offset_time is not None:
            self.ax2.text(0.98, 0.04, "Glissez les marqueurs pour corriger",
                         transform=self.ax2.transAxes,
                         ha='right', va='bottom', fontsize=7,
                         alpha=0.4, color='black')
        
        mode_text = "VOAT Adaptatif" if self.adaptive_threshold.get() else "Manuel"
        filter_text = " (Filtré)" if self.noise_reduction.get() else ""
        self.ax2.set_title(f"Détection {mode_text}{filter_text}")
        self.ax2.set_xlabel("Temps (sec)")
        self.ax2.set_ylabel("Amplitude RMS")
        self.ax2.legend(loc='upper right', fontsize=8)
        self.ax2.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def save_current(self):
        """Sauvegarde les résultats du fichier actuel"""
        if not self.wav_files:
            return
        
        filename = Path(self.wav_files[self.current_index]).stem
        
        result = {
            'filename': filename,
            'was_cut': self.is_cut_applied,
            'cut_start_ms': self.cut_start_ms.get() if self.is_cut_applied else 0,
            'cut_end_ms': self.cut_end_ms.get() if self.is_cut_applied else 0,
            'onset_time_s': self.onset_time if self.onset_time is not None else np.nan,
            'offset_time_s': self.offset_time if self.offset_time is not None else np.nan,
            'duration_s': (self.offset_time - self.onset_time) if (self.onset_time is not None and self.offset_time is not None) else np.nan,
            'onset_time_ms': self.onset_time * 1000 if self.onset_time is not None else np.nan,
            'offset_time_ms': self.offset_time * 1000 if self.offset_time is not None else np.nan,
            'duration_ms': (self.offset_time - self.onset_time) * 1000 if (self.onset_time is not None and self.offset_time is not None) else np.nan,
            'sample_rate': self.sample_rate,
            'detection_mode': 'adaptive' if self.adaptive_threshold.get() else 'manual',
            'noise_reduction': self.noise_reduction.get(),
            'highpass_freq': self.highpass_freq.get() if self.noise_reduction.get() else np.nan,
            'frame_length_ms': self.frame_length_ms.get(),
            'hop_length_ms': self.hop_length_ms.get(),
            'min_duration_ms': self.min_duration_ms.get(),
            'offset_min_duration_ms': self.offset_min_duration_ms.get(),
            'manually_corrected_onset': self.manual_onset_override is not None,
            'manually_corrected_offset': self.manual_offset_override is not None
        }
        
        if self.adaptive_threshold.get() and self.detection_result:
            result.update({
                'nSD': self.nSD.get(),
                'minTH': self.minTH.get(),
                'noise_window_ms': self.noise_window_ms.get(),
                'search_delay_ms': self.search_delay_ms.get(),
                'noise_mean': self.detection_result['noise_mean'],
                'noise_std': self.detection_result['noise_std'],
                'threshold_nSD': self.detection_result['threshold_nSD'],
                'threshold_minTH': self.detection_result['threshold_minTH'],
                'threshold_final': self.detection_result['threshold'],
                'peak_amplitude': self.detection_result['peak']
            })
        else:
            result.update({
                'manual_onset_threshold': self.manual_onset_threshold.get(),
                'manual_offset_threshold': self.manual_offset_threshold.get(),
                'manual_search_delay_ms': self.manual_search_delay_ms.get()
            })
        
        # 4b: Deduplication runs before append — verified correct
        self.results = [r for r in self.results if r['filename'] != filename]
        self.results.append(result)
        self._refresh_file_list()
        self.save_session()

    def process_all(self):
        """Traite tous les fichiers automatiquement et sauvegarde tous les résultats"""
        if not self.wav_files:
            messagebox.showwarning("Attention", "Aucun fichier à traiter")
            return
        
        response = messagebox.askyesno("Confirmation", 
            f"Traiter automatiquement tous les {len(self.wav_files)} fichiers?\n\n"
            f"• Les fichiers avec paramètres personnalisés conserveront leurs paramètres\n"
            f"• Les autres fichiers utiliseront les paramètres adaptatifs par défaut\n"
            f"• Tous les résultats seront sauvegardés automatiquement")
        
        if not response:
            return

        self._batch_mode = True
        self._batch_warnings = []
        self._suppress_batch_warnings = False

        progress_window = tk.Toplevel(self.root)
        progress_window.title("Traitement en cours")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()

        status_label = tk.Label(progress_window, text="Initialisation...", font=("Arial", 10))
        status_label.pack(pady=10)

        progress_bar = ttk.Progressbar(progress_window, length=350,
                                    mode='determinate', maximum=len(self.wav_files))
        progress_bar.pack(pady=10)

        counter_label = tk.Label(progress_window, text="0 / " + str(len(self.wav_files)), font=("Arial", 9))
        counter_label.pack(pady=5)

        self.results = []

        def process_next(index):
            if index < len(self.wav_files):
                # 4a: os.path.basename on wav_files[index] is already a basename
                filename = os.path.basename(self.wav_files[index])
                status_label.config(text=f"Traitement: {filename}")
                counter_label.config(text=f"{index + 1} / {len(self.wav_files)}")

                self.current_index = index
                try:
                    self.load_current_file()
                    if self.audio_data is not None:
                        self.save_current()
                except Exception as e:
                    self._warn("Erreur de traitement", f"Impossible de traiter {filename}: {e}")

                progress_bar['value'] = index + 1
                progress_window.update()

                self.root.after(10, lambda: process_next(index + 1))
            else:
                progress_window.destroy()
                self._batch_mode = False
                self._refresh_file_list()
                self.save_session()
                message = (f"Traitement terminé!\n\n"
                          f"• {len(self.results)} fichiers traités\n"
                          f"• Résultats sauvegardés en mémoire\n\n"
                          f"Cliquez sur 'Exporter CSV' pour sauvegarder les résultats")
                if self._batch_warnings:
                    message += f"\n\n⚠ {len(self._batch_warnings)} avertissement(s) survenu(s) pendant le traitement"
                messagebox.showinfo("Terminé", message)

        process_next(0)

    def export_csv(self):
        """Exporte les résultats en CSV"""
        if not self.results:
            messagebox.showwarning("Attention", "Aucun résultat à exporter")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="voice_key_results.csv")
        
        if filepath:
            df = pd.DataFrame(self.results)
            # 4g: Missing keys produce NaN columns — pandas handles this correctly, no crash
            priority_cols = ['filename', 'was_cut', 'onset_time_ms', 'offset_time_ms', 'duration_ms', 'detection_mode']
            other_cols = [col for col in df.columns if col not in priority_cols]
            df = df[priority_cols + other_cols]
            df.to_csv(filepath, index=False)
            messagebox.showinfo("Export", f"Résultats exportés vers:\n{filepath}\n\n{len(df)} fichiers exportés")
    
    def previous_file(self):
        """Fichier précédent"""
        if self.wav_files and self.current_index > 0:
            self.stop_audio()
            self.current_index -= 1
            self.load_current_file()
    
    def next_file(self):
        """Fichier suivant"""
        if self.wav_files and self.current_index < len(self.wav_files) - 1:
            self.stop_audio()
            self.current_index += 1
            self.load_current_file()

if __name__ == "__main__":
    root = tk.Tk()
    sv_ttk.set_theme("light")
    app = VoiceKeyAnalyzer(root)
    root.mainloop()