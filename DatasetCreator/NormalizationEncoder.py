import torch
import numpy as np
import pickle
import logging
from typing import Optional, Dict, Any, List, Union, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class NormalizationEncoder:
    """
    Encoder robusto e ottimizzato per dataset di scacchi (PyG Data).

    Caratteristiche:
    - Normalizzazione rating/clock con StandardScaler o RobustScaler.
    - Gestione di valori mancanti (rating o clock assenti).
    - Clipping opzionale dei valori normalizzati entro un intervallo.
    - Salvataggio/caricamento dei parametri su disco.
    - OTTIMIZZAZIONI:
      1. Tensori scalari -> float32 per valori continui, long per indici.
      2. x -> float32 + rimozione colonne costanti.
      3. edge_index -> int32, edge_attr -> float32.
      4. Rimozione/gestione efficiente di stringhe superflue.
      5. Deduplicazione: elimina best_move_idx se uguale a y.
      6. Aggiunta automatica di value_target se mancante.
    """

    def __init__(
        self,
        scaler_type: str = "standard",           # "standard" o "robust"
        clip_range: Optional[Tuple[float, float]] = (-5.0, 5.0),
        handle_missing: str = "default",         # "default", "skip", "raise"
        default_rating: float = 1500.0,
        default_clock: float = 30.0,
        copy: bool = True,
        remove_constant_x_cols: bool = True,     # Rimuove colonne costanti da x
        remove_strings: bool = True,             # Rimuove stringhe non necessarie
        keep_puzzle_id: bool = True,             # Tiene puzzle_id per debug
    ):
        """
        Args:
            scaler_type: "standard" (StandardScaler) o "robust" (RobustScaler).
            clip_range: intervallo di clipping per valori normalizzati (es. (-5,5)).
            handle_missing: "default" -> usa default_rating/clock;
                            "skip" -> ignora il campione;
                            "raise" -> solleva eccezione.
            default_rating, default_clock: valori di default per missing.
            copy: se True, i tensori originali non vengono modificati.
            remove_constant_x_cols: rimuove colonne costanti da x (risparmio memoria).
            remove_strings: rimuove campi stringa non necessari.
            keep_puzzle_id: mantiene puzzle_id se remove_strings=True.
        """
        self.scaler_type = scaler_type
        self.clip_range = clip_range
        self.handle_missing = handle_missing
        self.default_rating = default_rating
        self.default_clock = default_clock
        self.copy = copy
        self.remove_constant_x_cols = remove_constant_x_cols
        self.remove_strings = remove_strings
        self.keep_puzzle_id = keep_puzzle_id

        # Inizializza lo scaler scelto
        if scaler_type == "standard":
            self.rating_scaler = StandardScaler()
            self.clock_scaler = StandardScaler()
        elif scaler_type == "robust":
            self.rating_scaler = RobustScaler()
            self.clock_scaler = RobustScaler()
        else:
            raise ValueError(f"scaler_type deve essere 'standard' o 'robust', ricevuto '{scaler_type}'")

        self._is_fitted = False
        self._constant_columns = None  # Salviamo quali colonne di x sono costanti

    # ============================
    # 1. METODI DI NORMALIZZAZIONE
    # ============================

    def _extract_rating_and_clock(self, data: Data) -> Tuple[Optional[float], Optional[float]]:
        """Estrae rating e clock da un oggetto Data, applicando le policy di missing."""
        # Rating
        if hasattr(data, 'rating') and data.rating is not None:
            rating = data.rating
            if isinstance(rating, torch.Tensor):
                if rating.numel() == 1:
                    rating = rating.item()
                else:
                    rating = float(rating.mean().item())
            else:
                rating = float(rating)
        else:
            if self.handle_missing == "raise":
                raise ValueError(f"Campo 'rating' mancante in {data} e handle_missing='raise'")
            elif self.handle_missing == "default":
                rating = self.default_rating
                logger.debug(f"Rating mancante, usato default {self.default_rating}")
            else:  # skip
                rating = None

        # Clock
        if hasattr(data, 'clock_seconds') and data.clock_seconds is not None:
            clock = data.clock_seconds
            if isinstance(clock, torch.Tensor):
                if clock.numel() == 1:
                    clock = clock.item()
                else:
                    clock = float(clock.mean().item())
            else:
                clock = float(clock)
        else:
            if self.handle_missing == "raise":
                raise ValueError(f"Campo 'clock_seconds' mancante in {data} e handle_missing='raise'")
            elif self.handle_missing == "default":
                clock = self.default_clock
                logger.debug(f"Clock mancante, usato default {self.default_clock}")
            else:
                clock = None

        return rating, clock

    def fit(self, data_list: List[Data]) -> 'NormalizationEncoder':
        """
        Addestra gli scaler su una lista di oggetti Data.
        Estrae rating e clock da TUTTI i campioni (gestendo i missing).
        """
        ratings = []
        clocks = []

        for item in data_list:
            r, c = self._extract_rating_and_clock(item)
            if r is not None:
                ratings.append(r)
            if c is not None:
                clocks.append(c)

        if not ratings:
            raise ValueError("Nessun rating valido trovato per l'addestramento dello scaler.")
        if not clocks:
            logger.warning("Nessun clock valido trovato; lo scaler per clock sarà addestrato su array di soli zeri.")
            clocks = [0.0]

        ratings_2d = np.array(ratings).reshape(-1, 1)
        clocks_2d = np.array(clocks).reshape(-1, 1)

        self.rating_scaler.fit(ratings_2d)
        self.clock_scaler.fit(clocks_2d)
        self._is_fitted = True

        logger.info(f"✅ Scaler '{self.scaler_type}' addestrati su {len(ratings)} rating e {len(clocks)} clock.")
        if self.scaler_type == "standard":
            logger.info(f"   Rating: media={self.rating_scaler.mean_[0]:.1f}, std={self.rating_scaler.scale_[0]:.1f}")
            logger.info(f"   Clock:  media={self.clock_scaler.mean_[0]:.1f}, std={self.clock_scaler.scale_[0]:.1f}")
        else:
            logger.info(f"   Rating: mediana={self.rating_scaler.center_[0]:.1f}, IQR={self.rating_scaler.scale_[0]:.1f}")
            logger.info(f"   Clock:  mediana={self.clock_scaler.center_[0]:.1f}, IQR={self.clock_scaler.scale_[0]:.1f}")
        return self

    def transform(self, data: Data) -> Data:
        """
        Normalizza rating e clock in un singolo oggetto Data.
        Applica anche OTTIMIZZAZIONI sui tipi di dato.
        """
        if not self._is_fitted:
            raise RuntimeError("Chiamare prima fit() con i dati di training.")

        # Copia se richiesto
        if self.copy:
            data = data.clone() if hasattr(data, 'clone') else data.__class__(**data.to_dict())

        # ---- NORMALIZZAZIONE RATING ----
        if hasattr(data, 'rating') and data.rating is not None:
            rating_val = self._extract_rating_and_clock(data)[0]
            if rating_val is not None:
                norm_val = self.rating_scaler.transform([[rating_val]])[0][0]
                if self.clip_range is not None:
                    norm_val = np.clip(norm_val, self.clip_range[0], self.clip_range[1])
                data.rating = torch.tensor([norm_val], dtype=torch.float32)

        # ---- NORMALIZZAZIONE CLOCK ----
        if hasattr(data, 'clock_seconds') and data.clock_seconds is not None:
            clock_val = self._extract_rating_and_clock(data)[1]
            if clock_val is not None:
                norm_val = self.clock_scaler.transform([[clock_val]])[0][0]
                if self.clip_range is not None:
                    norm_val = np.clip(norm_val, self.clip_range[0], self.clip_range[1])
                data.clock_seconds = torch.tensor([norm_val], dtype=torch.float32)

        # ---- APPLICA OTTIMIZZAZIONI ----
        data = self._optimize_data(data)

        return data

    def fit_transform(self, data_list: List[Data]) -> List[Data]:
        """Addestra e trasforma immediatamente una lista di dati."""
        self.fit(data_list)
        return [self.transform(item) for item in data_list]

    # ============================
    # 2. OTTIMIZZAZIONI DEI DATI
    # ============================

    def _optimize_data(self, data: Data) -> Data:
        """
        Applica TUTTE le 6 ottimizzazioni a un oggetto Data.
        """
        # ---- OTTIMIZZAZIONE 1: Tensori scalari ----
        scalar_mappings = {
            'rating': (torch.float32, 1),
            'clock_seconds': (torch.float32, 1),
            'clock_is_real': (torch.long, 1),
            'ply': (torch.long, 1),
            'mate_n': (torch.long, 1),
            'y': (torch.long, 1),
            'best_move_idx': (torch.long, 1),
            'value_target': (torch.float32, 1),
        }

        for attr, (dtype, shape) in scalar_mappings.items():
            if hasattr(data, attr):
                val = getattr(data, attr)
                if isinstance(val, torch.Tensor):
                    if val.numel() == 1:
                        # Prendi il valore scalare e ricrea il tensore
                        setattr(data, attr, torch.tensor([val.item()], dtype=dtype))
                    else:
                        # Converti dtype mantenendo la forma
                        setattr(data, attr, val.to(dtype))

        # ---- OTTIMIZZAZIONE 2: Feature dei nodi (x) ----
        if hasattr(data, 'x') and isinstance(data.x, torch.Tensor):
            # Converti a float32
            data.x = data.x.to(torch.float32)

            # Rimuovi colonne costanti se richiesto
            if self.remove_constant_x_cols and data.x.numel() > 0:
                # Se abbiamo già calcolato le colonne costanti, usale
                if self._constant_columns is not None:
                    keep_cols = [i for i in range(data.x.size(1)) if i not in self._constant_columns]
                    if len(keep_cols) < data.x.size(1):
                        data.x = data.x[:, keep_cols]
                else:
                    # Calcoliamo le colonne costanti (solo se il batch è già stato collazionato)
                    # In process() singolo, non possiamo farlo perché abbiamo un solo grafo.
                    # Verrà fatto dopo il collate (vedi _detect_constant_columns)
                    pass

        # ---- OTTIMIZZAZIONE 3: Archi ----
        if hasattr(data, 'edge_index') and isinstance(data.edge_index, torch.Tensor):
            data.edge_index = data.edge_index.to(torch.int32)

        if hasattr(data, 'edge_attr') and isinstance(data.edge_attr, torch.Tensor):
            # Se edge_attr è piatto [N] lo lasciamo così, ma convertiamo a float32
            if data.edge_attr.dim() == 1:
                data.edge_attr = data.edge_attr.to(torch.float32)
            elif data.edge_attr.dim() == 2:
                data.edge_attr = data.edge_attr.to(torch.float32)

        # ---- OTTIMIZZAZIONE 4: Stringhe ----
        if self.remove_strings:
            # Lista di attributi stringa da rimuovere se vuoti
            string_attrs = ['fen', 'best_move_uci', 'game_id', 'clock_source', 'source']
            for attr in string_attrs:
                if hasattr(data, attr):
                    val = getattr(data, attr)
                    if isinstance(val, str) and (val == "" or val is None or val == " "):
                        delattr(data, attr)

            # Gestione puzzle_id: tienilo solo se richiesto e se non è vuoto
            if hasattr(data, 'puzzle_id'):
                if not self.keep_puzzle_id:
                    delattr(data, 'puzzle_id')
                else:
                    # Se è una stringa vuota o None, elimina
                    val = getattr(data, 'puzzle_id')
                    if isinstance(val, str) and (val == "" or val is None):
                        delattr(data, 'puzzle_id')

            # Se esiste 'problem_id' e 'puzzle_id', tieni solo 'puzzle_id'
            if hasattr(data, 'problem_id') and hasattr(data, 'puzzle_id'):
                delattr(data, 'problem_id')

        # ---- OTTIMIZZAZIONE 5: Deduplicazione best_move_idx / y ----
        if hasattr(data, 'best_move_idx') and hasattr(data, 'y'):
            # Controlla se sono uguali (stesso valore)
            if isinstance(data.best_move_idx, torch.Tensor) and isinstance(data.y, torch.Tensor):
                if data.best_move_idx.numel() == 1 and data.y.numel() == 1:
                    if data.best_move_idx.item() == data.y.item():
                        delattr(data, 'best_move_idx')
                elif torch.equal(data.best_move_idx, data.y):
                    delattr(data, 'best_move_idx')
            elif data.best_move_idx == data.y:
                delattr(data, 'best_move_idx')

        # Se esiste solo best_move_idx e non y, rinominiamo
        if hasattr(data, 'best_move_idx') and not hasattr(data, 'y'):
            data.y = data.best_move_idx
            delattr(data, 'best_move_idx')

        # ---- OTTIMIZZAZIONE 6: value_target (aggiungi se manca) ----
        if not hasattr(data, 'value_target') or data.value_target is None:
            # Se è puzzle, è +1; se è game, dovrebbe già esserci
            if hasattr(data, 'source') and data.source == 'puzzle':
                data.value_target = torch.tensor([1.0], dtype=torch.float32)
            else:
                # Fallback: 0 (patta)
                data.value_target = torch.tensor([0.0], dtype=torch.float32)

        return data

    def detect_constant_columns(self, data_list: List[Data], x_attr: str = 'x') -> List[int]:
        """
        Scansiona una lista di Data objects e rileva quali colonne di x sono costanti
        in TUTTI i campioni. Salva il risultato in self._constant_columns.
        """
        if not data_list:
            return []

        # Raccogli tutte le x
        x_tensors = []
        for d in data_list:
            if hasattr(d, x_attr) and isinstance(d.x, torch.Tensor):
                x_tensors.append(d.x)

        if not x_tensors:
            return []

        # Assumiamo che tutte le x abbiano la stessa shape
        # (devono essere già state collazionate o essere tutte della stessa dimensione)
        first_shape = x_tensors[0].shape
        if len(first_shape) != 2:
            # Se x è 3D (es. batch), la appiattiamo
            if len(first_shape) == 3:
                x_tensors = [x.view(x.size(0), -1) for x in x_tensors]
            else:
                return []

        # Calcoliamo la varianza per ogni colonna su TUTTI i campioni
        # Per farlo, concateniamo lungo la dimensione 0
        x_cat = torch.cat(x_tensors, dim=0)  # shape [N_total, num_cols]
        variances = torch.var(x_cat, dim=0, unbiased=False)  # varianza per colonna

        # Trova colonne con varianza < 1e-6 (costanti)
        constant_cols = (variances < 1e-6).nonzero(as_tuple=True)[0].tolist()

        self._constant_columns = constant_cols
        logger.info(f"🔍 Rilevate {len(constant_cols)} colonne costanti in x: {constant_cols}")
        return constant_cols

    # ============================
    # 3. SALVATAGGIO / CARICAMENTO
    # ============================

    def save(self, filepath: str):
        """Salva i parametri dello scaler su disco (formato pickle)."""
        if not self._is_fitted:
            raise RuntimeError("Impossibile salvare: lo scaler non è stato addestrato.")
        params = {
            "scaler_type": self.scaler_type,
            "clip_range": self.clip_range,
            "handle_missing": self.handle_missing,
            "default_rating": self.default_rating,
            "default_clock": self.default_clock,
            "copy": self.copy,
            "remove_constant_x_cols": self.remove_constant_x_cols,
            "remove_strings": self.remove_strings,
            "keep_puzzle_id": self.keep_puzzle_id,
            "rating_scaler": self.rating_scaler,
            "clock_scaler": self.clock_scaler,
            "constant_columns": self._constant_columns,
            "_is_fitted": self._is_fitted,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        logger.info(f"Parametri dello scaler salvati in {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'NormalizationEncoder':
        """Carica uno scaler precedentemente salvato."""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        encoder = cls(
            scaler_type=params["scaler_type"],
            clip_range=params["clip_range"],
            handle_missing=params["handle_missing"],
            default_rating=params["default_rating"],
            default_clock=params["default_clock"],
            copy=params["copy"],
            remove_constant_x_cols=params.get("remove_constant_x_cols", True),
            remove_strings=params.get("remove_strings", True),
            keep_puzzle_id=params.get("keep_puzzle_id", True),
        )
        encoder.rating_scaler = params["rating_scaler"]
        encoder.clock_scaler = params["clock_scaler"]
        encoder._constant_columns = params.get("constant_columns")
        encoder._is_fitted = params["_is_fitted"]
        logger.info(f"Parametri dello scaler caricati da {filepath}")
        return encoder

    # ============================
    # 4. INVERSIONE (per debug)
    # ============================

    def inverse_transform_rating(self, normalized_rating: Union[float, np.ndarray]) -> np.ndarray:
        """Riporta un rating normalizzato alla scala originale."""
        if not self._is_fitted:
            raise RuntimeError("Lo scaler non è stato addestrato.")
        return self.rating_scaler.inverse_transform(np.array(normalized_rating).reshape(-1, 1)).flatten()

    def inverse_transform_clock(self, normalized_clock: Union[float, np.ndarray]) -> np.ndarray:
        """Riporta un clock normalizzato alla scala originale."""
        if not self._is_fitted:
            raise RuntimeError("Lo scaler non è stato addestrato.")
        return self.clock_scaler.inverse_transform(np.array(normalized_clock).reshape(-1, 1)).flatten()