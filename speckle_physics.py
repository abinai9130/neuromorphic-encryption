"""
Speckle Correlography & Event Simulation Engine
================================================
Implements the forward physics model from the paper:
  1. Coherent scattering: plaintext → speckle pattern  (Eq. 1)
  2. Event generation:    speckle motion → event stream  (Eq. 5-6)
  3. Autocorrelation:     accumulated events → Γ_events  (Eq. 7)

Reference: Zhu et al., "Neuromorphic encryption: combining speckle
correlography and event data for enhanced security", APN 2024.
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift


def generate_speckle_pattern(image: np.ndarray, random_phase_seed: int = None) -> np.ndarray:
    """
    Simulate coherent scattering to produce a speckle pattern from a plaintext image.
    
    I(u) = |F[f_n(v)]|^2   (Eq. 1)
    where f_n(v) = |f_0(v) * exp(j * phi_n(v))|
    
    Args:
        image: 2D numpy array (grayscale, float [0,1]), the plaintext.
        random_phase_seed: Seed for reproducible random phase (acts as part of the "key").
    
    Returns:
        speckle: 2D numpy array, the speckle intensity pattern (ciphertext basis).
    """
    if random_phase_seed is not None:
        rng = np.random.RandomState(random_phase_seed)
    else:
        rng = np.random.RandomState()

    h, w = image.shape
    # Random phase from the diffuse surface (key component)
    phi = rng.uniform(0, 2 * np.pi, (h, w))
    
    # Object field modulated by random phase
    f_n = image.astype(np.float64) * np.exp(1j * phi)
    
    # Far-field propagation (Fourier transform)
    F_n = fft2(f_n)
    
    # Speckle intensity pattern
    speckle = np.abs(F_n) ** 2
    
    return speckle


def shift_speckle(speckle: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift a speckle pattern by (dx, dy) pixels to simulate object motion."""
    return np.roll(np.roll(speckle, dx, axis=1), dy, axis=0)


def simulate_events(speckle_frames: list, threshold: float = 0.2) -> np.ndarray:
    """
    Simulate event camera output from a sequence of shifted speckle patterns.
    
    An event e_k = (r_k, t_k, p_k) is triggered when:
        ΔL(r_k, t_k) = L(r_{k+1}, t_{k+1}) - L(r_k, t_k) = p_k * T   (Eq. 5)
    
    Accumulated events:
        I_events = Σ p_k * T * δ(x - x_k, y - y_k)   (Eq. 6)
    
    Args:
        speckle_frames: List of 2D speckle patterns (shifted versions).
        threshold: Temporal contrast threshold T.
    
    Returns:
        accumulated_events: 2D array of accumulated event intensities.
    """
    h, w = speckle_frames[0].shape
    accumulated = np.zeros((h, w), dtype=np.float64)
    
    for i in range(1, len(speckle_frames)):
        prev = speckle_frames[i - 1]
        curr = speckle_frames[i]
        
        # Avoid log(0) by adding small epsilon
        eps = 1e-10
        log_prev = np.log(prev + eps)
        log_curr = np.log(curr + eps)
        
        # Intensity change in log domain
        delta_L = log_curr - log_prev
        
        # Generate events where change exceeds threshold
        pos_events = delta_L > threshold
        neg_events = delta_L < -threshold
        
        # Accumulate with polarity
        accumulated[pos_events] += threshold
        accumulated[neg_events] -= threshold
    
    return accumulated


def compute_autocorrelation(accumulated_events: np.ndarray) -> np.ndarray:
    """
    Compute the autocorrelation of accumulated event data.
    
    Γ_events = F(I_events)   (Eq. 7, interpreted as power spectral density)
    
    The autocorrelation is computed via the Wiener-Khinchin theorem:
        autocorrelation = IFFT(|FFT(I_events)|^2)
    
    Args:
        accumulated_events: 2D accumulated event image.
    
    Returns:
        autocorrelation: 2D autocorrelation map (centered).
    """
    F = fft2(accumulated_events)
    power_spectrum = np.abs(F) ** 2
    autocorr = np.real(ifft2(power_spectrum))
    autocorr = fftshift(autocorr)
    
    # Normalize to [0, 1]
    autocorr = autocorr - autocorr.min()
    if autocorr.max() > 0:
        autocorr = autocorr / autocorr.max()
    
    return autocorr


def encrypt_image(image: np.ndarray, key_seed: int = 42, 
                  num_shifts: int = 3, shift_pixels: int = 2,
                  event_threshold: float = 0.2) -> tuple:
    """
    Full encryption pipeline: plaintext image → autocorrelation ciphertext.
    
    This simulates the entire neuromorphic encryption process:
      1. Generate speckle pattern from plaintext (coherent scattering)
      2. Create shifted speckle frames (simulate object motion)
      3. Generate and accumulate events (event camera simulation)
      4. Compute autocorrelation (preprocessing for decryption)
    
    Args:
        image: 2D numpy array (grayscale, float [0,1]).
        key_seed: Random seed (encryption key).
        num_shifts: Number of shifted frames (motion simulation).
        shift_pixels: Pixels to shift per frame.
        event_threshold: Event camera threshold.
    
    Returns:
        autocorrelation: The encrypted representation (input to U-Net).
        intermediates: Dict with speckle, events, etc. for visualization.
    """
    # Step 1: Generate speckle pattern
    speckle = generate_speckle_pattern(image, random_phase_seed=key_seed)
    
    # Step 2: Create shifted frames (simulating object motion on translation stage)
    frames = [speckle]
    for i in range(1, num_shifts + 1):
        shifted = shift_speckle(speckle, dx=i * shift_pixels, dy=0)
        frames.append(shifted)
    
    # Step 3: Generate accumulated events
    accumulated_events = simulate_events(frames, threshold=event_threshold)
    
    # Step 4: Compute autocorrelation
    autocorr = compute_autocorrelation(accumulated_events)
    
    intermediates = {
        'speckle': speckle,
        'accumulated_events': accumulated_events,
        'autocorrelation': autocorr,
        'key_seed': key_seed,
        'num_shifts': num_shifts,
        'shift_pixels': shift_pixels,
        'event_threshold': event_threshold,
    }
    
    return autocorr, intermediates


def generate_training_pair(image: np.ndarray, key_seed: int = None,
                           num_shifts: int = 3, shift_pixels: int = 2,
                           target_size: int = 256) -> tuple:
    """
    Generate a single (autocorrelation, ground_truth) training pair.
    
    Uses different random scattering conditions each time for data augmentation,
    mimicking the paper's approach of using 9 scattering transport modes.
    
    Args:
        image: Input image (grayscale, float [0,1]).
        key_seed: If None, random seed is used (different scattering each time).
        num_shifts: Number of motion shifts.
        shift_pixels: Pixels per shift.
        target_size: Output image size.
    
    Returns:
        autocorr: Autocorrelation image (model input).
        ground_truth: Original image (model target).
    """
    from PIL import Image
    
    # Resize image to target size
    if image.shape[0] != target_size or image.shape[1] != target_size:
        img_pil = Image.fromarray((image * 255).astype(np.uint8))
        img_pil = img_pil.resize((target_size, target_size), Image.BILINEAR)
        image = np.array(img_pil).astype(np.float64) / 255.0
    
    if key_seed is None:
        key_seed = np.random.randint(0, 2**31)
    
    # Encrypt
    autocorr, _ = encrypt_image(
        image, 
        key_seed=key_seed,
        num_shifts=num_shifts,
        shift_pixels=shift_pixels
    )
    
    return autocorr.astype(np.float32), image.astype(np.float32)
