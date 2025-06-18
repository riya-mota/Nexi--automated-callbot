# Quick Fix for F5-TTS Voice Generation
# Run this script after the model has loaded successfully

import torch
import torchaudio
import torchaudio.transforms as T
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_voice_generation():
    """Quick voice generation with parameter fixes"""
    
    # Configuration
    ref_audio_path = "./reference.wav"
    ref_text = "You just tell me what works, date, time, and your preferred location. We'll arrange everything for you. No pressure. Seriously. I'm just here to help you"
    target_text = "Hey Welcome to Nexi, I'm your personal assistant to help you with all your jaguar land rover queries. We have a wide range of car from SUV to sports, tell me what should i help you with."
    output_path = "./generated_voice.wav"
    
    # Check if reference audio exists
    if not Path(ref_audio_path).exists():
        logger.error(f"Reference audio not found: {ref_audio_path}")
        return False
    
    try:
        # Method 1: Try F5-TTS API (simplest)
        logger.info("Trying F5-TTS API approach...")
        
        from f5_tts.api import F5TTS
        
        f5tts = F5TTS()
        
        # Generate audio
        audio = f5tts.infer(
            ref_file=ref_audio_path,
            ref_text=ref_text,
            gen_text=target_text,
            cross_fade_duration=0.15,
            speed=1.0
        )
        
        # Handle audio output
        if isinstance(audio, tuple):
            audio_tensor = audio[0]
            sample_rate = audio[1] if len(audio) > 1 else 24000
        else:
            audio_tensor = audio
            sample_rate = 24000
        
        # Convert to proper format
        if hasattr(audio_tensor, 'numpy'):
            audio_np = audio_tensor.numpy()
        elif hasattr(audio_tensor, 'detach'):
            audio_np = audio_tensor.detach().cpu().numpy()
        else:
            audio_np = audio_tensor
        
        # Ensure correct shape
        if len(audio_np.shape) > 1:
            if audio_np.shape[0] > audio_np.shape[1]:
                audio_np = audio_np[0]  # Take first channel
            else:
                audio_np = audio_np.flatten()
        
        # Save using soundfile
        import soundfile as sf
        sf.write(output_path, audio_np, sample_rate)
        
        logger.info(f"✅ Voice generated successfully using API: {output_path}")
        return True
        
    except Exception as api_error:
        logger.error(f"API method failed: {api_error}")
    
    # Method 2: Manual preprocessing approach
    try:
        logger.info("Trying manual preprocessing approach...")
        
        # Import required modules
        from f5_tts.model import DiT
        from f5_tts.infer.utils_infer import load_checkpoint, infer_process
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model (assuming it's already cached)
        model_cfg = {
            'dim': 1024, 'depth': 22, 'heads': 16,
            'ff_mult': 2, 'text_dim': 512, 'conv_layers': 4
        }
        
        model = DiT(**model_cfg)
        
        # Try to load from cache or download
        try:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(
                repo_id="SWivid/F5-TTS",
                filename="F5TTS_Base/model_1200000.safetensors"
            )
            
            from safetensors.torch import load_file
            state_dict = load_file(model_path)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
            
        except Exception as model_error:
            logger.error(f"Model loading failed: {model_error}")
            return False
        
        # Manual audio preprocessing
        waveform, sample_rate = torchaudio.load(ref_audio_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 24kHz
        if sample_rate != 24000:
            resampler = T.Resample(sample_rate, 24000)
            waveform = resampler(waveform)
        
        ref_audio = waveform.to(device)
        
        # Try inference with different parameter combinations
        try:
            final_wave, final_sample_rate, _ = infer_process(
                ref_audio=ref_audio,
                ref_text=ref_text,
                gen_text=target_text,
                model=model,
                cross_fade_duration=0.15,
                speed=1.0
            )
        except TypeError:
            # Try without optional parameters
            final_wave, final_sample_rate, _ = infer_process(
                ref_audio, ref_text, target_text, model
            )
        
        # Handle output format
        if isinstance(final_wave, tuple):
            final_wave = final_wave[0]
        
        if len(final_wave.shape) == 1:
            final_wave = final_wave.unsqueeze(0)
        elif len(final_wave.shape) == 3:
            final_wave = final_wave.squeeze(0)
        
        # Save audio
        torchaudio.save(output_path, final_wave.cpu(), final_sample_rate)
        
        logger.info(f"✅ Voice generated successfully: {output_path}")
        return True
        
    except Exception as manual_error:
        logger.error(f"Manual method failed: {manual_error}")
    
    return False

def main():
    """Main execution"""
    logger.info("🚀 Quick Fix for F5-TTS Voice Generation")
    
    if quick_voice_generation():
        logger.info("🎉 Success! Check your generated_voice.wav file")
    else:
        logger.error("❌ All methods failed")
        logger.info("\nTroubleshooting:")
        logger.info("1. Ensure reference_audio.wav exists")
        logger.info("2. Check if the audio file is valid (not corrupted)")
        logger.info("3. Try with a shorter reference audio (10-30 seconds)")
        logger.info("4. Make sure you have enough free disk space")

if __name__ == "__main__":
    main()